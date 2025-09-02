#!/bin/bash
clear
set -e
# ==============================================
# 常量定义
# ==============================================
readonly DEBUGGABLE=1
readonly REQUIRED_UBUNTU_VERSION="24.04"
readonly CONDA_ENV_NAME="common-python-3.12"
readonly PYTHON_VERSION="3.12"
readonly NVIDIA_DRIVER_VERSION="550.163.01"  # Tesla 100兼容版本
readonly CUDA_TOOLKIT_VERSION="12.4"         # PyTorch安装指定版本
readonly DIR_HF_MODELS="$HOME/hf_models"
readonly DIR_CURRENT=$(pwd)
readonly DIR_WORKSPACE="$DIR_CURRENT/workspace"

export CUDA_VISIBLE_DEVICES=1,2 # 只用第1张显卡（编号从0开始），因为当前环境中，0号为16GB，1和2为32GB
# ==============================================
# 函数定义：检查Ubuntu版本
# ==============================================
check_ubuntu_version() {
    local current_version
    current_version=$(lsb_release -rs)
    
    if [[ "$current_version" != "$REQUIRED_UBUNTU_VERSION" ]]; then
        echo "❌ 当前Ubuntu版本为$current_version，必须为$REQUIRED_UBUNTU_VERSION才能继续执行。"
        exit 1
    else
        echo "✅ Ubuntu版本检查通过：$current_version"
    fi
}

# ==============================================
# 函数定义：检查conda环境是否存在，不存在则创建并激活
# ==============================================
check_and_create_conda_env() {
    if conda env list | grep -q "^$CONDA_ENV_NAME\s"; then
        echo "✅ conda环境'$CONDA_ENV_NAME'已存在。"
        eval "$(conda shell.bash hook)"
        conda activate "$CONDA_ENV_NAME" || {
            echo "❌ 激活环境'$CONDA_ENV_NAME'失败，请手动检查。"
            exit 1
        }
        echo "-- 已激活环境：$(conda info --envs | grep '*' | awk '{print $1}')"
    else
        echo "❌ conda环境'$CONDA_ENV_NAME'不存在，正在创建..."
        conda create -n "$CONDA_ENV_NAME" python="$PYTHON_VERSION" -y
        eval "$(conda shell.bash hook)"
        conda activate "$CONDA_ENV_NAME" || {
            echo "❌ 激活环境'$CONDA_ENV_NAME'失败，请手动检查。"
            exit 1
        }
        echo "✅ 环境'$CONDA_ENV_NAME'创建并激活成功。"
    fi
}
# ==============================================
# 函数定义：下载Hugging Face模型/数据集（可指定下载路径）
# ==============================================
download_huggingface_package() {
    local package_name="$1"
    local target_dir="$2"
    if [[ -z "$package_name" ]]; then
        echo "❌ 未指定Hugging Face的包名"
        return 1
    fi
    if [[ -z "$target_dir" ]]; then
        target_dir="$DIR_HF_MODELS/$package_name"
    fi
    echo "-- 正在下载Hugging Face包：$package_name 到 $target_dir ..."
    export HF_ENDPOINT="https://hf-mirror.com"
    hf download "$package_name" --repo-type model --local-dir "$target_dir"

    # 检查下载目录是否存在且非空
    if [[ -d "$target_dir" ]] && [[ $(ls -A "$target_dir" 2>/dev/null) ]]; then
        echo "✅ 下载完成且内容完整：$package_name"
    else
        echo "❌ 下载失败或内容为空，请检查网络或包名。"
        return 2
    fi
}

# ==============================================
# 函数定义：安装transformers等依赖
# ==============================================
install_transformers_dependencies() {
    echo "-- 正在安装 transformers、accelerate、sentencepiece llama-cpp-python等依赖..."
    sudo apt-get install -y libgomp1
    pip install --upgrade pip
    export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
    pip install transformers accelerate sentencepiece llama-cpp-python
    echo "✅ transformers及相关依赖安装完成。"
}
# ==============================================
# 函数定义：测试llama-cpp-python推理速度
# ==============================================
test_llama_token_speed() {
    local model_file="$1"
    echo "测试llama-cpp-python推理速度"
    if [[ -z "$model_file" ]]; then
        echo "❌ 未指定gguf模型文件"
        return 1
    fi
    python - <<EOF
import time
from llama_cpp import Llama

llm = Llama(
    model_path="$model_file",
    use_gpu=True,
    n_gpu_layers=-1,
    n_ctx=2048,
    verbose=False
)
start = time.time()
llm.create_chat_completion(messages=[{"role": "user", "content": "Hello"}], max_tokens=100)
print(f"GPU推理时间: {time.time() - start:.2f}s")
EOF
}
# ==============================================
# 函数定义：加载Hugging Face下载的模型
# ==============================================
load_huggingface_model() { 
    local model_path="$1"
    local user_input="$2"

    if [[ -z "$model_path" ]]; then
        echo "❌ 未指定模型路径"
        return 1
    fi

    export CUDA_VISIBLE_DEVICES=1 # 只用第1张显卡（编号从0开始），因为当前环境中，0号为16GB，1和2为32GB

    if [[ -f "$model_path" ]]; then
        if [[ "$model_path" == *gguf* ]]; then
            load_huggingface_model_with_llama "$model_path" "$user_input"
        else
            echo "❌ 文件存在但不包含gguf关键字，无法识别加载方式。"
            return 1
        fi
    elif [[ -d "$model_path" ]]; then
        echo "-- 检测到模型目录，正在使用transformers加载..."
        load_huggingface_model_with_transformers "$model_path"
    else
        echo "❌ 路径不存在：$model_path"
        return 2
    fi
}
# ==============================================
# 函数定义：使用llama-cpp-python加载gguf模型
# ==============================================
load_huggingface_model_with_llama(){
    model_file=$1
    user_input=$2
    local max_context_num=3276

    python - <<EOF
from llama_cpp import Llama

llm = Llama(
    model_path="$model_file",
    use_gpu=True,
    n_gpu_layers=-1,  # 使用全部GPU层
    n_ctx=$max_context_num,
    verbose=True     # 启用详细日志
)
output = llm(prompt="$user_input", max_tokens=$max_context_num)
print(f"{output['choices'][0]['text'] if 'choices' in output else output}")
EOF
}

# ==============================================
# 函数定义：使用transformers加载模型
# ==============================================
load_huggingface_model_with_transformers() {
    local model_dir="$1"
    if [[ -z "$model_dir" ]]; then
        echo "❌ 未指定模型目录"
        return 1
    fi
    if [[ ! -d "$model_dir" ]]; then
        echo "❌ 模型目录不存在：$model_dir"
        return 2
    fi
    python - <<EOF
import sys
from pathlib import Path
import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM

model_path = Path("$model_dir")
if not model_path.exists():
    print("❌ 模型目录不存在：", model_path)
    sys.exit(1)

def chat_with_model():
    print(">> 对话开始！输入'exit'退出。")
    chat_history = []  # 存储对话历史（可选，用于上下文连贯性）

    while True:
        user_input = input("\n>> 你: ").strip()
        if user_input.lower() == "exit":
            print(">> 再见！")
            break

        # 将用户输入添加到对话历史（可选）
        chat_history.append(user_input)

        # 编码输入文本（加入特殊token，如<|im_start|>、<|im_end|>）
        inputs = tokenizer(
            "\n".join(chat_history),  # 拼接历史对话
            return_tensors="pt",
            max_length=512,  # 限制输入长度（避免超出模型最大上下文）
            truncation=True   # 截断过长文本
        ).to(device)


        # max_new_tokens：控制生成回复的长度（避免过长导致偏离主题）；
        # do_sample=True：启用随机采样（若设为False，模型将选择概率最高的token，回复更保守）；
        # top_k=50：仅从概率最高的50个token中采样（减少不合理回复的概率）；
        # temperature=0.7：值越小，回复越确定（如0.1时模型更保守；如1.0时更随机

        # 生成模型回复（关键参数说明）
        with torch.no_grad():  # 禁用梯度计算（加速推理）
            outputs = model.generate(
                **inputs,
                max_new_tokens=10000,   # 生成的最大token数（控制回复长度）
                do_sample=True,         # 启用采样（增加回复多样性）
                top_k=50,               # 限制采样范围（top-k过滤）
                top_p=0.95,             # 核采样（top-p过滤，保留概率最高的95% token）
                temperature=0.7         # 控制随机性（0.7为常用值，平衡多样性与确定性）
            )

        # 解码生成的token为文本（去除特殊token）
        reply = tokenizer.decode(outputs[0][:], skip_special_tokens=True)
        print(f"\n>> 模型: {reply}")

        # 将模型回复添加到对话历史（可选）
        chat_history.append(reply)

# AutoTokenizer：自动识别模型对应的分词器（如OPT模型使用Byte-Pair Encoding分词）；
# AutoModelForCausalLM：适用于因果语言模型（如OPT、GPT系列），支持文本生成任务；
try:
    # 加载分词器（将文本转为模型可理解的token）
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="cuda:1"  # 自动分配模型层到GPU/CPU
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="cuda:1"  # 自动分配模型层到GPU/CPU
    )
    print(f"✅ 模型加载成功：{model_path}")
    chat_with_model()
except Exception as e:
    print(f"❌ 加载模型失败: {e}")
    sys.exit(2)
EOF
}
# ==============================================
# 主函数：脚本入口
# ==============================================
main() {
    model_package_name="Ttimofeyka/MistralRP-Noromaid-NSFW-Mistral-7B-GGUF"
    check_ubuntu_version

    # 检查conda环境并创建/激活
    check_and_create_conda_env

    # 下载Hugging Face模型
    # 需要计算这个函数执行的时间是多少秒，如果大于1分钟则以分钟为单位显示，如果大于1小时则以小时为单位
    start_time=$(date +%s)
    download_huggingface_package "$model_package_name"

    echo "-- 正在安装依赖..."
    conda install -c conda-forge libstdcxx-ng gcc
    export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda-$CUDA_TOOLKIT_VERSION/lib64:$LD_LIBRARY_PATH

    if [ $DEBUGGABLE ]; then
        test_llama_token_speed "$DIR_HF_MODELS/$model_package_name/MistralRP-Noromaid-NSFW-7B-Q8_0.gguf"
    fi

    end_time=$(date +%s)
    elapsed=$((end_time - start_time))
    if ((elapsed >= 3600)); then
        printf "✅ 下载耗时: %d小时%d分钟%d秒\n" $((elapsed/3600)) $(( (elapsed%3600)/60 )) $((elapsed%60))
    elif ((elapsed >= 60)); then
        printf "✅ 下载耗时: %d分钟%d秒\n" $((elapsed/60)) $((elapsed%60))
    else
        printf "✅ 下载耗时: %d秒\n" "$elapsed"
    fi

    # 加载Hugging Face模型
    mkdir -p "$DIR_WORKSPACE/output"
    load_huggingface_model "$DIR_HF_MODELS/$model_package_name/MistralRP-Noromaid-NSFW-7B-Q8_0.gguf" \
        "写一段50字的情色口播。要求更多的语气词,使用中文，不要换行空格" > "$DIR_WORKSPACE/output/model.prompt2txt.txt"
    cat "$DIR_WORKSPACE/output/model.prompt2txt.txt"
}
main