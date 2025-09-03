#!/bin/bash
clear
set -e
# ==============================================
# 常量定义
# ==============================================
readonly REQUIRED_UBUNTU_VERSION="24.04"
readonly PYTHON_VERSION="3.12"
readonly CONDA_ENV_NAME="spark-tts-python-$PYTHON_VERSION"
readonly NVIDIA_DRIVER_VERSION="550.163.01"  # Tesla 100兼容版本
readonly CUDA_TOOLKIT_VERSION="12.4"         # PyTorch安装指定版本
readonly DIR_HF_MODELS="$HOME/hf_models"
readonly DIR_CURRENT=$(pwd)
readonly DIR_WORKSPACE="$DIR_CURRENT/workspace"
readonly DIR_REPO="$DIR_WORKSPACE/Spark-TTS"

readonly URL_SOURCE_PIP_TSINGHUA="https://pypi.tuna.tsinghua.edu.cn/simple"
readonly URL_SOURCE_APT_TSINGHUA="https://mirrors.tuna.tsinghua.edu.cn/ubuntu"
readonly URL_SOURCE_CONDA="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
readonly FILE_UBUNTU_SOURCES="/etc/apt/sources.list.d/ubuntu.sources"
readonly URL_SOURCE_CUDA="https://mirrors.tuna.tsinghua.edu.cn/cloud/whl/cu124"
readonly URL_SOURCE_CUDA_LLAMA="https://jllllll.github.io/llama-cpp-python-cuBLAS-wheels/whl/cu124"

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
# 函数定义：检查并安装Hugging Face CLI
# ==============================================
install_huggingface_cli() {
    echo "-- 正在检查 huggingface_hub 和 hf 命令是否安装..."
    pip install hf huggingface_hub
    echo "✅ huggingface_hub 已安装：$(python -c 'import huggingface_hub; print(huggingface_hub.__version__)')"
    echo "✅ hf 命令已安装：$(hf --version 2>/dev/null || echo '未检测到')"
}

# ==============================================
# 函数定义：安装pytorch（基于cuda124）
# ==============================================
install_pytorch() {
    echo "-- 正在检查PyTorch是否安装..."
    if ! python -c "import torch; print(torch.__version__)" &> /dev/null; then
        echo "❌ 未检测到PyTorch，正在安装基于cuda124的版本..."
        pip install --upgrade pip
        CUDA124_UPDATE_PARAMS="--force-reinstall --no-cache-dir --index-url $URL_SOURCE_PIP_TSINGHUA --extra-index-url $URL_SOURCE_CUDA"
        pip install \
            torch \
            torchvision \
            torchaudio \
            transformers \
            accelerate \
            sentencepiece \
            $CUDA124_UPDATE_PARAMS
        echo "✅ PyTorch安装完成：$(python -c "import torch; print(torch.__version__ + ' (CUDA: ' + torch.version.cuda + ')')")"
    else
        echo "✅ PyTorch已安装：$(python -c "import torch; print(torch.__version__ + ' (CUDA: ' + torch.version.cuda + ')')")"
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
# 函数定义：询问是否删除conda环境
# ==============================================
ask_delete_conda_env() {
    local is_delete_conda_env_flag=$1
    if [[ "$is_delete_conda_env_flag" == "-y" ]]; then
        echo "-- 正在删除conda环境'$CONDA_ENV_NAME'..."
        conda remove --name "$CONDA_ENV_NAME" --all -y
        echo "✅ conda环境'$CONDA_ENV_NAME'已删除。"
    else
        read -p "是否要删除当前conda环境'$CONDA_ENV_NAME'? [y/N]: " answer
        answer=${answer:-n}
        if [[ "$answer" =~ ^[Yy]$ ]]; then
            echo "-- 正在删除conda环境'$CONDA_ENV_NAME'..."
            conda remove --name "$CONDA_ENV_NAME" --all -y
            echo "✅ conda环境'$CONDA_ENV_NAME'已删除。"
        else
            echo "未删除conda环境。"
        fi
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
# 函数定义：安装Spark-TTS
# ==============================================
install_spark_tts() {
    local message=$1
    if [[ ! -d "$DIR_REPO" ]]; then
        echo "-- Spark-TTS 未检测到，正在克隆仓库..."
        git clone https://github.com/SparkAudio/Spark-TTS.git "$DIR_REPO"
    else
        echo "✅ Spark-TTS 已存在：$DIR_REPO"
    fi
    cd "$DIR_REPO"
    echo "-- 安装 Spark-TTS 依赖..."
    if [ ! -f "$DIR_WORKSPACE/status/spark-tts.requirements_installed" ]; then
        pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --force-reinstall --trusted-host=mirrors.aliyun.com
        pip install soundfile
        touch "$DIR_WORKSPACE/status/spark-tts.requirements_installed"
    else
        echo "✅ Spark-TTS 依赖已安装，无需重复安装。"
    fi
    echo "✅ Spark-TTS 依赖安装完成。"
}
download_spark_tts_model(){
    mkdir -p pretrained_models
    download_huggingface_package "SparkAudio/Spark-TTS-0.5B" "$DIR_REPO/pretrained_models"
}
text2voice(){
    # 文字转语音
    echo "清理：$DIR_REPO/result/"
    rm -rf "$DIR_REPO/result/"
    python -m cli.inference \
        --text "$message" \
        --device 0 \
        --save_dir "$DIR_REPO/result" \
        --model_dir "$DIR_REPO/pretrained_models" \
        --prompt_text "吃燕窝就选燕之屋，本节目由26年专注高品质燕窝的燕之屋冠名播出。豆奶牛奶换着喝，营养更均衡，本节目由豆本豆豆奶特约播出。" \
        --prompt_speech_path "$DIR_REPO/example/prompt_audio.wav" \
        --speed "very_low"


    # 开始播放
    file_output_wav=/home/x/Documents/smart_tools/pickme/workspace/Spark-TTS/result/$(ls "$DIR_REPO/result/")
    echo "正在播放：$file_output_wav"
    ffplay -nodisp -autoexit "$file_output_wav"

}
# ==============================================
# 主函数：脚本入口
# ==============================================
main() {
    message=$1
    check_ubuntu_version

    # 询问是否删除conda环境
    ask_delete_conda_env

    # 检查conda环境并创建/激活
    check_and_create_conda_env

    if [ ! -n "$message" ]; then
        message=$(cat "$DIR_WORKSPACE/output/prompt2txt.result.txt")
    fi
     # fix：确保输入文本格式正确（如无特殊字符、空格），且tokenizer能正确转换为token IDs
    # 清理message中的特殊字符、多余空格、换行符和空行
    message=$(echo "$message" | tr -d '\r' | sed '/^[[:space:]]*$/d' | sed 's/[[:space:]]\+/ /g' | sed 's/^[ \t]*//;s/[ \t]*$//')

    echo "======================================================"
    echo "$message"
    echo "======================================================"

    # 安装 Hugging Face CLI
    install_huggingface_cli
    # 安装 PyTorch
    # install_pytorch
    # 安装 Spark-TTS
    install_spark_tts
    # 下载 Spark-TTS 模型
    start_download_time=$(date +%s)
    download_spark_tts_model
    end_download_time=$(date +%s)
    download_duration=$((end_download_time - start_download_time))

    if (( download_duration >= 3600 )); then
        hours=$((download_duration / 3600))
        mins=$(((download_duration % 3600) / 60))
        secs=$((download_duration % 60))
        echo "⏱️ Spark-TTS模型下载耗时: ${hours}小时${mins}分钟${secs}秒"
    elif (( download_duration >= 60 )); then
        mins=$((download_duration / 60))
        secs=$((download_duration % 60))
        echo "⏱️ Spark-TTS模型下载耗时: ${mins}分钟${secs}秒"
    else
        echo "⏱️ Spark-TTS模型下载耗时: ${download_duration}秒"
    fi
    # 文字转语音
    start_time=$(date +%s)
    text2voice "$message"
    end_time=$(date +%s)
    duration=$((end_time - start_time))

    if (( duration >= 3600 )); then
        hours=$((duration / 3600))
        mins=$(((duration % 3600) / 60))
        secs=$((duration % 60))
        echo "⏱️ 总耗时: ${hours}小时${mins}分钟${secs}秒"
    elif (( duration >= 60 )); then
        mins=$((duration / 60))
        secs=$((duration % 60))
        echo "⏱️ 总耗时: ${mins}分钟${secs}秒"
    else
        echo "⏱️ 总耗时: ${duration}秒"
    fi
}
main $1