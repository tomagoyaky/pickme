#!/bin/bash
clear
set -e
# ==============================================
# 常量定义
# ==============================================
readonly REQUIRED_UBUNTU_VERSION="24.04"
readonly CONDA_ENV_NAME="spark-tts-python-3.12"
readonly PYTHON_VERSION="3.12"
readonly NVIDIA_DRIVER_VERSION="550.163.01"  # Tesla 100兼容版本
readonly CUDA_TOOLKIT_VERSION="12.4"         # PyTorch安装指定版本
readonly DIR_HF_MODELS="$HOME/hf_models"
readonly DIR_CURRENT=$(pwd)
readonly DIR_WORKSPACE="$DIR_CURRENT/workspace"
readonly DIR_REPO="$DIR_WORKSPACE/Spark-TTS"

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
        pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com
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
    python -m cli.inference \
        --text "$message" \
        --device 0 \
        --save_dir "$DIR_REPO/result" \
        --model_dir "$DIR_REPO/pretrained_models" \
        --prompt_text "吃燕窝就选燕之屋，本节目由26年专注高品质燕窝的燕之屋冠名播出。豆奶牛奶换着喝，营养更均衡，本节目由豆本豆豆奶特约播出。" \
        --prompt_speech_path "$DIR_REPO/example/prompt_audio.wav"

    ls -lah "$DIR_REPO/result"
}
# ==============================================
# 主函数：脚本入口
# ==============================================
main() {
    message=$1
    check_ubuntu_version

    # 检查conda环境并创建/激活
    check_and_create_conda_env

    if [ -f "$DIR_WORKSPACE/output/model.prompt2txt.txt" ]; then
        echo "✅ 找到文本文件：$DIR_WORKSPACE/output/model.prompt2txt.txt"
        echo "======================================================"
        cat "$DIR_WORKSPACE/output/model.prompt2txt.txt"
        echo "======================================================"
        # 文字转音频
        cat "$DIR_WORKSPACE/output/model.prompt2txt.txt" > message
    else
        echo "======================================================"
        cat "$message"
        echo "======================================================"
    fi
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
main