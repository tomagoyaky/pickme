#!/bin/bash
clear
set -e
# ==============================================
# 常量定义
# ==============================================
readonly REQUIRED_UBUNTU_VERSION="24.04"
readonly CONDA_ENV_NAME="common-python-3.12"
readonly PYTHON_VERSION="3.12"
readonly NVIDIA_DRIVER_VERSION="550.163.01"  # Tesla 100兼容版本
readonly CUDA_TOOLKIT_VERSION="12.4"         # PyTorch安装指定版本
readonly DIR_HF_MODELS="$HOME/hf_models"
readonly DIR_CURRENT=$(pwd)
readonly DIR_WORKSPACE="$DIR_CURRENT/workspace"
readonly DIR_REPO="$DIR_WORKSPACE/ComfyUI"

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
# 函数定义：Git获取ComfyUI并安装依赖
# ==============================================
install_comfyui() {
    if [[ -d "$DIR_REPO" ]]; then
        echo "✅ ComfyUI 已存在于 $DIR_REPO"
    else
        echo "-- 正在克隆 ComfyUI 到 $DIR_REPO ..."
        git clone https://github.com/comfyanonymous/ComfyUI.git "$DIR_REPO"
    fi
    echo "-- 安装 ComfyUI 依赖 ..."
    pip install -r "$DIR_REPO/requirements.txt"
}
# ==============================================
# 函数定义：文字转图像
# ==============================================
text2image() {
    local message="$1"
    if [[ -z "$message" ]]; then
        echo "❌ 未提供输入文本"
        return 1
    fi
    echo "-- 正在将文本转换为图像：$message ..."
    # 这里调用实际的文本转图像命令
    # 例如：python scripts/txt2img.py --prompt "$message"
}

# 函数定义：下载 ComfyUI 所需模型
# ==============================================
download_comfyui_model() {
    local comfyui_models_dir="$DIR_WORKSPACE/ComfyUI/models"
    mkdir -p "$comfyui_models_dir"
    # 示例：下载Stable Diffusion模型
    local sd_model="runwayml/stable-diffusion-v1-5"
    download_huggingface_package "$sd_model" "$comfyui_models_dir/stable-diffusion-v1-5"
    # 可根据需要添加更多模型下载
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
    # 安装 ComfyUI
    install_comfyui
    # 下载 ComfyUI 模型
    local model_start_time=$(date +%s)
    download_comfyui_model
    local model_end_time=$(date +%s)
    local model_duration=$((model_end_time - model_start_time))
    if (( model_duration >= 3600 )); then
        printf "⏱️ ComfyUI模型下载时间: %.2f 小时\n" "$(echo "$model_duration/3600" | bc -l)"
    elif (( model_duration >= 60 )); then
        printf "⏱️ ComfyUI模型下载时间: %.2f 分钟\n" "$(echo "$model_duration/60" | bc -l)"
    else
        echo "⏱️ ComfyUI模型下载时间: ${model_duration} 秒"
    fi
    # 文字转图像
    local start_time=$(date +%s)
    text2image "$message"
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    if (( duration >= 3600 )); then
        printf "⏱️ 执行时间: %.2f 小时\n" "$(echo "$duration/3600" | bc -l)"
    elif (( duration >= 60 )); then
        printf "⏱️ 执行时间: %.2f 分钟\n" "$(echo "$duration/60" | bc -l)"
    else
        echo "⏱️ 执行时间: ${duration} 秒"
    fi
}
main