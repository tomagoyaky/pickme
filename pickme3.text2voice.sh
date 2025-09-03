#!/bin/bash
clear
set -e

source ./pickme0.common.sh
readonly DIR_REPO="$DIR_WORKSPACE/Spark-TTS"

readonly PYTHON_VERSION="3.12"
readonly CONDA_ENV_NAME="spark-tts-python-${PYTHON_VERSION}"
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
# 脚本入口
# ==============================================
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