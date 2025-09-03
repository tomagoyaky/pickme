#!/bin/bash
clear
set -e
source ./pickme0.common.sh
# ==============================================
# 脚本入口
# ==============================================
is_delete_conda_env_flag=$1
echo "-- 开始执行系统配置脚本..."
mkdir -p $DIR_WORKSPACE/status

# 1. 检查Ubuntu版本
check_ubuntu_version

# 1.5. 询问是否删除当前conda环境
ask_delete_conda_env $is_delete_conda_env_flag

# 2. 更新APT源为清华源
update_apt_source_to_tsinghua

# [tips]: If purge executed, more network bandwidth may be required, please be patient
# pip cache purge

# 3. 检查并安装conda
install_conda_if_not_exist

# 4. 检查conda环境并创建/激活
check_and_create_conda_env

# 5. 检查并安装docker（snap）
install_docker_via_snap

# 6. 打印conda环境列表
print_conda_envs

# 7. 检查并安装nvidia驱动
install_nvidia_driver

# 8. 检查并安装CUDA Toolkit
install_cuda_toolkit

# 9. 安装其他依赖
install_llama

# 10. 安装pytorch
install_pytorch

# 11. 检查并安装Hugging Face CLI
install_huggingface_cli
