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

export CUDA_VISIBLE_DEVICES=1 # 只用第1张显卡（编号从0开始），因为当前环境中，0号为16GB，1和2为32GB
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
# 函数定义：更新APT源为清华源
# ==============================================
update_apt_source_to_tsinghua() {
    local tsinghua_uri="https://mirrors.tuna.tsinghua.edu.cn/ubuntu"
    if grep -q "$tsinghua_uri" /etc/apt/sources.list.d/ubuntu.sources 2>/dev/null; then
        echo "✅ 已是清华源，无需替换。"
        return
    fi

    echo "-- 正在备份原有APT源列表..."
    sudo cp /etc/apt/sources.list.d/ubuntu.sources /etc/apt/sources.list.d/ubuntu.sources.bak

    echo "-- 正在替换为清华源..."
    sudo tee /etc/apt/sources.list.d/ubuntu.sources > /dev/null <<EOF
Types: deb
URIs: https://mirrors.tuna.tsinghua.edu.cn/ubuntu
Suites: noble noble-updates noble-backports
Components: main restricted universe multiverse
Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg

# 默认注释了源码镜像以提高 apt update 速度，如有需要可自行取消注释
# Types: deb-src
# URIs: https://mirrors.tuna.tsinghua.edu.cn/ubuntu
# Suites: noble noble-updates noble-backports
# Components: main restricted universe multiverse
# Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg

# 以下安全更新软件源包含了官方源与镜像站配置，如有需要可自行修改注释切换
Types: deb
URIs: https://mirrors.tuna.tsinghua.edu.cn/ubuntu
Suites: noble-security
Components: main restricted universe multiverse
Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg

# Types: deb-src
# URIs: https://mirrors.tuna.tsinghua.edu.cn/ubuntu
# Suites: noble-security
# Components: main restricted universe multiverse
# Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg

# 预发布软件源，不建议启用

# Types: deb
# URIs: https://mirrors.tuna.tsinghua.edu.cn/ubuntu
# Suites: noble-proposed
# Components: main restricted universe multiverse
# Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg

# # Types: deb-src
# # URIs: https://mirrors.tuna.tsinghua.edu.cn/ubuntu
# # Suites: noble-proposed
# # Components: main restricted universe multiverse
# # Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg
EOF

    echo "-- 正在更新APT包索引..."
    sudo apt update -y
    echo "✅ APT源更新完成。"
}

# ==============================================
# 函数定义：检查并安装conda
# ==============================================
install_conda_if_not_exist() {
    if ! command -v conda &> /dev/null; then
        echo " 未检测到conda，正在安装Miniconda..."
        wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
        bash miniconda.sh -b -p $HOME/miniconda
        rm miniconda.sh
        echo 'export PATH="$HOME/miniconda/bin:$PATH"' >> ~/.bashrc
        source ~/.bashrc
        echo "✅ Miniconda安装完成，conda已加入PATH。"
    else
        echo "✅ conda已安装：$(conda --version)"
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
# 函数定义：检查并安装docker（使用snap）
# ==============================================
install_docker_via_snap() {
    if ! command -v docker &> /dev/null; then
        echo "❌ 未检测到docker，正在通过snap安装..."
        sudo snap install docker
        echo "✅ docker安装完成。"
    else
        echo "✅ docker已安装：$(docker --version)"
    fi
}

# ==============================================
# 函数定义：打印当前所有conda环境名称
# ==============================================
print_conda_envs() {
    echo "-- 当前conda环境列表："
    conda env list | grep -E '^\w+\s' | awk '{print $1}'
}

# ==============================================
# 函数定义：检查并安装nvidia驱动（Tesla 100）
# ==============================================
install_nvidia_driver() {
    echo "-- 正在检查NVIDIA驱动是否安装..."
    if ! command -v nvidia-smi &> /dev/null; then
        echo "❌ 未检测到NVIDIA驱动，正在安装版本$NVIDIA_DRIVER_VERSION..."
        
        # 添加NVIDIA CUDA APT源
        echo "-- 添加NVIDIA CUDA APT源..."
        wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu$(lsb_release -cs)/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/cuda-keyring.deb
        sudo dpkg -i /tmp/cuda-keyring.deb
        sudo apt update
        
        # 安装指定版本驱动
        sudo apt install -y nvidia-driver-"${NVIDIA_DRIVER_VERSION%.*}"=*"${NVIDIA_DRIVER_VERSION}"-0ubuntu1 nvidia-dkms-"${NVIDIA_DRIVER_VERSION%.*}"=*"${NVIDIA_DRIVER_VERSION}"-0ubuntu1 nvidia-utils-"${NVIDIA_DRIVER_VERSION%.*}"=*"${NVIDIA_DRIVER_VERSION}"-0ubuntu1
        sudo apt-mark hold nvidia-driver-"${NVIDIA_DRIVER_VERSION%.*}" nvidia-dkms-"${NVIDIA_DRIVER_VERSION%.*}" nvidia-utils-"${NVIDIA_DRIVER_VERSION%.*}"  # 锁定版本，防止自动更新
        
        echo "✅ NVIDIA驱动安装完成，请重启系统以生效：sudo reboot"
    else
        echo "✅ NVIDIA驱动已安装：$(nvidia-smi --query-gpu=driver_version --format=csv,noheader)"
    fi
}

# ==============================================
# 函数定义：安装pytorch（基于cuda124）
# ==============================================
install_pytorch() {
    echo "-- 正在检查PyTorch是否安装..."
    if ! python -c "import torch; print(torch.__version__)" &> /dev/null; then
        echo "❌ 未检测到PyTorch，正在安装基于cuda124的版本..."
        pip install --upgrade pip
        # conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y
        pip install pytorch torchvision torchaudio --upgrade --force-reinstall --index-url https://pypi.tuna.tsinghua.edu.cn/simple --extra-index-url https://mirrors.tuna.tsinghua.edu.cn/cloud/whl/cu124
        echo "✅ PyTorch安装完成：$(python -c "import torch; print(torch.__version__ + ' (CUDA: ' + torch.version.cuda + ')')")"
    else
        echo "✅ PyTorch已安装：$(python -c "import torch; print(torch.__version__ + ' (CUDA: ' + torch.version.cuda + ')')")"
    fi
}

# ==============================================
# 函数定义：安装transformers等依赖
# ==============================================
install_transformers_dependencies() {
    echo "-- 正在安装 transformers、accelerate、sentencepiece llama-cpp-python等依赖..."
    pip install --upgrade pip
    pip install transformers accelerate sentencepiece --upgrade --force-reinstall --index-url https://pypi.tuna.tsinghua.edu.cn/simple --extra-index-url https://mirrors.tuna.tsinghua.edu.cn/cloud/whl/cu124

    sudo apt-get install -y libgomp1
    export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
    pip install llama-cpp-python --upgrade --force-reinstall --extra-index-url https://jllllll.github.io/llama-cpp-python-cuBLAS-wheels/whl/cu124
    echo "✅ transformers及相关依赖安装完成。"
}
# ==============================================
# 函数定义：检查并安装Hugging Face CLI
# ==============================================
install_huggingface_cli() {
    echo "-- 正在检查 huggingface_hub 和 hf 命令是否安装..."
    pip show hf &> /dev/null || pip install hf
    echo "✅ huggingface_hub 已安装：$(python -c 'import huggingface_hub; print(huggingface_hub.__version__)')"
    echo "✅ hf 命令已安装：$(hf --version 2>/dev/null || echo '未检测到')"
}
# ==============================================
# 主函数：脚本入口
# ==============================================
main() {
    echo "-- 开始执行系统配置脚本..."
    mkdir -p $DIR_WORKSPACE/status

    # 1. 检查Ubuntu版本
    check_ubuntu_version

    # 2. 更新APT源为清华源
    update_apt_source_to_tsinghua
    
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
    
    # 8. 安装pytorch
    install_pytorch

    # 9. 安装其他依赖
    install_transformers_dependencies

    # 10. 检查并安装Hugging Face CLI
    install_huggingface_cli
}

main
