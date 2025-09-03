#!/bin/bash
clear
set -e
# ==============================================
# 常量定义
# ==============================================
readonly DEBUGGABLE=1
readonly REQUIRED_UBUNTU_VERSION="24.04"
readonly REQUIRED_UBUNTU_VERSION_STR="2404"
readonly PYTHON_VERSION="3.12"
readonly CONDA_ENV_NAME="common-python-${PYTHON_VERSION}"

readonly NVIDIA_DRIVER_CUDA_TOOLKIT_VERSION="550.54.14"             # Toolkit版本 对于CUDA 12.4.0只能支持到550.54.14，安装的时候会提示是否覆盖，不覆盖就是了
readonly NVIDIA_DRIVER_VERSION=${NVIDIA_DRIVER_CUDA_TOOLKIT_VERSION}
readonly NVIDIA_DRIVER_DOWNLOAD_URL="https://cn.download.nvidia.com/tesla/550.163.01/nvidia-driver-local-repo-ubuntu2404-550.54.14_1.0-1_amd64.deb"
readonly CUDA_TOOLKIT_DOWNLOAD_URL="https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.14_linux.run"  # 官方说支持Ubuntu 22.04，但是也可以用在24.04上
readonly CUDA_TOOLKIT_VERSION="12.4"         # PyTorch安装指定版本
readonly CUDA_TOOLKIT_VERSION_NUM="12.4.0"   # CUDA Toolkit版本
readonly CUDA_TOOLKIT_HOME=/usr/local/cuda-${CUDA_TOOLKIT_VERSION}

readonly DIR_HF_MODELS="$HOME/hf_models"
readonly DIR_CURRENT=$(pwd)
readonly DIR_WORKSPACE="$DIR_CURRENT/workspace"
readonly URL_SOURCE_PIP_TSINGHUA="https://pypi.tuna.tsinghua.edu.cn/simple"
readonly URL_SOURCE_APT_TSINGHUA="https://mirrors.tuna.tsinghua.edu.cn/ubuntu"
readonly URL_SOURCE_CONDA="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
readonly FILE_UBUNTU_SOURCES="/etc/apt/sources.list.d/ubuntu.sources"
readonly URL_SOURCE_CUDA="https://mirrors.tuna.tsinghua.edu.cn/cloud/whl/cu124"
readonly URL_SOURCE_CUDA_LLAMA="https://jllllll.github.io/llama-cpp-python-cuBLAS-wheels/whl/cu124"

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
# 函数定义：更新APT源为清华源
# ==============================================
update_apt_source_to_tsinghua() {
    local tsinghua_uri="$URL_SOURCE_APT_TSINGHUA"
    if grep -q "$tsinghua_uri" $FILE_UBUNTU_SOURCES 2>/dev/null; then
        echo "✅ 已是清华源，无需替换。"
        return
    fi

    echo "-- 正在备份原有APT源列表..."
    sudo cp $FILE_UBUNTU_SOURCES $FILE_UBUNTU_SOURCES.bak

    echo "-- 正在替换为清华源..."
    sudo tee $FILE_UBUNTU_SOURCES > /dev/null <<EOF
Types: deb
URIs: $URL_SOURCE_APT_TSINGHUA
Suites: noble noble-updates noble-backports
Components: main restricted universe multiverse
Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg

# 默认注释了源码镜像以提高 apt update 速度，如有需要可自行取消注释
# Types: deb-src
# URIs: $URL_SOURCE_APT_TSINGHUA
# Suites: noble noble-updates noble-backports
# Components: main restricted universe multiverse
# Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg

# 以下安全更新软件源包含了官方源与镜像站配置，如有需要可自行修改注释切换
Types: deb
URIs: $URL_SOURCE_APT_TSINGHUA
Suites: noble-security
Components: main restricted universe multiverse
Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg

# Types: deb-src
# URIs: $URL_SOURCE_APT_TSINGHUA
# Suites: noble-security
# Components: main restricted universe multiverse
# Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg

# 预发布软件源，不建议启用

# Types: deb
# URIs: $URL_SOURCE_APT_TSINGHUA
# Suites: noble-proposed
# Components: main restricted universe multiverse
# Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg

# # Types: deb-src
# # URIs: $URL_SOURCE_APT_TSINGHUA
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
        wget $URL_SOURCE_CONDA -O miniconda.sh
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
    # 检查是否在docker容器中
    if grep -qE '/docker/' /proc/1/cgroup 2>/dev/null || [ -f /.dockerenv ]; then
        echo "⚠️ 当前已在docker容器中，跳过docker安装。"
        return
    fi

    if ! command -v docker &> /dev/null; then
        echo "❌ 未检测到docker，正在通过snap安装..."
        sudo snap install docker
        echo "✅ docker安装完成。"
    else
        echo "✅ docker已安装：$(docker --version)"
    fi

    echo "-- 配置 /var/snap/docker/current/config/daemon.json 使用清华镜像..."
    sudo mkdir -p /var/snap/docker/current/config
    sudo tee /var/snap/docker/current/config/daemon.json > /dev/null <<EOF
{
  "registry-mirrors": ["https://docker.mirrors.tuna.tsinghua.edu.cn"]
}
EOF
    sudo snap restart docker
    sudo systemctl restart snap.docker.dockerd

    echo "-- 等待5秒以确保docker服务重启完成..."
    sleep 5
    docker info
    echo "✅ docker镜像源已设置为清华镜像。"
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
    if command -v nvidia-smi &> /dev/null; then
        echo "✅ NVIDIA驱动已安装：$(nvidia-smi --query-gpu=driver_version --format=csv,noheader)"
        read -p "是否要重装Nvidia GPU驱动？目标驱动版本${NVIDIA_DRIVER_VERSION} [y/N]: " reinstall_answer
        reinstall_answer=${reinstall_answer:-n}
    fi
    if [[ "$reinstall_answer" =~ ^[Yy]$ ]]; then
        echo "-- 正在重新安装Nvidia GPU驱动..."
        if [ ! -f $DIR_WORKSPACE/nvidia-ubuntu-${REQUIRED_UBUNTU_VERSION}-${NVIDIA_DRIVER_VERSION}-driver.deb ];then
            wget $NVIDIA_DRIVER_DOWNLOAD_URL -O $DIR_WORKSPACE/nvidia-ubuntu-${REQUIRED_UBUNTU_VERSION}-${NVIDIA_DRIVER_VERSION}-driver.deb
        fi
        sudo dpkg -i $DIR_WORKSPACE/nvidia-ubuntu-${REQUIRED_UBUNTU_VERSION}-${NVIDIA_DRIVER_VERSION}-driver.deb
        echo "✅ Nvidia GPU驱动重装完成，请重启系统以生效：sudo reboot"
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
        CUDA124_UPDATE_PARAMS="--force-reinstall --index-url $URL_SOURCE_PIP_TSINGHUA --extra-index-url $URL_SOURCE_CUDA"
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
# 函数定义：检查并安装CUDA Toolkit
# ==============================================
install_cuda_toolkit() {
    echo "-- 正在检查CUDA Toolkit是否安装..."
    if ! $CUDA_TOOLKIT_HOME/bin/nvcc --version &> /dev/null; then
        echo "❌ 未检测到CUDA Toolkit，正在安装版本$CUDA_TOOLKIT_VERSION_NUM..."

        echo "- 正在下载CUDA Toolkit安装程序..."
        if [ ! -f "$DIR_WORKSPACE/cuda_${CUDA_TOOLKIT_VERSION_NUM}_${NVIDIA_DRIVER_VERSION}_toolkit.run" ];then
            wget $CUDA_TOOLKIT_DOWNLOAD_URL \
                -O $DIR_WORKSPACE/cuda_${CUDA_TOOLKIT_VERSION_NUM}_${NVIDIA_DRIVER_VERSION}_toolkit.run
        fi

        echo "- 正在安装CUDA Toolkit..."
        sudo chmod u+x $DIR_WORKSPACE/cuda_${CUDA_TOOLKIT_VERSION_NUM}_${NVIDIA_DRIVER_VERSION}_toolkit.run
        sudo sh $DIR_WORKSPACE/cuda_${CUDA_TOOLKIT_VERSION_NUM}_${NVIDIA_DRIVER_VERSION}_toolkit.run --silent --toolkit
        # rm $DIR_WORKSPACE/cuda_toolkit.run
        grep -qxF "export PATH=\"$CUDA_TOOLKIT_HOME/bin:\$PATH\"" ~/.bashrc || echo "export PATH=\"$CUDA_TOOLKIT_HOME/bin:\$PATH\"" >> ~/.bashrc
        grep -qxF "export LD_LIBRARY_PATH=\"$CUDA_TOOLKIT_HOME/lib64:\$LD_LIBRARY_PATH\"" ~/.bashrc || echo "export LD_LIBRARY_PATH=\"$CUDA_TOOLKIT_HOME/lib64:\$LD_LIBRARY_PATH\"" >> ~/.bashrc
        source ~/.bashrc
        if [ -f $CUDA_TOOLKIT_HOME/bin/nvcc ]; then
            echo "✅ CUDA Toolkit $CUDA_TOOLKIT_VERSION_NUM 安装成功：$($CUDA_TOOLKIT_HOME/bin/nvcc --version | grep release)"
        else
            echo "❌ CUDA Toolkit安装失败，请检查安装日志。"
            exit 1
        fi
        echo "✅ CUDA Toolkit $CUDA_TOOLKIT_VERSION_NUM 安装完成。"
    else
        echo "✅ CUDA Toolkit已安装：$($CUDA_TOOLKIT_HOME/bin/nvcc --version | grep release)"
    fi
}
# ==============================================
# 函数定义：安装llama
# ==============================================
install_llama() {
    echo "-- 正在安装 llama-cpp-python ..."
    pip install --upgrade pip

    # [fix]: For llama-cpp-python dependencies must be installed with the following command
    conda install -c conda-forge libstdcxx-ng gcc -y

    # [fix]: For llama-cpp-python dependencies must be installed with the following command
    # pip install MarkupSafe==2.1.3 $CUDA124_UPDATE_PARAMS

    # [tips]: Because llama-cpp-python relies on specific system libraries, we need to ensure they are installed
    source ~/.bashrc

    # [fix]: For llama-cpp-python compile must be installed with the following command
    #        - libopenblas-dev is required used as Math library
    sudo apt-get install -y \
        libgomp1 \
        libopenblas-dev \
        autoconf \
        automake \
        libtool \
        ninja-build
    export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda-$CUDA_TOOLKIT_VERSION/lib64:$LD_LIBRARY_PATH

    export CMAKE_CUDA_TOOLKIT_PARAMS="-DCUDA_PATH=/usr/local/cuda-$CUDA_TOOLKIT_VERSION \
        -DCUDAToolkit_ROOT=/usr/local/cuda-$CUDA_TOOLKIT_VERSION \
        -DCMAKE_CUDA_COMPILER=/usr/local/cuda-$CUDA_TOOLKIT_VERSION/bin/nvcc \
        -DCMAKE_CUDA_ARCHITECTURES=\"native\" \
        -DCUDAToolkit_INCLUDE_DIR=/usr/local/cuda-$CUDA_TOOLKIT_VERSION/include \
        -DCUDAToolkit_LIBRARY_DIR=/usr/local/cuda-$CUDA_TOOLKIT_VERSION/lib64"

    LLAMA_UPDATE_PARAMS="--force-reinstall \
        --index-url $URL_SOURCE_PIP_TSINGHUA \
        --extra-index-url $URL_SOURCE_CUDA"

    CMAKE_ARGS_PARAMS="-DGGML_CUDA=ON \
            -DGGML_BLAS=ON \
            -DGGML_BLAS_VENDOR=OpenBLAS \
            -DGGML_AVX512=ON \
            -DGGML_CUDA_FA_ALL_QUANTS=ON \
            -DGGML_CUDA_MMV_Y=2 \
            -DGGML_CUDA_PEER_MAX_BATCH_SIZE=4096 \
            -DGGML_CUDA_USE_GRAPHS=ON \
            $CMAKE_CUDA_TOOLKIT_PARAMS"
    if [ $DEBUGGABLE ]; then
        CMAKE_ARGS=$CMAKE_ARGS_PARAMS \
        pip install --no-binary :all: llama-cpp-python --verbose $LLAMA_UPDATE_PARAMS
    else
        CMAKE_ARGS=$CMAKE_ARGS_PARAMS \
        pip install --no-binary :all: llama-cpp-python           $LLAMA_UPDATE_PARAMS
    fi
    # 检查llama-cpp-python是否为GPU版本
    # echo "-- 检查llama-cpp-python是否为GPU版本..."
    # if pip show llama-cpp-python | grep -E 'cuBLAS|cu124'; then
    #     echo "✅ llama-cpp-python GPU版本安装成功。"
    # else
    #     echo "❌ llama-cpp-python GPU版本未检测到，请检查安装参数。"
    #     exit 1
    # fi
    echo "✅ llama-cpp-python安装完成。"
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
}

main
