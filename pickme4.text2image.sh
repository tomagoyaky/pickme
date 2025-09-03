#!/bin/bash
clear
set -e

source ./pickme0.common.sh
readonly DIR_REPO="$DIR_WORKSPACE/ComfyUI"

readonly PYTHON_VERSION="3.12"
readonly CONDA_ENV_NAME="comfyui-python-${PYTHON_VERSION}"
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
    cd "$DIR_REPO"
    pip install -r "$DIR_REPO/requirements.txt"
}
comfyui_server() {
    cd "$DIR_REPO"
    # 清理历史进程
    pkill -f "python main.py" || true

    # 后台启动 ComfyUI
    mkdir -p $DIR_WORKSPACE/logs
    nohup python main.py \
        --output-directory $DIR_WORKSPACE/output \
        --cuda-device 0 \
        --force-fp16 \
        > $DIR_WORKSPACE/logs/comfyui.log 2>&1 &
    echo "-- ComfyUI 已启动，请访问 http://127.0.0.1:8188 进行使用。"
}
modles_setup(){

    # text2Image 模型
    if [ ! -f $DIR_REPO/models/checkpoints/v1-5-pruned-emaonly-fp16.safetensors ];then
        wget https://huggingface.co/Comfy-Org/stable-diffusion-v1-5-archive/resolve/main/v1-5-pruned-emaonly-fp16.safetensors?download=true \
            -O $DIR_REPO/models/checkpoints/v1-5-pruned-emaonly-fp16.safetensors
    fi
}
# ==============================================
# 函数定义：文字转图像
# ==============================================
text2image() {
    local message="$1"
    local check_point_model_file_name="$2"
    local step_num=$3
    if [[ -z "$message" ]]; then
        echo "❌ 未提供输入文本"
        return 1
    fi
    echo -e "-- 正在将文本转换为图像：$message ...\n"
    # 这里调用实际的文本转图像命令
    cd "$DIR_REPO"
    local json=$(cat <<EOF
{
  "prompt": {
    "3": {
      "inputs": {
        "seed": 357088847916134,
        "steps": $step_num,
        "cfg": 8,
        "sampler_name": "euler",
        "scheduler": "normal",
        "denoise": 1,
        "model": [
          "4",
          0
        ],
        "positive": [
          "6",
          0
        ],
        "negative": [
          "7",
          0
        ],
        "latent_image": [
          "5",
          0
        ]
      },
      "class_type": "KSampler",
      "_meta": {
        "title": "K采样器"
      }
    },
    "4": {
      "inputs": {
        "ckpt_name": "$check_point_model_file_name"
      },
      "class_type": "CheckpointLoaderSimple",
      "_meta": {
        "title": "Checkpoint加载器（简易）"
      }
    },
    "5": {
      "inputs": {
        "width": 512,
        "height": 512,
        "batch_size": 1
      },
      "class_type": "EmptyLatentImage",
      "_meta": {
        "title": "空Latent图像"
      }
    },
    "6": {
      "inputs": {
        "text": "$message",
        "clip": [
          "4",
          1
        ]
      },
      "class_type": "CLIPTextEncode",
      "_meta": {
        "title": "CLIP文本编码"
      }
    },
    "7": {
      "inputs": {
        "text": "text, watermark",
        "clip": [
          "4",
          1
        ]
      },
      "class_type": "CLIPTextEncode",
      "_meta": {
        "title": "CLIP文本编码"
      }
    },
    "8": {
      "inputs": {
        "samples": [
          "3",
          0
        ],
        "vae": [
          "4",
          2
        ]
      },
      "class_type": "VAEDecode",
      "_meta": {
        "title": "VAE解码"
      }
    },
    "9": {
      "inputs": {
        "filename_prefix": "ComfyUI",
        "images": [
          "8",
          0
        ]
      },
      "class_type": "SaveImage",
      "_meta": {
        "title": "保存图像"
      }
    }
  }
}
EOF
    )
    curl -X POST http://127.0.0.1:8188/prompt \
        -H "Content-Type: application/json" \
        -d "$json"
    echo -e "\n"
}

# ==============================================
# 函数定义：图像转图像（图生图）
# ==============================================
image2image() {
  local input_image="$1"
  local prompt="$2"
  local check_point_model_file_name="$3"
  local step_num=$4
  if [[ ! -f "$input_image" ]]; then
    echo "❌ 输入图像文件不存在: $input_image"
    return 1
  fi
  if [[ -z "$prompt" ]]; then
    echo "❌ 未提供输入文本"
    return 1
  fi
  echo -e "-- 正在将图像和文本转换为新图像：$input_image, $prompt ...\n"
  cd "$DIR_REPO"
  local json=$(cat <<EOF
{
  "prompt": {
    "3": {
      "inputs": {
        "seed": 280823642470253,
        "steps": $step_num,
        "cfg": 8,
        "sampler_name": "dpmpp_2m",
        "scheduler": "normal",
        "denoise": 0.8700000000000001,
        "model": [
          "14",
          0
        ],
        "positive": [
          "6",
          0
        ],
        "negative": [
          "7",
          0
        ],
        "latent_image": [
          "12",
          0
        ]
      },
      "class_type": "KSampler",
      "_meta": {
        "title": "K采样器"
      }
    },
    "6": {
      "inputs": {
        "text": "$prompt",
        "clip": [
          "14",
          1
        ]
      },
      "class_type": "CLIPTextEncode",
      "_meta": {
        "title": "CLIP文本编码"
      }
    },
    "7": {
      "inputs": {
        "text": "watermark, text\n",
        "clip": [
          "14",
          1
        ]
      },
      "class_type": "CLIPTextEncode",
      "_meta": {
        "title": "CLIP文本编码"
      }
    },
    "8": {
      "inputs": {
        "samples": [
          "3",
          0
        ],
        "vae": [
          "14",
          2
        ]
      },
      "class_type": "VAEDecode",
      "_meta": {
        "title": "VAE解码"
      }
    },
    "9": {
      "inputs": {
        "filename_prefix": "ComfyUI",
        "images": [
          "8",
          0
        ]
      },
      "class_type": "SaveImage",
      "_meta": {
        "title": "保存图像"
      }
    },
    "10": {
      "inputs": {
        "image": "$input_image"
      },
      "class_type": "LoadImage",
      "_meta": {
        "title": "加载图像"
      }
    },
    "12": {
      "inputs": {
        "pixels": [
          "18",
          0
        ],
        "vae": [
          "14",
          2
        ]
      },
      "class_type": "VAEEncode",
      "_meta": {
        "title": "VAE编码"
      }
    },
    "14": {
      "inputs": {
        "ckpt_name": "$check_point_model_file_name"
      },
      "class_type": "CheckpointLoaderSimple",
      "_meta": {
        "title": "Checkpoint加载器（简易）"
      }
    },
    "18": {
      "inputs": {
        "upscale_method": "nearest-exact",
        "megapixels": 0.25,
        "image": [
          "10",
          0
        ]
      },
      "class_type": "ImageScaleToTotalPixels",
      "_meta": {
        "title": "缩放图像（像素）"
      }
    }
  }
}
EOF
  )
  curl -X POST http://127.0.0.1:8188/prompt \
    -H "Content-Type: application/json" \
    -d "$json"
  echo -e "\n"
}
# 查询 ComfyUI 队列状态，判断模型是否生成完毕
check_queue_status() {
    local api_url="http://127.0.0.1:8188/queue"
    local max_attempts=60
    local attempt=1
    local sleep_time=5

    echo "-- 正在查询 ComfyUI 队列状态..."

    while (( attempt <= max_attempts )); do
        local response=$(curl -s "$api_url")
        echo "[debug] $response"
        local pending=$(echo "$response" | grep -o '"pending": *[0-9]*' | grep -o '[0-9]*')
        local running=$(echo "$response" | grep -o '"running": *[0-9]*' | grep -o '[0-9]*')
        echo "pending: $pending, running: $running"

        if [[ "$pending" == "0" && "$running" == "0" ]]; then
            echo "✅ 模型生成已完成。"
            return 0
        else
            echo "⏳ 队列中还有任务，等待中... (pending: $pending, running: $running)"
            sleep $sleep_time
            ((attempt++))
        fi
    done

    echo "⚠️ 超时：模型生成未完成。"
    return 1
}
# 函数定义：下载 ComfyUI 所需模型
# ==============================================
download_comfyui_model() {
    local comfyui_models_dir="$DIR_WORKSPACE/ComfyUI/models"
    mkdir -p "$comfyui_models_dir"
    # 示例：下载Stable Diffusion模型
    local sd_model="runwayml/stable-diffusion-v1-5"
    download_huggingface_package "$sd_model" "$comfyui_models_dir/stable-diffusion-v1-5"
    # download_huggingface_package "$sd_model"
    # 可根据需要添加更多模型下载
}
# ==============================================
# 主函数：脚本入口
# ==============================================
message=$1
check_ubuntu_version
# 检查conda环境并创建/激活
check_and_create_conda_env


if [ ! -n "$message" ]; then
    message=$(cat "$DIR_WORKSPACE/output/prompt2txt.result.txt")
fi
    # fix：确保输入文本格式正确（如无特殊字符、空格），且tokenizer能正确转换为token IDs
# 清理message中的特殊字符、多余空格、换行符和空行
message=$(echo "$message" | tr -d '\r' | sed '/^[[:space:]]*$/d' | sed 's/[[:space:]]\+/ /g' | sed 's/^[ \t]*//;s/[ \t]*$//')

echo "======================================================"
echo "$message\n"
echo "======================================================"

# 安装PyTorch
install_pytorch
# 安装Hugging Face CLI
install_huggingface_cli
# 安装 ComfyUI
install_comfyui
# 下载 ComfyUI 模型
# download_comfyui_model  # 这个会把模型仓库全部都下载下来，太大了

# 模型下载
# modles_setup
# 启动 ComfyUI 服务器
if [ ! -f "$DIR_WORKSPACE/logs/comfyui.log" ]; then
    comfyui_server
else
    echo "✅ ComfyUI 服务器已启动，日志文件位于 $DIR_WORKSPACE/logs/comfyui.log"
fi

step_num=120

# 文字转图像
json=$(cat <<EOF
[
  "一只可爱的猫，宫崎骏漫画风格",
  "一只可爱的兔子，宫崎骏漫画风格",
  "一只凶猛的老虎，宫崎骏漫画风格",
  "一只草原上的长颈鹿，宫崎骏漫画风格",
  "一只侏罗纪时期的恐龙，宫崎骏漫画风格"
]
EOF
)
# for i in {0..4}; do
#   echo "第 $i 次生成图像..."
#   prompt=$(jq -r ".[$i]" <<< "$json")
#   echo "prompt: $prompt"

#   # 异步的提交生图请求
#   text2image "$prompt" "v1-5-pruned-emaonly-fp16.safetensors" $step_num
# done
json2=$(cat <<EOF
[
  "红色衣服，皮衣，黑色短发的女性战士",
  "黄色衣服，毛衣，白色短发的女修士",
  "吊带衣服，裙子，马尾辫的女学生"
]
EOF
)

for i in {0..2}; do
  echo "第 $i 次图片生成图像..."
  prompt=$(jq -r ".[$i]" <<< "$json2")
  echo "prompt: $prompt"

  # 异步的提交生图请求
  image2image "/home/x/Documents/smart_tools/pickme/workspace/input.png" "$prompt" "v1-5-pruned-emaonly-fp16.safetensors" $step_num
done