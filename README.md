# Grounded-Segment-Anything (My Custom Implementation)

本项目基于 [IDEA-Research/Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything)，实现了基于文本提示的物体检测与分割 (Grounding DINO + SAM)。

## 🛠️ 环境安装 (Installation)

推荐使用 Anaconda 创建虚拟环境：

```bash
# 1. 创建并激活环境
conda create -n grounded-sam python=3.8
conda activate grounded-sam

# 2. 安装 PyTorch (请根据你的 CUDA 版本调整)
pip install torch torchvision --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)

# 3. 安装依赖库
pip install -r requirements.txt
python -m pip install -e segment_anything
python -m pip install -e GroundingDINO
```

## 📥 模型权重下载 (Model Weights)

**注意：** 由于模型权重文件较大，未包含在仓库中。请在运行前手动下载以下权重文件，并放置在项目根目录下。

### 1. 下载 GroundingDINO 权重
```bash
wget [https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth](https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth)
```

### 2. 下载 SAM 权重 (ViT-H)
```bash
wget [https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)
```

> **提示**：如果你无法使用 `wget`，请直接点击链接下载后手动上传到服务器。

## 🚀 快速开始 (Quick Start)

### 运行 Demo
使用以下命令对图片进行检测和分割：

```bash
export CUDA_VISIBLE_DEVICES=0
python grounded_sam_demo.py \
  --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --grounded_checkpoint groundingdino_swint_ogc.pth \
  --sam_checkpoint sam_vit_h_4b8939.pth \
  --input_image assets/demo1.jpg \
  --output_dir "outputs" \
  --box_threshold 0.3 \
  --text_threshold 0.25 \
  --text_prompt "bear" \
  --device "cuda"
```

运行成功后，结果将保存在 `outputs/` 文件夹中。

## 📂 目录结构说明
* `GroundingDINO/`: 检测模型源码
* `segment_anything/`: 分割模型源码
* `assets/`: 测试图片
* `outputs/`: 结果输出目录 (默认被 git 忽略)
* `weights/`: (可选) 存放权重的目录

## 🔗 引用与致谢
本项目参考自官方仓库：[IDEA-Research/Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything)
