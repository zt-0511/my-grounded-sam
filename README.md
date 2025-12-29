# Agri-Grounded-SAM: 基于大模型与 RAG 的农作物病虫害智能检测系统

> **西北农林科技大学 - 大学生科创项目**
>
> 本项目基于 [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything) 进行二次开发，融合了 **Qwen-VL 大模型**与 **RAG (检索增强生成)** 技术，旨在解决复杂背景下的农作物病害与虫害的自适应检测与分割问题。

---

## ✨ 核心创新点 (Key Features)

1. **全配置驱动 (Config-Driven)**  
   摒弃繁琐命令行参数，所有运行参数（模型路径、阈值、API 密钥）均通过 `config.yaml` 统一管理，操作更简便。

2. **自适应阈值策略 (Adaptive Thresholding)**  
   - **虫害 (Pest)**：采用严格面积限制 (`max_area_threshold_pest`)，防止背景误检为微小害虫。  
   - **病害 (Disease)**：采用宽容面积限制 (`max_area_threshold_disease`)，允许病斑覆盖大面积叶片。

3. **大模型辅助推理**  
   集成 Qwen-VL API，通过 RAG 模块自动分析图片内容并生成精准检测提示词 (Text Prompt)，无需人工输入类别。

---

## Installation

The code requires `python>=3.8`, as well as `pytorch>=1.7` and `torchvision>=0.8`. Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies. Installing both PyTorch and TorchVision with CUDA support is strongly recommended.



### Install without Docker

You should set the environment variable manually as follows if you want to build a local GPU environment for Grounded-SAM:


### 1. 基础环境配置

建议使用 Conda 创建独立虚拟环境（Python 3.8+）：

```bash
conda create -n grounded-sam-rag python=3.8
conda activate grounded-sam-rag
```

```bash
export AM_I_DOCKER=False
export BUILD_WITH_CUDA=True
export CUDA_HOME=/path/to/cuda-11.3/  ??不一定，可能会遇到错误
```

Install Segment Anything:

```bash
python -m pip install -e segment_anything
```

Install Grounding DINO:

```bash
pip install --no-build-isolation -e GroundingDINO
```

Install diffusers:

```bash
pip install --upgrade diffusers[torch]
```


Install RAM & Tag2Text:

```bash
git clone https://github.com/xinyu1205/recognize-anything.git
pip install -r ./recognize-anything/requirements.txt
pip install -e ./recognize-anything/
```

---

### Optional Dependencies

The following optional dependencies are necessary for mask post-processing, saving masks in COCO format, running example notebooks, and exporting the model in ONNX format. `jupyter` is also required to run the example notebooks.

```bash
pip install opencv-python pycocotools matplotlib onnxruntime onnx ipykernel
```



## Step 1: Download the pretrained weights

```bash
cd Grounded-Segment-Anything

wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

## Step 2: Running Project

```bash
export CUDA_VISIBLE_DEVICES=0
python grounded_sam_with_RAG.py
```

## 📥 模型权重准备 (Model Weights)

如果上面 wget 命令没有正常请下载以下核心权重文件，并建议放置在项目根目录下（需在 `config.yaml` 中修改对应路径）：

| 模型名称                 | 说明                   | 下载地址         |
|--------------------------|------------------------|------------------|
| `groundingdino_swint_ogc.pth` | 用于目标检测           | [点击下载](#)    |
| `sam_vit_h_4b8939.pth`        | 用于图像分割 (SAM Huge)| [点击下载](#)    |
| `bert-base-uncased`           | 文本编码器（首次运行自动下载） | —                |

---

## ⚙️ 配置文件说明 (Configuration)

本项目运行完全依赖 `config.yaml`。运行前，请务必根据环境修改以下参数。

**文件位置**: `./config.yaml`

```yaml
# ================= 模型路径配置 =================
config: "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
# ⚠️ 请修改为你的本地绝对路径
grounded_checkpoint: "/path/to/your/groundingdino_swint_ogc.pth" 
sam_checkpoint: "/path/to/your/sam_vit_h_4b8939.pth"
bert_base_uncased_path: null  # 留 null 即可自动下载

# ================= SAM 配置 =================
sam_version: "vit_h"
use_sam_hq: false

# ================= 输入输出 =================
# 待检测图片的路径
input_image: "./data/test_image.jpg"
# 结果保存目录
output_dir: "./outputs"

# ================= 核心阈值参数 =================
device: "cuda"          # 使用 GPU
box_threshold: 0.15     # 检测框置信度阈值
text_threshold: 0.15    # 文本匹配阈值
max_area_threshold: 0.5 # 通用最大面积阈值

# --- 🎯 创新点：病虫害差异化阈值 ---
# 虫害模式：严格限制，防止把整片叶子当成虫子 (建议 0.05 - 0.15)
max_area_threshold_pest: 0.2

# 病害模式：宽容限制，允许病斑覆盖大半个叶子 (建议 0.50 - 0.80)
max_area_threshold_disease: 0.60

# ================= 大模型 API 配置 =================
# 是否启用 LLM 自动识别
use_api: true
# ⚠️ 替换为你的 DashScope/Qwen API Key
access_key_id: "YOUR_API_KEY_HERE" 
access_key_secret: "unused_placeholder"

# 若 use_api: false，则需手动填写下方 prompt
# text_prompt: "rice blast on leaves"
```

---

## 🚀 运行步骤 (Usage)

### 步骤 1: 修改配置

打开 `config.yaml`，填入你的图片路径 (`input_image`) 和 API Key。

### 步骤 2: 运行主程序

直接运行主脚本，程序会自动读取配置并执行检测流程：

```bash
python grounded_sam_with_RAG.py
```

### 步骤 3: 查看结果

运行完成后，前往 `output_dir` 配置的目录查看生成的结果图片和 JSON 数据。

---

## 📂 目录结构 (File Structure)

```
.
├── config.yaml                 # [核心] 项目配置文件
├── grounded_sam_with_RAG.py    # [核心] 主运行脚本
├── requirements.txt            # 依赖列表
├── GroundingDINO/              # 检测模块源码
├── segment_anything/           # 分割模块源码
├── data/                       # 存放输入图片
└── outputs/                    # 存放输出结果 (自动生成)
```

---

## 🔗 引用 (Citation)

**Original Implementation**: [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything)
