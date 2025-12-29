import argparse
import os
import sys
import yaml
import json
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))
sys.path.append(os.path.join(os.getcwd(), "segment_anything"))

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# SAM
from segment_anything import (
    sam_model_registry,
    sam_hq_model_registry,
    SamPredictor
)


def load_image(image_path):
    image_pil = Image.open(image_path).convert("RGB")
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image, _ = transform(image_pil, None)
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, bert_base_uncased_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    if bert_base_uncased_path:
        args.bert_base_uncased_path = bert_base_uncased_path
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def filter_boxes_by_area(boxes, logits, max_area_threshold=0.30):
    """
    根据面积比例过滤过大的框（防止框住整个叶片）。
    """
    if boxes.shape[0] == 0:
        return boxes, logits

    # 计算面积 (w * h)
    areas = boxes[:, 2] * boxes[:, 3]
    
    # 生成保留掩码 (True 表示保留)
    keep_mask = areas < max_area_threshold
    
    # 打印调试信息
    num_original = len(boxes)
    num_kept = keep_mask.sum().item()
    if num_original - num_kept > 0:
        print(f"🧹 [面积过滤] 已剔除 {num_original - num_kept} 个过大的框 (面积占比 > {max_area_threshold})")
    
    return boxes[keep_mask], logits[keep_mask]


def get_grounding_output(model, image, caption, box_threshold, text_threshold, max_area_threshold=0.30, with_logits=True, device="cpu"):
    caption = caption.lower().strip()
    if not caption.endswith("."):
        caption += "."
    
    model = model.to(device)
    image = image.to(device)
    
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    
    logits = outputs["pred_logits"].cpu().sigmoid()[0]
    boxes = outputs["pred_boxes"].cpu()[0]

    # 1. 基础置信度过滤
    filt_mask = logits.max(dim=1)[0] > box_threshold
    logits_filt = logits[filt_mask]
    boxes_filt = boxes[filt_mask]

    # 2. 调用面积过滤函数
    boxes_filt, logits_filt = filter_boxes_by_area(boxes_filt, logits_filt, max_area_threshold)

    # 3. 生成文本标签
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    pred_phrases = []
    
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)
            
    return boxes_filt, pred_phrases


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
    ax.text(x0, y0, label)


def save_mask_data(output_dir, mask_list, box_list, label_list):
    value = 0
    mask_img = torch.zeros(mask_list.shape[-2:])
    for idx, mask in enumerate(mask_list):
        mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
    plt.figure(figsize=(10, 10))
    plt.imshow(mask_img.numpy())
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'mask.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)

    json_data = [{'value': value, 'label': 'background'}]
    for label, box in zip(label_list, box_list):
        value += 1
        name, logit = label.split('(')
        logit = logit[:-1]
        json_data.append({
            'value': value,
            'label': name,
            'logit': float(logit),
            'box': box.numpy().tolist(),
        })
    with open(os.path.join(output_dir, 'mask.json'), 'w') as f:
        json.dump(json_data, f, indent=2)


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Grounded-SAM with Qwen-VL API (Config Support)")
    parser.add_argument("--config_file", type=str, default="config.yaml", help="Path to YAML config file")
    parser.add_argument("--input_image", type=str, help="Override input image path")
    parser.add_argument("--output_dir", type=str, help="Override output directory")
    parser.add_argument("--device", type=str, help="Override device (cuda/cpu)")
    parser.add_argument("--use_api", action="store_true", help="Force enable API mode")
    parser.add_argument("--access_key_id", type=str, help="Override AccessKey ID")
    parser.add_argument("--access_key_secret", type=str, help="Override AccessKey Secret")

    args = parser.parse_args()

    config = load_config(args.config_file)

    def override(key, default=None):
        return getattr(args, key) if getattr(args, key) is not None else config.get(key, default)

    input_image = override("input_image")
    output_dir = override("output_dir")
    device = override("device", "cpu")
    use_api = args.use_api or config.get("use_api", False)

    access_key_id = (
        args.access_key_id or
        config.get("access_key_id") or
        os.getenv("ALIBABA_CLOUD_ACCESS_KEY_ID")
    )
    access_key_secret = (
        args.access_key_secret or
        config.get("access_key_secret") or
        os.getenv("ALIBABA_CLOUD_ACCESS_KEY_SECRET")
    )

    # === 1. 初始化阈值变量 (从 yaml 读取) ===
    box_thresh = config.get("box_threshold", 0.3)
    text_thresh = config.get("text_threshold", 0.25)
    
    # 默认阈值
    default_area_thresh = config.get("max_area_threshold", 0.30)
    # 虫害阈值 (微小)
    pest_area_thresh = config.get("max_area_threshold_pest", 0.10)
    # 病害阈值 (成片)
    disease_area_thresh = config.get("max_area_threshold_disease", 0.60)
    
    # 当前使用的阈值 (先设为默认)
    current_max_area = default_area_thresh

    # >>>>>>>>>>>>>>> 调用多模态大模型（API） <<<<<<<<<<<<<<<<<
    if use_api:
        from multimodal_expert import get_plant_diagnosis_via_api

        if not access_key_id or not access_key_secret:
            raise ValueError(
                "启用了 --use_api，但未提供 access_key_id 和 access_key_secret。\n"
                "请在 config.yaml 中填写，或通过命令行/环境变量提供。"
            )

        print("🔍 正在调用 Qwen-VL API 分析病虫害...")
        diagnosis = get_plant_diagnosis_via_api(
            image_path=input_image,
            access_key_id=access_key_id,
            access_key_secret=access_key_secret,
            region_id=config.get("region_id", "cn-beijing")
        )

        print(f"✅ 病虫害名称: {diagnosis.get('disease_name', '未知')}")
        print(f"🔤 检测提示词(英文): {diagnosis.get('english_prompt', 'N/A')}")
        print(f"📍 区域描述: {diagnosis.get('region_description', '')}")
        print(f"🛡️ 防治建议: {diagnosis.get('control_advice', '')}")
        
        # 获取大模型判断的目标类型
        target_type = diagnosis.get("target_type", "")
        print(f"⚖️ 模型判定目标类型: {target_type}")

        # === 2. 核心逻辑：根据目标类型动态切换阈值 ===
        if "微小个体" in target_type:
            current_max_area = pest_area_thresh
            print(f"⚙️ 策略调整：应用【虫害严格模式】 (Max Area = {current_max_area})")
        elif "成片纹理" in target_type:
            current_max_area = disease_area_thresh
            print(f"⚙️ 策略调整：应用【病害宽容模式】 (Max Area = {current_max_area})")
        else:
            current_max_area = default_area_thresh
            print(f"⚙️ 策略保持：未触发特殊规则，使用默认阈值 (Max Area = {current_max_area})")
        # ==========================================

        if diagnosis["disease_name"] in ["无法确定", "API 调用失败", "图像读取失败"]:
            text_prompt = "plant disease symptoms"
        else:
            text_prompt = diagnosis.get('english_prompt', "plant disease symptoms")
            
    else:
        text_prompt = config.get("text_prompt")
        if not text_prompt:
            raise ValueError("未启用 API 模式，请在 config.yaml 中设置 text_prompt")
        diagnosis = None
        # 非 API 模式使用默认配置
        current_max_area = default_area_thresh
    # >>>>>>>>>>>>>>> API 调用结束 <<<<<<<<<<<<<<<<<

    gdino_config = config["config"]
    gdino_ckpt = config["grounded_checkpoint"]
    bert_path = config.get("bert_base_uncased_path")
    sam_version = config["sam_version"]
    sam_ckpt = config["sam_checkpoint"]
    sam_hq_ckpt = config.get("sam_hq_checkpoint")
    use_sam_hq = config.get("use_sam_hq", False)
    
    os.makedirs(output_dir, exist_ok=True)
    image_pil, image = load_image(input_image)
    image_pil.save(os.path.join(output_dir, "raw_image.jpg"))

    model = load_model(gdino_config, gdino_ckpt, bert_path, device=device)
    
    # === 3. 调用检测函数，传入动态计算后的 current_max_area ===
    print(f"🚀 正在使用提示词进行检测: '{text_prompt}' (Box阈值: {box_thresh}, 动态面积阈值: {current_max_area})")
    
    boxes_filt, pred_phrases = get_grounding_output(
        model, 
        image, 
        text_prompt, 
        box_thresh, 
        text_thresh, 
        max_area_threshold=current_max_area, # <--- 关键修改点：使用动态变量
        device=device
    )

    print(f"🔍 GroundingDINO 检测到了 {boxes_filt.shape[0]} 个目标")
    if boxes_filt.shape[0] == 0:
        print("⚠️ 警告：未检测到任何病害区域！SAM 将跳过执行以防止崩溃。")
        print("💡 建议：1. 检查 config.yaml 中的 box_threshold 是否过高 (建议 0.15)")
        print("          2. 确保 multimodal_expert.py 已更新并输出了正确的英文提示词")
        
        if use_api and diagnosis:
            report_path = os.path.join(output_dir, "diagnosis_report.txt")
            with open(report_path, "w", encoding="utf-8") as f:
                f.write("🌱 智能植保诊断报告\n")
                f.write("=" * 40 + "\n")
                f.write(f"病虫害名称：{diagnosis.get('disease_name')}\n")
                f.write("【注意】虽然确诊了病害，但算法未在图中定位到具体病斑。\n")
                f.write(f"防治建议：\n{diagnosis.get('control_advice')}\n")
            print(f"📄 诊断报告已保存至: {report_path}")
        
        sys.exit(0)

    if use_sam_hq:
        predictor = SamPredictor(sam_hq_model_registry[sam_version](checkpoint=sam_hq_ckpt).to(device))
    else:
        predictor = SamPredictor(sam_model_registry[sam_version](checkpoint=sam_ckpt).to(device))

    image_cv = cv2.imread(input_image)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_cv)

    W, H = image_pil.size
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]

    boxes_filt = boxes_filt.cpu()
    transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image_cv.shape[:2]).to(device)
    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )

    plt.figure(figsize=(10, 10))
    plt.imshow(image_cv)
    for mask in masks:
        show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
    for box, label in zip(boxes_filt, pred_phrases):
        show_box(box.numpy(), plt.gca(), label)
    plt.axis('off')
    plt.savefig(
        os.path.join(output_dir, "grounded_sam_output.jpg"),
        bbox_inches="tight", dpi=300, pad_inches=0.0
    )

    save_mask_data(output_dir, masks, boxes_filt, pred_phrases)

    if use_api:
        report_path = os.path.join(output_dir, "diagnosis_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("🌱 智能植保诊断报告\n")
            f.write("=" * 40 + "\n")
            f.write(f"病虫害名称：{diagnosis.get('disease_name')}\n\n")
            f.write(f"区域描述：{diagnosis.get('region_description')}\n\n")
            f.write(f"防治建议：\n{diagnosis.get('control_advice')}\n")
        print(f"📄 诊断报告已保存至: {report_path}")

    print(f"✅ 处理完成！结果目录: {output_dir}")