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

# 添加路径依赖
sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))
sys.path.append(os.path.join(os.getcwd(), "segment_anything"))

# Grounding DINO 导入
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# SAM 导入
from segment_anything import (
    sam_model_registry,
    sam_hq_model_registry,
    SamPredictor
)

# ==========================================
# 核心算法增强模块 (Core Enhancement Modules)
# ==========================================

def soft_nms_pytorch(dets, box_scores, sigma=0.1, thresh=0.001):
    """
    [Stage 1 Enhancement] Soft-NMS (软非极大值抑制)
    sigma=0.1: 温和抑制，允许较高程度的重叠（针对密集虫害）。
    """
    if dets.shape[0] == 0:
        return torch.tensor([]).long(), torch.tensor([])

    N = dets.shape[0]
    indexes = torch.arange(0, N, dtype=torch.long).view(N)
    dets = dets.float()
    box_scores = box_scores.float()

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    for i in range(N):
        tscore = box_scores[i].clone()
        pos = i + 1
        if i != N - 1:
            maxscore, maxpos = torch.max(box_scores[pos:], dim=0)
            if tscore < maxscore:
                dets[i], dets[maxpos + pos] = dets[maxpos + pos].clone(), dets[i].clone()
                box_scores[i], box_scores[maxpos + pos] = box_scores[maxpos + pos].clone(), box_scores[i].clone()
                areas[i], areas[maxpos + pos] = areas[maxpos + pos].clone(), areas[i].clone()
                indexes[i], indexes[maxpos + pos] = indexes[maxpos + pos].clone(), indexes[i].clone()

        xx1 = torch.maximum(dets[i, 0], dets[i+1:, 0])
        yy1 = torch.maximum(dets[i, 1], dets[i+1:, 1])
        xx2 = torch.minimum(dets[i, 2], dets[i+1:, 2])
        yy2 = torch.minimum(dets[i, 3], dets[i+1:, 3])

        w = torch.maximum(torch.tensor(0.0), xx2 - xx1)
        h = torch.maximum(torch.tensor(0.0), yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[i+1:] - inter)

        weight = torch.exp(-(ovr * ovr) / sigma)
        box_scores[i+1:] = box_scores[i+1:] * weight

    keep = box_scores > thresh
    return indexes[keep], box_scores[keep]


def refine_mask(mask_tensor, is_tiny_object=False):
    """
    [Stage 2 Enhancement] Mask Refinement
    根据 is_tiny_object 动态调整形态学策略。
    """
    mask_np = (mask_tensor.cpu().numpy() * 255).astype(np.uint8)
    
    if is_tiny_object:
        # === 虫害策略：绝对禁用腐蚀！===
        # 只做极小的闭运算连接断点
        kernel = np.ones((2, 2), np.uint8)
        mask_result = cv2.morphologyEx(mask_np, cv2.MORPH_CLOSE, kernel)
    
    else:
        # === 病害模式 (修正点) ===
        kernel = np.ones((3, 3), np.uint8)
        
        # 2. 闭运算 (Closing)
        # 作用：连接断裂的孢子点
        mask_result = cv2.morphologyEx(mask_np, cv2.MORPH_CLOSE, kernel)
        kernel = np.ones((2, 2), np.uint8) # 核稍微大一点，增强连接能力
        mask_result = cv2.morphologyEx(mask_np, cv2.MORPH_CLOSE, kernel)
        
       
        mask_result = cv2.GaussianBlur(mask_result, (3, 3), 0)
        _, mask_result = cv2.threshold(mask_result, 127, 255, cv2.THRESH_BINARY)
    
    return torch.from_numpy(mask_result > 0).bool()


def save_structured_result(output_dir, image_name, diagnosis_data, masks, boxes, scores, labels):
    result = {
        "image_id": image_name,
        "ai_diagnosis": {
            "disease_name": diagnosis_data.get("disease_name", "Unknown") if diagnosis_data else "Local Mode",
            "target_type": diagnosis_data.get("target_type", "Unknown") if diagnosis_data else "Unknown",
            "advice": diagnosis_data.get("control_advice", "N/A") if diagnosis_data else "N/A"
        },
        "detections": []
    }
    
    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        clean_label = label.split('(')[0] if '(' in label else label
        result["detections"].append({
            "id": i + 1,
            "label": clean_label,
            "confidence": round(float(score), 4),
            "bbox": box.tolist(),
        })
        
    json_path = os.path.join(output_dir, "analysis_result.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
    print(f"📊 结构化数据已保存: {json_path}")


# ==========================================
# 基础辅助函数
# ==========================================

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
    if boxes.shape[0] == 0:
        return boxes, logits

    areas = boxes[:, 2] * boxes[:, 3]
    keep_mask = areas < max_area_threshold
    
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

    filt_mask = logits.max(dim=1)[0] > box_threshold
    logits_filt = logits[filt_mask]
    boxes_filt = boxes[filt_mask]

    # === 调用面积过滤 ===
    boxes_filt, logits_filt = filter_boxes_by_area(boxes_filt, logits_filt, max_area_threshold)

    # === Soft-NMS ===
    scores = logits_filt.max(dim=1)[0]
    if len(scores) > 0:
        keep_indices, updated_scores = soft_nms_pytorch(boxes_filt, scores, sigma=0.1, thresh=box_threshold)
        boxes_filt = boxes_filt[keep_indices]
        logits_filt = logits_filt[keep_indices]

    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    pred_phrases = []
    
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)
            
    return boxes_filt, pred_phrases, logits_filt


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


# ==========================================
# 主程序
# ==========================================

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

    # === 1. 初始化阈值变量 (从 config.yaml 读取) ===
    # 注意：这里我们使用 config.get(key, default_value) 
    # 这样既能读取你的配置，又能在配置没写时有合理的默认值
    base_box_thresh = config.get("box_threshold", 0.3)
    text_thresh = config.get("text_threshold", 0.25)
    
    # [UPDATED] 从 Config 读取你指定的特殊阈值，不再写死
    pest_area_thresh = config.get("max_area_threshold_pest", 0.20)
    disease_area_thresh = config.get("max_area_threshold_disease", 0.60)
    default_area_thresh = config.get("max_area_threshold", 0.30)
    
    current_max_area = default_area_thresh
    current_box_thresh = base_box_thresh
    is_tiny_mode = False 

    if use_api:
        from multimodal_expert import get_plant_diagnosis_via_api

        if not access_key_id or not access_key_secret:
            raise ValueError("API Key Missing")

        print("🔍 正在调用 Qwen-VL API 分析病虫害...")
        diagnosis = get_plant_diagnosis_via_api(
            image_path=input_image,
            access_key_id=access_key_id,
            access_key_secret=access_key_secret,
            region_id=config.get("region_id", "cn-beijing")
        )

        print(f"✅ 病虫害名称: {diagnosis.get('disease_name', '未知')}")
        target_type = diagnosis.get("target_type", "")
        print(f"⚖️ 模型判定目标类型: {target_type}")

        # === 2. 核心逻辑：根据目标类型动态切换阈值 (Config-Driven) ===
        if "微小个体" in target_type:
            current_max_area = pest_area_thresh
            current_box_thresh = 0.20 
            is_tiny_mode = True 
            print(f"⚙️ 策略调整：应用【虫害严格模式】 (Max Area={current_max_area}, Box Thresh={current_box_thresh})")
            print("   -> 已启用：禁用 Mask 腐蚀，降低 Soft-NMS 惩罚")
            
        elif "成片纹理" in target_type:
            current_max_area = disease_area_thresh
            current_box_thresh = 0.25 
            is_tiny_mode = False
            print(f"⚙️ 策略调整：应用【病害宽容模式】 (Max Area={current_max_area}, Box Thresh={current_box_thresh})")
        else:
            current_max_area = default_area_thresh
            is_tiny_mode = False
            print(f"⚙️ 策略保持：默认阈值 (Max Area={current_max_area})")

        if diagnosis["disease_name"] in ["无法确定", "API 调用失败", "图像读取失败"]:
            text_prompt = "plant disease symptoms"
        else:
            text_prompt = diagnosis.get('english_prompt', "plant disease symptoms")
            
    else:
        text_prompt = config.get("text_prompt")
        diagnosis = None
        current_max_area = default_area_thresh
        is_tiny_mode = False

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
    
    print(f"🚀 正在检测: '{text_prompt}' (Box Thresh: {current_box_thresh}, Area Limit: {current_max_area})")
    
    boxes_filt, pred_phrases, logits_filt = get_grounding_output(
        model, 
        image, 
        text_prompt, 
        current_box_thresh, 
        text_thresh, 
        max_area_threshold=current_max_area,
        device=device
    )

    print(f"🔍 检测到 {boxes_filt.shape[0]} 个目标")

    if boxes_filt.shape[0] == 0:
        print("⚠️ 未检测到目标，程序结束。")
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
    
    print("🎨 正在运行 SAM 分割...")
    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )

    # === 4. Mask Refinement (传递 correct flag) ===
    print(f"✨ 正在优化 Mask (小目标模式: {is_tiny_mode})...")
    refined_masks = []
    for mask in masks:
        refined_masks.append(refine_mask(mask[0], is_tiny_object=is_tiny_mode)) 
    masks = torch.stack(refined_masks).unsqueeze(1).to(device)
    # ==========================================

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

    final_scores = logits_filt.max(dim=1)[0].cpu().numpy()
    save_structured_result(output_dir, os.path.basename(input_image), diagnosis, masks, boxes_filt, final_scores, pred_phrases)
    save_mask_data(output_dir, masks, boxes_filt, pred_phrases)

    print(f"✅ 处理完成！结果目录: {output_dir}")