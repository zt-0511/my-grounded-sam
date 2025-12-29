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
from tqdm import tqdm  # 进度条库，如果没有请 pip install tqdm

# 添加路径依赖
sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))
sys.path.append(os.path.join(os.getcwd(), "segment_anything"))

# 导入 RAG 引擎 (必须确保 rag_engine.py 在同级目录)
try:
    sys.path.append(os.getcwd()) # 强制把当前目录加入搜索路径
    from rag_engine import PlantRAGSystem
except ImportError as e:
    print(f"❌ 致命错误: 无法导入 rag_engine。")
    print(f"🔍 真实报错信息: {e}")  # <--- 这行代码会告诉你真相
    print(f"📂 当前工作目录: {os.getcwd()}")
    print("💡 提示: 如果报错是 'No module named sentence_transformers'，请运行 pip install sentence-transformers")
    sys.exit(1)

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
    [Stage 1 Enhancement] Soft-NMS
    sigma=0.1: 极温和抑制，最大程度保留密集重叠的真实目标
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
    [Stage 2 Enhancement] 自适应形态学修复
    - 微小虫害: 闭运算连接 (禁止腐蚀)
    - 条纹病害: 开运算切断 (禁止粘连)
    """
    mask_np = (mask_tensor.cpu().numpy() * 255).astype(np.uint8)
    
    if is_tiny_object:
        # === 虫害策略 (Tiny Mode) ===
        # 作用：连接断裂的虫腿/触角，严禁腐蚀
        kernel = np.ones((2, 2), np.uint8)
        mask_result = cv2.morphologyEx(mask_np, cv2.MORPH_CLOSE, kernel)
    else:
        # === 病害策略 (Texture Mode) ===
        # 针对条锈病优化：使用微小开运算切断条纹间的粘连
        kernel = np.ones((2, 2), np.uint8)
        mask_result = cv2.morphologyEx(mask_np, cv2.MORPH_OPEN, kernel)
        
        # 边缘平滑
        mask_result = cv2.GaussianBlur(mask_result, (3, 3), 0)
        _, mask_result = cv2.threshold(mask_result, 127, 255, cv2.THRESH_BINARY)
    
    return torch.from_numpy(mask_result > 0).bool()


def save_structured_result(output_dir, image_name, diagnosis_data, masks, boxes, scores, labels, rag_info=None):
    """
    [Stage 3 Enhancement] 生成包含 RAG 信息的结构化报告
    """
    result = {
        "image_id": image_name,
        "ai_diagnosis": {
            "disease_name": diagnosis_data.get("disease_name", "Unknown") if diagnosis_data else "Local Mode",
            "target_type": diagnosis_data.get("target_type", "Unknown") if diagnosis_data else "Unknown",
        },
        "rag_metadata": rag_info if rag_info else "RAG Not Triggered",
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


# ==========================================
# 基础辅助函数
# ==========================================

# def load_image(image_path):
#     image_pil = Image.open(image_path).convert("RGB")
#     transform = T.Compose([
#         T.RandomResize([800], max_size=1333),
#         T.ToTensor(),
#         T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#     ])
#     image, _ = transform(image_pil, None)
#     return image_pil, image


# def load_model(model_config_path, model_checkpoint_path, bert_base_uncased_path, device):
#     args = SLConfig.fromfile(model_config_path)
#     args.device = device
#     if bert_base_uncased_path:
#         args.bert_base_uncased_path = bert_base_uncased_path
#     model = build_model(args)
#     checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
#     model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
#     model.eval()
#     return model


# def filter_boxes_by_area(boxes, logits, max_area_threshold=0.30):
#     if boxes.shape[0] == 0:
#         return boxes, logits
#     areas = boxes[:, 2] * boxes[:, 3]
#     keep_mask = areas < max_area_threshold
#     return boxes[keep_mask], logits[keep_mask]


# def get_grounding_output(model, image, caption, box_threshold, text_threshold, max_area_threshold=0.30, with_logits=True, device="cpu"):
#     caption = caption.lower().strip()
#     if not caption.endswith("."):
#         caption += "."
    
#     model = model.to(device)
#     image = image.to(device)
    
#     with torch.no_grad():
#         outputs = model(image[None], captions=[caption])
    
#     logits = outputs["pred_logits"].cpu().sigmoid()[0]
#     boxes = outputs["pred_boxes"].cpu()[0]

#     filt_mask = logits.max(dim=1)[0] > box_threshold
#     logits_filt = logits[filt_mask]
#     boxes_filt = boxes[filt_mask]

#     boxes_filt, logits_filt = filter_boxes_by_area(boxes_filt, logits_filt, max_area_threshold)

#     scores = logits_filt.max(dim=1)[0]
#     if len(scores) > 0:
#         # Soft-NMS 调用
#         keep_indices, updated_scores = soft_nms_pytorch(boxes_filt, scores, sigma=0.1, thresh=box_threshold)
#         boxes_filt = boxes_filt[keep_indices]
#         logits_filt = logits_filt[keep_indices]

#     tokenlizer = model.tokenizer
#     tokenized = tokenlizer(caption)
#     pred_phrases = []
    
#     for logit, box in zip(logits_filt, boxes_filt):
#         pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
#         if with_logits:
#             pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
#         else:
#             pred_phrases.append(pred_phrase)
            
#     return boxes_filt, pred_phrases, logits_filt


# def show_mask(mask, ax, random_color=False):
#     if random_color:
#         color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
#     else:
#         color = np.array([30/255, 144/255, 255/255, 0.6])
#     h, w = mask.shape[-2:]
#     mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
#     ax.imshow(mask_image)


# def show_box(box, ax, label):
#     x0, y0 = box[0], box[1]
#     w, h = box[2] - box[0], box[3] - box[1]
#     ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
#     ax.text(x0, y0, label)


# def save_mask_data(output_dir, mask_list, box_list, label_list):
#     value = 0
#     mask_img = torch.zeros(mask_list.shape[-2:])
#     for idx, mask in enumerate(mask_list):
#         mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
#     plt.figure(figsize=(10, 10))
#     plt.imshow(mask_img.numpy())
#     plt.axis('off')
#     plt.savefig(os.path.join(output_dir, 'mask.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)
#     plt.close() # 必须关闭，否则批量处理会爆内存

#     json_data = [{'value': value, 'label': 'background'}]
#     for label, box in zip(label_list, box_list):
#         value += 1
#         name, logit = label.split('(')
#         logit = logit[:-1]
#         json_data.append({
#             'value': value,
#             'label': name,
#             'logit': float(logit),
#             'box': box.numpy().tolist(),
#         })
#     with open(os.path.join(output_dir, 'mask.json'), 'w') as f:
#         json.dump(json_data, f, indent=2)


# def load_config(config_path):
#     with open(config_path, 'r', encoding='utf-8') as f:
#         return yaml.safe_load(f)


# # ==========================================
# # 单张图片处理管线 (Pipeline)
# # ==========================================
# def process_single_image(input_image_path, root_output_dir, models, config, args, rag_system):
#     filename = os.path.basename(input_image_path)
#     file_stem = os.path.splitext(filename)[0]
#     current_output_dir = os.path.join(root_output_dir, file_stem)
#     os.makedirs(current_output_dir, exist_ok=True)

#     gdino_model, sam_predictor = models
#     device = args.device
    
#     # 默认值
#     current_max_area = config.get("max_area_threshold", 0.30)
#     current_box_thresh = config.get("box_threshold", 0.30)
#     text_thresh = config.get("text_threshold", 0.25)
#     text_prompt = config.get("text_prompt", "plant disease")
#     is_tiny_mode = False
#     rag_info = None
#     diagnosis = None

#     if args.use_api:
#         from multimodal_expert import get_plant_diagnosis_via_api
#         try:
#             diagnosis = get_plant_diagnosis_via_api(
#                 image_path=input_image_path,
#                 access_key_id=args.access_key_id or config.get("access_key_id"),
#                 access_key_secret=args.access_key_secret or config.get("access_key_secret"),
#                 region_id=config.get("region_id", "cn-beijing")
#             )
     
#             disease_name_en = diagnosis.get('english_name', 'Unknown')
#             visual_desc = diagnosis.get('english_prompt', '')
            

#             #level 2 当不命中时使用这个参数
#             if "微小个体" in target_type:
#                 current_max_area = config.get("max_area_threshold_pest", 0.20)
#                 current_box_thresh = 0.20 
#                 is_tiny_mode = True 
#                 print(f"      策略调整: [通用虫害模式] (Area<{current_max_area})")
#             elif "成片纹理" in target_type:
#                 current_max_area = config.get("max_area_threshold_disease", 0.60)
#                 current_box_thresh = 0.25 
#                 is_tiny_mode = False
#                 print(f"      策略调整: [通用病害模式] (Area<{current_max_area})")


#             # 过滤掉无意义的默认值
#             query_parts = []
            
#             # 1. 加入英文名 (权重最高)
#             if disease_name_en and disease_name_en not in ["Unknown", "Error", "None"]:
#                 query_parts.append(disease_name_en)
            
#             # 2. 加入视觉描述 (作为辅助特征，增加匹配度)
#             # 只有当描述不是默认的占位符时才加
#             if visual_desc and "plant disease symptoms" not in visual_desc:
#                 query_parts.append(visual_desc)
            
#             # 3. 合并成一个长句子
#             if query_parts:
#                 search_query = " ".join(query_parts)
#             else:
#                 # 兜底：如果都提取失败，就用中文名碰运气（虽然大概率匹配不到）
#                 search_query = diagnosis.get('disease_name', 'Unknown')

#             print(f"   -> 🔍 RAG 复合检索词: '{search_query}'")
            
#             # 发起检索
#             rag_knowledge = rag_system.search(search_query)
            
#             if rag_knowledge:
#                 print(f"   -> 📚 [RAG 命中] 匹配: {rag_knowledge['disease_name']}")
#                 text_prompt = rag_knowledge['grounding_prompt']
#                 current_box_thresh = rag_knowledge['thresholds']['box']
#                 current_max_area = rag_knowledge['thresholds']['area']
#                 strategy = rag_knowledge.get('refine_strategy', 'normal')
#                 is_tiny_mode = (strategy == "tiny_mode")
                
#                 rag_info = {
#                     "matched_disease": rag_knowledge['disease_name'],
#                     "strategy": strategy,
#                     "prompt_used": text_prompt
#                 }
#             else:
#                 print("   -> ⚠️ [RAG 未命中] 使用默认配置")
#                 text_prompt = diagnosis.get('english_prompt', text_prompt)

#         except Exception as e:
#             print(f"   ❌ API/RAG 错误: {e}")

#     # 检测
#     image_pil, image = load_image(input_image_path)
#     # 保存原图
#     image_pil.save(os.path.join(current_output_dir, "raw_image.jpg"))

#     print(f"   -> 🚀 检测: '{text_prompt}'")
#     boxes_filt, pred_phrases, logits_filt = get_grounding_output(
#         gdino_model, image, text_prompt, current_box_thresh, text_thresh, 
#         max_area_threshold=current_max_area, device=device
#     )

#     if boxes_filt.shape[0] == 0:
#         print("   ⚠️ 无目标，跳过。")
#         return

#     # SAM 分割
#     image_cv = cv2.imread(input_image_path) # BGR
#     image_cv_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB) # RGB
#     sam_predictor.set_image(image_cv_rgb)

#     W, H = image_pil.size
#     for i in range(boxes_filt.size(0)):
#         boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
#         boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
#         boxes_filt[i][2:] += boxes_filt[i][:2]

#     boxes_filt = boxes_filt.cpu()
#     transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_filt, image_cv_rgb.shape[:2]).to(device)
    
#     masks, _, _ = sam_predictor.predict_torch(
#         point_coords=None, point_labels=None, boxes=transformed_boxes, multimask_output=False,
#     )

#     # 优化 Mask
#     refined_masks = []
#     for mask in masks:
#         refined_masks.append(refine_mask(mask[0], is_tiny_object=is_tiny_mode)) 
#     masks = torch.stack(refined_masks).unsqueeze(1).to(device)


#     # 绘图总览
#     plt.figure(figsize=(10, 10))
#     plt.imshow(image_cv_rgb)
#     for mask in masks:
#         show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
#     for box, label in zip(boxes_filt, pred_phrases):
#         show_box(box.numpy(), plt.gca(), label)
#     plt.axis('off')
#     plt.savefig(os.path.join(current_output_dir, "grounded_sam_output.jpg"), bbox_inches="tight", dpi=300, pad_inches=0.0)
#     plt.close()

#     final_scores = logits_filt.max(dim=1)[0].cpu().numpy()
#     save_structured_result(current_output_dir, filename, diagnosis, masks, boxes_filt, final_scores, pred_phrases, rag_info)

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
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.eval()
    return model


def filter_boxes_by_area(boxes, logits, max_area_threshold=0.30):
    if boxes.shape[0] == 0:
        return boxes, logits
    areas = boxes[:, 2] * boxes[:, 3]
    keep_mask = areas < max_area_threshold
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

    # 面积过滤
    boxes_filt, logits_filt = filter_boxes_by_area(boxes_filt, logits_filt, max_area_threshold)

    scores = logits_filt.max(dim=1)[0]
    if len(scores) > 0:
        # Soft-NMS 调用
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
    plt.close()

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
# 单张图片处理管线 (Pipeline)
# ==========================================
def process_single_image(input_image_path, root_output_dir, models, config, args, rag_system):
    filename = os.path.basename(input_image_path)
    file_stem = os.path.splitext(filename)[0]
    
    current_output_dir = os.path.join(root_output_dir, file_stem)
    os.makedirs(current_output_dir, exist_ok=True)

    gdino_model, sam_predictor = models
    device = args.device
    
    # === Level 3: 加载默认兜底参数 (Lowest Priority) ===
    current_max_area = config.get("max_area_threshold", 0.30)
    current_box_thresh = config.get("box_threshold", 0.30)
    text_thresh = config.get("text_threshold", 0.25)
    text_prompt = config.get("text_prompt", "plant disease")
    is_tiny_mode = False
    rag_info = None
    diagnosis = None

    if args.use_api:
        from multimodal_expert import get_plant_diagnosis_via_api
        try:
            diagnosis = get_plant_diagnosis_via_api(
                image_path=input_image_path,
                access_key_id=args.access_key_id or config.get("access_key_id"),
                access_key_secret=args.access_key_secret or config.get("access_key_secret"),
                region_id=config.get("region_id", "cn-beijing")
            )
            
            # 提取 API 返回的信息
            disease_name_cn = diagnosis.get('disease_name', 'Unknown')
            disease_name_en = diagnosis.get('english_name', 'Unknown')
            visual_desc = diagnosis.get('english_prompt', '')
            # [关键修复] 必须在这里提取 target_type，Level 2 才能用
            target_type = diagnosis.get("target_type", "") 
            
            print(f"   -> 🤖 诊断结果: {disease_name_cn} (Eng: {disease_name_en})")
            print(f"   -> ⚖️ 目标类型: {target_type}")

            # === Level 2: 根据目标类型应用通用规则 (Middle Priority) ===
            # 如果 RAG 没命中，这套参数就是生效的“最佳替补”
            if "微小个体" in target_type:
                current_max_area = config.get("max_area_threshold_pest", 0.20)
                current_box_thresh = 0.20 
                is_tiny_mode = True 
                print(f"      策略调整: [通用虫害模式] (Area<{current_max_area})")
            elif "成片纹理" in target_type:
                current_max_area = config.get("max_area_threshold_disease", 0.60)
                current_box_thresh = 0.25 
                is_tiny_mode = False
                print(f"      策略调整: [通用病害模式] (Area<{current_max_area})")

            # === Level 1: RAG 专家系统 (Highest Priority) ===
            # 构建复合查询词
            query_parts = []
            if disease_name_en and disease_name_en not in ["Unknown", "Error", "None"]:
                query_parts.append(disease_name_en)
            if visual_desc and "plant disease symptoms" not in visual_desc:
                query_parts.append(visual_desc)
            
            if query_parts:
                search_query = " ".join(query_parts)
            else:
                search_query = disease_name_cn # 兜底

            print(f"   -> 🔍 RAG 复合检索词: '{search_query}'")
            
            rag_knowledge = rag_system.search(search_query)
            
            if rag_knowledge:
                print(f"   -> 📚 [RAG 命中] 使用 '{rag_knowledge['disease_name']}' 专家配置")
                # 覆盖 Prompt
                text_prompt = rag_knowledge['grounding_prompt']
                # 覆盖阈值 (这里会覆盖掉 Level 2 的设置)
                current_box_thresh = rag_knowledge['thresholds']['box']
                current_max_area = rag_knowledge['thresholds']['area']
                # 覆盖策略
                strategy = rag_knowledge.get('refine_strategy', 'normal')
                is_tiny_mode = (strategy == "tiny_mode")
                
                rag_info = {
                    "matched_disease": rag_knowledge['disease_name'],
                    "strategy": strategy,
                    "prompt_used": text_prompt
                }
            else:
                print("   -> ⚠️ [RAG 未命中] 保持 Level 2 通用配置")
                text_prompt = visual_desc if visual_desc else text_prompt

        except Exception as e:
            print(f"   ❌ API/RAG 流程错误: {e}")

    # === 2. Grounding DINO 检测 ===
    image_pil, image = load_image(input_image_path)
    image_pil.save(os.path.join(current_output_dir, "raw_image.jpg"))

    print(f"   -> 🚀 检测提示词: '{text_prompt}' (Box>{current_box_thresh}, Area<{current_max_area})")
    boxes_filt, pred_phrases, logits_filt = get_grounding_output(
        gdino_model, image, text_prompt, current_box_thresh, text_thresh, 
        max_area_threshold=current_max_area, device=device
    )

    if boxes_filt.shape[0] == 0:
        print("   ⚠️ 未检测到目标，跳过后续步骤。")
        return

    # === 3. SAM 分割 ===
    image_cv = cv2.imread(input_image_path)
    image_cv_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    sam_predictor.set_image(image_cv_rgb)

    W, H = image_pil.size
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]

    boxes_filt = boxes_filt.cpu()
    transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_filt, image_cv_rgb.shape[:2]).to(device)
    
    masks, _, _ = sam_predictor.predict_torch(
        point_coords=None, point_labels=None, boxes=transformed_boxes, multimask_output=False,
    )

    # === 4. Mask 优化 ===
    refined_masks = []
    for mask in masks:
        refined_masks.append(refine_mask(mask[0], is_tiny_object=is_tiny_mode)) 
    masks = torch.stack(refined_masks).unsqueeze(1).to(device)

    # === 5. 结果保存 ===
    
    # B. 保存可视化图
    plt.figure(figsize=(10, 10))
    plt.imshow(image_cv_rgb)
    for mask in masks:
        show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
    for box, label in zip(boxes_filt, pred_phrases):
        show_box(box.numpy(), plt.gca(), label)
    plt.axis('off')
    plt.savefig(os.path.join(current_output_dir, "grounded_sam_output.jpg"), bbox_inches="tight", dpi=300, pad_inches=0.0)
    plt.close()

    # C. 保存数据
    final_scores = logits_filt.max(dim=1)[0].cpu().numpy()
    save_structured_result(current_output_dir, filename, diagnosis, masks, boxes_filt, final_scores, pred_phrases, rag_info)
    save_mask_data(current_output_dir, masks, boxes_filt, pred_phrases)
# ==========================================
# 主入口
# ==========================================

# ==========================================
# 主入口 (修正版：支持纯 Config 启动)
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Grounded-SAM Batch Processing with RAG")
    parser.add_argument("--config_file", type=str, default="config.yaml")
    
    # [关键修改] 去掉了 required=True，允许命令行不传这两个参数
    parser.add_argument("--input_image", type=str, help="可以是单张图片路径，也可以是文件夹路径")
    parser.add_argument("--output_dir", type=str)
    
    parser.add_argument("--device", type=str)
    parser.add_argument("--use_api", action="store_true")
    parser.add_argument("--access_key_id", type=str)
    parser.add_argument("--access_key_secret", type=str)

    args = parser.parse_args()
    config = load_config(args.config_file)
    
    # === 参数优先级处理逻辑 (命令行 > Config > 报错) ===
    
    # 1. 处理输入路径
    input_path = args.input_image # 先看命令行
    if input_path is None:        # 命令行没传，去 Config 找
        input_path = config.get("input_image")
    
    if input_path is None:        # Config 也没写，报错
        raise ValueError("❌ 错误: 未指定输入图片路径！请在命令行使用 --input_image 或在 config.yaml 中配置 input_image")

    # 2. 处理输出路径
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = config.get("output_dir")
        
    if output_dir is None:
        raise ValueError("❌ 错误: 未指定输出目录！请在命令行使用 --output_dir 或在 config.yaml 中配置 output_dir")

    # 3. 处理其他参数
    device = args.device if args.device else config.get("device", "cuda")
    use_api = args.use_api or config.get("use_api", False)

    # ==========================================

    # 1. 初始化 RAG 系统
    print("📚 初始化 RAG 知识库...")
    try:
        rag_system = PlantRAGSystem("knowledge_base.json")
    except Exception as e:
        print(f"⚠️ RAG 初始化失败: {e}，将使用普通模式运行")
        rag_system = None

    # 2. 加载模型
    print("⏳ 加载视觉模型...")
    gdino_model = load_model(config["config"], config["grounded_checkpoint"], config.get("bert_base_uncased_path"), device)
    
    use_sam_hq = config.get("use_sam_hq", False)
    if use_sam_hq:
        sam = sam_hq_model_registry[config["sam_version"]](checkpoint=config.get("sam_hq_checkpoint"))
    else:
        sam = sam_model_registry[config["sam_version"]](checkpoint=config["sam_checkpoint"])
    sam.to(device=device)
    sam_predictor = SamPredictor(sam)
    
    models = (gdino_model, sam_predictor)

    # 3. 准备文件列表
    image_files = []
    
    if os.path.isdir(input_path):
        print(f"📂 批量处理目录: {input_path}")
        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        # 增加对大小写后缀的兼容
        image_files = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.lower().endswith(valid_exts)]
    else:
        print(f"📄 处理单张图片: {input_path}")
        image_files = [input_path]

    if len(image_files) == 0:
        print(f"❌ 错误: 在路径 {input_path} 下未找到任何图片文件！")
        sys.exit(1)

    # 4. 开始循环处理
    print(f"🚀 开始处理 {len(image_files)} 张图片...")
    
    # 重新打包 args，确保 process_single_image 能拿到合并后的参数
    class MergedArgs:
        pass
    merged_args = MergedArgs()
    merged_args.device = device
    merged_args.use_api = use_api
    merged_args.access_key_id = args.access_key_id
    merged_args.access_key_secret = args.access_key_secret

    # 如果有 tqdm 就用，没有就普通循环
    iterator = tqdm(image_files) if 'tqdm' in sys.modules else image_files

    for img_path in iterator:
        try:
            process_single_image(img_path, output_dir, models, config, merged_args, rag_system)
        except Exception as e:
            # 打印详细错误栈，方便调试
            import traceback
            print(f"\n❌ 处理失败: {img_path}")
            traceback.print_exc()

    print("\n🎉 全部完成！")