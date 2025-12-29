# multimodal_expert.py (硅基流动版 - 支持英文检测提示词)
import base64
import re
import os
from openai import OpenAI

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_plant_diagnosis_via_api(image_path, access_key_id, access_key_secret=None, region_id=None):
    """
    使用硅基流动 (SiliconFlow) API 进行诊断。
    """
    
    # === 配置部分 ===
    # 硅基流动的 Base URL
    BASE_URL = "https://api.siliconflow.cn/v1"
    # 使用的模型
    MODEL_NAME = "Qwen/Qwen2-VL-72B-Instruct" 
    # ================

    # 初始化 OpenAI 客户端
    api_key = access_key_id
    if not api_key or not api_key.startswith("sk-"):
        return {
            "disease_name": "配置错误",
            "english_prompt": "plant disease symptoms", # 默认值
            "region_description": "",
            "control_advice": "请在 access_key_id 中填入正确的硅基流动 API Key (sk-...)"
        }

    client = OpenAI(api_key=api_key, base_url=BASE_URL)

    try:
        base64_image = encode_image(image_path)
    except Exception as e:
        return {
            "disease_name": "图像读取失败",
            "english_prompt": "plant disease symptoms",
            "region_description": "",
            "control_advice": str(e)
        }




    


    user_prompt = (
    "你是一位农业植保与计算机视觉专家。请根据提供的作物图像，严格按以下格式回答：\n\n"
    "【病虫害名称】：(中文)\n"
    "【英文学名】：(严格的英文学术名称，用于检索数据库)\n"
    "【检测提示词(英文)】：\n"
    "【区域描述】：\n"
    "【防治建议】：\n\n"
    "核心要求：\n"
    "1. 病虫害名称：使用中文标准学名（如“小麦条锈病”、“麦蚜”、“粘虫”）；\n"
    "2. 【目标类型判定】：为了辅助下游检测算法设定阈值，请严格从以下两个选项中**二选一**填写：\n"
    "    - 选项A：'微小个体' (适用于：独立个体害虫，或极早期的独立小病斑)\n"
    "    - 选项B：'成片纹理' (适用于：覆盖连片的病害)\n"
    "3. 【检测提示词(英文)】：这是给视觉Grounding模型（如GroundingDINO）使用的核心指令，必须极度精确，请严格遵守以下规则：\n"
    "    - **【通用铁律】严禁出现植物部位名称**：绝对不要包含 'leaf', 'leaves', 'plant', 'stem', 'wheat', 'crop', 'background' 等词汇，否则模型会分割整个背景叶片！\n"
    "    - **【分流策略】根据目标类型选择描述方式**：\n"
    "        * **若是病害 (Disease)** -> **重材质 (Texture)**：\n"
    "             - 必须描述颜色和质感。\n"
    "        * **若是虫害 (Pest)** -> **重结构 (Structure ONLY)**：\n"
    "             - **【绝对禁令】严禁描述颜色**：对于虫害，禁止使用 'green', 'black', 'red', 'white' 等颜色词！因为虫子往往有保护色，描述颜色会导致模型误检背景叶片。只描述形态结构。\n"
    "    - **【避坑指南】(Structure over Color/Shape)**：\n"
    "        * 避免单独使用简单的形状词（如 'round', 'oval'），必须配合生物特征（如 'legs', 'head'）。\n"
    "        * 针对**小麦条锈病**：虽然叫“条锈”，但严禁使用 'stripe'！请用 'rows of powdery fungal spores' (成排粉状真菌孢子)。\n"
    "        * 针对**保护色害虫（如青虫/蚜虫）**：**禁止提及颜色**！不要说 'green caterpillar'，必须说 'cylindrical larva body' (圆柱状幼虫身体) 或 'insect with legs'。\n"
    "        * 针对密集微小害虫, 禁止使用'clusters', 'dense dots'，必须描述单体特征如 'tiny insect body'。"
    "    - **【正确示例】**：\n"
    "        * 病害（带颜色）：'yellow powdery pustules', 'white mold patch', 'brown necrotic lesions'\n"
    "        * 虫害（**无颜色**）：'shiny beetle with hard shell', 'segmented larva body', 'insect with antennae', 'tiny soft body'\n"
    "4. 区域描述：说明病虫害的具体位置（如叶背、茎秆、心叶）和分布形态；\n"
    "5. 防治建议：包括化学药剂（针对性杀虫剂/杀菌剂）和农艺措施；\n"
    "6. 若无法判断，写“无法确定”。"
) 
#     user_prompt = (
#     "你是一位农业植保与计算机视觉专家。请根据提供的作物图像，严格按以下格式回答：\n\n"
#     "【病虫害名称】：\n"
#     "【目标类型判定】：\n"
#     "【检测提示词(英文)】：\n"
#     "【区域描述】：\n"
#     "【防治建议】：\n\n"
#     "核心要求：\n"
#     "1. 病虫害名称：使用中文标准学名（如“小麦条锈病”、“麦蚜”、“粘虫”）；\n"
#     "2. 【目标类型判定】：为了辅助下游检测算法设定阈值，请严格从以下两个选项中**二选一**填写：\n"
#     "   - 选项A：'微小个体' (适用于：蚜虫、甲虫、红蜘蛛、幼虫、粉虱等独立个体害虫，或极早期的独立小病斑)\n"
#     "   - 选项B：'成片纹理' (适用于：条锈病、白粉病、霉病、大面积叶斑病等覆盖连片的病害)\n"
#     "3. 【检测提示词(英文)】：这是给视觉Grounding模型（如GroundingDINO）使用的核心指令，必须极度精确，请严格遵守以下规则：\n"
#     "   - **【通用铁律】严禁出现植物部位名称**：绝对不要包含 'leaf', 'leaves', 'plant', 'stem', 'wheat', 'crop', 'background' 等词汇，否则模型会分割整个背景叶片！\n"
#     "   - **【分流策略】根据目标类型选择描述方式**：\n"
#     "       * **若是病害 (Disease)** -> **重材质 (Texture)**：使用 'powdery' (粉末状), 'moldy' (霉状), 'necrotic' (坏死), 'spores' (孢子) 等词汇。\n"
#     "       * **若是虫害 (Pest)** -> **重本体 (Body/Morphology)**：\n"
#     "           - **成虫**：使用 'insect body', 'shiny beetle', 'legs', 'wings', 'hard shell'。\n"
#     "           - **幼虫**：使用 'larva', 'caterpillar', 'segmented body' (分节身体), 'worm-like'。\n"
#     "   - **【避坑指南】(Texture/Body over Shape)**：\n"
#     "       * 避免单独使用简单的形状词（如 'stripe', 'line', 'oval'），除非配合材质或生物特征使用，否则容易误检叶脉。\n"
#     "       * 针对**小麦条锈病**：虽然叫“条锈”，但严禁使用 'stripe'！请用 'rows of yellow fungal powder' (成排黄色真菌粉末)。\n"
#     "       * 针对**保护色害虫（如青虫）**：不要只说 'green object'，必须说 'green cylindrical caterpillar' (绿色圆柱状毛虫) 或 'insect larva'。\n"
#     "       * 针对密集微小害虫,禁止使用'clusters of tiny insects', 'dense small dots', 'aggregation of aphids'。"
#     "   - **【正确示例】**：\n"
#     "       * 病害：'yellow powdery pustules', 'white mold patch', 'brown necrotic lesions'\n"
#     "       * 虫害：'black shiny beetle', 'green segmented larva', 'caterpillar with hairs'\n"
#     "4. 区域描述：说明病虫害的具体位置（如叶背、茎秆、心叶）和分布形态；\n"
#     "5. 防治建议：包括化学药剂（针对性杀虫剂/杀菌剂）和农艺措施；\n"
#     "6. 若无法判断，写“无法确定”。"
# )
#    



    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": user_prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            temperature=0.1, # 降低随机性
        )
        
        content = response.choices[0].message.content
        
    except Exception as error:
        return {
            "disease_name": "API 调用失败",
            "english_prompt": "plant disease symptoms",
            "region_description": "",
            "control_advice": f"SiliconFlow Error: {str(error)}"
        }

    # === 正则提取逻辑更新 ===
    name = re.search(r"【病虫害名称】：\s*(.+)", content)
    eng_name_match = re.search(r"【英文学名】：\s*(.+)", content)
    target_type_match = re.search(r"【目标类型判定】：\s*(.+)", content)
    eng_prompt = re.search(r"【检测提示词\(英文\)】：\s*(.+)", content)
    region = re.search(r"【区域描述】：\s*(.+)", content)
    advice = re.search(r"【防治建议】：\s*(.+?)(?:\n|$)", content, re.DOTALL)

    return {
        "disease_name": (name.group(1).strip() if name else "无法确定"),
        "english_name": (eng_name_match.group(1).strip() if eng_name_match else "Unknown"),
        "target_type": (target_type_match.group(1).strip() if target_type_match else ""),

        # 如果提取成功用提取的，否则用默认值
        "english_prompt": (eng_prompt.group(1).strip() if eng_prompt else "plant disease symptoms"),
        "region_description": (region.group(1).strip() if region else ""),
        "control_advice": (advice.group(1).strip() if advice else "")
    }