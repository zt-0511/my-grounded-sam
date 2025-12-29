import json
import torch
import os
import sys
# === 修正点：SentenceTransformer 首字母必须大写 ===
from sentence_transformers import SentenceTransformer, util
import numpy as np

class PlantRAGSystem:
    def __init__(self, knowledge_path="knowledge_base.json"):
        print("📚 [RAG] 正在初始化知识库检索引擎...")
        self.knowledge_path = knowledge_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 设定本地模型路径
        local_model_path = "./weights/all-MiniLM-L6-v2"
        
        if not os.path.exists(local_model_path):
            print(f"❌ 错误: 找不到本地模型文件夹: {local_model_path}")
            print(f"📂 当前工作目录是: {os.getcwd()}")
            sys.exit(1)

        print(f"⏳ 正在加载本地模型权重: {local_model_path} ...")
        
        try:
            self.embedder = SentenceTransformer(local_model_path, device=self.device)
            print("✅ 模型加载成功！")
        except Exception as e:
            print(f"❌ 模型文件加载失败，请检查文件是否完整。错误信息: {e}")
            sys.exit(1)
            
        self.knowledge_base = []
        self.corpus_embeddings = None
        self.load_knowledge()

    def load_knowledge(self):
        if not os.path.exists(self.knowledge_path):
            print(f"❌ 错误: 找不到知识库文件 {self.knowledge_path}")
            sys.exit(1)

        with open(self.knowledge_path, 'r', encoding='utf-8') as f:
            self.knowledge_base = json.load(f)
        
        corpus = []
        for item in self.knowledge_base:
            aliases = item.get('aliases', [])
            text = f"{item['disease_name']} {' '.join(aliases)}"
            corpus.append(text)
        
        print("⚙️ 正在构建向量索引...")
        self.corpus_embeddings = self.embedder.encode(corpus, convert_to_tensor=True)
        print(f"✅ [RAG] 知识库构建完成，共包含 {len(self.knowledge_base)} 种病虫害知识。")

    def search(self, query_disease_name, score_threshold=0.4):
        if not query_disease_name or not isinstance(query_disease_name, str):
            return None

        query_embedding = self.embedder.encode(query_disease_name, convert_to_tensor=True)
        cos_scores = util.cos_sim(query_embedding, self.corpus_embeddings)[0]
        
        top_result = torch.topk(cos_scores, k=1)
        score = top_result.values.item()
        idx = top_result.indices.item()
        
        if score > score_threshold:
            match_data = self.knowledge_base[idx]
            print(f"🔍 [RAG检索] Query: '{query_disease_name}' -> Match: '{match_data['disease_name']}' (相似度: {score:.4f})")
            return match_data
        else:
            print(f"⚠️ [RAG失效] 知识库中未找到 '{query_disease_name}' 的相关记录 (最高相似度: {score:.4f})")
            return None

if __name__ == "__main__":
    print("--- 开始 RAG 自测 ---")
    rag = PlantRAGSystem()
    test_queries = ["发现了一些黄色的锈病", "叶子上全是白色粉末"]
    for q in test_queries:
        print(f"\n❓ 测试提问: {q}")
        res = rag.search(q)