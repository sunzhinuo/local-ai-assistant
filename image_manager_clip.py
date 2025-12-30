import os
import json
import hashlib
from typing import List, Dict, Any
from PIL import Image
import torch
import chromadb
from chromadb.config import Settings
import config


class ImageManager:
    def __init__(self):
        self.config = config.Config()

        # 尝试加载CLIP模型，如果失败则使用简化版本
        self.clip_model = None
        self.clip_processor = None

        print("初始化图像管理器...")

        # 初始化向量数据库
        print(f"初始化图像向量数据库: {self.config.VECTOR_DB_DIR}")
        try:
            self.client = chromadb.PersistentClient(path=self.config.VECTOR_DB_DIR)
        except Exception as e:
            print(f"数据库初始化失败: {e}")
            self.client = chromadb.Client(Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=self.config.VECTOR_DB_DIR
            ))

        # 获取或创建图像集合
        self.image_collection = self.client.get_or_create_collection(
            name="images",
            metadata={"description": "图像向量数据库"}
        )

        print("图像管理器初始化完成")

    def _extract_simple_features(self, image_path: str):
        """简化的图像特征提取（基于文件名和基本属性）"""
        try:
            import numpy as np

            with Image.open(image_path) as img:
                # 获取图像基本信息
                width, height = img.size

                # 从文件名提取关键词特征
                filename = os.path.basename(image_path).lower()

                # 定义关键词及其权重
                keywords = [
                    "sunset", "sunrise", "beach", "ocean", "sea",
                    "mountain", "landscape", "city", "urban",
                    "portrait", "person", "people",
                    "animal", "cat", "dog", "bird",
                    "car", "vehicle", "building", "architecture",
                    "flower", "plant", "tree", "nature",
                    "abstract", "art", "modern", "vintage"
                ]

                # 创建特征向量
                features = []

                # 添加基本属性
                features.append(width / 1000.0)
                features.append(height / 1000.0)
                features.append((width * height) / 1000000.0)  # 面积

                # 添加关键词匹配特征
                for keyword in keywords:
                    if keyword in filename:
                        features.append(1.0)
                    else:
                        features.append(0.0)

                # 归一化
                features = np.array(features)
                norm = np.linalg.norm(features)
                if norm > 0:
                    features = features / norm

                return features.tolist()

        except Exception as e:
            print(f"特征提取失败 {image_path}: {e}")
            return [0.0] * 32  # 返回固定长度的零向量

    def add_image(self, image_path: str):
        """添加单张图像到数据库"""
        if not os.path.exists(image_path):
            print(f"文件不存在: {image_path}")
            return False

        filename = os.path.basename(image_path)
        print(f"处理图像: {filename}")

        # 计算文件哈希
        with open(image_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()

        # 检查是否已处理
        try:
            existing = self.image_collection.get(where={"file_hash": file_hash})
            if existing and len(existing['ids']) > 0:
                print(f"图像已处理: {filename}")
                return True
        except:
            pass  # 集合可能不存在

        # 提取图像特征
        features = self._extract_simple_features(image_path)
        if not features:
            print(f"无法提取图像特征: {filename}")
            return False

        # 生成描述
        description = self._generate_description(filename)

        # 添加到向量数据库
        try:
            self.image_collection.add(
                embeddings=[features],
                metadatas=[{
                    "file_path": image_path,
                    "file_hash": file_hash,
                    "filename": filename,
                    "description": description,
                    "feature_dim": len(features)
                }],
                ids=[file_hash]
            )

            print(f"图像已添加到数据库: {filename}")
            return True

        except Exception as e:
            print(f"数据库添加失败: {e}")
            return False

    def _generate_description(self, filename: str) -> str:
        """根据文件名生成简单描述"""
        filename_lower = filename.lower()

        # 关键词到描述的映射
        descriptions = {
            "sunset": "美丽的日落景色",
            "sunrise": "日出美景",
            "beach": "海滩和海洋",
            "ocean": "广阔的海洋",
            "sea": "大海景色",
            "mountain": "山峦风景",
            "landscape": "自然风光",
            "city": "城市景观",
            "urban": "都市风貌",
            "portrait": "人物肖像",
            "person": "人物照片",
            "people": "人群",
            "animal": "野生动物",
            "cat": "猫咪",
            "dog": "狗狗",
            "bird": "鸟类",
            "car": "汽车",
            "vehicle": "交通工具",
            "building": "建筑物",
            "architecture": "建筑艺术",
            "flower": "花卉",
            "plant": "植物",
            "tree": "树木",
            "nature": "自然景色",
            "abstract": "抽象艺术",
            "art": "艺术作品",
            "modern": "现代风格"
        }

        # 查找匹配的关键词
        for keyword, desc in descriptions.items():
            if keyword in filename_lower:
                return desc

        # 默认描述
        return "图片"

    def process_images_directory(self, directory: str = None):
        """批量处理图像目录"""
        if directory is None:
            directory = self.config.IMAGES_DIR

        if not os.path.exists(directory):
            print(f"目录不存在: {directory}")
            return

        # 支持的图像格式
        import glob
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif', '*.webp']
        image_files = []

        for ext in extensions:
            image_files.extend(glob.glob(os.path.join(directory, '**', ext), recursive=True))

        print(f"找到 {len(image_files)} 张图像")

        if not image_files:
            print("未找到图像文件")
            return

        # 处理图像
        success_count = 0
        for i, image_path in enumerate(image_files, 1):
            print(f"[{i}/{len(image_files)}] 处理: {os.path.basename(image_path)}")
            if self.add_image(image_path):
                success_count += 1

        print(f"\n完成！成功处理 {success_count}/{len(image_files)} 张图像")

    def search_image(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """以文搜图：通过文本描述搜索图像"""
        print(f"搜索图像描述: '{query}'")

        # 将文本查询转换为特征向量
        query_features = self._text_to_features(query)

        # 查询向量数据库
        try:
            results = self.image_collection.query(
                query_embeddings=[query_features],
                n_results=n_results,
                include=["metadatas", "distances"]
            )
        except Exception as e:
            print(f"数据库查询失败: {e}")
            return []

        if not results['ids'][0]:
            print("未找到相关图像")
            return []

        # 整理结果
        search_results = []
        for i in range(len(results['ids'][0])):
            metadata = results['metadatas'][0][i]
            distance = results['distances'][0][i]

            # 检查文件是否存在
            if os.path.exists(metadata['file_path']):
                search_results.append({
                    'file_path': metadata['file_path'],
                    'filename': metadata['filename'],
                    'description': metadata.get('description', ''),
                    'similarity': 1.0 - distance,  # 距离转换为相似度
                    'distance': distance
                })

        # 按相似度排序
        search_results.sort(key=lambda x: x['similarity'], reverse=True)

        return search_results

    def _text_to_features(self, text: str):
        """将文本转换为特征向量"""
        import numpy as np

        text_lower = text.lower()

        # 定义关键词及其权重（与图像特征提取保持一致）
        keywords = [
            "sunset", "sunrise", "beach", "ocean", "sea",
            "mountain", "landscape", "city", "urban",
            "portrait", "person", "people",
            "animal", "cat", "dog", "bird",
            "car", "vehicle", "building", "architecture",
            "flower", "plant", "tree", "nature",
            "abstract", "art", "modern", "vintage"
        ]

        # 创建特征向量
        features = np.zeros(len(keywords) + 3)  # 3个基本属性占位

        # 设置基本属性为中等值（0.5）
        features[0] = 0.5  # 宽度
        features[1] = 0.5  # 高度
        features[2] = 0.5  # 面积

        # 关键词匹配
        for i, keyword in enumerate(keywords):
            if keyword in text_lower:
                features[i + 3] = 1.0

        # 归一化
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm

        return features.tolist()