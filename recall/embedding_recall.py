import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Tuple
import pickle
import os
from collections import defaultdict

class EmbeddingRecall:
    """
    基于嵌入的召回
    使用Word2Vec训练物品嵌入向量，通过向量相似度进行召回
    """
    
    def __init__(self, embedding_dim: int = 128, window: int = 5, 
                 min_count: int = 1, workers: int = 4, top_k: int = 100):
        """
        初始化嵌入召回器
        
        Args:
            embedding_dim: 嵌入维度
            window: 窗口大小
            min_count: 最小词频
            workers: 并行线程数
            top_k: 召回数量
        """
        self.embedding_dim = embedding_dim
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.top_k = top_k
        
        self.model = None
        self.item_embeddings = None
        self.item_sim_matrix = None
        
    def fit(self, train_df: pd.DataFrame):
        """
        训练嵌入模型
        
        Args:
            train_df: 训练数据
        """
        print("开始训练嵌入召回模型...")
        
        # 构建用户行为序列
        user_sequences = self._build_user_sequences(train_df)
        
        # 训练Word2Vec模型
        self._train_word2vec(user_sequences)
        
        # 构建物品嵌入矩阵
        self._build_item_embeddings(train_df)
        
        # 计算物品相似度矩阵
        self._compute_item_similarity()
        
        print("嵌入召回模型训练完成")
    
    def _build_user_sequences(self, train_df: pd.DataFrame) -> List[List[str]]:
        """构建用户行为序列"""
        # 按用户和时间排序
        train_df = train_df.sort_values(['user_id_encoded', 'timestamp'])
        
        sequences = []
        for user_id in train_df['user_id_encoded'].unique():
            user_items = train_df[train_df['user_id_encoded'] == user_id]['movie_id_encoded'].tolist()
            # 转换为字符串（Word2Vec需要字符串输入）
            sequences.append([str(item_id) for item_id in user_items])
        
        print(f"构建了 {len(sequences)} 个用户行为序列")
        return sequences
    
    def _train_word2vec(self, sequences: List[List[str]]):
        """训练Word2Vec模型"""
        self.model = Word2Vec(
            sentences=sequences,
            vector_size=self.embedding_dim,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            sg=1,  # Skip-gram
            epochs=10
        )
        
        print(f"Word2Vec模型训练完成，词汇表大小: {len(self.model.wv.key_to_index)}")
    
    def _build_item_embeddings(self, train_df: pd.DataFrame):
        """构建物品嵌入矩阵"""
        max_item_id = train_df['movie_id_encoded'].max() + 1
        self.item_embeddings = np.zeros((max_item_id, self.embedding_dim))
        
        # 为每个物品获取嵌入向量
        for item_id in range(max_item_id):
            item_str = str(item_id)
            if item_str in self.model.wv:
                self.item_embeddings[item_id] = self.model.wv[item_str]
        
        print(f"物品嵌入矩阵构建完成: {self.item_embeddings.shape}")
    
    def _compute_item_similarity(self):
        """计算物品相似度矩阵"""
        self.item_sim_matrix = cosine_similarity(self.item_embeddings)
        
        # 将对角线设为0
        np.fill_diagonal(self.item_sim_matrix, 0)
        
        print(f"物品相似度矩阵计算完成: {self.item_sim_matrix.shape}")
    
    def recall(self, user_id: int, user_history: List[int], n_recall: int = 100) -> List[Tuple[int, float]]:
        """
        基于用户历史行为召回物品
        
        Args:
            user_id: 用户ID
            user_history: 用户历史行为序列
            n_recall: 召回数量
            
        Returns:
            召回物品列表
        """
        if self.item_sim_matrix is None:
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        if not user_history:
            return []
        
        # 计算用户对每个物品的兴趣分数
        scores = np.zeros(self.item_sim_matrix.shape[0])
        
        for item_id in user_history:
            if item_id < len(self.item_sim_matrix):
                # 获取该物品与其他物品的相似度
                item_similarities = self.item_sim_matrix[item_id]
                scores += item_similarities
        
        # 归一化
        scores = scores / len(user_history)
        
        # 过滤用户历史中的物品
        scores[user_history] = -1
        
        # 获取top-k物品
        top_indices = np.argsort(scores)[::-1][:n_recall]
        top_scores = scores[top_indices]
        
        # 过滤负分
        valid_indices = top_scores > 0
        top_indices = top_indices[valid_indices]
        top_scores = top_scores[valid_indices]
        
        return list(zip(top_indices, top_scores))
    
    def batch_recall(self, user_histories: Dict[int, List[int]], 
                    n_recall: int = 100) -> Dict[int, List[Tuple[int, float]]]:
        """
        批量召回
        
        Args:
            user_histories: 用户历史行为字典
            n_recall: 召回数量
            
        Returns:
            用户召回结果字典
        """
        results = {}
        
        for user_id, history in user_histories.items():
            results[user_id] = self.recall(user_id, history, n_recall)
        
        return results
    
    def get_item_embedding(self, item_id: int) -> np.ndarray:
        """获取物品嵌入向量"""
        if self.item_embeddings is None:
            raise ValueError("模型尚未训练")
        
        if item_id >= len(self.item_embeddings):
            return np.zeros(self.embedding_dim)
        
        return self.item_embeddings[item_id]
    
    def save_model(self, save_path: str = "./models/embedding_model.pkl"):
        """保存模型"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, "wb") as f:
            pickle.dump({
                'model': self.model,
                'item_embeddings': self.item_embeddings,
                'item_sim_matrix': self.item_sim_matrix,
                'embedding_dim': self.embedding_dim,
                'top_k': self.top_k
            }, f)
        
        print(f"嵌入模型已保存到: {save_path}")
    
    def load_model(self, load_path: str = "./models/embedding_model.pkl"):
        """加载模型"""
        with open(load_path, "rb") as f:
            model_data = pickle.load(f)
            
        self.model = model_data['model']
        self.item_embeddings = model_data['item_embeddings']
        self.item_sim_matrix = model_data['item_sim_matrix']
        self.embedding_dim = model_data['embedding_dim']
        self.top_k = model_data['top_k']
        
        print(f"嵌入模型已从 {load_path} 加载")

def main():
    """嵌入召回测试"""
    from data_loader import MovieLensDataLoader
    
    # 加载数据
    data_loader = MovieLensDataLoader()
    train_df, test_df, movies_df, users_df = data_loader.load_processed_data(load_path="../processed_data")
    
    # 初始化嵌入召回器
    embedding_recall = EmbeddingRecall(embedding_dim=64, top_k=100)
    
    # 训练模型
    embedding_recall.fit(train_df)
    
    # 获取用户历史
    user_histories = data_loader.get_user_history(train_df)
    
    # 测试召回
    test_users = list(user_histories.keys())[:3]
    test_histories = {user_id: user_histories[user_id] for user_id in test_users}
    
    recall_results = embedding_recall.batch_recall(test_histories, n_recall=20)
    
    # 打印结果
    for user_id in test_users:
        print(f"用户 {user_id} 的召回结果:")
        for item_id, score in recall_results[user_id][:5]:
            print(f"  电影ID: {item_id}, 分数: {score:.4f}")
        print()
    
    # 保存模型
    embedding_recall.save_model()

if __name__ == "__main__":
    main() 