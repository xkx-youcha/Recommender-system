import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Tuple
import pickle
import os

class ItemCFRecall:
    """
    基于物品的协同过滤召回
    使用物品相似度矩阵为用户推荐相似物品
    """
    
    def __init__(self, top_k: int = 100):
        """
        初始化ItemCF召回器
        
        Args:
            top_k: 召回物品数量
        """
        self.top_k = top_k
        self.item_sim_matrix = None
        self.user_item_matrix = None
        self.item_means = None
        
    def fit(self, train_df: pd.DataFrame):
        """
        训练ItemCF模型
        
        Args:
            train_df: 训练数据，包含user_id_encoded, movie_id_encoded, rating列
        """
        print("开始训练ItemCF召回模型...")
        
        # 构建用户-物品评分矩阵
        self._build_user_item_matrix(train_df)
        
        # 计算物品相似度矩阵
        self._compute_item_similarity()
        
        print("ItemCF召回模型训练完成")
    
    def _build_user_item_matrix(self, train_df: pd.DataFrame):
        """构建用户-物品评分矩阵"""
        # 创建稀疏矩阵
        users = train_df['user_id_encoded'].values
        items = train_df['movie_id_encoded'].values
        ratings = train_df['rating'].values
        
        # 获取用户和物品的最大ID
        max_user_id = train_df['user_id_encoded'].max() + 1
        max_item_id = train_df['movie_id_encoded'].max() + 1
        
        # 构建稀疏矩阵
        self.user_item_matrix = csr_matrix(
            (ratings, (users, items)), 
            shape=(max_user_id, max_item_id)
        )
        
        # 计算物品平均评分
        self.item_means = np.array(self.user_item_matrix.mean(axis=0)).flatten()
        
        print(f"用户-物品矩阵构建完成: {self.user_item_matrix.shape}")
    
    def _compute_item_similarity(self):
        """计算物品相似度矩阵"""
        # 中心化评分矩阵（减去物品平均评分）
        centered_matrix = self.user_item_matrix.copy()
        
        # 对每个物品减去平均评分
        for item_id in range(centered_matrix.shape[1]):
            if self.item_means[item_id] > 0:
                item_ratings = centered_matrix[:, item_id]
                centered_matrix[:, item_id] = item_ratings - self.item_means[item_id]
        
        # 计算余弦相似度
        self.item_sim_matrix = cosine_similarity(centered_matrix.T)
        
        # 将对角线设为0（物品与自身的相似度）
        np.fill_diagonal(self.item_sim_matrix, 0)
        
        print(f"物品相似度矩阵计算完成: {self.item_sim_matrix.shape}")
    
    def recall(self, user_id: int, n_recall: int = 100) -> List[Tuple[int, float]]:
        """
        为用户召回物品
        
        Args:
            user_id: 用户ID
            n_recall: 召回数量
            
        Returns:
            召回物品列表，格式为[(item_id, score), ...]
        """
        if self.user_item_matrix is None or self.item_sim_matrix is None:
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        # 获取用户已评分的物品
        user_ratings = self.user_item_matrix[user_id].toarray().flatten()
        rated_items = np.where(user_ratings > 0)[0]
        
        if len(rated_items) == 0:
            return []
        
        # 计算用户对每个物品的预测评分
        scores = np.zeros(self.item_sim_matrix.shape[0])
        
        for rated_item in rated_items:
            # 获取该物品与其他物品的相似度
            item_similarities = self.item_sim_matrix[rated_item]
            
            # 加权求和
            scores += item_similarities * user_ratings[rated_item]
        
        # 归一化
        similarity_sums = np.sum(self.item_sim_matrix[rated_items], axis=0)
        similarity_sums[similarity_sums == 0] = 1  # 避免除零
        scores = scores / similarity_sums
        
        # 过滤已评分的物品
        scores[rated_items] = -1
        
        # 获取top-k物品
        top_indices = np.argsort(scores)[::-1][:n_recall]
        top_scores = scores[top_indices]
        
        # 过滤负分
        valid_indices = top_scores > 0
        top_indices = top_indices[valid_indices]
        top_scores = top_scores[valid_indices]
        
        return list(zip(top_indices, top_scores))
    
    def batch_recall(self, user_ids: List[int], n_recall: int = 100) -> Dict[int, List[Tuple[int, float]]]:
        """
        批量召回
        
        Args:
            user_ids: 用户ID列表
            n_recall: 召回数量
            
        Returns:
            用户召回结果字典
        """
        results = {}
        
        for user_id in user_ids:
            results[user_id] = self.recall(user_id, n_recall)
        
        return results
    
    def save_model(self, save_path: str = "./models/itemcf_model.pkl"):
        """保存模型"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, "wb") as f:
            pickle.dump({
                'item_sim_matrix': self.item_sim_matrix,
                'user_item_matrix': self.user_item_matrix,
                'item_means': self.item_means,
                'top_k': self.top_k
            }, f)
        
        print(f"ItemCF模型已保存到: {save_path}")
    
    def load_model(self, load_path: str = "./models/itemcf_model.pkl"):
        """加载模型"""
        with open(load_path, "rb") as f:
            model_data = pickle.load(f)
            
        self.item_sim_matrix = model_data['item_sim_matrix']
        self.user_item_matrix = model_data['user_item_matrix']
        self.item_means = model_data['item_means']
        self.top_k = model_data['top_k']
        
        print(f"ItemCF模型已从 {load_path} 加载")

def main():
    """ItemCF召回测试"""
    import sys
    sys.path.append("..")
    from data_loader import MovieLensDataLoader
    
    # 加载数据
    data_loader = MovieLensDataLoader()
    train_df, test_df, movies_df, users_df = data_loader.load_processed_data(load_path="../processed_data")
    
    # 初始化ItemCF召回器
    itemcf = ItemCFRecall(top_k=100)
    
    # 训练模型
    itemcf.fit(train_df)
    
    # 测试召回
    test_users = [0, 1, 2]  # 测试用户
    recall_results = itemcf.batch_recall(test_users, n_recall=20)
    
    # 打印结果
    for user_id in test_users:
        print(f"用户 {user_id} 的召回结果:")
        for item_id, score in recall_results[user_id][:5]:
            print(f"  电影ID: {item_id}, 分数: {score:.4f}")
        print()
    
    # 保存模型
    itemcf.save_model()

if __name__ == "__main__":
    main()