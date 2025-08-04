import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import pickle
import os

class HotRecall:
    """
    热门召回
    基于电影的热门程度（评分、观看次数等）进行召回
    """
    
    def __init__(self, top_k: int = 100):
        """
        初始化热门召回器
        
        Args:
            top_k: 召回数量
        """
        self.top_k = top_k
        self.hot_items = None
        self.item_scores = None
        
    def fit(self, train_df: pd.DataFrame, movies_df: pd.DataFrame = None):
        """
        训练热门召回模型
        
        Args:
            train_df: 训练数据
            movies_df: 电影数据（可选）
        """
        print("开始训练热门召回模型...")
        
        # 计算物品热门分数
        self._compute_item_popularity(train_df, movies_df)
        
        print("热门召回模型训练完成")
    
    def _compute_item_popularity(self, train_df: pd.DataFrame, movies_df: pd.DataFrame = None):
        """计算物品热门程度"""
        # 计算每个电影的统计信息
        item_stats = train_df.groupby('movie_id_encoded').agg({
            'rating': ['count', 'mean', 'sum'],
            'user_id_encoded': 'nunique'
        }).reset_index()
        
        item_stats.columns = ['movie_id_encoded', 'rating_count', 'rating_mean', 'rating_sum', 'user_count']
        
        # 计算热门分数（综合考虑评分次数、平均评分等）
        item_stats['popularity_score'] = (
            item_stats['rating_count'] * 0.4 +  # 观看次数权重
            item_stats['rating_mean'] * 0.4 +   # 平均评分权重
            item_stats['user_count'] * 0.2      # 用户数权重
        )
        
        # 按热门分数排序
        self.hot_items = item_stats.sort_values('popularity_score', ascending=False)
        self.item_scores = dict(zip(self.hot_items['movie_id_encoded'], 
                                   self.hot_items['popularity_score']))
        
        print(f"热门物品计算完成，共 {len(self.hot_items)} 个物品")
    
    def recall(self, user_id: int, user_history: List[int] = None, 
               n_recall: int = 100) -> List[Tuple[int, float]]:
        """
        热门召回
        
        Args:
            user_id: 用户ID
            user_history: 用户历史行为（可选，用于过滤）
            n_recall: 召回数量
            
        Returns:
            召回物品列表
        """
        if self.hot_items is None:
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        # 获取热门物品
        hot_items = self.hot_items.head(n_recall * 2)  # 多取一些，用于过滤
        
        # 过滤用户历史中的物品
        if user_history:
            hot_items = hot_items[~hot_items['movie_id_encoded'].isin(user_history)]
        
        # 取top-k
        top_items = hot_items.head(n_recall)
        
        # 构建结果
        results = []
        for _, row in top_items.iterrows():
            item_id = row['movie_id_encoded']
            score = row['popularity_score']
            results.append((item_id, score))
        
        return results
    
    def batch_recall(self, user_histories: Dict[int, List[int]] = None, 
                    n_recall: int = 100) -> Dict[int, List[Tuple[int, float]]]:
        """
        批量召回
        
        Args:
            user_histories: 用户历史行为字典（可选）
            n_recall: 召回数量
            
        Returns:
            用户召回结果字典
        """
        if user_histories is None:
            user_histories = {}
        
        results = {}
        
        for user_id, history in user_histories.items():
            results[user_id] = self.recall(user_id, history, n_recall)
        
        return results
    
    def get_global_hot_items(self, n_items: int = 100) -> List[Tuple[int, float]]:
        """
        获取全局热门物品
        
        Args:
            n_items: 物品数量
            
        Returns:
            热门物品列表
        """
        if self.hot_items is None:
            raise ValueError("模型尚未训练")
        
        top_items = self.hot_items.head(n_items)
        return [(row['movie_id_encoded'], row['popularity_score']) 
                for _, row in top_items.iterrows()]
    
    def save_model(self, save_path: str = "./models/hot_model.pkl"):
        """保存模型"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, "wb") as f:
            pickle.dump({
                'hot_items': self.hot_items,
                'item_scores': self.item_scores,
                'top_k': self.top_k
            }, f)
        
        print(f"热门模型已保存到: {save_path}")
    
    def load_model(self, load_path: str = "./models/hot_model.pkl"):
        """加载模型"""
        with open(load_path, "rb") as f:
            model_data = pickle.load(f)
            
        self.hot_items = model_data['hot_items']
        self.item_scores = model_data['item_scores']
        self.top_k = model_data['top_k']
        
        print(f"热门模型已从 {load_path} 加载")

def main():
    """热门召回测试"""
    import sys
    sys.path.append("..")
    from data_loader import MovieLensDataLoader
    
    # 加载数据
    data_loader = MovieLensDataLoader()
    train_df, test_df, movies_df, users_df = data_loader.load_processed_data()
    
    # 初始化热门召回器
    hot_recall = HotRecall(top_k=100)
    
    # 训练模型
    hot_recall.fit(train_df, movies_df)
    
    # 获取用户历史
    user_histories = data_loader.get_user_history(train_df)
    
    # 测试召回
    test_users = list(user_histories.keys())[:3]
    test_histories = {user_id: user_histories[user_id] for user_id in test_users}
    
    recall_results = hot_recall.batch_recall(test_histories, n_recall=20)
    
    # 打印结果
    for user_id in test_users:
        print(f"用户 {user_id} 的热门召回结果:")
        for item_id, score in recall_results[user_id][:5]:
            print(f"  电影ID: {item_id}, 热门分数: {score:.4f}")
        print()
    
    # 获取全局热门物品
    global_hot = hot_recall.get_global_hot_items(10)
    print("全局热门电影:")
    for item_id, score in global_hot:
        print(f"  电影ID: {item_id}, 热门分数: {score:.4f}")
    
    # 保存模型
    hot_recall.save_model()

if __name__ == "__main__":
    main()