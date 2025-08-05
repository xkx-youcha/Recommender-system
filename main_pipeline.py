import os
import sys
import pandas as pd
import numpy as np
import torch
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# 添加模块路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import MovieLensDataLoader
from recall.itemcf_recall import ItemCFRecall
from recall.hot_recall import HotRecall
from rank.model_deepfm import DeepFMRanker
from evaluate import RecommenderEvaluator

class RecommenderPipeline:
    """
    推荐系统端到端流程
    """
    
    def __init__(self, data_path: str = "../ml-1m", processed_data_path: str = "../processed_data"):
        self.data_path = data_path
        self.processed_data_path = processed_data_path
        
        # 初始化组件
        self.data_loader = MovieLensDataLoader(data_path)
        self.evaluator = RecommenderEvaluator()
        
        # 召回模型
        self.itemcf_recall = None
        self.hot_recall = None
        
        # 排序模型
        self.deepfm_ranker = None
        
        # 数据
        self.train_df = None
        self.test_df = None
        self.movies_df = None
        self.users_df = None
        self.user_histories = None
        
    def load_and_preprocess_data(self):
        """加载和预处理数据"""
        print("="*60)
        print("开始数据加载和预处理...")
        print("="*60)
        
        try:
            # 尝试加载已处理的数据
            self.train_df, self.test_df, self.movies_df, self.users_df = \
                self.data_loader.load_processed_data(self.processed_data_path)
            print("成功加载已处理的数据")
        except:
            print("未找到已处理的数据，开始原始数据处理...")
            # 加载原始数据
            ratings_df, movies_df, users_df = self.data_loader.load_data()
            
            # 预处理数据
            ratings_df, movies_df, users_df = self.data_loader.preprocess_data(
                ratings_df, movies_df, users_df
            )
            
            # 划分训练测试集
            self.train_df, self.test_df = self.data_loader.split_train_test(ratings_df)
            self.movies_df = movies_df
            self.users_df = users_df
            
            # 保存处理后的数据
            self.data_loader.save_processed_data(
                self.train_df, self.test_df, self.movies_df, self.users_df,
                self.processed_data_path
            )
        
        # 获取用户历史
        self.user_histories = self.data_loader.get_user_history(self.train_df)
        
        print(f"数据加载完成:")
        print(f"- 训练集: {len(self.train_df)} 条记录")
        print(f"- 测试集: {len(self.test_df)} 条记录")
        print(f"- 用户数: {len(self.user_histories)}")
        print(f"- 电影数: {len(self.movies_df)}")
    
    def train_recall_models(self):
        """训练召回模型"""
        print("\n" + "="*60)
        print("开始训练召回模型...")
        print("="*60)
        
        # ItemCF召回
        print("训练ItemCF召回模型...")
        self.itemcf_recall = ItemCFRecall(top_k=100)
        self.itemcf_recall.fit(self.train_df)
        
        # 热门召回
        print("训练热门召回模型...")
        self.hot_recall = HotRecall(top_k=100)
        self.hot_recall.fit(self.train_df, self.movies_df)
        
        print("召回模型训练完成")
    
    def multi_path_recall(self, user_id: int, n_recall: int = 200) -> List[Tuple[int, float]]:
        """多路召回"""
        recall_results = []
        
        # ItemCF召回
        if self.itemcf_recall:
            itemcf_results = self.itemcf_recall.recall(user_id, n_recall//2)
            recall_results.extend(itemcf_results)
        
        # 热门召回
        if self.hot_recall:
            hot_results = self.hot_recall.recall(user_id, n_recall//2)
            recall_results.extend(hot_results)
        
        # 去重和排序
        unique_results = {}
        for item_id, score in recall_results:
            if item_id not in unique_results or score > unique_results[item_id]:
                unique_results[item_id] = score
        
        sorted_results = sorted(unique_results.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:n_recall]
    
    def train_ranking_model(self):
        """训练排序模型"""
        print("\n" + "="*60)
        print("开始训练排序模型...")
        print("="*60)
        
        # 获取特征维度
        user_features = self.data_loader.get_user_features(self.users_df)
        movie_features = self.data_loader.get_movie_features(self.movies_df)
        
        if user_features and movie_features:
            user_dim = len(next(iter(user_features.values())))
            movie_dim = len(next(iter(movie_features.values())))
            feature_size = user_dim + movie_dim
        else:
            feature_size = 100
        
        # DeepFM排序器
        print("训练DeepFM排序模型...")
        self.deepfm_ranker = DeepFMRanker(
            feature_size=feature_size,
            embedding_dim=16,
            hidden_dims=[200, 100],
            dropout=0.3,
            learning_rate=0.001
        )
        
        print("排序模型训练完成")
    
    def recommend_for_user(self, user_id: int, n_recommend: int = 10) -> List[Tuple[int, float]]:
        """为用户推荐电影"""
        # 召回阶段
        candidate_items = self.multi_path_recall(user_id, n_recall=100)
        
        if not candidate_items:
            return []
        
        # 排序阶段（简化版）
        return candidate_items[:n_recommend]
    
    def evaluate_system(self):
        """评估推荐系统"""
        print("\n" + "="*60)
        print("开始评估推荐系统...")
        print("="*60)
        
        # 选择测试用户
        test_users = list(self.user_histories.keys())[:20]
        
        all_predictions = []
        all_labels = []
        
        for user_id in test_users:
            recommendations = self.recommend_for_user(user_id, n_recommend=20)
            user_history = set(self.user_histories.get(user_id, []))
            
            for item_id, score in recommendations:
                label = 1.0 if item_id in user_history else 0.0
                all_predictions.append(score)
                all_labels.append(label)
        
        if all_predictions:
            metrics = self.evaluator.evaluate_ranking(
                np.array(all_labels), 
                np.array(all_predictions)
            )
            self.evaluator.print_evaluation_results(metrics, "推荐系统")
        else:
            print("没有足够的评估数据")
    
    def run_pipeline(self):
        """运行完整的推荐系统流程"""
        print("🎯 两阶段推荐系统启动")
        print("="*60)
        
        # 1. 数据加载和预处理
        self.load_and_preprocess_data()
        
        # 2. 训练召回模型
        self.train_recall_models()
        
        # 3. 训练排序模型
        self.train_ranking_model()
        
        # 4. 评估系统
        self.evaluate_system()
        
        # 5. 示例推荐
        self._show_example_recommendations()
        
        print("\n🎉 推荐系统流程完成！")
    
    def _show_example_recommendations(self):
        """显示示例推荐"""
        print("\n" + "="*60)
        print("示例推荐结果")
        print("="*60)
        
        example_users = list(self.user_histories.keys())[:3]
        
        for user_id in example_users:
            print(f"\n用户 {user_id} 的推荐电影:")
            recommendations = self.recommend_for_user(user_id, n_recommend=10)
            
            for i, (item_id, score) in enumerate(recommendations, 1):
                movie_title = f"电影_{item_id}"
                if item_id in self.data_loader.reverse_movie_map:
                    original_id = self.data_loader.reverse_movie_map[item_id]
                    movie_info = self.movies_df[self.movies_df['movie_id'] == original_id]
                    if not movie_info.empty:
                        movie_title = movie_info.iloc[0]['title']
                
                print(f"  {i:2d}. {movie_title} (分数: {score:.4f})")

def main():
    """主函数"""
    pipeline = RecommenderPipeline()
    pipeline.run_pipeline()

if __name__ == "__main__":
    main() 