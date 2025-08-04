import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
import os
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')

class MovieLensDataLoader:
    """
    MovieLens 1M 数据加载器
    负责数据读取、预处理、特征工程和训练测试集划分
    """
    
    def __init__(self, data_path: str = "../ml-1m"):
        """
        初始化数据加载器
        
        Args:
            data_path: MovieLens 1M 数据集路径
        """
        self.data_path = data_path
        self.user_encoder = LabelEncoder()
        self.movie_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
        # 编码后的用户和电影映射
        self.user_id_map = {}
        self.movie_id_map = {}
        self.reverse_user_map = {}
        self.reverse_movie_map = {}
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        加载 MovieLens 1M 数据集
        
        Returns:
            ratings_df: 评分数据
            movies_df: 电影数据  
            users_df: 用户数据
        """
        print("正在加载 MovieLens 1M 数据集...")
        
        # 加载评分数据
        ratings_df = pd.read_csv(
            os.path.join(self.data_path, "ratings.dat"),
            sep="::", 
            names=["user_id", "movie_id", "rating", "timestamp"],
            engine="python"
        )
        
        # 加载电影数据
        movies_df = pd.read_csv(
            os.path.join(self.data_path, "movies.dat"),
            sep="::",
            names=["movie_id", "title", "genres"],
            engine="python",
            encoding="latin-1"
        )
        
        # 加载用户数据
        users_df = pd.read_csv(
            os.path.join(self.data_path, "users.dat"),
            sep="::",
            names=["user_id", "gender", "age", "occupation", "zipcode"],
            engine="python"
        )
        
        print(f"数据加载完成:")
        print(f"- 评分数据: {len(ratings_df)} 条记录")
        print(f"- 电影数据: {len(movies_df)} 部电影")
        print(f"- 用户数据: {len(users_df)} 个用户")
        
        return ratings_df, movies_df, users_df
    
    def preprocess_data(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame, 
                       users_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        数据预处理：时间排序、ID编码、特征工程
        
        Args:
            ratings_df: 原始评分数据
            movies_df: 原始电影数据
            users_df: 原始用户数据
            
        Returns:
            预处理后的数据
        """
        print("开始数据预处理...")
        
        # 1. 时间排序
        ratings_df = ratings_df.sort_values('timestamp')
        
        # 2. 用户和电影ID编码
        ratings_df['user_id_encoded'] = self.user_encoder.fit_transform(ratings_df['user_id'])
        ratings_df['movie_id_encoded'] = self.movie_encoder.fit_transform(ratings_df['movie_id'])
        
        # 保存映射关系
        self.user_id_map = dict(zip(ratings_df['user_id'], ratings_df['user_id_encoded']))
        self.movie_id_map = dict(zip(ratings_df['movie_id'], ratings_df['movie_id_encoded']))
        self.reverse_user_map = {v: k for k, v in self.user_id_map.items()}
        self.reverse_movie_map = {v: k for k, v in self.movie_id_map.items()}
        
        # 3. 电影特征工程
        movies_df = self._process_movie_features(movies_df)
        
        # 4. 用户特征工程
        users_df = self._process_user_features(users_df, ratings_df)
        
        # 5. 评分数据特征工程
        ratings_df = self._process_rating_features(ratings_df, movies_df, users_df)
        
        print("数据预处理完成")
        return ratings_df, movies_df, users_df
    
    def _process_movie_features(self, movies_df: pd.DataFrame) -> pd.DataFrame:
        """处理电影特征"""
        # 提取年份
        movies_df['year'] = movies_df['title'].str.extract(r'\((\d{4})\)').astype(float)
        movies_df['year'].fillna(movies_df['year'].median(), inplace=True)
        
        # 处理类别特征
        genres = movies_df['genres'].str.get_dummies(sep='|')
        movies_df = pd.concat([movies_df, genres], axis=1)
        
        return movies_df
    
    def _process_user_features(self, users_df: pd.DataFrame, ratings_df: pd.DataFrame) -> pd.DataFrame:
        """处理用户特征"""
        # 用户统计特征
        user_stats = ratings_df.groupby('user_id').agg({
            'rating': ['count', 'mean', 'std'],
            'movie_id': 'nunique'
        }).reset_index()
        user_stats.columns = ['user_id', 'rating_count', 'rating_mean', 'rating_std', 'movie_count']
        user_stats.fillna(0, inplace=True)
        
        users_df = users_df.merge(user_stats, on='user_id', how='left')
        
        # 编码分类特征
        users_df['gender_encoded'] = (users_df['gender'] == 'M').astype(int)
        
        return users_df
    
    def _process_rating_features(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame, 
                                users_df: pd.DataFrame) -> pd.DataFrame:
        """处理评分特征"""
        # 添加电影特征
        movie_features = movies_df[['movie_id', 'year'] + 
                                 [col for col in movies_df.columns if col not in ['movie_id', 'title', 'genres', 'year']]]
        ratings_df = ratings_df.merge(movie_features, on='movie_id', how='left')
        
        # 添加用户特征
        user_features = users_df[['user_id', 'age', 'gender_encoded', 'rating_count', 'rating_mean', 'rating_std']]
        ratings_df = ratings_df.merge(user_features, on='user_id', how='left')
        
        return ratings_df
    
    def split_train_test(self, ratings_df: pd.DataFrame, test_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        按时间划分训练集和测试集
        
        Args:
            ratings_df: 评分数据
            test_ratio: 测试集比例
            
        Returns:
            train_df: 训练集
            test_df: 测试集
        """
        print(f"按时间划分训练集和测试集 (测试集比例: {test_ratio})")
        
        # 按时间排序
        ratings_df = ratings_df.sort_values('timestamp')
        
        # 计算分割点
        split_idx = int(len(ratings_df) * (1 - test_ratio))
        
        train_df = ratings_df.iloc[:split_idx].copy()
        test_df = ratings_df.iloc[split_idx:].copy()
        
        print(f"训练集: {len(train_df)} 条记录")
        print(f"测试集: {len(test_df)} 条记录")
        
        return train_df, test_df
    
    def get_user_history(self, ratings_df: pd.DataFrame, n_recent: int = 5) -> Dict[int, List[int]]:
        """
        获取用户历史行为序列
        
        Args:
            ratings_df: 评分数据
            n_recent: 最近N个行为
            
        Returns:
            user_history: 用户历史行为字典
        """
        user_history = {}
        
        for user_id in ratings_df['user_id_encoded'].unique():
            user_ratings = ratings_df[ratings_df['user_id_encoded'] == user_id].sort_values('timestamp')
            recent_movies = user_ratings['movie_id_encoded'].tail(n_recent).tolist()
            user_history[user_id] = recent_movies
        
        return user_history
    
    def get_movie_features(self, movies_df: pd.DataFrame) -> Dict[int, np.ndarray]:
        """
        获取电影特征向量
        
        Args:
            movies_df: 电影数据
            
        Returns:
            movie_features: 电影特征字典
        """
        movie_features = {}
        
        # 选择数值特征
        feature_cols = ['year'] + [col for col in movies_df.columns 
                                 if col not in ['movie_id', 'title', 'genres', 'year']]
        
        for _, row in movies_df.iterrows():
            movie_id = row['movie_id']
            if movie_id in self.movie_id_map:
                encoded_id = self.movie_id_map[movie_id]
                features = row[feature_cols].fillna(0).values.astype(np.float32)
                movie_features[encoded_id] = features
        
        return movie_features
    
    def get_user_features(self, users_df: pd.DataFrame) -> Dict[int, np.ndarray]:
        """
        获取用户特征向量
        
        Args:
            users_df: 用户数据
            
        Returns:
            user_features: 用户特征字典
        """
        user_features = {}
        
        # 选择数值特征
        feature_cols = ['age', 'gender_encoded', 'rating_count', 'rating_mean', 'rating_std']
        
        for _, row in users_df.iterrows():
            user_id = row['user_id']
            if user_id in self.user_id_map:
                encoded_id = self.user_id_map[user_id]
                features = row[feature_cols].fillna(0).values.astype(np.float32)
                user_features[encoded_id] = features
        
        return user_features
    
    def save_processed_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame, 
                           movies_df: pd.DataFrame, users_df: pd.DataFrame, 
                           save_path: str = "./processed_data"):
        """保存处理后的数据"""
        os.makedirs(save_path, exist_ok=True)
        
        # 保存数据
        train_df.to_csv(os.path.join(save_path, "train.csv"), index=False)
        test_df.to_csv(os.path.join(save_path, "test.csv"), index=False)
        movies_df.to_csv(os.path.join(save_path, "movies.csv"), index=False)
        users_df.to_csv(os.path.join(save_path, "users.csv"), index=False)
        
        # 保存编码器
        with open(os.path.join(save_path, "encoders.pkl"), "wb") as f:
            pickle.dump({
                'user_encoder': self.user_encoder,
                'movie_encoder': self.movie_encoder,
                'user_id_map': self.user_id_map,
                'movie_id_map': self.movie_id_map,
                'reverse_user_map': self.reverse_user_map,
                'reverse_movie_map': self.reverse_movie_map
            }, f)
        
        print(f"数据已保存到: {save_path}")
    
    def load_processed_data(self, load_path: str = "../processed_data"):
        """加载处理后的数据"""
        train_df = pd.read_csv(os.path.join(load_path, "train.csv"))
        test_df = pd.read_csv(os.path.join(load_path, "test.csv"))
        movies_df = pd.read_csv(os.path.join(load_path, "movies.csv"))
        users_df = pd.read_csv(os.path.join(load_path, "users.csv"))
        
        with open(os.path.join(load_path, "encoders.pkl"), "rb") as f:
            encoders = pickle.load(f)
            self.user_encoder = encoders['user_encoder']
            self.movie_encoder = encoders['movie_encoder']
            self.user_id_map = encoders['user_id_map']
            self.movie_id_map = encoders['movie_id_map']
            self.reverse_user_map = encoders['reverse_user_map']
            self.reverse_movie_map = encoders['reverse_movie_map']
        
        return train_df, test_df, movies_df, users_df

def main():
    """数据加载器测试"""
    # 初始化数据加载器
    data_loader = MovieLensDataLoader()
    
    # 加载原始数据
    ratings_df, movies_df, users_df = data_loader.load_data()
    
    # 预处理数据
    ratings_df, movies_df, users_df = data_loader.preprocess_data(ratings_df, movies_df, users_df)
    
    # 划分训练测试集
    train_df, test_df = data_loader.split_train_test(ratings_df)
    
    # 获取特征
    user_history = data_loader.get_user_history(train_df)
    movie_features = data_loader.get_movie_features(movies_df)
    user_features = data_loader.get_user_features(users_df)
    
    # 保存处理后的数据
    data_loader.save_processed_data(train_df, test_df, movies_df, users_df)
    
    print("数据加载器测试完成！")
    print(f"用户数量: {len(user_features)}")
    print(f"电影数量: {len(movie_features)}")
    print(f"训练集大小: {len(train_df)}")
    print(f"测试集大小: {len(test_df)}")

if __name__ == "__main__":
    main() 