import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple
import pickle
import os

class FM(nn.Module):
    """
    Factorization Machine层
    用于捕捉低阶特征交互
    """
    
    def __init__(self, feature_size: int, k: int = 10):
        """
        初始化FM层
        
        Args:
            feature_size: 特征数量
            k: 隐向量维度
        """
        super(FM, self).__init__()
        self.feature_size = feature_size
        self.k = k
        
        # 一阶特征权重
        self.linear = nn.Linear(feature_size, 1)
        
        # 二阶特征隐向量
        self.v = nn.Parameter(torch.randn(feature_size, k))
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入特征 [batch_size, feature_size]
            
        Returns:
            fm_output: FM层输出
        """
        # 一阶特征
        linear_part = self.linear(x)
        
        # 二阶特征交互
        square_of_sum = torch.mm(x, self.v).pow(2).sum(1)
        sum_of_square = torch.mm(x.pow(2), self.v.pow(2)).sum(1)
        interaction_part = 0.5 * (square_of_sum - sum_of_square)
        
        fm_output = linear_part.squeeze() + interaction_part
        
        return fm_output

class DeepFM(nn.Module):
    """
    DeepFM模型
    结合FM和DNN，同时捕捉低阶和高阶特征交互
    """
    
    def __init__(self, feature_size: int, embedding_dim: int = 10, 
                 hidden_dims: List[int] = [400, 400, 400], dropout: float = 0.5):
        """
        初始化DeepFM模型
        
        Args:
            feature_size: 特征数量
            embedding_dim: FM隐向量维度
            hidden_dims: DNN隐藏层维度列表
            dropout: Dropout比例
        """
        super(DeepFM, self).__init__()
        
        self.feature_size = feature_size
        self.embedding_dim = embedding_dim
        
        # FM层
        self.fm = FM(feature_size, embedding_dim)
        
        # DNN层
        self.dnn_layers = nn.ModuleList()
        input_dim = feature_size
        
        for hidden_dim in hidden_dims:
            self.dnn_layers.append(nn.Linear(input_dim, hidden_dim))
            input_dim = hidden_dim
        
        # 输出层
        self.output_layer = nn.Linear(embedding_dim + hidden_dims[-1], 1)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入特征 [batch_size, feature_size]
            
        Returns:
            output: 模型输出
        """
        # FM部分
        fm_output = self.fm(x)
        
        # DNN部分
        dnn_input = x
        for layer in self.dnn_layers:
            dnn_input = F.relu(layer(dnn_input))
            dnn_input = self.dropout(dnn_input)
        
        # 拼接FM和DNN输出
        combined = torch.cat([fm_output.unsqueeze(1), dnn_input], dim=1)
        
        # 输出层
        output = torch.sigmoid(self.output_layer(combined))
        
        return output.squeeze()
    
    def predict(self, x):
        """
        预测方法
        
        Args:
            x: 输入特征
            
        Returns:
            predictions: 预测结果
        """
        self.eval()
        with torch.no_grad():
            return self.forward(x)

class DeepFMRanker:
    """
    DeepFM排序器
    封装DeepFM模型，提供训练、预测等功能
    """
    
    def __init__(self, feature_size: int, embedding_dim: int = 10,
                 hidden_dims: List[int] = [400, 400, 400], dropout: float = 0.5,
                 learning_rate: float = 0.001, device: str = 'cpu'):
        """
        初始化DeepFM排序器
        
        Args:
            feature_size: 特征数量
            embedding_dim: FM隐向量维度
            hidden_dims: DNN隐藏层维度
            dropout: Dropout比例
            learning_rate: 学习率
            device: 设备
        """
        self.feature_size = feature_size
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.device = device
        
        # 初始化模型
        self.model = DeepFM(feature_size, embedding_dim, hidden_dims, dropout)
        self.model.to(device)
        
        # 优化器和损失函数
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.BCELoss()
        
    def train_step(self, features: torch.Tensor, labels: torch.Tensor) -> float:
        """
        训练一步
        
        Args:
            features: 特征张量
            labels: 标签张量
            
        Returns:
            loss: 损失值
        """
        self.model.train()
        
        # 前向传播
        predictions = self.model(features)
        loss = self.criterion(predictions, labels)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def predict(self, features: torch.Tensor) -> torch.Tensor:
        """
        预测
        
        Args:
            features: 特征张量
            
        Returns:
            predictions: 预测结果
        """
        return self.model.predict(features)
    
    def save_model(self, save_path: str = "./models/deepfm_model.pth"):
        """保存模型"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'feature_size': self.feature_size,
            'embedding_dim': self.embedding_dim,
            'hidden_dims': self.hidden_dims,
            'dropout': self.dropout,
            'learning_rate': self.learning_rate
        }, save_path)
        
        print(f"DeepFM模型已保存到: {save_path}")
    
    def load_model(self, load_path: str = "./models/deepfm_model.pth"):
        """加载模型"""
        checkpoint = torch.load(load_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"DeepFM模型已从 {load_path} 加载")

def main():
    """DeepFM模型测试"""
    # 创建测试数据
    batch_size = 32
    feature_size = 100
    
    # 随机特征和标签
    features = torch.randn(batch_size, feature_size)
    labels = torch.randint(0, 2, (batch_size,)).float()
    
    # 初始化模型
    ranker = DeepFMRanker(feature_size=feature_size)
    
    # 训练几步
    for epoch in range(5):
        loss = ranker.train_step(features, labels)
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}")
    
    # 预测
    predictions = ranker.predict(features)
    print(f"预测结果形状: {predictions.shape}")
    print(f"预测值范围: [{predictions.min():.4f}, {predictions.max():.4f}]")
    
    # 保存模型
    ranker.save_model()

if __name__ == "__main__":
    main() 