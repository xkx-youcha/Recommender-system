import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple
import pickle
import os

class AttentionLayer(nn.Module):
    """
    注意力层
    用于计算用户历史行为与候选物品的注意力权重
    """
    
    def __init__(self, embedding_dim: int, attention_dim: int = 16):
        """
        初始化注意力层
        
        Args:
            embedding_dim: 嵌入维度
            attention_dim: 注意力隐藏层维度
        """
        super(AttentionLayer, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.attention_dim = attention_dim
        
        # 注意力网络
        self.attention_net = nn.Sequential(
            nn.Linear(embedding_dim * 4, attention_dim),
            nn.ReLU(),
            nn.Linear(attention_dim, 1)
        )
        
    def forward(self, user_behavior_emb, candidate_emb):
        """
        前向传播
        
        Args:
            user_behavior_emb: 用户行为嵌入 [batch_size, seq_len, embedding_dim]
            candidate_emb: 候选物品嵌入 [batch_size, embedding_dim]
            
        Returns:
            attention_output: 注意力输出
            attention_weights: 注意力权重
        """
        batch_size, seq_len, embedding_dim = user_behavior_emb.size()
        
        # 扩展候选物品嵌入
        candidate_emb_expanded = candidate_emb.unsqueeze(1).expand(-1, seq_len, -1)
        
        # 计算注意力输入
        attention_input = torch.cat([
            user_behavior_emb,
            candidate_emb_expanded,
            user_behavior_emb - candidate_emb_expanded,
            user_behavior_emb * candidate_emb_expanded
        ], dim=-1)
        
        # 计算注意力权重
        attention_weights = self.attention_net(attention_input).squeeze(-1)
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # 加权求和
        attention_output = torch.sum(user_behavior_emb * attention_weights.unsqueeze(-1), dim=1)
        
        return attention_output, attention_weights

class DIN(nn.Module):
    """
    Deep Interest Network
    使用注意力机制建模用户对不同候选物品的兴趣差异
    """
    
    def __init__(self, feature_size: int, embedding_dim: int = 16, 
                 attention_dim: int = 16, hidden_dims: List[int] = [200, 80],
                 dropout: float = 0.3):
        """
        初始化DIN模型
        
        Args:
            feature_size: 特征数量
            embedding_dim: 嵌入维度
            attention_dim: 注意力隐藏层维度
            hidden_dims: DNN隐藏层维度
            dropout: Dropout比例
        """
        super(DIN, self).__init__()
        
        self.feature_size = feature_size
        self.embedding_dim = embedding_dim
        self.attention_dim = attention_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        
        # 嵌入层
        self.embedding = nn.Linear(feature_size, embedding_dim)
        
        # 注意力层
        self.attention = AttentionLayer(embedding_dim, attention_dim)
        
        # DNN层
        self.dnn_layers = nn.ModuleList()
        input_dim = embedding_dim * 2  # 用户兴趣表示 + 候选物品嵌入
        
        for hidden_dim in hidden_dims:
            self.dnn_layers.append(nn.Linear(input_dim, hidden_dim))
            input_dim = hidden_dim
        
        # 输出层
        self.output_layer = nn.Linear(hidden_dims[-1], 1)
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, user_features, candidate_features, user_behavior_seq=None):
        """
        前向传播
        
        Args:
            user_features: 用户特征 [batch_size, feature_size]
            candidate_features: 候选物品特征 [batch_size, feature_size]
            user_behavior_seq: 用户行为序列 [batch_size, seq_len, feature_size]
            
        Returns:
            output: 模型输出
        """
        # 嵌入用户特征和候选物品特征
        user_emb = self.embedding(user_features)
        candidate_emb = self.embedding(candidate_features)
        
        # 注意力机制
        if user_behavior_seq is not None:
            # 嵌入用户行为序列
            batch_size, seq_len, _ = user_behavior_seq.size()
            user_behavior_emb = self.embedding(user_behavior_seq.view(-1, self.feature_size))
            user_behavior_emb = user_behavior_emb.view(batch_size, seq_len, self.embedding_dim)
            
            # 计算注意力
            user_interest_emb, attention_weights = self.attention(user_behavior_emb, candidate_emb)
        else:
            # 如果没有行为序列，直接使用用户嵌入
            user_interest_emb = user_emb
        
        # 拼接用户兴趣表示和候选物品嵌入
        combined = torch.cat([user_interest_emb, candidate_emb], dim=1)
        
        # DNN层
        dnn_input = combined
        for layer in self.dnn_layers:
            dnn_input = F.relu(layer(dnn_input))
            dnn_input = self.dropout_layer(dnn_input)
        
        # 输出层
        output = torch.sigmoid(self.output_layer(dnn_input))
        
        return output.squeeze()
    
    def predict(self, user_features, candidate_features, user_behavior_seq=None):
        """
        预测方法
        
        Args:
            user_features: 用户特征
            candidate_features: 候选物品特征
            user_behavior_seq: 用户行为序列
            
        Returns:
            predictions: 预测结果
        """
        self.eval()
        with torch.no_grad():
            return self.forward(user_features, candidate_features, user_behavior_seq)

class DINRanker:
    """
    DIN排序器
    封装DIN模型，提供训练、预测等功能
    """
    
    def __init__(self, feature_size: int, embedding_dim: int = 16,
                 attention_dim: int = 16, hidden_dims: List[int] = [200, 80],
                 dropout: float = 0.3, learning_rate: float = 0.001, device: str = 'cpu'):
        """
        初始化DIN排序器
        
        Args:
            feature_size: 特征数量
            embedding_dim: 嵌入维度
            attention_dim: 注意力隐藏层维度
            hidden_dims: DNN隐藏层维度
            dropout: Dropout比例
            learning_rate: 学习率
            device: 设备
        """
        self.feature_size = feature_size
        self.embedding_dim = embedding_dim
        self.attention_dim = attention_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.device = device
        
        # 初始化模型
        self.model = DIN(feature_size, embedding_dim, attention_dim, hidden_dims, dropout)
        self.model.to(device)
        
        # 优化器和损失函数
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.BCELoss()
        
    def train_step(self, user_features: torch.Tensor, candidate_features: torch.Tensor,
                   labels: torch.Tensor, user_behavior_seq: torch.Tensor = None) -> float:
        """
        训练一步
        
        Args:
            user_features: 用户特征
            candidate_features: 候选物品特征
            labels: 标签
            user_behavior_seq: 用户行为序列
            
        Returns:
            loss: 损失值
        """
        self.model.train()
        
        # 前向传播
        predictions = self.model(user_features, candidate_features, user_behavior_seq)
        loss = self.criterion(predictions, labels)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def predict(self, user_features: torch.Tensor, candidate_features: torch.Tensor,
                user_behavior_seq: torch.Tensor = None) -> torch.Tensor:
        """
        预测
        
        Args:
            user_features: 用户特征
            candidate_features: 候选物品特征
            user_behavior_seq: 用户行为序列
            
        Returns:
            predictions: 预测结果
        """
        return self.model.predict(user_features, candidate_features, user_behavior_seq)
    
    def save_model(self, save_path: str = "./models/din_model.pth"):
        """保存模型"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'feature_size': self.feature_size,
            'embedding_dim': self.embedding_dim,
            'attention_dim': self.attention_dim,
            'hidden_dims': self.hidden_dims,
            'dropout': self.dropout,
            'learning_rate': self.learning_rate
        }, save_path)
        
        print(f"DIN模型已保存到: {save_path}")
    
    def load_model(self, load_path: str = "./models/din_model.pth"):
        """加载模型"""
        checkpoint = torch.load(load_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"DIN模型已从 {load_path} 加载")

def main():
    """DIN模型测试"""
    # 创建测试数据
    batch_size = 32
    feature_size = 100
    seq_len = 10
    
    # 随机特征和标签
    user_features = torch.randn(batch_size, feature_size)
    candidate_features = torch.randn(batch_size, feature_size)
    user_behavior_seq = torch.randn(batch_size, seq_len, feature_size)
    labels = torch.randint(0, 2, (batch_size,)).float()
    
    # 初始化模型
    ranker = DINRanker(feature_size=feature_size)
    
    # 训练几步
    for epoch in range(5):
        loss = ranker.train_step(user_features, candidate_features, labels, user_behavior_seq)
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}")
    
    # 预测
    predictions = ranker.predict(user_features, candidate_features, user_behavior_seq)
    print(f"预测结果形状: {predictions.shape}")
    print(f"预测值范围: [{predictions.min():.4f}, {predictions.max():.4f}]")
    
    # 保存模型
    ranker.save_model()

if __name__ == "__main__":
    main() 