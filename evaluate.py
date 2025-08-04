import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss
from typing import Dict, List, Tuple
import torch

class RecommenderEvaluator:
    """
    推荐系统评估器
    提供各种评估指标：AUC、LogLoss、Recall@K、NDCG@K等
    """
    
    def __init__(self):
        """初始化评估器"""
        pass
    
    def calculate_auc(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        计算AUC
        
        Args:
            y_true: 真实标签
            y_pred: 预测概率
            
        Returns:
            auc: AUC值
        """
        try:
            return roc_auc_score(y_true, y_pred)
        except ValueError:
            return 0.0
    
    def calculate_logloss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        计算LogLoss
        
        Args:
            y_true: 真实标签
            y_pred: 预测概率
            
        Returns:
            logloss: LogLoss值
        """
        try:
            return log_loss(y_true, y_pred)
        except ValueError:
            return float('inf')
    
    def calculate_recall_at_k(self, y_true: np.ndarray, y_pred: np.ndarray, k: int = 10) -> float:
        """
        计算Recall@K
        
        Args:
            y_true: 真实标签 [n_samples]
            y_pred: 预测分数 [n_samples]
            k: Top-K
            
        Returns:
            recall_at_k: Recall@K值
        """
        # 获取top-k索引
        top_k_indices = np.argsort(y_pred)[::-1][:k]
        
        # 计算召回率
        relevant_in_top_k = np.sum(y_true[top_k_indices])
        total_relevant = np.sum(y_true)
        
        if total_relevant == 0:
            return 0.0
        
        return relevant_in_top_k / total_relevant
    
    def calculate_ndcg_at_k(self, y_true: np.ndarray, y_pred: np.ndarray, k: int = 10) -> float:
        """
        计算NDCG@K
        
        Args:
            y_true: 真实标签 [n_samples]
            y_pred: 预测分数 [n_samples]
            k: Top-K
            
        Returns:
            ndcg_at_k: NDCG@K值
        """
        # 获取top-k索引
        top_k_indices = np.argsort(y_pred)[::-1][:k]
        
        # 计算DCG
        dcg = 0.0
        for i, idx in enumerate(top_k_indices):
            dcg += y_true[idx] / np.log2(i + 2)  # log2(i+2) 因为i从0开始
        
        # 计算IDCG（理想DCG）
        ideal_scores = np.sort(y_true)[::-1][:k]
        idcg = 0.0
        for i, score in enumerate(ideal_scores):
            idcg += score / np.log2(i + 2)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    def evaluate_ranking(self, y_true: np.ndarray, y_pred: np.ndarray, 
                        k_list: List[int] = [5, 10, 20]) -> Dict[str, float]:
        """
        评估排序效果
        
        Args:
            y_true: 真实标签
            y_pred: 预测分数
            k_list: K值列表
            
        Returns:
            metrics: 评估指标字典
        """
        metrics = {}
        
        # 基础指标
        metrics['auc'] = self.calculate_auc(y_true, y_pred)
        metrics['logloss'] = self.calculate_logloss(y_true, y_pred)
        
        # Top-K指标
        for k in k_list:
            metrics[f'recall@{k}'] = self.calculate_recall_at_k(y_true, y_pred, k)
            metrics[f'ndcg@{k}'] = self.calculate_ndcg_at_k(y_true, y_pred, k)
        
        return metrics
    
    def print_evaluation_results(self, metrics: Dict[str, float], model_name: str = "Model"):
        """
        打印评估结果
        
        Args:
            metrics: 评估指标字典
            model_name: 模型名称
        """
        print(f"\n{'='*50}")
        print(f"{model_name} 评估结果")
        print(f"{'='*50}")
        
        # 基础指标
        if 'auc' in metrics:
            print(f"AUC: {metrics['auc']:.4f}")
        if 'logloss' in metrics:
            print(f"LogLoss: {metrics['logloss']:.4f}")
        
        # Top-K指标
        k_metrics = {}
        for key, value in metrics.items():
            if '@' in key:
                k = key.split('@')[1]
                metric_type = key.split('@')[0]
                if k not in k_metrics:
                    k_metrics[k] = {}
                k_metrics[k][metric_type] = value
        
        for k in sorted(k_metrics.keys(), key=int):
            print(f"\nTop-{k} 指标:")
            for metric_type, value in k_metrics[k].items():
                print(f"  {metric_type.upper()}@{k}: {value:.4f}")
        
        print(f"{'='*50}")

def main():
    """评估器测试"""
    # 创建测试数据
    np.random.seed(42)
    n_samples = 1000
    
    # 模拟真实标签和预测分数
    y_true = np.random.randint(0, 2, n_samples)
    y_pred = np.random.random(n_samples)
    
    # 初始化评估器
    evaluator = RecommenderEvaluator()
    
    # 评估排序效果
    metrics = evaluator.evaluate_ranking(y_true, y_pred)
    
    # 打印结果
    evaluator.print_evaluation_results(metrics, "测试模型")

if __name__ == "__main__":
    main() 