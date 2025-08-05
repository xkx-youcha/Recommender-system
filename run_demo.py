#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
推荐系统演示脚本
快速测试推荐系统的各个组件
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

# 添加模块路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_data_loader():
    """测试数据加载器"""
    print("🔍 测试数据加载器...")
    
    try:
        from data_loader import MovieLensDataLoader
        
        # 初始化数据加载器
        data_loader = MovieLensDataLoader()
        
        # 尝试加载已处理的数据
        try:
            train_df, test_df, movies_df, users_df = data_loader.load_processed_data()
            print("✅ 成功加载已处理的数据")
            print(f"   - 训练集: {len(train_df)} 条记录")
            print(f"   - 测试集: {len(test_df)} 条记录")
            print(f"   - 电影数: {len(movies_df)}")
            print(f"   - 用户数: {len(users_df)}")
            return train_df, test_df, movies_df, users_df, data_loader
        except:
            print("⚠️  未找到已处理的数据，需要先运行数据预处理")
            return None, None, None, None, data_loader
            
    except Exception as e:
        print(f"❌ 数据加载器测试失败: {e}")
        return None, None, None, None, None

def test_recall_models(train_df, data_loader):
    """测试召回模型"""
    print("\n🔍 测试召回模型...")
    
    if train_df is None:
        print("❌ 无法测试召回模型，缺少训练数据")
        return
    
    try:
        from recall.itemcf_recall import ItemCFRecall
        from recall.hot_recall import HotRecall
        
        # 测试ItemCF召回
        print("   - 测试ItemCF召回...")
        itemcf = ItemCFRecall(top_k=50)
        itemcf.fit(train_df)
        
        # 测试召回
        test_user = train_df['user_id_encoded'].iloc[0]
        recall_results = itemcf.recall(test_user, n_recall=10)
        print(f"   ✅ ItemCF召回测试成功，为用户{test_user}召回{len(recall_results)}个物品")
        
        # 测试热门召回
        print("   - 测试热门召回...")
        hot_recall = HotRecall(top_k=50)
        hot_recall.fit(train_df, None)  # 简化测试
        
        hot_results = hot_recall.recall(test_user, n_recall=10)
        print(f"   ✅ 热门召回测试成功，为用户{test_user}召回{len(hot_results)}个物品")
        
    except Exception as e:
        print(f"❌ 召回模型测试失败: {e}")

def test_ranking_models():
    """测试排序模型"""
    print("\n🔍 测试排序模型...")
    
    try:
        from rank.model_deepfm import DeepFMRanker
        
        # 创建测试数据
        batch_size = 32
        feature_size = 50
        
        features = torch.randn(batch_size, feature_size)
        labels = torch.randint(0, 2, (batch_size,)).float()
        
        # 测试DeepFM
        print("   - 测试DeepFM排序模型...")
        ranker = DeepFMRanker(feature_size=feature_size)
        
        # 训练几步
        for epoch in range(3):
            loss = ranker.train_step(features, labels)
            print(f"     Epoch {epoch+1}, Loss: {loss:.4f}")
        
        # 预测
        predictions = ranker.predict(features)
        print(f"   ✅ DeepFM测试成功，预测形状: {predictions.shape}")
        
    except Exception as e:
        print(f"❌ 排序模型测试失败: {e}")

def test_evaluator():
    """测试评估器"""
    print("\n🔍 测试评估器...")
    
    try:
        from evaluate import RecommenderEvaluator
        
        # 创建测试数据
        np.random.seed(42)
        n_samples = 1000
        y_true = np.random.randint(0, 2, n_samples)
        y_pred = np.random.random(n_samples)
        
        # 测试评估
        evaluator = RecommenderEvaluator()
        metrics = evaluator.evaluate_ranking(y_true, y_pred)
        
        print("   ✅ 评估器测试成功")
        print(f"   - AUC: {metrics['auc']:.4f}")
        print(f"   - LogLoss: {metrics['logloss']:.4f}")
        
    except Exception as e:
        print(f"❌ 评估器测试失败: {e}")

def show_system_info():
    """显示系统信息"""
    print("🎯 两阶段推荐系统 - 系统信息")
    print("="*50)
    
    # 检查Python版本
    print(f"Python版本: {sys.version}")
    
    # 检查关键依赖
    dependencies = ['pandas', 'numpy', 'sklearn', 'torch', 'scipy']
    missing_deps = []
    
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"✅ {dep}: 已安装")
        except ImportError:
            print(f"❌ {dep}: 未安装")
            missing_deps.append(dep)
    
    if missing_deps:
        print(f"\n⚠️  缺少依赖: {', '.join(missing_deps)}")
        print("请运行: pip install " + " ".join(missing_deps))
    
    print("="*50)

def main():
    """主函数"""
    print("🚀 推荐系统演示开始")
    print("="*60)
    
    # 显示系统信息
    show_system_info()
    
    # 测试数据加载器
    train_df, test_df, movies_df, users_df, data_loader = test_data_loader()
    
    # 测试召回模型
    test_recall_models(train_df, data_loader)
    
    # 测试排序模型
    test_ranking_models()
    
    # 测试评估器
    test_evaluator()
    
    print("\n" + "="*60)
    print("🎉 推荐系统演示完成！")
    print("\n📝 下一步:")
    print("1. 运行 'python data_loader.py' 进行数据预处理")
    print("2. 运行 'python main_pipeline.py' 启动完整流程")
    print("3. 查看 README.md 了解详细使用方法")
    print("="*60)

if __name__ == "__main__":
    main()