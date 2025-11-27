import pickle
import pandas as pd
import os
from django.conf import settings

class RecommendationEngine:
    def __init__(self):
        self.model = None
        self.data = None
        self.load_model()
        self.load_data()
    
    def load_model(self):
        """加载训练好的模型"""
        model_path = "D:/python/movie_recommendation_model.pkl"
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            print("模型加载成功")
        except Exception as e:
            print(f"模型加载失败: {e}")
    
    def load_data(self):
        """加载处理后的数据"""
        data_path = "D:/python/processed_data.csv"
        try:
            self.data = pd.read_csv(data_path)
            print("数据加载成功")
        except Exception as e:
            print(f"数据加载失败: {e}")
    
    def get_recommendations(self, user_id, n_recommendations=10):
        """为用户生成推荐"""
        if self.model is None or self.data is None:
            return []
        
        try:
            # 这里需要根据你的KNN模型实际接口进行调整
            # 假设你的模型有predict方法或类似功能
            # 以下为示例代码，需要根据你的模型实际情况修改
            
            # 获取用户未评分的电影
            user_ratings = self.data[self.data['user_id'] == user_id]
            rated_movies = user_ratings['movie_id'].tolist()
            all_movies = self.data['movie_id'].unique()
            unrated_movies = [movie for movie in all_movies if movie not in rated_movies]
            
            # 为未评分电影预测评分
            predictions = []
            for movie_id in unrated_movies[:100]:  # 限制数量以提高性能
                # 这里需要根据你的模型预测接口进行调整
                # predicted_rating = self.model.predict(user_id, movie_id)
                # 示例：随机预测（请替换为实际模型预测）
                import random
                predicted_rating = random.uniform(3.0, 5.0)
                predictions.append((movie_id, predicted_rating))
            
            # 按预测评分排序，取前n个
            predictions.sort(key=lambda x: x[1], reverse=True)
            top_recommendations = [movie_id for movie_id, _ in predictions[:n_recommendations]]
            
            # 关键修复：将NumPy类型转换为Python原生类型
            top_recommendations = [int(movie_id) for movie_id in top_recommendations]
            
            return top_recommendations
        
        except Exception as e:
            print(f"生成推荐时出错: {e}")
            return []

# 创建全局推荐引擎实例
recommendation_engine = RecommendationEngine()