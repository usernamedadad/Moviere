import pandas as pd

# 定义数据集路径
data_path = r'D:\下载\ml-100k\ml-100k\u.data'

# 读取数据（假设数据文件是 u.data，使用制表符分隔）
data = pd.read_csv(data_path, delimiter='\t', header=None, names=['user_id', 'movie_id', 'rating', 'timestamp'])

# 显示数据的前几行，查看数据结构
print(data.head())

# 去除重复数据
data.drop_duplicates(inplace=True)

# 去除空值（如果有的话）
data.dropna(inplace=True)

# 确认去重和去空值后数据的形状
print(f'数据集的形状：{data.shape}')

# 如果需要保存处理后的数据，可以使用：
# data.to_csv('processed_data.csv', index=False)
import pandas as pd

# 1. 读取用户评分数据
data_path = r'D:\下载\ml-100k\ml-100k\u.data'
data = pd.read_csv(data_path, delimiter='\t', header=None, names=['user_id', 'movie_id', 'rating', 'timestamp'])

# 2. 确保评分字段是浮动类型，并且在合理范围内（1到5之间）
data['rating'] = pd.to_numeric(data['rating'], errors='coerce')  # 强制转换为数值类型，避免非数字错误
data = data[(data['rating'] >= 1) & (data['rating'] <= 5)]  # 去除不在1-5范围内的评分

# 3. 读取电影信息数据（u.item），使用 latin-1 编码处理文件中的特殊字符
movies_data_path = r'D:\下载\ml-100k\ml-100k\u.item'
movies_columns = ['movie_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL', 
                  'unknown', 'action', 'adventure', 'animation', 'children', 
                  'comedy', 'crime', 'documentary', 'drama', 'fantasy', 
                  'film_noir', 'horror', 'musical', 'mystery', 'romance', 
                  'sci_fi', 'thriller', 'war', 'western']

# 读取电影信息数据
movies_data = pd.read_csv(movies_data_path, sep='|', header=None, names=movies_columns, encoding='latin-1')

# 4. 格式化电影类型（Genres）：转换为电影类型的列表
movie_types = movies_data.columns[5:]  # 从第5列到最后一列是电影类型列
movies_data['genres'] = movies_data[movie_types].apply(lambda row: [movie_types[i] for i in range(len(row)) if row[i] == 1], axis=1)

# 5. 合并评分数据与电影信息
merged_data = pd.merge(data, movies_data[['movie_id', 'title', 'genres']], on='movie_id', how='left')

# 6. 格式化时间戳：确保时间戳被转换为 datetime 类型，并提取年、月、日等特征
merged_data['timestamp'] = pd.to_datetime(merged_data['timestamp'], unit='s')  # 转换时间戳
merged_data['year'] = merged_data['timestamp'].dt.year  # 提取年份
merged_data['month'] = merged_data['timestamp'].dt.month  # 提取月份
merged_data['day'] = merged_data['timestamp'].dt.day  # 提取日期
merged_data['hour'] = merged_data['timestamp'].dt.hour  # 提取小时

# 7. 格式化电影标题：去除多余的空格
merged_data['title'] = merged_data['title'].str.strip()


# 8. 进一步提取统计特征（用户的平均评分）
user_avg_rating = merged_data.groupby('user_id')['rating'].mean().reset_index()
user_avg_rating.rename(columns={'rating': 'avg_rating'}, inplace=True)

# 9将用户的平均评分合并到原数据中
merged_data = pd.merge(merged_data, user_avg_rating, on='user_id', how='left')

# 10. 显示包含用户平均评分的合并数据
print(merged_data[['user_id', 'movie_id', 'rating', 'avg_rating', 'title', 'genres', 'year', 'month', 'day', 'hour']].head())
# 11.保存处理后的数据到 CSV 文件
output_path = r'D:\下载\ml-100k\ml-100k\processed_data.csv'
merged_data.to_csv(output_path, index=False)

print(f"处理后的数据已保存到: {output_path}")

import pandas as pd
from surprise import Reader, Dataset, KNNBasic
from surprise.model_selection import train_test_split, GridSearchCV
from surprise import accuracy
import pickle
# 1. 加载和准备数据
data_path = r"D:\下载\ml-100k\ml-100k\processed_data.csv"
data = pd.read_csv(data_path)

print(data.head())

# 创建 Surprise 的 Reader 对象
reader = Reader(rating_scale=(1, 5))

# 转换数据为 Surprise 的 Dataset 格式
dataset = Dataset.load_from_df(data[['user_id', 'movie_id', 'rating']], reader)

# 查看数据集的一些信息
trainset = dataset.build_full_trainset()
print(f"Number of users: {trainset.n_users}")
print(f"Number of items: {trainset.n_items}")

# 2. 使用协同过滤算法（基于用户的 KNN 算法）
# 将数据拆分为训练集和测试集，使用 80% 训练集和 20% 测试集
trainset, testset = train_test_split(dataset, test_size=0.2, random_state=42)  # 固定随机种子以保证可重复性

# 3. 超参数调优（使用 GridSearchCV）
param_grid = {
    'k': [10, 20, 30, 40, 50],  # 邻居数
    'sim_options': {
        'name': ['cosine'],  # 使用余弦相似度
        'user_based': [True]  # 基于用户的协同过滤
    }
}

# 使用 GridSearchCV 进行超参数调优
gs = GridSearchCV(KNNBasic, param_grid, measures=['rmse'], cv=3)
gs.fit(dataset)

# 输出最佳参数组合和RMSE值
print(f"Best parameters: {gs.best_params}")
print(f"Best RMSE: {gs.best_score['rmse']}")

# 使用最佳参数训练模型
best_model = gs.best_estimator['rmse']
best_model.fit(trainset)

# 在测试集上进行预测
predictions = best_model.test(testset)

# 计算RMSE
rmse = accuracy.rmse(predictions)
print(f'RMSE: {rmse}')

# 4. 查看推荐结果
# 获取用户ID为1的用户，预测该用户对所有电影的评分
user_id = 1
all_movie_ids = list(range(1, trainset.n_items + 1))

# 对所有电影进行预测
predictions = [best_model.predict(user_id, movie_id) for movie_id in all_movie_ids]

# 将预测结果按评分从高到低排序，选择前5个推荐
top_5_recommendations = sorted(predictions, key=lambda x: x.est, reverse=True)[:5]

# 显示前5个推荐电影（电影ID和预测评分）
for prediction in top_5_recommendations:
    print(f"Movie ID: {prediction.iid}, Predicted Rating: {prediction.est}")

# 5. 保存训练后的模型
model_filename = 'movie_recommendation_model.pkl'
with open(model_filename, 'wb') as f:
    pickle.dump(best_model, f)

print(f"Model saved as {model_filename}")



with open('movie_recommendation_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# 使用加载的模型进行预测
user_id = 1
all_movie_ids = list(range(1, trainset.n_items + 1))
predictions = [loaded_model.predict(user_id, movie_id) for movie_id in all_movie_ids]

# 按评分排序并推荐前5个
top_5_recommendations = sorted(predictions, key=lambda x: x.est, reverse=True)[:5]
for prediction in top_5_recommendations:
    print(f"Movie ID: {prediction.iid}, Predicted Rating: {prediction.est}")







