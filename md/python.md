# 一.对于数据的预处理：

## 1.清洗数据

``` py

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

#保存处理后的数据
# data.to_csv('processed_data.csv', index=False)
```

## 2.数据格式化和特征工程

```py

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
```

### **详细步骤解释**

1. **读取用户评分数据**：我们读取了 `u.data` 文件，包含了用户对电影的评分，格式为：`user_id, movie_id, rating, timestamp`。

2. **读取电影信息数据**：我们读取了 `u.item` 文件，并提取了电影的 ID、标题、类型等信息。

3. **提取电影类型（Genres）**：电影类型信息保存在 `u.item` 文件的第 6 列到最后一列，我们通过检查这些列的值（0 或 1），生成每个电影的类型列表。例如，如果某部电影的 `action` 列为 1，则说明它属于动作片类型。

4. **合并数据**：将 `u.data` 中的评分数据与电影的基本信息（包括类型）进行合并，得到了每个评分条目的电影信息。

5. **时间戳处理**：我们将时间戳转化为日期，并从中提取出年份、月份、日期和小时等信息。这样可以更好地理解用户评分的时间分布情况。

6. **用户平均评分**：我们计算了每个用户的平均评分，并将其合并到数据中，这对于某些推荐算法可能有帮助，例如基于用户的协同过滤。

7. **最终数据**：最终，你的数据集包括了电影的基本信息（标题、类型）、用户的评分信息、时间特征、以及用户的平均评分等。

   # 二.训练模型

   1.实现推荐算法：

   使用 **余弦相似度** 度量方法，并且使用 **交叉验证** 来进行超参数调优，确保模型不会因为过拟合而表现不佳。训练和测试数据的拆分比例将是 **80% 训练集和 20% 测试集**，并且在训练过程中，使用 `train_test_split` 来确保每次都随机选择不同的测试集和训练集。

   ```py
   import pandas as pd
   from surprise import Reader, Dataset, KNNBasic
   from surprise.model_selection import train_test_split, GridSearchCV
   from surprise import accuracy
   import pickle
   
   # 1. 加载和准备数据
   # 加载预处理后的数据
   data_path = r"D:\下载\ml-100k\ml-100k\processed_data.csv"
   data = pd.read_csv(data_path)
   
   # 查看数据的前几行
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
   ```
   
   ### 2.**修改和改进的部分：**

   1. **交叉验证与超参数调优：**
      - 我使用了 `GridSearchCV` 来对 `k` 值和相似度度量方法进行调优。这样做能够确保你选择到最优的超参数（例如最优的邻居数量 `k`）。
      - 在 `param_grid` 中，我只使用了 `cosine` 作为相似度度量方法，并调整了 `k` 值的范围（`[10, 20, 30, 40, 50]`）。
      - 交叉验证的次数设置为 3（即 `cv=3`），表示数据将被分为 3 个子集，依次用其中 2 个进行训练，剩下的 1 个进行验证。
   2. **训练集与测试集的拆分：**
      - 使用 `train_test_split(dataset, test_size=0.2, random_state=42)` 来将数据集拆分为 80% 训练集和 20% 测试集。`random_state=42` 确保每次运行时能够得到相同的拆分结果。
   3. **保存训练后的模型：**
      - 使用 `pickle.dump` 将训练好的模型保存到本地，文件名为 `movie_recommendation_model.pkl`。可以稍后加载这个模型进行推荐或评估。
   
   ### 3.**其他注意事项：**

   - **调优 `k` 值**：根据 `GridSearchCV` 的结果选择最佳的 `k` 值，进一步提升模型的性能。
   - **RMSE评估**：通过 `accuracy.rmse(predictions)` 计算 RMSE，用于评估模型的预测精度。
   - **推荐结果**：我提供了获取用户（ID=1）对所有电影的预测评分，并选出评分最高的 5 个电影进行推荐。
   
   # 三.**模型的评估和预测：**

   采用 **RMSE** 作为验证指标，因为它是最常用且直观的评估推荐系统准确度的方法。已经在训练阶段计算了 **RMSE**，通过 **加载已保存的模型** 来对 **测试集** 进行验证，并计算 **RMSE**。

   ```py
   import pandas as pd
   from surprise import Reader, Dataset, KNNBasic
   from surprise.model_selection import train_test_split  # 确保导入了 train_test_split
   from surprise import accuracy
   import pickle
   
   # 1. 加载数据
   data_path = r"D:\下载\ml-100k\ml-100k\processed_data.csv"
   data = pd.read_csv(data_path)
   
   # 创建 Surprise 的 Reader 对象
   reader = Reader(rating_scale=(1, 5))
   
   # 转换数据为 Surprise 的 Dataset 格式
   dataset = Dataset.load_from_df(data[['user_id', 'movie_id', 'rating']], reader)
   
   # 将数据拆分为训练集和测试集，使用 80% 训练集和 20% 测试集
   trainset, testset = train_test_split(dataset, test_size=0.2, random_state=42)
   
   # 2. 加载保存的模型
   model_filename = 'movie_recommendation_model.pkl'
   
   with open(model_filename, 'rb') as f:
       best_model = pickle.load(f)
   
   # 3. 在测试集上进行预测
   predictions = best_model.test(testset)
   
   # 4. 计算并输出 RMSE
   rmse = accuracy.rmse(predictions)
   print(f'RMSE: {rmse}')
   
   #运行结果：
   #RMSE: 1.0187
   #RMSE: 1.018721342662647
   #达到预期
   ```
   
   # 四：通过命令行实现交换：
   
   ```py
   import pickle
   import pandas as pd
   from sklearn.metrics.pairwise import cosine_similarity
   import numpy as np
   
   # 加载模型
   with open('movie_recommendation_model.pkl', 'rb') as f:
       model = pickle.load(f)
   
   # 加载数据
   df = pd.read_csv('processed_data.csv')
   user_movie_matrix = df.pivot_table(index='user_id', columns='movie_id', values='rating')
   user_movie_matrix.fillna(0, inplace=True)
   user_similarity = cosine_similarity(user_movie_matrix)
   user_similarity_df = pd.DataFrame(user_similarity, index=user_movie_matrix.index, columns=user_movie_matrix.index)
   def recommend_for_user(user_id, top_n=10):
       # 找到最相似的用户
       similar_users = user_similarity_df[user_id].sort_values(ascending=False).index[1:11]  # 排除自己
       
       # 获取目标用户未评分的电影
       user_rated_movies = set(df[df['user_id'] == user_id]['movie_id'])
       all_movies = set(df['movie_id'])
       unrated_movies = all_movies - user_rated_movies
       
       # 预测评分：使用相似用户的评分加权平均
       movie_scores = {}
       for movie in unrated_movies:
           weighted_sum = 0
           sim_sum = 0
           for sim_user in similar_users:
               sim = user_similarity_df.loc[user_id, sim_user]
               rating = user_movie_matrix.loc[sim_user, movie]
               if rating > 0:
                   weighted_sum += sim * rating
                   sim_sum += sim
           if sim_sum > 0:
               movie_scores[movie] = weighted_sum / sim_sum
       
       # 返回评分最高的电影
       recommended_movies = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
       return recommended_movies, similar_users
   if __name__ == "__main__":
       user_id = int(input("请输入用户ID: "))
       
       if user_id not in df['user_id'].values:
           print("用户ID不存在")
       else:
           recommendations, similar_users = recommend_for_user(user_id)
           
           print(f"\n与用户 {user_id} 最相似的其他用户：")
           for i, sim_user in enumerate(similar_users, 1):
               print(f"{i}. 用户 {sim_user}")
           
           print(f"\n为用户 {user_id} 推荐的电影(电影ID - 预测评分）：")
           for i, (movie_id, score) in enumerate(recommendations, 1):
               movie_title = df[df['movie_id'] == movie_id]['title'].iloc[0]
               print(f"{i}. {movie_title} (ID: {movie_id}) - 预测评分: {score:.2f}")
   
   ```
   
   
   
   