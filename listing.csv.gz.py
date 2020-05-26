# -*- coding: utf-8 -*-
from matplotlib.font_manager import FontProperties
def getChineseFont():
    return FontProperties(fname='/Users/yangziyi/PycharmProjects/Airbnb/venv/lib/python3.7/site-packages/matplotlib/mpl-data/fonts/ttf/msyh.ttf')
import pandas as pd
import numpy as np

from sklearn.preprocessing import scale, StandardScaler, RobustScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold, StratifiedKFold, \
    RandomizedSearchCV
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV, LinearRegression, ElasticNet, HuberRegressor
from sklearn.metrics import mean_squared_error, r2_score
import xgboost
from xgboost import XGBRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.utils import resample
# Model interpretation modules
import eli5
from eli5 import show_prediction
#创建解释器
import lime
import lime.lime_tabular
import shap
shap.initjs()
import warnings
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] # 显示中文不乱码
plt.rcParams['axes.unicode_minus']= False # 显示负数不乱码
plt.style.use('seaborn')
import seaborn as sns
import pygal


warnings.filterwarnings("ignore")

import vincenty
from geopy.distance import vincenty
import options

from random import randint

df_initial = pd.read_csv('listings.csv')
print(df_initial)
# 显示省略数据
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
font = '/Users/yangziyi/PycharmProjects/Airbnb/venv/lib/python3.7/site-packages/matplotlib/mpl-data/fonts/ttf/msyh.ttf'
Combined_data = pd.read_csv('listings.csv')
# ###########word cloud
# from wordcloud import WordCloud, ImageColorGenerator
# plt.rcParams['font.sans-serif'] = ['Arial Unicode MS'] # 显示中文不乱码
# plt.rcParams['font.serif'] = ['Arial Unicode MS']
# text = " ".join(str(each) for each in Combined_data.name)
# # Create and generate a word cloud image:
# wordcloud = WordCloud(max_font_size=None, max_words=200, background_color="lightgrey",
#                       width=3000, height=2000,font_path=font).generate(text)
# print(wordcloud)
# plt.figure(figsize=(10,6))
# plt.figure(figsize=(15,10))
# # Display the generated image:
# plt.imshow(wordcloud, interpolation='Bilinear')
# plt.axis("off")
# plt.show()
# #let's comeback now to the 'name' column as it will require litte bit more coding and continue to analyze it!
# #initializing empty list where we are going to put our name strings
# _names_=[]
# #getting name strings from the column and appending it to the list
# for name in Combined_data.name:
#     _names_.append(name)
# #setting a function that will split those name strings into separate words
# def split_name(name):
#     spl=str(name).split()
#     return spl
# #initializing empty list where we are going to have words counted
# _names_for_count_=[]
# #getting name string from our list and using split function, later appending to list above
# for x in _names_:
#     for word in split_name(x):
#         word=word.lower()
#         _names_for_count_.append(word)
# #we are going to use counter
# from collections import Counter
# #let's see top 25 used words by host to name their listing
# _top_20_w=Counter(_names_for_count_).most_common()
# _top_20_w=_top_20_w[0:20]
#
# #now let's put our findings in dataframe for further visualizations
# sub_w=pd.DataFrame(_top_20_w)
# sub_w.rename(columns={0:'Words', 1:'Count'}, inplace=True)
# #we are going to use barplot for this visualization
# plt.rcParams['font.sans-serif'] = ['Arial Unicode MS'] # 显示中文不乱码
# plt.rcParams['font.serif'] = ['Arial Unicode MS']
# plt.figure(figsize=(10,6))
# viz_5=sns.barplot(x='Words', y='Count', data=sub_w)
# viz_5.set_title('Counts of the top 20 used words for listing names')
# viz_5.set_ylabel('Count of words')
# viz_5.set_xlabel('Words')
# viz_5.set_xticklabels(viz_5.get_xticklabels(), rotation=80)
# plt.show()
# #######



# 显示数据形式
print("数据所含 {} 行和 {} 列.".format(*df_initial.shape))

# 重复行查询
print("包含 {} 重复行.".format(df_initial.duplicated().sum()))
print("第一行数据为:")
print(df_initial.head(10))

# 检查当前拥有的列
print("显示当前所拥有的列：")
print(df_initial.columns)
# 定义所需要的列
columns_to_keep = ['id', 'space', 'description', 'host_has_profile_pic', 'neighbourhood', 'neighbourhood_cleansed',
                   'latitude', 'longitude', 'property_type', 'room_type', 'accommodates', 'bathrooms',
                   'bedrooms', 'bed_type', 'amenities', 'square_feet', 'price', 'cleaning_fee',
                   'security_deposit', 'extra_people', 'guests_included', 'minimum_nights',
                   'instant_bookable', 'is_business_travel_ready', 'cancellation_policy']
for i in range(41360):
    a = str(df_initial['neighbourhood_cleansed'][i])
    # print("a!!", a)
    a = a[0:3]
    df_initial['neighbourhood_cleansed'][i] = a
    # print("a", a)
    # print("数值：", (df_raw['neighbourhood_cleansed'][i + 1]))
df_raw = df_initial[columns_to_keep].set_index('id')

print("数据还有 {} 行和 {} 列(删去无关列)".format(*df_raw.shape))
print("各类房间类型：")
print(df_raw.room_type.value_counts(normalize=True))
# print(type(df_raw.room_type.value_counts(normalize=True)))
#各类房间类型画图
pie_chart1 = pygal.Pie()
pie_chart1.title = '各类房间类型：'
pie_chart1.add('Entire home/apt', df_raw.room_type.value_counts(normalize=True)[0])
pie_chart1.add('Private room', df_raw.room_type.value_counts(normalize=True)[1])
pie_chart1.add('Shared room', df_raw.room_type.value_counts(normalize=True)[2])
pie_chart1.render_to_file('各类房间类型bar_chart.svg')
print("各类房屋类型")
print(df_raw.property_type.value_counts(normalize=True))
print(len(df_raw.property_type.value_counts(normalize=True)))
#各类房屋类型画图
pie_chart2 = pygal.Pie()
pie_chart2.title = '各类房屋类型：'
for i in range(len(df_raw.property_type.value_counts(normalize=True))):
    pie_chart2.add(df_raw.property_type.value_counts(normalize=True).index[i], df_raw.property_type.value_counts(normalize=True).values[i])
    i = i + 1
pie_chart2.render_to_file('各类房屋类型bar_chart.svg')
#价格相关列
print(df_raw[['price', 'cleaning_fee', 'extra_people', 'security_deposit']].head(3))
# price列中无意义的数据
print("price列中缺失的数据")
print(df_raw.price.isna().sum())
# cleaning_fee列中无意义的数据
print("cleaning_fee列中缺失的数据")
print(df_raw.cleaning_fee.isna().sum())
print("security_deposit列中缺失的数据")
print(df_raw.security_deposit.isna().sum())
print("extra_people列中缺失的数据")
print(df_raw.extra_people.isna().sum())
# 使用0去代替其中的缺失数据
df_raw.cleaning_fee.fillna('$0.00', inplace=True)
print("处理后cleaning_fee列中缺失的数据")
print(df_raw.cleaning_fee.isna().sum())
df_raw.security_deposit.fillna('$0.00', inplace=True)
print("处理后security_deposit列中缺失的数据")
print(df_raw.security_deposit.isna().sum())

# 删除所有四列中的美元符号，并将字符串值转换为数字值
df_raw.price = df_raw.price.str.replace('$', '').str.replace(',', '').astype(float)
df_raw.cleaning_fee = df_raw.cleaning_fee.str.replace('$', '').str.replace(',', '').astype(float)
df_raw.security_deposit = df_raw.security_deposit.str.replace('$', '').str.replace(',', '').astype(float)
df_raw.extra_people = df_raw.extra_people.str.replace('$', '').str.replace(',', '').astype(float)
print("price的描述性统计：")
print(df_raw['price'].describe())
red_square = dict(markerfacecolor='r', markeredgecolor='r', marker='.')
df_raw['price'].plot(kind='box', xlim=(0, 20000), vert=False, flierprops=red_square, figsize=(16, 2))
# plt.show()
df_raw.drop(df_raw[(df_raw.price > 7500) | (df_raw.price == 0)].index, axis=0, inplace=True)
print("新的price的描述性统计：")
print(df_raw['price'].describe())
print("数据有 {} 行和 {} 列(处理价格信息后)".format(*df_raw.shape))
# 统计各列缺失值的个数
print("各列缺失值个数：")
print(df_raw.isna().sum())


# 由于缺失值过多删除对应列
df_raw.drop(columns=['square_feet', 'space', 'neighbourhood'], inplace=True)
# 存在少量缺失值删除对应行
df_raw.dropna(subset=['bathrooms', 'bedrooms', ], inplace=True)
#查看host_has_profile_pic列中所含的元素
print(df_raw.host_has_profile_pic.unique())
# 用"f"代替缺失值位置，表示否定
df_raw.host_has_profile_pic.fillna(value='f', inplace=True)
#重新查看host_has_profile_pic列中所含的元素
df_raw.host_has_profile_pic.unique()
#重新统计各列缺失值的个数
print(df_raw.isna().sum())
print("数据有 {} 行和 {} 列(缺失值处理后)".format(*df_raw.shape))

#特征工程1：定义各房源到北京市中心的距离函数
def distance_to_mid(lat, lon):
    beijing_centre = (39.9, 116.42)
    accommodation = (lat, lon)
    return vincenty(beijing_centre, accommodation).km

#计算每个房源到市中心的距离并记入新列
df_raw['distance'] = df_raw.apply(lambda x: distance_to_mid(x.latitude, x.longitude), axis=1)
print(df_raw.head(2))#查看新数据集的前两行
#特征工程2：房屋大小
#因为列square_feet含有很多缺失值 将此列删除，由于表示房屋大小的并不是square_feet而是square meters

#list(df_raw.description[:10])
print("description列中缺失的数据")
print(df_raw.description.isna().sum())
# 处理size列
df_raw['size'] = df_raw['description'].str.extract('(\d{2,3}\s?[smSM])', expand=True)
df_raw['size'] = df_raw['size'].str.replace("\D", "")

# 将数据改变成float类型
df_raw['size'] = df_raw['size'].astype(float)

print('NaNs in size_column absolute:     ', df_raw['size'].isna().sum())
print('NaNs in size_column in percentage:', round(df_raw['size'].isna().sum()/len(df_raw),3), '%')
print(df_raw[['description', 'size']].head(10))
#list(df_raw.description[:10])
# 删除description列
df_raw.drop(['description'], axis=1, inplace=True)
df_raw.info()

# 确定变量
sub_df = df_raw[['accommodates', 'bathrooms', 'bedrooms',  'price', 'cleaning_fee',
                 'security_deposit', 'extra_people', 'guests_included', 'distance', 'size']]
#去除size数据
train_data = sub_df[sub_df['size'].notnull()]
test_data  = sub_df[sub_df['size'].isnull()]

#定义X自变量
X_train = train_data.drop('size', axis=1)
X_test  = test_data.drop('size', axis=1)

#定义y因变量
y_train = train_data['size']
print("Shape of Training Data:", train_data.shape)
print("Shape of Test Data:    ",test_data.shape)
print("\nShape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("\nShape of y_train:", y_train.shape)


from sklearn.linear_model import LinearRegression


# instantiate
linreg = LinearRegression()

# 建立回归模型
linreg.fit(X_train, y_train)
print("模型R^2：",linreg.score(X_train, y_train))
#进行预测
y_test = linreg.predict(X_test)
y_test = pd.DataFrame(y_test)
y_test.columns = ['size']
print('y_test.shape',y_test.shape)
print('y_test',y_test.head())
print('X_test.shape',X_test.shape)
print('X_test',X_test.head())
# 建立index
beijing_index = pd.DataFrame(X_test.index)
beijing_index.columns = ['beijing']

#合并数据集
y_test = pd.concat([y_test, beijing_index], axis=1)
y_test.set_index(['beijing'], inplace=True)
print(y_test.head())
new_test_data = pd.concat([X_test, y_test], axis=1)
print(new_test_data.shape)
print(new_test_data.head())
new_test_data['size'].isna().sum()
# 结合训练集和测试集数据
sub_df_new = pd.concat([new_test_data, train_data], axis=0)

print(sub_df_new.shape)
print(sub_df_new.head())
print(sub_df_new['size'].isna().sum())
# 整理列变量
df_raw.drop(['accommodates', 'bathrooms', 'bedrooms', 'price', 'cleaning_fee',
             'security_deposit', 'extra_people', 'guests_included', 'distance', 'size'],
            axis=1, inplace=True)
# 合成最终的数据表格
df = pd.concat([sub_df_new, df_raw], axis=1)

print(df.shape)
print(df.head(2))
#调差生成列
print(df['size'].isna().sum())
print(df['size'].describe())
red_square = dict(markerfacecolor='r', markeredgecolor='r', marker='.')
df['size'].plot(kind='box', xlim=(0, 1700), vert=False, flierprops=red_square, figsize=(16,2))
# plt.show()
#特征工程：便利设施
from collections import Counter
results = Counter()
df['amenities'].str.strip('{}')\
               .str.replace('"', '')\
               .str.lstrip('\"')\
               .str.rstrip('\"')\
               .str.split(',')\
               .apply(results.update)

print(results.most_common(30))
#建立设备数量信息表格
sub_df = pd.DataFrame(results.most_common(30), columns=['amenity', 'count'])
# 绘出前20的便利设施排名
sub_df.sort_values(by=['count'], ascending=True).plot(kind='barh', x='amenity', y='count',
                                                      figsize=(10,7), legend=False, color='darkgrey',
                                                      title='Amenities')
plt.xlabel('Count')
# plt.show()

df.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4, figsize=(10,7),
        c="price", cmap="gist_heat_r", colorbar=True, sharex=False)
# plt.show()
df['neighbourhood_cleansed'].value_counts().sort_values().plot(kind='barh', color='darkgrey')
plt.title('Number of Accommodations per District',fontproperties=getChineseFont())
plt.rcParams['font.sans-serif'] = ['SimHei'] # 显示中文不乱码
plt.rcParams['font.serif'] = ['SimHei']
# plt.show()

# 按照地区进行分类
df_grouped = pd.DataFrame(df.groupby(['neighbourhood_cleansed'])['price'].agg(np.median))
df_grouped.reset_index(inplace=True)

# 绘图
df_grouped.sort_values(by=['price'], ascending=True)\
          .plot(kind='barh', x='neighbourhood_cleansed', y='price',
                figsize=(10,6), legend=False, color='salmon')

plt.xlabel('\nMedian Price', fontsize=12)
plt.ylabel('District\n', fontsize=12)
plt.title('\nMedian Prices by Neighbourhood\n', fontsize=14, fontweight='bold')


red_square = dict(markerfacecolor='salmon', markeredgecolor='salmon', marker='.')

df.boxplot(column='price', by='neighbourhood_cleansed',
           flierprops=red_square, vert=False, figsize=(10,8))

plt.xlabel('\nMedian Price', fontsize=12)
plt.ylabel('District\n', fontsize=12)
plt.title('\nBoxplot: Prices by Neighbourhood\n', fontsize=14, fontweight='bold')

# 自动生成boxplot标题
plt.suptitle('')


df.plot.scatter(x="distance", y="price", figsize=(9,6), c='dimgrey')
plt.title('\nRelation between Distance & Median Price\n', fontsize=14, fontweight='bold')

sns.jointplot(x=df["distance"], y=df["price"], kind='hex')
plt.title('\nRelation between Distance & Median Price\n', fontsize=14, fontweight='bold')
###
sns.set_style("white")
cmap = sns.cubehelix_palette(rot=-.2, as_cmap=True)

fig, ax = plt.subplots(figsize=(12,7))
ax = sns.scatterplot(x="size", y="price", size='cleaning_fee', sizes=(5, 200),
                      hue='size', palette=cmap,  data=df)

plt.title('\nRelation between Size & Median Price\n', fontsize=14, fontweight='bold')

# 制定绘图范围
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.);
###
plt.figure(figsize=(6,6))
sns.heatmap(df.groupby(['neighbourhood_cleansed', 'bedrooms']).price.median().unstack(),
            cmap='Reds', annot=True, fmt=".0f")

plt.xlabel('\nBedrooms', fontsize=12)
plt.ylabel('District\n', fontsize=12)
plt.title('\nHeatmap: Median Prices by Neighbourhood and Number of Bedrooms\n\n', fontsize=14, fontweight='bold',fontproperties=getChineseFont())
plt.rcParams['font.sans-serif'] = ['SimHei'] # 显示中文不乱码
plt.rcParams['font.serif'] = ['SimHei']
###
# 按照取消政策的严格性进行讨论
df_grouped = pd.DataFrame(df.groupby(['cancellation_policy'])['price'].agg(np.median))
df_grouped.reset_index(inplace=True)

# 绘图
df_grouped.sort_values(by=['price'], ascending=True)\
          .plot(kind='barh', x='cancellation_policy', y='price',
                figsize=(9,5), legend=False, color='darkblue')

plt.xlabel('\nMedian Price', fontsize=12)
plt.ylabel('Cancellation Policy\n', fontsize=12)
plt.title('\nMedian Prices by Cancellation Policy\n', fontsize=14, fontweight='bold',fontproperties=getChineseFont())
plt.rcParams['font.sans-serif'] = ['SimHei'] # 显示中文不乱码
plt.rcParams['font.serif'] = ['SimHei']
plt.show()

#########################################
print(df.columns)
df.info()
df.drop(['latitude', 'longitude', 'neighbourhood_cleansed', 'property_type'], axis=1, inplace=True)
for col in ['host_has_profile_pic', 'room_type', 'bed_type', 'instant_bookable',
            'is_business_travel_ready', 'cancellation_policy']:
    df[col] = df[col].astype('category')
#定义目标函数
target = df[["price"]]

# 定义特征值
features = df.drop(["price"], axis=1)
num_feats = features.select_dtypes(include=['float64', 'int64', 'bool']).copy()

#0-1变量的转码
cat_feats = features.select_dtypes(include=['category']).copy()
categorical_features_one_hot = pd.get_dummies(cat_feats)
cat_feats = pd.get_dummies(cat_feats)
features_recoded = pd.concat([num_feats, cat_feats], axis=1)
print(features_recoded.head(2))

# import train_test_split function
from sklearn.model_selection import train_test_split
# import metrics
from sklearn.metrics import mean_squared_error, r2_score
numerical_features = df.select_dtypes(exclude=['object'])
y = numerical_features.price
numerical_features = numerical_features.drop(['price'], axis=1)
X_df = pd.concat([numerical_features, categorical_features_one_hot], axis=1)
# split our data
X_train, X_test, y_train, y_test = train_test_split(features_recoded, target, test_size=0.2)
# scale data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test  = sc.transform(X_test)
# create a baseline
booster = xgboost.XGBRegressor()

from sklearn.model_selection import GridSearchCV

# create Grid
param_grid = {'n_estimators': [100, 150, 200],
              'learning_rate': [0.01, 0.05, 0.1],
              'max_depth': [3, 4, 5, 6, 7],
              'colsample_bytree': [0.6, 0.7, 1],
              'gamma': [0.0, 0.1, 0.2]}

# instantiate the tuned random forest
booster_grid_search = GridSearchCV(booster, param_grid, cv=3, n_jobs=-1)

# train the tuned random forest
booster_grid_search.fit(X_train, y_train)

# print best estimator parameters found during the grid search
print(booster_grid_search.best_params_)

best_random = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=30,
                      max_features='sqrt', max_leaf_nodes=None,
                      min_impurity_decrease=0.0, min_impurity_split=None,
                      min_samples_leaf=1, min_samples_split=5,
                      min_weight_fraction_leaf=0.0, n_estimators=1400,
                      n_jobs=None, oob_score=False, random_state=42, verbose=0,
                      warm_start=False)
from eli5.sklearn import PermutationImportance
perm = PermutationImportance(best_random.fit(X_train, y_train), random_state=1).fit(X_test, y_test)
eli5.show_weights(perm, feature_names=list(X_df.columns))