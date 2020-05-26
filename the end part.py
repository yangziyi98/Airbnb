# -*- coding: utf-8 -*-
import mathplot as mathplot
import matplotlib
import numpy as np
from matplotlib.font_manager import FontProperties
def getChineseFont():
    return FontProperties(fname='/opt/anaconda3/lib/python3.7/site-packages/matplotlib/mpl-data/fonts/ttf/simhei.ttf')

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['Arial Unicode MS']
plt.rcParams['axes.unicode_minus']=False
sns.set(font='Arial Unicode MS')  # 解决Seaborn中文显示问题
import sklearn
plt.style.use('ggplot')
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly as py
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import cufflinks as cf
from plotly.offline import iplot, init_notebook_mode
init_notebook_mode(connected=True)
cf.go_offline(connected=True)
import collections
import itertools
import scipy.stats as stats
from scipy.stats import norm
from scipy.special import boxcox1p
import statsmodels
import statsmodels.api as sm

print(statsmodels.__version__)

from sklearn.preprocessing import scale, StandardScaler, RobustScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold, StratifiedKFold, \
    RandomizedSearchCV
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV, LinearRegression, ElasticNet, HuberRegressor
from sklearn.metrics import mean_squared_error, r2_score
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


warnings.filterwarnings("ignore", category=FutureWarning)
Combined_data = pd.read_csv('北京列表的摘要信息和指标-适合可视化.csv')
print(Combined_data.head())
print('Number of features: {}'.format(Combined_data.shape[1]))
print('Number of examples: {}'.format(Combined_data.shape[0]))
# for c in df.columns:
#    print(c, dtype(df_train[c]))
print(Combined_data.dtypes)
Combined_data['last_review'] = pd.to_datetime(Combined_data['last_review'], infer_datetime_format=True)
total = Combined_data.isnull().sum().sort_values(ascending=False)
percent = (Combined_data.isnull().sum()) / Combined_data.isnull().count().sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'], sort=False).sort_values('Total',

                                                                                   ascending=False)
print(missing_data.head(40))


###########word cloud
font = '/Users/yangziyi/PycharmProjects/Airbnb/venv/lib/python3.7/site-packages/matplotlib/mpl-data/fonts/ttf/msyh.ttf'
from wordcloud import WordCloud, ImageColorGenerator
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS'] # 显示中文不乱码
plt.rcParams['font.serif'] = ['Arial Unicode MS']
text = " ".join(str(each) for each in Combined_data.name)
# Create and generate a word cloud image:
wordcloud = WordCloud(max_font_size=None, max_words=200, background_color="lightgrey",
                      width=3000, height=2000,font_path=font).generate(text)
plt.figure(figsize=(10,6))
plt.figure(figsize=(15,10))
# Display the generated image:
plt.imshow(wordcloud, interpolation='Bilinear')
plt.axis("off")
plt.show()
#let's comeback now to the 'name' column as it will require litte bit more coding and continue to analyze it!
#initializing empty list where we are going to put our name strings
_names_=[]
#getting name strings from the column and appending it to the list
for name in Combined_data.name:
    _names_.append(name)
#setting a function that will split those name strings into separate words
def split_name(name):
    spl=str(name).split()
    return spl
#initializing empty list where we are going to have words counted
_names_for_count_=[]
#getting name string from our list and using split function, later appending to list above
for x in _names_:
    for word in split_name(x):
        word=word.lower()
        _names_for_count_.append(word)
#we are going to use counter
from collections import Counter
#let's see top 25 used words by host to name their listing
_top_20_w=Counter(_names_for_count_).most_common()
_top_20_w=_top_20_w[0:20]

#now let's put our findings in dataframe for further visualizations
sub_w=pd.DataFrame(_top_20_w)
sub_w.rename(columns={0:'Words', 1:'Count'}, inplace=True)
#we are going to use barplot for this visualization
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS'] # 显示中文不乱码
plt.rcParams['font.serif'] = ['Arial Unicode MS']
plt.figure(figsize=(10,6))
viz_5=sns.barplot(x='Words', y='Count', data=sub_w)
viz_5.set_title('Counts of the top 20 used words for listing names')
viz_5.set_ylabel('Count of words')
viz_5.set_xlabel('Words')
viz_5.set_xticklabels(viz_5.get_xticklabels(), rotation=80)
plt.show()
#######

Combined_data.drop(['host_name', 'name', 'neighbourhood_group'], axis=1, inplace=True)
print(Combined_data[Combined_data['number_of_reviews'] == 0.0].shape)
Combined_data['reviews_per_month'] = Combined_data['reviews_per_month'].fillna(0)
earliest = min(Combined_data['last_review'])
Combined_data['last_review'] = Combined_data['last_review'].fillna(earliest)
Combined_data['last_review'] = Combined_data['last_review'].apply(lambda x: x.toordinal() - earliest.toordinal())
total = Combined_data.isnull().sum().sort_values(ascending=False)
percent = (Combined_data.isnull().sum()) / Combined_data.isnull().count().sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'], sort=False).sort_values('Total',
                                                                                                      ascending=False)
print(missing_data.head(40))
#######数据正态化
fig, axes = plt.subplots(1, 3, figsize=(21, 6))
sns.distplot(Combined_data['price'], ax=axes[0])
sns.distplot(np.log1p(Combined_data['price']), ax=axes[1])
axes[1].set_xlabel('log(1+price)')
sm.qqplot(np.log1p(Combined_data['price']), stats.norm, fit=True, line='45', ax=axes[2])
plt.show()
Combined_data = Combined_data[np.log1p(Combined_data['price']) < 10]
Combined_data = Combined_data[np.log1p(Combined_data['price']) > 4]
print(Combined_data.head(40))
fig, axes = plt.subplots(1, 3, figsize=(21, 6))
sns.distplot(Combined_data['price'], ax=axes[0])
sns.distplot(np.log1p(Combined_data['price']), ax=axes[1])
axes[1].set_xlabel('log(1+price)')
sm.qqplot(np.log1p(Combined_data['price']), stats.norm, fit=True, line='45', ax=axes[2])
plt.show()
Combined_data['price'] = np.log1p(Combined_data['price'])
#######

print(Combined_data.columns)
Combined_data = Combined_data.drop(['host_id', 'id'], axis=1)
# sns.catplot(x='neighbourhood', kind='count', data=Combined_data)
sns.countplot(x='neighbourhood',data=Combined_data)
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS'] # 显示中文不乱码
plt.rcParams['font.serif'] = ['Arial Unicode MS']
plt.show()

plt.figure(figsize=(10,6))
sns.scatterplot(Combined_data.longitude,Combined_data.latitude,hue=Combined_data.neighbourhood)
plt.ioff()

fig, axes = plt.subplots(1, 3, figsize=(21, 6))
sns.distplot(Combined_data['latitude'], ax=axes[0])
sns.distplot(Combined_data['longitude'], ax=axes[1])
sns.scatterplot(x=Combined_data['latitude'], y=Combined_data['longitude'])
plt.show()

sns.catplot(x='room_type', kind='count', data=Combined_data)
fig = plt.gcf()
fig.set_size_inches(8, 6)
plt.show()

import plotly.offline as pyo
import plotly.graph_objs as go
roomdf = Combined_data.groupby('room_type').size()/Combined_data['room_type'].count()*100
labels = roomdf.index
values = roomdf.values
# Use `hole` to create a donut-like pie chart
fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.6)])
plt.show()

plt.figure(figsize=(10,6))
sns.countplot(x = 'room_type',hue = "neighbourhood",data = Combined_data)
plt.title("Room types occupied by the neighbourhood")
plt.show()

#catplot room type and price
plt.figure(figsize=(10,6))
sns.catplot(x="room_type", y="price", data=Combined_data)
plt.ioff()

fig, axes = plt.subplots(1, 2, figsize=(21, 6))

sns.distplot(Combined_data['minimum_nights'], rug=False, kde=False, color="green", ax=axes[0])
axes[0].set_yscale('log')
axes[0].set_xlabel('minimum stay [nights]')
axes[0].set_ylabel('count')

sns.distplot(np.log1p(Combined_data['minimum_nights']), rug=False, kde=False, color="green", ax=axes[1])
axes[1].set_yscale('log')
axes[1].set_xlabel('minimum stay [nights]')
axes[1].set_ylabel('count')
plt.show()

Combined_data['minimum_nights'] = np.log1p(Combined_data['minimum_nights'])
fig, axes = plt.subplots(1, 2, figsize=(18.5, 6))
sns.distplot(Combined_data[Combined_data['reviews_per_month'] < 17.5]['reviews_per_month'], rug=True, kde=False,
             color="green", ax=axes[0])
sns.distplot(np.sqrt(Combined_data[Combined_data['reviews_per_month'] < 17.5]['reviews_per_month']), rug=True,
             kde=False, color="green", ax=axes[1])
axes[1].set_xlabel('ln(reviews_per_month)')
plt.show()

fig, axes = plt.subplots(1, 1, figsize=(21, 6))
sns.scatterplot(x=Combined_data['availability_365'], y=Combined_data['reviews_per_month'])
plt.show()

Combined_data['reviews_per_month'] = Combined_data[Combined_data['reviews_per_month'] < 12.5]['reviews_per_month']

fig, axes = plt.subplots(1, 1, figsize=(18.5, 6))
sns.distplot(Combined_data['availability_365'], rug=False, kde=False, color="blue", ax=axes)
axes.set_xlabel('availability_365')
axes.set_xlim(0, 365)
plt.show()

Combined_data['all_year_avail'] = Combined_data['availability_365'] > 350
Combined_data['low_avail'] = Combined_data['availability_365'] < 15
Combined_data['no_reviews'] = Combined_data['reviews_per_month'] == 0

corrmatrix = Combined_data.corr()
f, ax = plt.subplots(figsize=(15, 12))
sns.heatmap(corrmatrix, vmax=0.8, square=True)
sns.set(font_scale=0.8)
plt.show()

# sns.pairplot(Combined_data.select_dtypes(exclude=['object']).values)
# plt.show()
# #
categorical_features = Combined_data.select_dtypes(include=['object'])
print('Categorical features: {}'.format(categorical_features.shape))
categorical_features_one_hot = pd.get_dummies(categorical_features)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
print(categorical_features_one_hot.head())
Combined_data['reviews_per_month'] = Combined_data['reviews_per_month'].fillna(0)

numerical_features = Combined_data.select_dtypes(exclude=['object'])
y = numerical_features.price
numerical_features = numerical_features.drop(['price'], axis=1)
print('Numerical features: {}'.format(numerical_features.shape))

X = np.concatenate((numerical_features, categorical_features_one_hot), axis=1)
X_df = pd.concat([numerical_features, categorical_features_one_hot], axis=1)
print('Dimensions of the design matrix: {}'.format(X.shape))
print('Dimension of the target vector: {}'.format(y.shape))

Processed_data = pd.concat([X_df, y], axis=1)
Processed_data.to_csv('Beijing_Airbnb_Processed.dat')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print('Dimensions of the training feature matrix: {}'.format(X_train.shape))
print('Dimensions of the training target vector: {}'.format(y_train.shape))
print('Dimensions of the test feature matrix: {}'.format(X_test.shape))
print('Dimensions of the test target vector: {}'.format(y_test.shape))

scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

n_folds = 5


# squared_loss
def rmse_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=91).get_n_splits(numerical_features)
    return cross_val_score(model, X_train, y_train, scoring='neg_mean_squared_error', cv=kf)


def rmse_lv_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=91).get_n_splits(numerical_features)
    return cross_val_score(model, Xlv_train, y_train, scoring='neg_mean_squared_error', cv=kf)


for Model in [LinearRegression, Ridge, Lasso, ElasticNet, RandomForestRegressor, XGBRegressor, HuberRegressor]:
    if Model == XGBRegressor:
        cv_res = rmse_cv(XGBRegressor(objective='reg:squarederror',max_iter=10000))
    else:
        cv_res = rmse_cv(Model())
    print('{}: {:.5f} +/- {:5f}'.format(Model.__name__, -cv_res.mean(), cv_res.std()))


alphas1 = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]
cv_ridge1 = [-rmse_cv(Ridge(alpha = alpha)).mean() for alpha in alphas1]
alphas2 = [0.5*i for i in range(4,12)]
cv_ridge2 = [-rmse_cv(Ridge(alpha = alpha)).mean() for alpha in alphas2]
cv_ridge1 = pd.Series(cv_ridge1, index = alphas1)
cv_ridge2 = pd.Series(cv_ridge2, index = alphas2)

fig, axes = plt.subplots(1,2,figsize=(21, 8))
cv_ridge1.plot(title = "Ridge Regression Cross-Validation", style='-o', ax = axes[0])
axes[0].set_xlabel("alpha")
axes[0].set_ylabel("rmse")
axes[0].set_xscale('log')

cv_ridge2.plot(title = "Ridge Regression Cross-Validation", style='-o', ax = axes[1])
axes[1].set_xlabel("alpha")
axes[1].set_ylabel("rmse")
axes[1].set_xscale('log')
plt.show()
print('alpha=',np.argmin(cv_ridge1))
RR_best = Ridge(alpha = np.argmin(cv_ridge1))
RR_best.fit(X_train, y_train)
predicted_prices = RR_best.predict(X_test)
print(predicted_prices)
fig= plt.subplots(1,2,figsize=(21, 8))
fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=alphas1,
        y=cv_ridge1,
        line=dict(color='royalBlue', width=2)
    ),
)

fig.update_layout(

    xaxis=go.layout.XAxis(
        title=go.layout.xaxis.Title(
            text="Penalty",
            font=dict(
                size=16
            )
        )
    ),

    yaxis=go.layout.YAxis(
        title=go.layout.yaxis.Title(
            text="Cross-validation error",
            font=dict(
                size=16
            )
        )
    ),
)
fig.update_layout(height=400,
                width = 600,
                title = 'Telescopic Search: Coarse level',
                  xaxis_type="log",
                  showlegend=False)

plt.show()

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=alphas2,
        y=cv_ridge2,
        line=dict(color='crimson', width=2)
    ),
)

fig.update_layout(

    xaxis=go.layout.XAxis(
        title=go.layout.xaxis.Title(
            text="Penalty",
            font=dict(
                size=16
            )
        )
    ),

    yaxis=go.layout.YAxis(
        title=go.layout.yaxis.Title(
            text="Cross-validation error",
            font=dict(
                size=16
            ),
        )
    ),
)
fig.update_layout(height=400,
                width = 600,
                title = 'Telescopic Search: Fine level',
                  xaxis_type="log",
                  showlegend=False)

plt.show()

def L_theta_new(intercept, coef, X, y, lamb):
    h = np.dot(X, coef) + intercept  # np.dot 表示矩阵乘法
    L_theta = 0.5 * mean_squared_error(h, y) + 0.5 * lamb * np.sum(np.square(coef))
    return L_theta
best_alpha = alphas2[np.argmin(cv_ridge2.values)]
print('alpha=',np.argmin(cv_ridge2.values))
RR_CV_best = -rmse_cv(Ridge(alpha = best_alpha))
RR = Ridge(alpha = best_alpha)
RR.fit(X_train, y_train)
print(RR.intercept_, RR.coef_)
print(L_theta_new(intercept=RR.intercept_, coef=RR.coef_.T, X=X_train, y=y_train, lamb=best_alpha))
y_train_RR = RR.predict(X_train)
y_test_RR = RR.predict(X_test)
ridge_results = pd.DataFrame({'algorithm':['Ridge Regression'],
            'CV error': RR_CV_best.mean(),
            'CV std': RR_CV_best.std(),
            'training error': [mean_squared_error(y_train, y_train_RR)],
            'test error': [mean_squared_error(y_test_RR, y_test_RR)],
            'training_r2_score': [r2_score(y_train, y_train_RR)],
            'test_r2_score': [r2_score(y_test, y_test_RR)]})
print(ridge_results)

shap.initjs()
explainer = lime.lime_tabular.LimeTabularExplainer(X_train, feature_names=X_df.columns, class_names=['price'], verbose=True, mode='regression')
i=25
exp = explainer.explain_instance(X_test[i], RR.predict, num_features=5)
exp.show_in_notebook(show_table=True, show_all=False)
exp.save_to_file('expi=25.html')
item = pd.DataFrame(scaler.inverse_transform(X_test[i].reshape(1,-1))[0], index=X_df.columns)
item.loc['minimum_nights'] = np.expm1(item.loc['minimum_nights'])
item.loc['true_price'] = np.exp(y_test.iloc[i])
print(exp.intercept)
print(exp.local_pred)
item.loc['predicted_price'] = np.exp(exp.local_pred)
print(item[(item.select_dtypes(include=['number']) != 0).any(1)])

i=0
exp = explainer.explain_instance(X_test[i], RR.predict, num_features=5)
item = pd.DataFrame(scaler.inverse_transform(X_test[i].reshape(1,-1))[0], index=X_df.columns)
item.loc['minimum_nights'] = np.expm1(item.loc['minimum_nights'])
item.loc['true_price'] = np.exp(y_test.iloc[i])
print(exp.intercept)
print(exp.local_pred)
item.loc['predicted_price'] = np.exp(exp.local_pred)
print(item[(item.select_dtypes(include=['number']) != 0).any(1)])
exp.show_in_notebook(show_table=True, show_all=False)
exp.save_to_file('expi=0.html')

i=78
exp = explainer.explain_instance(X_test[i], RR.predict, num_features=5)
item = pd.DataFrame(scaler.inverse_transform(X_test[i].reshape(1,-1))[0], index=X_df.columns)
item.loc['minimum_nights'] = np.expm1(item.loc['minimum_nights'])
item.loc['true_price'] = np.exp(y_test.iloc[i])
print(exp.intercept)
print(exp.local_pred)
item.loc['ridge_prediction_price'] = np.exp(exp.local_pred)
print(item[(item.select_dtypes(include=['number']) != 0).any(1)])
exp.show_in_notebook(show_table=True, show_all=False)
exp.save_to_file('expi=78.html')

i=395
exp = explainer.explain_instance(X_test[i], RR.predict, num_features=5)
item = pd.DataFrame(scaler.inverse_transform(X_test[i].reshape(1,-1))[0], index=X_df.columns)
item.loc['minimum_nights'] = np.expm1(item.loc['minimum_nights'])
item.loc['true_price'] = np.exp(y_test.iloc[i])
print(exp.intercept)
print(exp.local_pred)
item.loc['ridge_prediction_price'] = np.exp(exp.local_pred)
print(item[(item.select_dtypes(include=['number']) != 0).any(1)])
exp.show_in_notebook(show_table=True, show_all=False)
exp.save_to_file('expi=395.html')

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS'] # 显示中文不乱码
plt.rcParams['font.serif'] = ['Arial Unicode MS']
explainer_sh = shap.LinearExplainer(RR, X_train, feature_dependence='independent')
shap_values = explainer_sh.shap_values(X_test)
shap.summary_plot(shap_values, X_test, feature_names=X_df.columns)

######Lasso
alphas = [0.0001, 0.001, 0.005,0.01, 0.05, 0.1, 0.3, 1]
cv_lasso = [-rmse_cv(Lasso(alpha = alpha, max_iter=2000)).mean() for alpha in alphas]
cv_lasso = pd.Series(cv_lasso, index = alphas)
cv_lasso.plot(title = "LASSO Regression Cross-Validation", style='-+')
plt.xlabel("alpha")
plt.ylabel("rmse")
plt.xscale('log')
plt.show()
fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x = alphas,
        y= cv_lasso,
        line = dict(color='crimson', width=2)
        ),
)
fig.update_layout(

    xaxis=go.layout.XAxis(
        title=go.layout.xaxis.Title(
            text="Penalty",
            font=dict(
                size=16
            )
        )
    ),

    yaxis=go.layout.YAxis(
        title=go.layout.yaxis.Title(
            text="Cross-validation error",
            font=dict(
                size=16
            ),
        )
    ),
)
fig.update_layout(height=400,
                width = 600,
                title = 'Lasso penalty optimization',
                  xaxis_type="log",
                  showlegend=False)

plt.show()
best_alpha = alphas[np.argmin(cv_lasso.values)]
lasso_CV_best = -rmse_cv(Lasso(alpha = best_alpha))
print('best_alpha=',best_alpha)
lasso = Lasso(alpha = best_alpha)
lasso.fit(X_train, y_train)
y_train_lasso = lasso.predict(X_train)
y_test_lasso = lasso.predict(X_test)
lasso_results = pd.DataFrame({'algorithm':['LASSO Regression'],
            'CV error': lasso_CV_best.mean(),
            'CV std': lasso_CV_best.std(),
            'training error': [mean_squared_error(y_train_lasso, y_train)],
            'test error': [mean_squared_error(y_test_lasso, y_test)],
            'training_r2_score': [r2_score(y_train, y_train_lasso)],
            'test_r2_score': [r2_score(y_test, y_test_lasso)]})
print(lasso_results)
features = list(categorical_features_one_hot.columns) + list(numerical_features.columns)
coef = pd.Series(lasso.coef_, index = features)
print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
print(coef)
imp_coef = pd.concat([coef.sort_values().iloc[:10],
                     coef.sort_values().iloc[-10:]])
matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS'] # 显示中文不乱码
plt.rcParams['font.serif'] = ['Arial Unicode MS']
plt.title("Coefficients in the Lasso Model")
plt.show()
i=25
exp = explainer.explain_instance(X_test[i], lasso.predict, num_features=5)
exp.show_in_notebook(show_table=True)
exp.save_to_file('exp2_i=25.html')
explainer = shap.LinearExplainer(lasso, X_train, feature_dependence='independent')
shap_values = explainer.shap_values(X_test)
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS'] # 显示中文不乱码
plt.rcParams['font.serif'] = ['Arial Unicode MS']
shap.summary_plot(shap_values, X_test, feature_names=X_df.columns)

######Huber regression
alphas = [0.0001, 0.001, 0.005,0.01, 0.05, 0.1, 0.3, 1]
#cv_huber = [-rmse_cv(HuberRegressor(alpha = alpha, max_iter=2000)).mean() for alpha in alphas]
cv_huber = [0.20051906841425277, 0.20044833042114646, 0.20048899799050565, 0.200533996471012, 0.20051788009059482, 0.2005294886778608, 0.20052011204607623, 0.2004070661477452]
cv_huber = pd.Series(cv_huber, index = alphas)
cv_huber.plot(title = "Huber Regression Cross-Validation", style='-o')
plt.xlabel("alpha")
plt.ylabel("rmse")
plt.xscale('log')
plt.show()

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x = alphas,
        y= cv_huber,
        line = dict(color='crimson', width=2)
        ),
)
fig.update_layout(

    xaxis=go.layout.XAxis(
        title=go.layout.xaxis.Title(
            text="Penalty",
            font=dict(
                size=16
            )
        )
    ),

    yaxis=go.layout.YAxis(
        title=go.layout.yaxis.Title(
            text="Cross-validation error",
            font=dict(
                size=16
            ),
        )
    ),
)

fig.update_layout(height=400,
                  width=600,
                  title='Lasso penalty optimization',
                  xaxis_type="log",
                  showlegend=False)

plt.show()
best_alpha = alphas[np.argmin(cv_huber.values)]
print(best_alpha)
huber_CV_best = -rmse_cv(HuberRegressor(alpha=best_alpha))
huber = HuberRegressor(alpha=best_alpha)
huber.fit(X_train, y_train)
y_train_huber = huber.predict(X_train)
y_test_huber = huber.predict(X_test)
huber_results = pd.DataFrame({'algorithm':['Huber Regression'],
            'CV error': huber_CV_best.mean(),
            'CV std': huber_CV_best.std(),
            'training error': [mean_squared_error(y_train, y_train_huber)],
            'test error': [mean_squared_error(y_test, y_test_huber)],
            'training_r2_score': [r2_score(y_train, y_train_huber)],
            'test_r2_score': [r2_score(y_test, y_test_huber)]})
print(huber_results)

lasso_coef = coef[coef!=0]
Xlv = X_df[list(lasso_coef.index)]
#X_lasso_vars.shape
Xlv_train, Xlv_test, y_train, y_test = train_test_split(Xlv, y, test_size=0.2, random_state=42)
print('Dimensions of the training feature matrix for lasso variable selection: {}'.format(Xlv_train.shape))
print('Dimensions of the test feature matrix for lasso variable selection: {}'.format(Xlv_test.shape))

for Model in [LinearRegression, Ridge, Lasso, ElasticNet, RandomForestRegressor, XGBRegressor, HuberRegressor]:
    if Model == XGBRegressor: cv_res = rmse_cv(XGBRegressor(objective='reg:squarederror'))
    else: cv_res = rmse_lv_cv(Model())
    print('{}: {:.5f} +/- {:5f}'.format(Model.__name__, -cv_res.mean(), cv_res.std()))

alphas1 = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]
cv_ridge1 = [-rmse_lv_cv(Ridge(alpha = alpha)).mean() for alpha in alphas1]
alphas2 = [1.4+0.05*i for i in range(1,12)]
cv_ridge2 = [-rmse_lv_cv(Ridge(alpha = alpha)).mean() for alpha in alphas2]
cv_ridge1 = pd.Series(cv_ridge1, index = alphas1)
cv_ridge2 = pd.Series(cv_ridge2, index = alphas2)

fig, axes = plt.subplots(1,2,figsize=(21, 8))
cv_ridge1.plot(title = "Ridge Regression Cross-Validation", style='-o', ax = axes[0])
axes[0].set_xlabel("alpha")
axes[0].set_ylabel("rmse")
axes[0].set_xscale('log')

cv_ridge2.plot(title = "Ridge Regression Cross-Validation", style='-o', ax = axes[1])
axes[1].set_xlabel("alpha")
axes[1].set_ylabel("rmse")
#axes[1].set_xscale('log')

#RR_best = Ridge(alpha = np.argmin(cv_ridge)) RR_best.fit(X_train, y_train) predicted_prices = RR_best.predict(test_data)

best_alpha = alphas2[np.argmin(cv_ridge2.values)]
print(best_alpha)
RR_lassoVars_CV_best = -rmse_lv_cv(Ridge(alpha = best_alpha))
RR_lassoVars = Ridge(alpha = best_alpha)
RR_lassoVars.fit(Xlv_train, y_train)
y_train_RR_lassoVars = RR_lassoVars.predict(Xlv_train)
y_test_RR_lassoVars = RR_lassoVars.predict(Xlv_test)
ridge_lassoVars_results = pd.DataFrame({'algorithm':['Ridge Regression with LASSO variable selection'],
            'CV error': RR_lassoVars_CV_best.mean(),
            'CV std': RR_lassoVars_CV_best.std(),
            'training error': [mean_squared_error(y_train, y_train_RR_lassoVars)],
            'test error': [mean_squared_error(y_test, y_test_RR_lassoVars)],
            'training_r2_score': [r2_score(y_train, y_train_RR_lassoVars)],
            'test_r2_score': [r2_score(y_test, y_test_RR_lassoVars)]})
print(ridge_lassoVars_results)


rfr_CV_baseline = -rmse_cv(RandomForestRegressor(random_state=42))
rfr_baseline = RandomForestRegressor(random_state=42)
rfr_baseline.fit(X_train, y_train)
y_train_rfr = rfr_baseline.predict(X_train)
y_test_rfr = rfr_baseline.predict(X_test)
rfr_baseline_results = pd.DataFrame({'algorithm':['Random Forest Regressor [baseline]'],
            'CV error': rfr_CV_baseline.mean(),
            'CV std': rfr_CV_baseline.std(),
            'training error': [mean_squared_error(y_train_rfr, y_train)],
            'test error': [mean_squared_error(y_test_rfr, y_test)]})
print(rfr_baseline_results)
print(rfr_baseline.estimators_)
eli5.show_weights(rfr_baseline, feature_names=list(X_df.columns))

rf = RandomForestRegressor(random_state=42)
from pprint import pprint
print('Parameters currently in use: \n')
pprint(rf.get_params())
#Number of trees in the forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop=2000,num=10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
max_depth.append(None)
min_samples_split = [2,5,10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]

random_grid = {'n_estimators': n_estimators,
        'max_features': max_features,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'bootstrap': bootstrap}

pprint(random_grid)
rf_random = RandomizedSearchCV(estimator=rf, param_distributions = random_grid, n_iter=10, cv = 3, verbose=2, random_state=42, n_jobs=-1)

rf_random.fit(X_train, y_train)
#best_random = rf_random.best_estimator_
best_random = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=30,
                      max_features='sqrt', max_leaf_nodes=None,
                      min_impurity_decrease=0.0, min_impurity_split=None,
                      min_samples_leaf=1, min_samples_split=5,
                      min_weight_fraction_leaf=0.0, n_estimators=1400,
                      n_jobs=None, oob_score=False, random_state=42, verbose=0,
                      warm_start=False)
rfr_CV_best = -rmse_cv(best_random)
best_random.fit(X_train, y_train)
y_train_rfr = best_random.predict(X_train)
y_test_rfr = best_random.predict(X_test)
rfr_best_results = pd.DataFrame({'algorithm':['Random Forest Regressor'],
            'CV error': rfr_CV_best.mean(),
            'CV std': rfr_CV_best.std(),
            'training error': [mean_squared_error(y_train, y_train_rfr)],
            'test error': [mean_squared_error(y_test, y_test_rfr)],
            'training_r2_score': [r2_score(y_train, y_train_rfr)],
            'test_r2_score': [r2_score(y_test, y_test_rfr)]})
print(rfr_best_results)
eli5.show_weights(best_random, feature_names=list(X_df.columns))



pd.concat([ridge_results, lasso_results, ridge_lassoVars_results, huber_results,rfr_best_results], axis=0, ignore_index=True)
