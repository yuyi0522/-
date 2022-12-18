import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import  train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from xgboost.sklearn import XGBRegressor
from scipy.stats import spearmanr

train_data = pd.read_csv("dataset/novozymes-enzyme-stability-prediction/train.csv", index_col='seq_id')
train_data_update = pd.read_csv("dataset/novozymes-enzyme-stability-prediction/train_updates_20220929.csv", index_col='seq_id')
test_data = pd.read_csv("dataset/novozymes-enzyme-stability-prediction/test.csv", index_col='seq_id')

#以下為原始資料中有錯因此照著官方指示將原資料中的錯誤給修正
all_nan = train_data_update.isna().all(axis='columns')

drop_index = train_data_update[all_nan].index
train_data.drop(index=drop_index, inplace=True)

update_index = train_data_update[~all_nan].index    # ~ means not
train_data.loc[update_index, ['protein_sequence', 'pH', 'tm']] = train_data_update.loc[update_index, ['protein_sequence', 'pH', 'tm']]

#發現資料中pH值有缺失並以平均值補上
print(train_data.isnull().any())
train_data['pH'] = train_data['pH'].fillna(train_data["pH"].mean())

#將資料欄位中的蛋白質序列做拆解並計算每個序列出現的次數，以此做為判斷不同酶的依據
amino_acids= ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
for letter in amino_acids:
    train_data[letter] = train_data['protein_sequence'].str.count(letter)
    test_data[letter] = test_data['protein_sequence'].str.count(letter)

train_data.drop(['protein_sequence', 'data_source'], axis=1, inplace=True)
test_data.drop(['protein_sequence', 'data_source'], axis=1, inplace=True)

"""
畫出heatmap來檢查各項目對資料的重要性
sns.heatmap(train_data.astype('float').corr(), annot=True)
plt.show()
"""

#將目標欄位分割出來以及做資料切割
X = train_data.drop(['tm'], axis=1)
y = train_data['tm']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

#使用XGBRegressor做資料預測，xgb跟隨機森林相似，然而隨機森林是決策樹每每獨立而xgb為每棵樹相互影響
#但是xgb的優勢在於更多的超參數和更快的迭代速度，提供了更多參數調整的可能性
xgbr = XGBRegressor(n_estimators=2000,learning_rate=0.1
                    ,max_depth = 10,min_child_weight = 1
                    ,subsample = 0.8, colsample_bytree = 0.8
                    ,gamma = 0,reg_alpha = 0,reg_lambda = 1
                    ,tree_method= 'gpu_hist',base_score=0.5
                    ,max_cat_threshold=64,max_cat_to_onehot=4
                    ,gpu_id=0,random_state=7)

xgbr.fit(X_train, y_train)
#做資料預測
y_train_pred = xgbr.predict(X_train)
y_test_pred = xgbr.predict(X_test)
#輸出結果
print('MSE train: %.3f, test: %.3f' % (
    mean_squared_error(y_train, y_train_pred),
    mean_squared_error(y_test, y_test_pred)))

print('R^2 train: %.3f, test: %.3f' % (
    r2_score(y_train, y_train_pred),
    r2_score(y_test, y_test_pred)))


print('spearmanr train:', spearmanr(y_train, y_train_pred))
print('spearmanr test:', spearmanr(y_test, y_test_pred))
#最後把資料輸出成csv並將成果上傳至kaggle
submission = pd.read_csv('dataset/novozymes-enzyme-stability-prediction/sample_submission.csv')
submission['tm'] = xgbr.predict(test_data)
submission['tm'] = submission['tm'].astype('int64')
submission.to_csv('submission.csv', index=False)

"""
#以下為使用gridsearch做超參數的search
parameters_to_search = {'n_estimators': range(500,5100,100),'max_depth' : range(3,10,1)}
xgbr_cv = GridSearchCV(xgbr,parameters_to_search,cv=5,scoring = 'r2',n_jobs = -1)
xgbr_cv.fit(X_train,y_train)

best_estimator = xgbr_cv.best_estimator_
print(best_estimator)
"""
