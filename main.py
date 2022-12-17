import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import  train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from xgboost.sklearn import XGBRegressor

train_data = pd.read_csv("dataset/novozymes-enzyme-stability-prediction/train.csv", index_col='seq_id')
train_data_update = pd.read_csv("dataset/novozymes-enzyme-stability-prediction/train_updates_20220929.csv", index_col='seq_id')
test_data = pd.read_csv("dataset/novozymes-enzyme-stability-prediction/test.csv", index_col='seq_id')

all_nan = train_data_update.isna().all(axis='columns')

drop_index = train_data_update[all_nan].index
train_data.drop(index=drop_index, inplace=True)

update_index = train_data_update[~all_nan].index    # ~ means not
train_data.loc[update_index, ['protein_sequence', 'pH', 'tm']] = train_data_update.loc[update_index, ['protein_sequence', 'pH', 'tm']]

print(train_data.isnull().any())
train_data['pH'] = train_data['pH'].fillna(7.0)

# amino_acids = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
#                'U', 'V', 'W', 'X', 'Y', 'Z']
# for letter in amino_acids:
#     train_data[letter] = train_data['protein_sequence'].str.count(letter)
#     test_data[letter] = test_data['protein_sequence'].str.count(letter)
# print(train_data)
# letter_count_is_zero = {}
# for letter in amino_acids:
#     letter_count_is_zero[letter] = train_data[letter].isin([0]).all()
# print(letter_count_is_zero)

amino_acids= ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
for letter in amino_acids:
    train_data[letter] = train_data['protein_sequence'].str.count(letter)
    test_data[letter] = test_data['protein_sequence'].str.count(letter)

train_data.drop(['protein_sequence', 'data_source'], axis=1, inplace=True)
sns.heatmap(train_data.astype('float').corr(), annot=True)
# plt.show()

# print(train_data)
X = train_data.drop(['tm'], axis=1)
y = train_data['tm']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

xgbr = XGBRegressor(n_estimators=800,learning_rate=0.1
                    ,max_depth = 5,min_child_weight = 1
                    ,subsample = 0.8, colsample_bytree = 0.8
                    ,gamma = 0,reg_alpha = 0,reg_lambda = 1
                    ,tree_method= 'gpu_hist',base_score=0.5
                    ,max_cat_threshold=64,max_cat_to_onehot=4)

xgbr.fit(X_train, y_train)
y_train_pred = xgbr.predict(X_train)
y_test_pred = xgbr.predict(X_test)
print('MSE train: %.3f, test: %.3f' % (
    mean_squared_error(y_train, y_train_pred),
    mean_squared_error(y_test, y_test_pred)))

print('R^2 train: %.5f, test: %.5f' % (
    r2_score(y_train, y_train_pred),
    r2_score(y_test, y_test_pred)))

"""
parameters_to_search = {'learning_rate': np.linspace(0.1,1,10),'n_estimators': range(500,2000,100)}
xgbr_cv = GridSearchCV(xgbr,parameters_to_search,cv=5,scoring = 'r2',n_jobs = -1)
xgbr_cv.fit(X_train,y_train)

best_estimator = xgbr_cv.best_estimator_
print(best_estimator)
"""
