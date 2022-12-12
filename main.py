import pandas as pd
import numpy as np

train_data = pd.read_csv("./novozymes-enzyme-stability-prediction/train.csv", index_col='seq_id')
train_data_update = pd.read_csv("./novozymes-enzyme-stability-prediction/train_updates_20220929.csv", index_col='seq_id')

all_nan = train_data_update.isna().all(axis='columns')

drop_index = train_data_update[all_nan].index
train_data.drop(index=drop_index, inplace=True)

update_index = train_data_update[~all_nan].index    # ~ means not
train_data.loc[update_index, ['protein_sequence', 'pH', 'tm']] = train_data_update.loc[update_index, ['protein_sequence', 'pH', 'tm']]