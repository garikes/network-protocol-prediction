import pickle
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)

import model_best_hyperparameters
import columns

# read train data
ds = pd.read_csv("new_data.csv")
print('new data size', ds.shape)

# feature engineering
param_dict = pickle.load(open('param_dict.pickle', 'rb'))



# Outlier Engineering
for column in columns.outlier_columns:
    ds[column] = ds[column].astype(float)
    ds = ds[~ np.where(ds[column] > param_dict['upper_lower_limits'][column + '_upper_limit'], True,
                       np.where(ds[column] < param_dict['upper_lower_limits'][column + '_lower_limit'], True, False))]

le = LabelEncoder()
for column in columns.cat_columns:
    ds[column] = le.fit_transform(ds[column])


# Define target and features columns
X = ds[columns.X_columns]

# load the model and predict
rf = pickle.load(open('finalized_model.sav', 'rb'))
ds['Pred'] = rf.predict(X)
ds.to_csv('prediction_results.csv', index=False)


