import pickle
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# custom files
import model_best_hyperparameters
import columns

# read train data
ds = pd.read_csv("train_data.csv")

ds.drop(['Flow.ID','Bwd.PSH.Flags','Fwd.PSH.Flags','Fwd.URG.Flags','Bwd.URG.Flags','FIN.Flag.Count',
         'SYN.Flag.Count','RST.Flag.Count','PSH.Flag.Count','ACK.Flag.Count','URG.Flag.Count','CWE.Flag.Count','ECE.Flag.Count',
        'Fwd.Avg.Bytes.Bulk','Fwd.Avg.Packets.Bulk','Fwd.Avg.Bulk.Rate','Bwd.Avg.Bytes.Bulk','Bwd.Avg.Packets.Bulk','Bwd.Avg.Bulk.Rate','ProtocolName'
        ],axis=1, inplace=True)

def find_skewed_boundaries(df, variable, distance):
    df[variable] = pd.to_numeric(df[variable], errors='coerce')
    IQR = df[variable].quantile(0.75) - df[variable].quantile(0.25)
    lower_boundary = df[variable].quantile(0.25) - (IQR * distance)
    upper_boundary = df[variable].quantile(0.75) + (IQR * distance)
    return upper_boundary, lower_boundary


upper_lower_limits = dict()
for column in columns.outlier_columns:
    upper_lower_limits[column + '_upper_limit'], upper_lower_limits[column + '_lower_limit'] = find_skewed_boundaries(
        ds, column, 5)
for column in columns.outlier_columns:
    ds = ds[~ np.where(ds[column] > upper_lower_limits[column + '_upper_limit'], True,
                       np.where(ds[column] < upper_lower_limits[column + '_lower_limit'], True, False))]

le = LabelEncoder()
for column in columns.cat_columns:
    ds[column] = le.fit_transform(ds[column])

# save parameters
param_dict = {
              'upper_lower_limits': upper_lower_limits,
              }
with open('param_dict.pickle', 'wb') as handle:
    pickle.dump(param_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Define target and features columns
X = ds[columns.X_columns]
y = ds[columns.y_column]

# Let's say we want to split the data in 90:10 for train:test dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

# Building and train Random Forest Model
rf = RandomForestClassifier(**model_best_hyperparameters.params)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print('test set metrics: ', metrics.classification_report(y_test, y_pred))

filename = 'finalized_model.sav'
pickle.dump(rf, open(filename, 'wb'))