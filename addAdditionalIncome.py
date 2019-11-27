import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error,mean_absolute_error
# import matplotlib.pyplot as plt
# from feature_engine import missing_data_imputers as mdi
# from feature_engine.categorical_encoders import OneHotCategoricalEncoder
from sklearn.linear_model import LinearRegression
import pickle
train = pd.read_csv("combined.csv")
train['Yearly Income in addition to Salary (e.g. Rental Income)'] = train['Yearly Income in addition to Salary (e.g. Rental Income)'].str.replace(' EUR' ,'')
train['Yearly Income in addition to Salary (e.g. Rental Income)'] = train['Yearly Income in addition to Salary (e.g. Rental Income)'].astype(float)
sub_df = pd.DataFrame({'Instance':train['Instance'],
                       'Total Yearly Income [EUR]':(train['Total Yearly Income [EUR]']+train['Yearly Income in addition to Salary (e.g. Rental Income)'])/2})
print(sub_df.head())
sub_df.to_csv("sub191015_19.csv",index=False)
'done'