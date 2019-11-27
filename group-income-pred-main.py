import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error,mean_absolute_error
import pickle
train = pd.read_csv("tcd-ml-1920-group-income-train.csv")

test = pd.read_csv("tcd-ml-1920-group-income-test.csv")

data = pd.concat([train,test],ignore_index=True)


def tranformAge(data):


    data['Age'] = data['Age'].replace( np.NaN ,50)
    data['Age'] = data['Age'] / 10
    print(data['Age'].unique())
    return data


def tranformYearofRecord(data):

    data['Year of Record'] = data['Year of Record'].replace( '#N/A' ,2019)
    data['Year of Record'] = data['Year of Record'].replace( np.NaN ,2019)
    return data

def tranformHousing(data):

    data['Housing Situation'] = data['Housing Situation'].replace( 'nA' ,'missing_housing')
    data['Housing Situation'] = data['Housing Situation'].replace( '0' ,'missing_housing')
    data['Housing Situation'] = data['Housing Situation'].replace( 0 ,'missing_housing')
    print(data['Housing Situation'].unique())

    return data

def tranformWorkEx(data):

    data['Work Experience in Current Job [years]'] = data['Work Experience in Current Job [years]'].replace( '#N/A' ,25)
    data['Work Experience in Current Job [years]'] = data['Work Experience in Current Job [years]'].replace( np.NaN ,25)
    data['Work Experience in Current Job [years]'] = data['Work Experience in Current Job [years]'].replace( '#NUM!' ,25)
    data['Work Experience in Current Job [years]'] = data['Work Experience in Current Job [years]'].astype(float)
    print(data['Work Experience in Current Job [years]'].unique())
    return data

def tranformSatisfaction(data):

    data['Satisfation with employer'] = data['Satisfation with employer'].replace( np.NaN ,'missing_satisfaction')
    data['Satisfation with employer'] = data['Satisfation with employer'].replace( '#N/A' ,'missing_satisfaction')
    return data
def transformBodyHeight(data):

    data['Body Height [cm]'] = data['Body Height [cm]'] / 100
    print(data['Body Height [cm]'].unique())
    return data

def tranformGender(data):

    data.Gender = data.Gender.replace( '#N/A' ,'missing_gender')
    data.Gender = data.Gender.replace( 'unknown' ,'missing_gender')
    data.Gender = data.Gender.replace( '0' ,'missing_gender')
    data.Gender = data.Gender.replace( 0 ,'missing_gender')
    data.Gender = data.Gender.replace( 'f' ,'female')
    data.Gender = data.Gender.replace( np.NaN ,'missing_gender')
    print(data['Gender'].unique())
    return data

def tranformProfession(data):

    data['Profession'] = data['Profession'].replace( np.NaN ,'missing_prof')
    return data

def tranformUniversity(data):

    data['University Degree'] = data['University Degree'].replace( '#N/A' ,'missing_degree')
    data['University Degree'] = data['University Degree'].replace( '0' ,'missing_degree')
    data['University Degree'] = data['University Degree'].replace( 0 ,'missing_degree')
    data['University Degree'] = data['University Degree'].replace( np.NaN ,'missing_degree')
    return data

def transformCitySize(data):

    data['Size of City'] = data['Size of City'] / 100

    return data

def tranformHairColor(data):

    data['Hair Color'] = data['Hair Color'].replace( '#N/A' ,'missing_hair')
    data['Hair Color'] = data['Hair Color'].replace( '0' ,'missing_hair')
    data['Hair Color'] = data['Hair Color'].replace( 0 ,'missing_hair')
    data['Hair Color'] = data['Hair Color'].replace( 'Unknown' ,'missing_hair')
    data['Hair Color'] = data['Hair Color'].replace( np.NaN ,'missing_hair')

    return data

def tranformExtraIncome(data):

    data['Yearly Income in addition to Salary (e.g. Rental Income)'] = data['Yearly Income in addition to Salary (e.g. Rental Income)'].str.replace(' EUR' ,'')
    data['Yearly Income in addition to Salary (e.g. Rental Income)'] = data['Yearly Income in addition to Salary (e.g. Rental Income)'].astype(float)
    return data

def traindataPreprocessing(data):

    data = tranformAge(data)
    data = tranformYearofRecord(data)
    data = tranformHousing(data)
    data = tranformWorkEx(data)
    data = transformCitySize(data)
    data = tranformExtraIncome(data)
    data = transformBodyHeight(data)
    data = tranformGender(data)
    data = tranformUniversity(data)
    data = tranformProfession(data)
    data = tranformHairColor(data)
    data = tranformSatisfaction(data)

    # Dropping the additional income column. Adding it back after the predictions in addAdditionalIncome.py file to generate the final CSV. 
    data['Actual Income'] = data['Total Yearly Income [EUR]'] - data['Yearly Income in addition to Salary (e.g. Rental Income)']

    return data

traindataPreprocessing(data)

# referred this part of code from the Github of the Top scorer in first ML competetion.
def timeblockFrequencyEncoder(df,cats,cons,normalize=True):
    for i,cat in enumerate(cats):
        vc = df[cat].value_counts(dropna=False, normalize=normalize).to_dict()
        nm = cat + '_FE_FULL'
        df[nm] = df[cat].map(vc)
        df[nm] = df[nm].astype('float32')
        for j,con in enumerate(cons):
#             print("cat %s con %s"%(cat,con))
            new_col = cat +'_'+ con
            print('timeblock frequency encoding:', new_col)
            df[new_col] = df[cat].astype(str)+'_'+df[con].astype(str)
            temp_df = df[new_col]
            fq_encode = temp_df.value_counts(normalize=True).to_dict()
            df[new_col] = df[new_col].map(fq_encode)
            df[new_col] = df[new_col]/df[cat+'_FE_FULL']
    return df

cats = ['Year of Record','Crime Level in the City of Employement', 'Work Experience in Current Job [years]',
        'Satisfation with employer', 'Gender', 'Country',
        'Profession', 'University Degree','Wears Glasses',
        'Hair Color','Age']
cons = ['Size of City','Body Height [cm]']

data = timeblockFrequencyEncoder(data,cats,cons)

for col in train.dtypes[train.dtypes == 'object'].index.tolist():
    feat_le = LabelEncoder()
    feat_le.fit(data[col].unique().astype(str))
    data[col] = feat_le.transform(data[col].astype(str))

del_col = set(['Total Yearly Income [EUR]','Instance','Yearly Income in addition to Salary (e.g. Rental Income)','Actual Income'])
features_col =  list(set(data) - del_col)
print(features_col)
print(data.head(50))
print(data.tail(50))
X_train,X_test = data[features_col].iloc[:1048573],data[features_col].iloc[1048574:]
Y_train = data['Actual Income'].iloc[:1048573]
X_test_id = data['Instance'].iloc[1048574:]
x_train,x_val,y_train,y_val = train_test_split(X_train,Y_train,test_size=0.2,random_state=1234)

params = {
        #   'max_depth': 15,
          'learning_rate': 0.001,
          "boosting": "gbdt",
          "bagging_seed": 11,
          "metric": 'mae',
          "verbosity": -1,
          "num_leaves" : 25
         }
trn_data = lgb.Dataset(x_train, label=y_train)
val_data = lgb.Dataset(x_val, label=y_val)
clf = lgb.train(params, trn_data, 300000, valid_sets = [trn_data, val_data], verbose_eval=1000, early_stopping_rounds=500)
# save model
joblib.dump(clf, 'lgb9.pkl') #numbering my models to keep a backup
# load model
gbm_pickle = joblib.load('lgb9.pkl')

pre_test_lgb = gbm_pickle.predict(X_test)

pre_val_lgb = gbm_pickle.predict(x_val)
val_mse = mean_squared_error(y_val,pre_val_lgb)
val_mae = mean_absolute_error(y_val,pre_val_lgb)
val_rmse = np.sqrt(val_mse)
print(val_rmse)
print(val_mae)

sub_df = pd.DataFrame({'Instance':X_test_id,
                       'Total Yearly Income [EUR]':pre_test_lgb})
print(sub_df.head())
sub_df.to_csv("sub191015_17.csv",index=False)
'done'