import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine
import ModelAuto as ma
#pip install pymysql
import pymysql

engine = create_engine("mysql+pymysql://admin:@database-1.cqzv6mobzhz3.us-east-2.rds.amazonaws.com/datascience_project?charset=utf8mb4")

train_data = pd.read_sql_table("train_house_features",engine)
test_data = pd.read_sql_table("test_house_features",engine)
train_data.isna().sum() #Finding the missing values

train = train_data.copy()
test = test_data.copy()

train = ma.Datapreprocess.handel_nan(train) #In this step handel_nan will remove columns that has more than 50% null values
test = ma.Datapreprocess.handel_nan(test)

Y = train['SalePrice'] #Storing Saleprice column in a separate variable,   This will be used to select feature
X = train.drop(['SalePrice','Id'],axis=1)
test = test.drop(['Id'],axis=1)

train = ma.Datapreprocess.handel_standardization(X)
test = ma.Datapreprocess.handel_standardization(test)

count = ma.No_of_Catagorical(train)
count.head(5)

Train, Test = ma.Datapreprocess.handel_Catagorical(train,test)

Train.head(5)

feature = ma.FeatureSelection.backwardElimination(Train,Y) #The dataset - Train, Y-Target Column
# It gives the best value with PValue less than 0.5

feature.head()

model = ma.ModelSelection.Regress_model(feature,Y) #Choose the best model, It gives value for each model and also displays chart

subb = model.predict(Test.loc[:,feature.columns]) #Predicting the values with the best model

model

#submission_format = pd.read_csv()

Test

#df = pd.DataFrame({'Id':Test_data['Id'],'SalePrice':sub})
df = pd.DataFrame({'Id':test_data['Id'],'SalePrice':subb})


print(df.head())