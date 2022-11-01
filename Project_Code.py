#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd;


# In[2]:


import ModelAuto as ma;


# In[3]:


#Load train and test data into python data frame
train_data = pd.read_csv("C:\\Users\\trobi\\OneDrive\\Desktop\\Data Science and Technologies\\Project\\train.csv")
test_data = pd.read_csv("C:\\Users\\trobi\\OneDrive\\Desktop\\Data Science and Technologies\\Project\\test.csv")


# In[4]:


no_of_null_values = train_data.isna().sum()


# In[5]:


no_of_null_values.to_csv('C:\\Users\\trobi\\ain = ma.Datapreprocess.handel_nan(train) 
test = ma.Datapreprocess.handel_nan(test)OneDrive\\Desktop\\Data Science and Technologies\\Project\\no_of_null_values.csv',index=True)


# In[6]:


train = train_data.copy()
test = test_data.copy()


# In[9]:


#In this step handel_nan will remove columns that has more than 50% null values
train = ma.Datapreprocess.handel_nan(train) 
test = ma.Datapreprocess.handel_nan(test)


# In[8]:


train.to_csv('C:\\Users\\trobi\\OneDrive\\Desktop\\Data Science and Technologies\\Project\\train_new.csv',index=True)
#If I check the csv file the total number of columns present now is 77 previously it was 81


# In[10]:


#Storing Saleprice column in a separate variable this SalePrice will be the target column
Y = train['SalePrice'] 


# In[11]:


#Storing All other columns in a separate variable except for SalePrice, for all this columns  we need to find out SalePrice
X = train.drop(['SalePrice','Id'],axis=1)


# In[12]:


test = test.drop(['Id'],axis=1)


# In[13]:


#The below code will scale the data between 0,1 by default
train = ma.Datapreprocess.handel_standardization(X)
test = ma.Datapreprocess.handel_standardization(test)


# In[14]:


print(train)


# In[15]:


#Shows the number of categorical values in each column
count = ma.No_of_Catagorical(train)
count.head(5)


# In[17]:


#Assigns the value between 0 and 1 for the categorical variables
Train, Test = ma.Datapreprocess.handel_Catagorical(train,test)


# In[18]:


Train.head(5)


# In[19]:


#The dataset - Train (All Columns except SalePrice), Y-Target Column (SalePrice)
feature = ma.FeatureSelection.backwardElimination(Train,Y) 
# It gives the best value with PValue less than 0.5
# Accuracy is improving by the removal of attributes


# In[20]:


feature.head() 
# Out of 176 columns 60 columns are removed for the improvement of accuracy


# In[21]:


#Choose the best model, It gives value for each model and also displays chart
model = ma.ModelSelection.Regress_model(feature,Y) 


# In[22]:


#Predicting the values for the Test data with the best model
subb = model.predict(Test.loc[:,feature.columns]) 


# In[23]:


result_set = pd.DataFrame({'Id':test_data['Id'],'SalePrice':subb})


# In[24]:


print(result_set)


# In[25]:


#Exporting the results to csv
result_set.to_csv('C:\\Users\\trobi\\OneDrive\\Desktop\\Data Science and Technologies\\Project\\ds_project_results.csv',index=False)


# In[ ]:




