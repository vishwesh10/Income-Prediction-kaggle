#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import sklearn.preprocessing as pp
import sklearn.model_selection as ms
import sklearn.metrics as metrics
from sklearn.metrics import r2_score


# In[2]:


train_df = pd.read_csv('../input/tcd ml 2019-20 income prediction training (with labels).csv')
test_df = pd.read_csv('../input/tcd ml 2019-20 income prediction test (without labels).csv')


# In[ ]:


from fancyimpute import KNN 
train_cols = list(train_df)
train = pd.DataFrame(KNN(k=5).complete(train_df))
train.columns = train_cols


# In[ ]:


x_prime=train_df.ix[:,(1,2,3,4,5,6,7,8)].values
y_prime=train_df.ix[:,9].values
x_test = test_df.ix[:,(1,2,3,4,5,6,7,8)].values


# In[ ]:


#Import sklearn.preprocessing.StandardScaler
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler() # default=(0, 1)
numerical = ['Age', 'Year of Record', 'Size of City', 'Body Height [cm]']

features_log_minmax_transform = pd.DataFrame(data = x_prime)
features_log_minmax_transform[numerical] = scaler.fit_transform(data[numerical])

display(features_log_minmax_transform.head(n = 5))

features_log_minmax_transform = pd.DataFrame(data = x_test)
features_log_minmax_transform[numerical] = scaler.fit_transform(data[numerical])

display(features_log_minmax_transform.head(n = 5))


# In[ ]:


features_log_minmax_transform = features_log_minmax_transform.drop(['Hair Color','Wears Glasses'], axis=1)


# In[ ]:


newdf = features_log_minmax_transform
newdf.head(10)


# In[8]:


# import labelencoder
from sklearn.preprocessing import LabelEncoder
# instantiate labelencoder object
le = LabelEncoder()


# In[9]:


# apply le on categorical feature columns
newdf[categorical_cols] = newdf[categorical_cols].apply(lambda col: le.fit_transform(col))
newdf[categorical_cols].head(10)


# # k-fold cross validation

# In[ ]:


from sklearn.model_selection import KFold
X = x_prime
y = y_prime
kf = KFold(n_splits=10)
kf.get_n_splits(X)
print(kf)  


# In[ ]:


for train_index, test_index in kf.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]


# In[15]:


from xgboost import XGBRegressor
clf = XGBRegressor(objective ='reg:squarederror',n_estimators=2000,learning_rate=0.05,max_depth=6,colsample_bytree =1,nthread= 1,reg_alpha=7,subsample = 0.5,num_round = 10)
clf.fit(x_train, y_train, early_stopping_rounds=20, 
             eval_set=[(x_val_, y_val)], verbose=True)


# In[17]:


pred = clf.predict(x_test)


# In[18]:


from sklearn.metrics import mean_squared_error
from math import sqrt
#66126 #63523 #63388 #63814 #63495
rms = sqrt(mean_squared_error(y_val, pred))
print(rms)


# In[ ]:


test_df['Income'] = pred
test_df.to_csv('Submission.csv', columns = ['Income'])
test_df['Income']


# In[ ]:





# In[ ]:




