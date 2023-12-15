#!/usr/bin/env python
# coding: utf-8

# In[124]:


import pandas as pd
import numpy as np


# In[125]:


df=pd.read_csv(r"D:\NIT\DECEMBER\11 DEC  (SLR(SIMPLE))\11th - Regression model\SIMPLE LINEAR REGRESSION\Salary_Data.csv")


# In[126]:


df


# # mean

# In[127]:


df.mean()


# In[128]:


df["Salary"].mean()


# # median

# In[129]:


df.median() # this will give median of entire dataframe 


# In[130]:


df['Salary'].median() # this will give us median of that particular column 


# # Mode
# 

# In[131]:


df['Salary'].mode() # this will give us mode of that particular column 


# # Variance
# 

# In[132]:


df.var() # this will give variance of entire dataframe 


# In[133]:


df['Salary'].var() # this will give us variance of that particular column


# # Standard deviation
# 

# In[134]:


df.std() # this will give standard deviation of entire dataframe 


# In[135]:


df['Salary'].std() # this will give us standard deviation of that particular column


# # Correlation

# In[136]:


df.corr() # this will give correlation of entire dataframe


# In[137]:


df['Salary'].corr(df['YearsExperience']) # this will give us correlation between these tw


# # Skewness

# In[138]:


df.skew() # this will give skewness of entire dataframe 


# In[139]:


df['Salary'].skew() # this will give us skewness of that particular column


# # Standard Error

# In[140]:


df.sem() # this will give standard error of entire dataframe 


# In[141]:


df['Salary'].sem() # this will give us standard error of that particular column


# # Coefficient of variation(cv)
# 

# In[142]:


# for calculating cv we have to import a library first
from scipy.stats import variation
variation(df.values) # this will give cv of entire dataframe 


# In[143]:


variation(df['Salary']) # this will give us cv of that particular column


# # Z-score

# In[144]:


# for calculating Z-score we have to import a library first
import scipy.stats as stats
df.apply(stats.zscore) # this will give Z-score of entire dataframe


# In[145]:


stats.zscore(df['Salary']) # this will give us Z-score of that particular column


# # Degree of Freedom

# In[146]:


df.shape


# In[147]:


a = df.shape[0] # this will gives us no.of rows
b = df.shape[1] # this will give us no.of columns


# In[148]:


print(a)
print(b)


# In[149]:


degree_of_freedom = a-b
print(degree_of_freedom) # this will give us degree of freedom for entire dataset


# # Sum of Squares Regression (SSR)

# In[150]:


#First we have to separate dependent and independent variables
X=df.iloc[:,:-1].values #independent variable
y=df.iloc[:,1].values # dependent variable


# In[151]:


print(X,y)


# In[152]:


y_mean = np.mean(y) # this will calculate mean of dependent variable


# In[153]:


y_mean


# In[154]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=0)


# In[155]:


X_train,X_test,y_train,y_test


# In[156]:


from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train,y_train)
y_predict = reg.predict(X_test) # before doing this we have to train,test and split our 


# In[157]:


y_predict


# In[158]:


SSR = np.sum((y_predict-y_mean)**2)


# In[159]:


SSR


# # Sum of Squares Error (SSE)
# 

# In[160]:


#First we have to separate dependent and independent variables
X=df.iloc[:,:-1].values #independent variable
y=df.iloc[:,1].values # dependent variable


# In[161]:


print(X,y)


# In[162]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=0)


# In[163]:


X_train,X_test,y_train,y_test


# In[164]:


from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train,y_train)
y_predict = reg.predict(X_test) # before doing this we have to train,test and split our


# In[165]:


y_predict 


# In[166]:


y = y[0:6]


# In[167]:


y


# In[168]:


SSE = np.sum((y-y_predict)**2)
print(SSE)


# # Sum of Squares Total (SST)
# 

# In[169]:


np.mean(df.values) 


# In[170]:


mean_total = np.mean(df.values) # here df.to_numpy()will convert pandas Dataframe to Nump
SST = np.sum((df.values-mean_total)**2)
print(SST)


# # R-Square
# 

# In[171]:


r_square = 1-(SSR/SST)
r_square


# In[ ]:





# In[ ]:





# In[ ]:




