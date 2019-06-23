#!/usr/bin/env python
# coding: utf-8

# In[106]:


import numpy as np
from numpy import random   # some important libraries 
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# In[107]:


df=pd.read_csv('C:\Users\sohai\Desktop\\train.csv',delimiter=',')
df_t=pd.read_csv('C:\Users\sohai\Desktop\\test.csv',delimiter=',')
df=df.fillna(0) 

df1=np.asarray(df[['date','store','item','sales']])
df2=np.asarray(df_t[['date','store','item']])


# In[127]:




X1=np.asarray(df[['date']])
X2=np.asarray(df_t[['date']])
Y=np.asarray(df[['sales']])
Z=np.asarray(df[['item']])


# In[109]:


print(df1[0][0])


# In[22]:


plt.scatter(df['date'], df['sales'], color='red')
plt.title('sales on different dates', fontsize=14)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Sales', fontsize=14)
plt.grid(True)
plt.show()


# In[110]:


from pandas import DataFrame
from sklearn import linear_model
import statsmodels.api as sm


# In[111]:


i=0
while i<913000:
    df1[i][0]=X1[i][0][0]+X1[i][0][1]+X1[i][0][2]+X1[i][0][3]+X1[i][0][5]+X1[i][0][6]+X1[i][0][8]+X1[i][0][9]
    
    i=i+1




# In[112]:


i=0
while i<45000:
    df2[i][0]=X2[i][0][0]+X2[i][0][1]+X2[i][0][2]+X2[i][0][3]+X2[i][0][5]+X2[i][0][6]+X2[i][0][8]+X2[i][0][9]
    
    i=i+1


# In[113]:


dat1 = pd.DataFrame({'date':df1[:,0],'store':df1[:,1],'item':df1[:,2],'sales':df1[:,3]})
data_t=pd.DataFrame({'date':df2[:,0],'store':df2[:,1],'item':df2[:,2]})
X = dat1[['date','store','item']] # here we have 2 variables for multiple regression. If you just want to use one variable for simple linear regression, then use X = df['Interest_Rate'] for example.Alternatively, you may add additional variables within the brackets
Y = dat1['sales']


# In[114]:




regr = linear_model.LinearRegression()
regr.fit(X, Y)



# In[115]:


New_item = 1
New_store = 1
date=20180101
#print ('Predicted Stock Index Price: \n', regr.predict([[date,New_item ,New_store]]))
i=0
a= regr.predict(data_t)
while i<45000:
    
    
    print 'date:',df2[i][0],' store:',df2[i][1],' item:',df2[i][2],' predicted value:',a[i]
    i=i+1


# In[ ]:




