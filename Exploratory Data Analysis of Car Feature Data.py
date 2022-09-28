#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns #visualisation
import matplotlib.pyplot as plt #visualisation
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(color_codes=True)
from scipy import stats


# In[2]:


df = pd.read_csv('data.csv') #data downloaded from - https://www.kaggle.com/CooperUnion/cardataset
df.head()


# In[3]:


df.dtypes


# In[4]:


cols_to_drop = ["Engine Fuel Type", "Market Category", "Vehicle Style", "Popularity", "Number of Doors", "Vehicle Size"]
df.drop(cols_to_drop,axis=1,inplace=True)# keeping what i need
df.head()


# In[5]:


#for ease
rename_cols = {"Engine HP": "HP", "Engine Cylinders": "Cylinders", "Transmission Type": "Transmission", 
               "Driven_Wheels": "Drive Mode","highway MPG": "MPG_H", "city mpg": "MPG-C", "MSRP": "Price" }
df.rename(columns=rename_cols,inplace=True)
df.head()


# In[6]:


print(df.count())
df.drop_duplicates(inplace=True)
df.head()


# In[7]:


print(df.count())# around 1000 dropped


# In[8]:


df.isnull().sum()


# In[9]:


df.dropna(inplace=True)
df.isnull().sum()


# In[10]:


df['Price']=np.array(df['Price']).astype(np.float)
df.describe()# gives an idea about the type of data


# ### Plots

# In[11]:


plt.boxplot(df.Price,vert = 0,widths=10)
plt.show()  # Boxplot for Price


# In[12]:


plt.boxplot('HP',data=df,vert=0)
plt.show() # for HP


# In[13]:


a=df.columns[df.dtypes != object]
print(a)


# In[14]:


li=list(a)
li


# In[15]:


# removing outlier using IQR
df2=df.copy()
for x in li :
    q1 = df[x].quantile(0.25)
    q3 = df[x].quantile(0.75)
    iqr = q3-q1
    lower_limit =(q1-1.5*iqr)
    upper_limit = (q3 + 1.5*iqr)
    df2= df2[(df2[x]>=lower_limit) & (df2[x]<= upper_limit)]
    

df2.head()


# In[16]:


print(df.shape)
print(df2.shape)
# should give a count of ouliers removed
df.shape[0]-df2.shape[0]


# In[17]:


df2.value_counts()


# In[18]:


sns.distplot(df2['HP'])# distribution plot


# In[19]:


plt.figure(figsize=(12,6))
plt.hist(x=df2['Make'])
plt.xlabel(" make")
plt.ylabel("number of cars")
plt.xticks(rotation=90)
plt.title("Bar Plot")
plt.show()


# In[20]:


plt.figure(figsize=(15,15))
sns.scatterplot(x='HP',y='Price',data=df2)
#Scatter plot


# In[21]:


sns.pairplot(df2)#pair plot


# In[22]:


plt.figure(figsize=(15,5))
sns.lmplot(x='HP',y='Price',data=df2,hue='Transmission',fit_reg=False)


# In[ ]:




