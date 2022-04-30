#!/usr/bin/env python
# coding: utf-8

# # INDEX

# 
# 
# 1. Importing Required Libraries and Loading the Dataset
# 
# 2. Exploratory Data Analysis
# 
#     a. Understanding the dataset
# 
#     b. Data Cleaning
#         --> Checking NULL values
#         --> Checking for 0 value
# 3. Data Visualization
# 
#         Here we are going to plot :-
#             Count Plot :- to see if the dataset is balanced or not
#             Histograms :- to see if data is normally distributed or skewed
# 5. Handling Outliers
# 
# 6. Split the Data Frame into X and y
# 
# 7. TRAIN TEST SPLIT
# 
# 8. Apply Deep Neural Network model for the Trained Data

# # 1. Import required libraries and data

# In[35]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


# In[36]:


df = pd.read_csv('diabetes.csv')


# #  2. Exploratory Data Analysis

# # 2.a. Understanding the data

# In[37]:


df.head()


# In[6]:


df.shape


# In[7]:


df.info()


# In[8]:


df.describe()


# Conclusion :-  We observe that min value of some columns is 0 which cannot be possible medically.Hence in the data cleaning process we'll have to replace them with median/mean value depending on the distribution. Also in the max column we can see insulin levels as high as 846! We have to treat outliers.

# # 2.b. Data Cleaning

#     --> Checking NULL values
#     --> Checking for 0 value and replacing it :- It isn't medically possible for some data record to have 0 value such as Blood Pressure or Glucose levels. Hence we replace them with the mean value of that particular column.

# In[9]:


df.isnull().sum()


# Checking for 0 values in 5 columns , Age & DiabetesPedigreeFunction do not have have minimum 0 value so no need to replace , also no. of pregnancies as 0 is possible as observed in df.describe

# In[10]:


print(df[df['BloodPressure']==0].shape[0])
print(df[df['Glucose']==0].shape[0])
print(df[df['SkinThickness']==0].shape[0])
print(df[df['Insulin']==0].shape[0])
print(df[df['BMI']==0].shape[0])


# NOTE :-
# Some of the columns have a skewed distribution, so the mean is more affected by outliers than the median. Glucose and Blood Pressure have normal distributions hence we replace 0 values in those columns by mean value. SkinThickness, Insulin,BMI have skewed distributions hence median is a better choice as it is less affected by outliers.
# 
# Refer Histograms down below to see the distribution

# In[11]:


df['Glucose']=df['Glucose'].replace(0,df['Glucose'].mean())
df['BloodPressure']=df['BloodPressure'].replace(0,df['BloodPressure'].mean())
df['SkinThickness']=df['SkinThickness'].replace(0,df['SkinThickness'].median())
df['Insulin']=df['Insulin'].replace(0,df['Insulin'].median())
df['BMI']=df['BMI'].replace(0,df['BMI'].median())


# # 3. Data Visualization

# In[34]:


sns.countplot('Outcome',data=df)


# Conclusion :- We observe that number of people who do not have diabetes is far more than people who do which indicates that our data is imbalanced

# In[13]:


df.hist(bins=10,figsize=(10,10))
plt.show()


# Conclusion :- We observe that only Glucose and Blood Pressure are normally distributed rest others are skewed and have outliers

# In[25]:


col_norm =['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age']
df_norm = df[col_norm]
df1_norm = df[col_norm].apply(lambda x :( (x - x.min()) / (x.max()-x.min()) ) )
Y_Data = df["Outcome"]
X_Data = df1_norm
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X_Data,Y_Data, test_size=0.3,random_state=101)


# In[ ]:




