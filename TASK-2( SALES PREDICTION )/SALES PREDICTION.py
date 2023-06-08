#!/usr/bin/env python
# coding: utf-8

# # OASIS INFOBYTE DATA SCIENCE INTERNSHIP
# # TASK- 2 SALES PREDICTION USING PYTHON
# ## BY: MUZAMIL JAMIL CHANNA

# Sales prediction means predicting how much of a product people will buy based on factors
# such as the amount you spend to advertise your product, the segment of people you
# advertise for, or the platform you are advertising on about your product.
# Typically, a product and service-based business always need their Data Scientist to predict
# their future sales with every step they take to manipulate the cost of advertising their
# product. So let’s start the task of sales prediction with machine learning using Python.
# 
# 

# In[39]:


#import libiraies
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# In[3]:


#load Dataset
dataset = pd.read_csv("Advertising.csv")


# In[4]:


dataset


# In[5]:


dataset.info()


# In[6]:


## Descriptive statistics
dataset.describe()


# In[7]:


dataset.shape


# In[8]:


dataset.columns


# In[9]:


dataset = dataset.rename(columns= { "Unnamed: 0" : "id" })


# In[10]:


dataset


# In[55]:


dataset.isnull().sum()


# In[56]:


sales_mean = dataset['Sales'].mean()
print('Mean Sales:', sales_mean)


# In[57]:


tv_sum = dataset['TV'].sum()
print('Sum of TV:', tv_sum)


# In[59]:


tv_radio_diff = dataset['TV'] - dataset['Radio']
print('TV - Radio Difference:')
print(tv_radio_diff)


# In[62]:


newspaper_max = dataset['Newspaper'].max()
print('Max Newspaper:', newspaper_max)


# In[75]:


# Histogram of 'Sales' column
plt.hist(dataset['Sales'], bins=10)
plt.title('Distribution of Sales')
plt.xlabel('Sales')
plt.ylabel('Frequency')
plt.show()


# # Observation:
# The majority of sales values in the dataset are higher than 10, with a frequency of 40 instances.

# In[76]:


media_columns = ['TV', 'Radio', 'Newspaper']
media_sums = dataset[media_columns].sum()

plt.bar(media_columns, media_sums)
plt.title('Total Advertising Expenditure by Media')
plt.xlabel('Media')
plt.ylabel('Total Expenditure')
plt.show()


# 
# # Observation:
# The total advertising expenditure is highest for TV, followed by Radio and then Newspaper

# In[12]:


plt.figure(figsize=(12,6))
sns.histplot(dataset['TV'],kde=True,bins=10 ,color= "blue")
plt.ylabel("frequency")
plt.title("Distribution of TV")
plt.show()


# # Observation
# The majority of TV spending falls within the range of 0-50 and 200-250.

# In[13]:


# Kernel Density Estimation (KDE) plot
plt.figure(figsize=(10, 6))
sns.kdeplot(dataset['Sales'], shade=True, color='green')
plt.title('Kernel Density Estimation (KDE) Plot of Sales')
plt.xlabel('Sales')
plt.ylabel('Density')
plt.show()


# In[15]:


plt.figure(figsize=(16, 6))
sns.countplot(data=dataset, x='Newspaper', order=dataset['Newspaper'].value_counts().index[:30])
plt.title('Distribution of Newspaper')
plt.xlabel('Newspaper')
plt.ylabel('Count')
plt.show()


# In[16]:


# Histogram for 'Radio' column
plt.figure(figsize=(10, 6))
plt.hist(dataset['Radio'], bins=10, edgecolor='black')
plt.title('Distribution of Radio')
plt.xlabel('Radio')
plt.ylabel('Count')
plt.show()


# In[ ]:





# In[48]:


# Histogram for 'Newspaper' column
plt.figure(figsize=(10, 6))
plt.hist(dataset['Newspaper'], bins=12, edgecolor='black')
plt.title('Distribution of Newspaper')
plt.xlabel('Newspaper')
plt.ylabel('Count')
plt.show()


# In[49]:


data_without_id = dataset.drop('id', axis=1)
# Correlation matrix
corr_matrix = data_without_id.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


# In[31]:


plt.figure(figsize=(14, 6))
plt.subplot(131)
plt.scatter(dataset['TV'], dataset['Sales'], color='blue')
plt.title('TV vs Sales')
plt.xlabel('TV')
plt.ylabel('Sales')

plt.subplot(132)
plt.scatter(dataset['Radio'], dataset['Sales'], color='green')
plt.title('Radio vs Sales')
plt.xlabel('Radio')
plt.ylabel('Sales')

plt.subplot(133)
plt.scatter(dataset['Newspaper'], dataset['Sales'], color='red')
plt.title('Newspaper vs Sales')
plt.xlabel('Newspaper')
plt.ylabel('Sales')

plt.tight_layout()
plt.show()


# In[33]:


# Data preprocessing
X = dataset[['TV', 'Radio', 'Newspaper']]
y = dataset['Sales']


# In[36]:


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[37]:


# Model training
model = LinearRegression()
model.fit(X_train, y_train)


# In[40]:


# Model evaluation
y_train_pred = model.predict(X_train)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
print('Train RMSE:', train_rmse)


# In[41]:


y_test_pred = model.predict(X_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
print('Test RMSE:', test_rmse)


# In[42]:


# Coefficients and Intercept
print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)


# In[43]:


# Feature importance
feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': model.coef_})
feature_importance.sort_values(by='Importance', ascending=False, inplace=True)
print(feature_importance)


# In[50]:


# Predicting sales for new data
new_data = pd.DataFrame({'TV': [200], 'Radio': [80], 'Newspaper': [40]})
predicted_sales = model.predict(new_data)
print('Predicted Sales:', predicted_sales)


# In[52]:


# Residual analysis
residuals = y_test - y_test_pred
plt.scatter(y_test_pred, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residual Analysis')
plt.xlabel('Predicted Sales')
plt.ylabel('Residuals')
plt.show()


# In[54]:


# Histogram of residuals
plt.hist(residuals, bins=20)
plt.title('Residual Distribution')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()

