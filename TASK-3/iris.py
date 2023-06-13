#!/usr/bin/env python
# coding: utf-8

# # OASIS INFOBYTE DATA SCIENCE INTERNSHIP
# # TASK- 3  IRIS FLOWER CLASSIFICATION
# ## BY: MUZAMIL JAMIL CHANNA

# 
# Iris flower has three species; setosa, versicolor, and virginica, which differs according to their
# measurements. Now assume that you have the measurements of the iris flowers according to
# their species, and here your task is to train a machine learning model that can learn from the
# measurements of the iris species and classify them.
# 
# Although the Scikit-learn library provides a dataset for iris flower classification, you can also
# download the same dataset from here for the task of iris flower classification with Machine
# Learning. 

# In[171]:


# import libraries
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


# In[128]:


dataset=pd.read_csv('Iris.csv')
dataset.head(10)


# In[101]:


dataset.info()


# In[102]:


dataset.shape


# In[103]:


dataset.columns


# In[104]:


dataset.describe()


# In[105]:


dataset.isnull().sum()


# In[106]:


sn.swarmplot(x='Species',y='SepalLengthCm',data=dataset)
plt.show()


# In[186]:


sns.pairplot(dataset, hue='Species')


# In[108]:


sns.swarmplot(x='Species', y='SepalLengthCm', data=dataset)

plt.xlabel('Species')
plt.ylabel('Sepal Length (cm)')
plt.title('Swarm Plot of Sepal Length for Iris Flowers')
plt.grid(True)
plt.show()


# In[109]:


sns.swarmplot(x='Species', y='SepalLengthCm', data=dataset)
plt.xlabel('Species')
plt.ylabel('Sepal Length (cm)')
plt.title('Swarm Plot of Sepal Length for Iris Flowers')
plt.grid(True)
plt.show()


# In[110]:



sns.swarmplot(x='Species', y='PetalLengthCm', data=dataset, palette='viridis', size=5, alpha=0.8)
plt.xlabel('Species')
plt.ylabel('Petal Length (cm)')
plt.title('Swarm Plot of Petal Length for Iris Flowers')
plt.grid(True)
plt.ylim(0, 8)
legend_labels = ['Setosa', 'Versicolor', 'Virginica']
plt.legend(title='Species', labels=legend_labels)
plt.show()


# In[111]:


sns.swarmplot(x='Species', y='SepalLengthCm', data=dataset, palette='viridis', size=6, dodge=True)
plt.xlabel('Species')
plt.ylabel('Sepal Length (cm)')
plt.title('Swarm Plot of Sepal Length for Iris Flowers')
plt.grid(True)
plt.ylim(3, 9)
legend_labels = ['Setosa', 'Versicolor', 'Virginica']
plt.legend(title='Species', labels=legend_labels)
plt.show()


# In[112]:


X=dataset['SepalLengthCm'].values.reshape(-1,1)


# In[113]:


Y=dataset['SepalWidthCm'].values.reshape(-1,1)


# In[116]:


train,test=train_test_split(dataset, test_size = 0.25)
print(train.shape)
print(test.shape)


# In[117]:


train_X =train[['SepalLengthCm','SepalWidthCm',	'PetalLengthCm',	'PetalWidthCm']]
train_y=train.Species
test_X = test[['SepalLengthCm','SepalWidthCm',	'PetalLengthCm',	'PetalWidthCm']]
test_y=test.Species


# In[118]:


from sklearn.preprocessing import StandardScaler
Scaler=StandardScaler()


# In[119]:


Scaler.fit(train_X)


# In[160]:


standardized_data = Scaler.transform(train_X)


# In[121]:


model=LogisticRegression()
model.fit(train_X,train_y)
prediction = model.predict(test_X)
print('Accuracy:', metrics.accuracy_score(prediction,test_y))


# In[167]:


data =dataset.values
X = data[:,0:4]
Y = data[:,4]


# In[168]:


label_encoder = LabelEncoder()
Y = label_encoder.fit_transform(Y)


# In[169]:


# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# In[170]:


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, Y_train)


# In[172]:


# Evaluate the model's accuracy
accuracy = accuracy_score(Y_test, Y_pred)
print("Accuracy:", accuracy)


# In[179]:


from sklearn.neighbors import KNeighborsClassifier
     
# Create a k-NN classifier with k=3
knn = KNeighborsClassifier(n_neighbors=3)

# Train the classifier
knn.fit(X_train, Y_train)

# Make predictions on the test set
Y_pred = knn.predict(X_test)
     


# In[187]:


# Evaluate the accuracy of the classifier
accuracy = accuracy_score(Y_test, Y_pred)
print("Accuracy:", accuracy)
     


# In[188]:


# Split the dataset into features (X) and target variable (Y)
X = dataset[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].values
Y = dataset['Species'].values


# In[189]:


# Encode the categorical target variable
label_encoder = LabelEncoder()
Y = label_encoder.fit_transform(Y)


# In[190]:


# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# In[191]:


# Standardize the features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[192]:


#Train a Logistic Regression model
logistic_regression_model = LogisticRegression(max_iter=1000)
logistic_regression_model.fit(X_train_scaled, Y_train)


# In[193]:


# Make predictions on the test set using Logistic Regression
Y_pred_logistic_regression = logistic_regression_model.predict(X_test_scaled)


# In[194]:


# Evaluate the accuracy of the Logistic Regression model
accuracy_logistic_regression = accuracy_score(Y_test, Y_pred_logistic_regression)
print("Logistic Regression Accuracy:", accuracy_logistic_regression)


# In[195]:


# Train a k-NN classifier with k=3
knn_classifier = KNeighborsClassifier(n_neighbors=3)
knn_classifier.fit(X_train, Y_train)


# In[196]:


# Make predictions on the test set using k-NN
Y_pred_knn = knn_classifier.predict(X_test)


# In[197]:


# Evaluate the accuracy of the k-NN classifier
accuracy_knn = accuracy_score(Y_test, Y_pred_knn)
print("k-NN Accuracy:", accuracy_knn)


# In[ ]:




