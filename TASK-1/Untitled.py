#!/usr/bin/env python
# coding: utf-8

# # UNEMPLOYMENT ANALYSIS WITH PYTHON

# Unemployment is measured by the unemployment rate which is the number of people
# who are unemployed as a percentage of the total labour force. We have seen a sharp
# increase in the unemployment rate during Covid-19, so analyzing the unemployment rate
# can be a good data science project. 
# 
# 

# In[139]:


#import Libiries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[140]:


# load Dataset
unemployment_data = pd.read_csv('Unemployment in India.csv')
unemployment_rate_data = pd.read_csv('Unemployment_Rate_upto_11_2020.csv')


# In[141]:


#look dataset
unemployment_data.head(15)


# In[142]:


unemployment_data.columns


# In[143]:


#look dataset
unemployment_rate_data.head(15)


# In[144]:


#remove space form both dataset
unemployment_rate_data =unemployment_rate_data.rename(columns= lambda x:x.strip()) 


# In[145]:


unemployment_data =unemployment_data.rename(columns= lambda x:x.strip())


# In[146]:


unemployment_rate_data.info()


# In[147]:


unemployment_rate_data.describe()


# In[148]:


unemployment_rate_data.shape


# In[149]:


unemployment_data.columns


# In[150]:


x = unemployment_data['Estimated Unemployment Rate (%)'].mean()
print("average unemployment rate:",x )


# In[151]:


y = unemployment_data['Estimated Employed'].sum()
print("total number of employed individuals:", y)


# In[152]:


z= unemployment_data['Estimated Labour Participation Rate (%)'].mean()
print("labor participation rate:", z)


# #  Handling Missing Value
# ## unemployment_data dataset

# In[153]:


unemployment_data.isnull().sum()


# In[154]:


unemployment_data['Region']= unemployment_data['Region'].fillna(unemployment_data['Region'].mode()[0])


# In[155]:


#replace space of column
unemployment_data.columns=unemployment_data.columns.str.replace(' ' ,'')


# In[156]:


#replace missing value with mode
unemployment_data['Date']=unemployment_data['Date'].fillna(unemployment_data['Date'].mode()[0])


# In[157]:


unemployment_data['Frequency']= unemployment_data['Frequency'].fillna('Monthly')


# In[158]:


unemployment_data['Frequency']=unemployment_data['Frequency'].str.replace(' ' ,'')


# In[159]:


unemployment_data['EstimatedUnemploymentRate(%)']= unemployment_data['EstimatedUnemploymentRate(%)'].fillna(unemployment_data['EstimatedUnemploymentRate(%)'].mode()[0])


# In[160]:


unemployment_data['EstimatedEmployed'] =unemployment_data['EstimatedEmployed'].fillna(unemployment_data['EstimatedEmployed'].mode()[0])


# In[161]:


unemployment_data['EstimatedLabourParticipationRate(%)']=unemployment_data['EstimatedLabourParticipationRate(%)'].fillna(unemployment_data['EstimatedLabourParticipationRate(%)'].mode()[0])


# In[162]:


unemployment_data['Area']=unemployment_data['Area'].fillna(unemployment_data['Area'].mode()[0])


# In[191]:


unemployment_rate_data


# # average unemployment rate based on region and date: 

# In[192]:


grouped_data = unemployment_rate_data.groupby(['Region', 'Date'])
average_unemployment = grouped_data['Estimated Unemployment Rate (%)'].mean()
average_employment = grouped_data['Estimated Employed'].mean()
average_participation = grouped_data['Estimated Labour Participation Rate (%)'].mean()

plt.figure(figsize=(12, 8))
for region, data in average_unemployment.groupby('Region'):
    plt.plot(data.index.get_level_values('Date'), data.values , label=region)
plt.xlabel('Date')
plt.ylabel('Average Unemployment Rate')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()


# In[197]:


average_unemployment = unemployment_rate_data.groupby('Region')['Estimated Unemployment Rate (%)'].mean()
plt.figure(figsize=(10, 6))
average_unemployment.plot(kind='bar')
plt.title('Average Unemployment Rate by Region/Area')
plt.xlabel('Region/Area')
plt.ylabel('Unemployment Rate (%)')
plt.show()


# # Observations:
# 1. The region with the highest average unemployment rate is Haryana <br>
# 2. The region with the second-highest average unemployment rate is Tripura <br>

# In[208]:


unemployment_rate_data['Date'] = pd.to_datetime(unemployment_rate_data['Date'])
average_unemployment = unemployment_rate_data.groupby('Date')['Estimated Unemployment Rate (%)'].mean()
plt.figure(figsize=(10, 6))
average_unemployment.plot(kind='bar')
plt.title('Average Unemployment Rate over Time')
plt.xlabel('Date')
plt.ylabel('Unemployment Rate (%)')
plt.show()


# In[211]:


selected_date = '2020-01-31'
filtered_data = unemployment_rate_data[unemployment_rate_data['Date'] == selected_date]  # Filtering based on a specific date
plt.figure(figsize=(10, 6))
plt.bar(filtered_data['Region'], filtered_data['Estimated Unemployment Rate (%)'])
plt.title(f'Unemployment Rates on {selected_date}')
plt.xlabel('Region')
plt.ylabel('Unemployment Rate (%)')
plt.xticks(rotation=90)
plt.show()


# In[212]:


selected_date = '2020-04-30'
filtered_data = unemployment_rate_data[unemployment_rate_data['Date'] == selected_date]  # Filtering based on a specific date
plt.figure(figsize=(10, 6))
plt.bar(filtered_data['Region'], filtered_data['Estimated Unemployment Rate (%)'])
plt.title(f'Unemployment Rates on {selected_date}')
plt.xlabel('Region')
plt.ylabel('Unemployment Rate (%)')
plt.xticks(rotation=90)
plt.show()


# In[199]:


plt.figure(figsize=(10, 6))
colors = ["#FF7F50", "#6495ED"]
sns.barplot(x="Estimated Unemployment Rate (%)", y='Region', data=unemployment_rate_data, palette= colors)
plt.title("Unemployment Rate by Region")
plt.show()


# In[ ]:





# In[ ]:





# # Observation: 
# 1:Haryana have the highest estimated unemployment rate <br>
# 2: Tripura has the second-highest estimated unemployment rate. <br>
# 3: Bihar and jharkhand has the third-highest estimated unemployment rate,which is the same for both regions <br>

# In[165]:


unemployment_rate_data.columns


# In[182]:


import seaborn as sns
import matplotlib.pyplot as plt

# Set the plot size
plt.figure(figsize=(10, 6))
#set color
colors = ["#FF7F50", "#6495ED"] 
# Create the bar plot with custom color palette
sns.barplot(x="Estimated Employed", y='Region', data=unemployment_rate_data, palette=colors)
# Set the plot title
plt.title('Estimated Employed by Region')
# Display the plot
plt.show()


# # Observation
# 1: Uttar Pradesh has the highest estimated unemployment rate. <br>
# 2: Maharashtra has the second-highest estimated unemployment rate.<br>
# 3: West Bengal has the third-highest estimated unemployment rate.<br>

# In[167]:


plt.figure(figsize=(10, 6))
colors = ["#FF7F50", "#6495ED"]
sns.barplot(x="Estimated Labour Participation Rate (%)", y= 'Region' , data=unemployment_rate_data , palette=colors )
plt.title("Labor Participation Rate by region")


# # Obervation:
# 1: Meghalaya  has the highest Labor Participation Rate among regions.  <br>
# 2: Tripura has the second-highest Labor Participation Rate among regions. <br>
# 3: Telangana has the third-highest Labor Participation Rate among regions. <br>

# In[168]:



unemployment_rate_data['Date'] = pd.to_datetime(unemployment_rate_data['Date'])
plt.figure(figsize=(12, 6))
sns.lineplot(x='Date', y='Estimated Unemployment Rate (%)', data=unemployment_rate_data, label='Unemployment Rate')
sns.lineplot(x='Date', y='Estimated Labour Participation Rate (%)', data=unemployment_rate_data, label='Labor Participation Rate')

plt.title('Trends of Unemployment Rate and Labor Participation Rate')
plt.xlabel('Date')
plt.ylabel('Percentage')
plt.legend()

plt.show()


# In[169]:


column = ['Estimated Unemployment Rate (%)','Estimated Employed','Estimated Labour Participation Rate (%)'] 


# In[170]:


column_corr= unemployment_rate_data[column].corr()     


# In[171]:


sns.heatmap(data=column_corr ,annot=True,cmap='Greens' )
plt.title('Correlation Matrix')
plt.show()


# In[172]:


unemployment_rate_data =unemployment_rate_data.rename(columns={'Region.1': "Regionx"})


# In[173]:



region_counts = unemployment_rate_data['Regionx'].value_counts()
# Create a pie chart
plt.figure(figsize=(8, 8))
plt.pie(region_counts, labels=region_counts.index, autopct='%1.1f%%' )
plt.title('Partition of Regions')
plt.axis('equal')
plt.show()


# In[174]:


total_employed = unemployment_rate_data['Estimated Employed'].sum()
employment_by_region = unemployment_rate_data.groupby('Region')['Estimated Employed'].sum()
proportion_by_region = employment_by_region / total_employed
threshold = 0.02  # Exclude regions with less than 2% employment
filtered_proportions = proportion_by_region[proportion_by_region >= threshold]
excluded_proportion = proportion_by_region[proportion_by_region < threshold].sum()
filtered_proportions['Other'] = excluded_proportion
plt.figure(figsize=(12, 10))
plt.pie(filtered_proportions, labels=filtered_proportions.index, autopct='%1.1f%%')
plt.title('Proportion of Employment by Region')
plt.axis('equal')
plt.tight_layout()
plt.show()
correlation = unemployment_rate_data['Estimated Unemployment Rate (%)'].corr(unemployment_rate_data['Estimated Employed'])
print('Correlation between Unemployment Rate and Employed Individuals:', correlation)


# In[ ]:




