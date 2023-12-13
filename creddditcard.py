#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from sklearn.linear_model import LogisticRegression


# In[2]:


credit_card_data = pd.read_csv("creditcard.csv")


# In[3]:


credit_card_data


# In[4]:


ccd = credit_card_data


# In[5]:


ccd.info()


# In[6]:


ccd.isnull().sum()


# In[7]:


# distribution of fraudulent and legitimate classes
ccd['Class'].value_counts()


# In[8]:


print((ccd.groupby('Class')['Class'].count()/ccd['Class'].count())*100)
((ccd.groupby('Class')['Class'].count()/ccd['Class'].count())*100).plot.pie()


# In[9]:


classes = ccd['Class'].value_counts()
normal_value = round(classes[0]/ccd['Class'].count()*100,2)
fraud_values = round(classes[1]/ccd['Class'].count()*100,2)
print(normal_value)
print(fraud_values)


# In[10]:


corr = ccd.corr()
corr


# In[11]:


plt.figure(figsize=(27,19))
sns.heatmap(corr, cmap = 'spring', annot= True )
plt.show()


# In[12]:


legit = ccd[ccd.Class == 0]


# In[15]:


fraud = ccd[ccd.Class == 1]


# In[14]:


legit.Amount.describe()


# In[16]:


fraud.Amount.describe()


# In[17]:


ccd.groupby('Class').describe()


# In[18]:


ccd.groupby('Class').mean()


# In[19]:


normal_sample = legit.sample(n=492)


# In[20]:


new_dataset = pd.concat([normal_sample, fraud], axis = 0)
new_dataset


# In[21]:


new_dataset['Class'].value_counts()


# In[22]:


new_dataset.groupby('Class').mean() 


# In[23]:


# here we can drop the time feature and instead use a derived column using timedelta function of pandas to represent the duration that is difference between two time values
delta_time = pd.to_timedelta(new_dataset['Time'], unit = 's')
# create the derived column
new_dataset['time_hour']=(delta_time.dt.components.hours).astype(int)
# now drop the time column
new_dataset.drop(columns='Time', axis=1, inplace = True)
new_dataset


# In[24]:


x = new_dataset.drop('Class', axis=1)


# In[25]:


y = new_dataset['Class']


# In[26]:


x.shape
y.shape


# In[27]:


x.shape


# In[28]:


y.shape


# In[29]:


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.25, random_state = 3, stratify = y)


# In[30]:


#accumulating all the column names under one variable
cols = list(x.columns.values)


# In[31]:


normal_entries = new_dataset.Class==0
fraud_entries = new_dataset.Class==1

plt.figure(figsize=(20,70))
for n, col in enumerate(cols):
    plt.subplot(10,3,n+1)
    sns.histplot(x[col][normal_entries], color='blue', kde = True, stat = 'density')
    sns.histplot(x[col][fraud_entries], color='red', kde = True, stat = 'density')
    plt.title(col, fontsize=17)
plt.show()


# In[32]:


model = LogisticRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_train)
pred_test = model.predict(x_test)


# In[33]:


# creating confusion matrix
from sklearn.metrics import confusion_matrix
def Plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test,pred_test)
    plt.clf()
    plt.show()


# In[34]:


acc_score= round(accuracy_score(y_pred, y_train)*100,2)


# In[35]:


print('the accuracy score for training data of our model is :', acc_score)


# In[36]:


y_pred = model.predict(x_test)
acc_score = round(accuracy_score(y_pred, y_test)*100,2)


# In[37]:


print('the accuracy score of our model is :', acc_score)


# In[38]:


from sklearn import metrics


# In[39]:


score = round(model.score(x_test, y_test)*100,2)
print('score of our model is :', score)


# In[40]:


class_report = classification_report(y_pred, y_test)
print('classification report of our model: ', class_report)


# In[ ]:




