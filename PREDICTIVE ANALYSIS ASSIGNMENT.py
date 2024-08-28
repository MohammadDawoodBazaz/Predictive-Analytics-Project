#!/usr/bin/env python
# coding: utf-8

# # Importing libraries 

# In[135]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# # Importing csv file

# In[111]:


pre_assg= pd.read_csv('/Users/daoudbazaz/Desktop/project pa.csv')


# # Dataset Description

# In[112]:


#Showing the dataset's first six values

pre_assg.head(6)


# In[113]:


# Presenting the dataset's information.

print(pre_assg.info())


# In[114]:


#statistical discription of dataset
pre_assg.describe()


# ## Data Preparation (Cleaning)

# In[115]:


##Finding Null Values in the Dataset.

print(pre_assg.isnull().sum())


# In[116]:


#Managing missing values 'Age' column by assigning the median value to the missing values

median_age = pre_assg['Age'].median()
pre_assg['Age'].fillna(median_age, inplace=True)


# In[117]:


#locating and eliminating the number 0 that is present in the "Fare" column.

(pre_assg['Fare']==0).sum()
pre_assg=pre_assg[pre_assg['Fare']!=0]


# In[118]:


# Transforming the data: Converting character-type data to Boolean.
# In the 'Gender' column, where 'Male' is represented as 0 and 'Female' is represented as 1.

pre_assg = pd.get_dummies(pre_assg, columns=['Gender'], drop_first=True)


# In[119]:


#Result from conversion to Boolean.
pre_assg.head()


# In[120]:


#Finding if outliers is present in the dataset.
sns.boxplot(y=pre_assg['Age'], boxprops=dict(facecolor="skyblue"))
plt.show()
sns.boxplot(y=pre_assg['Fare'],boxprops=dict(facecolor="lightsalmon"))
plt.show()


# In[121]:


# Removing Outliers from dataset
def handle_outliers(column):
    Q1 = pre_assg[column].quantile(0.25)
    Q3 = pre_assg[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    pre_assg[column] = np.where(pre_assg[column] < lower_bound, lower_bound, pre_assg[column])
    pre_assg[column] = np.where(pre_assg[column] > upper_bound, upper_bound, pre_assg[column])
    
handle_outliers('Age')
handle_outliers('Fare')


# In[122]:


#Cross checking if outliers have been removed

# Boxplot for 'Age' with light green color
sns.boxplot(y=pre_assg['Age'], boxprops=dict(facecolor="skyblue"))
plt.show()


# Boxplot for 'Fare' with light salmon color
sns.boxplot(y=pre_assg['Fare'], boxprops=dict(facecolor="lightsalmon"))
plt.show()


# ## Predictive Analysis

# In[123]:


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# In[124]:


# Dividing the dataset into features (X) and target variable (y)
X = pre_assg.drop('Survived', axis=1)
y = pre_assg['Survived']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=60)


# In[125]:


# Random Forest Classifier Model
rf_model = RandomForestClassifier(random_state=60)

# Training the model on the training set
rf_model.fit(X_train, y_train)

# Making predictions on the test set
rf_predictions = rf_model.predict(X_test)


# In[126]:


# Logistic Regression
lr_model = LogisticRegression(random_state=60)

# Training the model on the training set
lr_model.fit(X_train, y_train)

# Making predictions on the test set
lr_predictions = lr_model.predict(X_test)


# In[127]:


# Support Vector Machine
svm_model = SVC(random_state=60)

# Training the model on the training set
svm_model.fit(X_train, y_train)

# Making predictions on the test set
svm_predictions = svm_model.predict(X_test)


# In[128]:


# Evaluation Metrics Function
def evaluate_model(model, predictions):
    # Calculating and printing accuracy
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy:.4f}")
    
    # Printing classification report
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    # Printing confusion matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))


# In[129]:


# Evaluate Random Forest model
print("Random Forest Model:")
evaluate_model(rf_model, rf_predictions)


# In[130]:


# Evaluate Logistic Regression model
print("\nLogistic Regression Model:")
evaluate_model(lr_model, lr_predictions)


# In[131]:


# Evaluate Support Vector Machine model
print("\nSupport Vector Machine Model:")
evaluate_model(svm_model, svm_predictions)


# # Visualisation

# In[134]:


# Assuming you have already trained and evaluated the models
accuracy_rf = accuracy_score(y_test, rf_predictions)
accuracy_lr = accuracy_score(y_test, lr_predictions)
accuracy_svm = accuracy_score(y_test, svm_predictions)

predict_model = ['Random Forest', 'Logistic Regression', 'Support Vector Machine']
scores = [accuracy_rf, accuracy_lr, accuracy_svm]

# Use a different set of colors for better visibility
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

plt.bar(predict_model, scores, color=colors)
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison of Prediction Models')
plt.ylim(0, 1)  # Assuming accuracy values are between 0 and 1

plt.show()



# In[ ]:




