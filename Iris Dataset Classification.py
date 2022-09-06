#!/usr/bin/env python
# coding: utf-8

# In[44]:


import pandas as pd
import mglearn
import numpy as np


# In[45]:


from sklearn.datasets import load_iris


# In[46]:


iris_dataset = load_iris()


# In[62]:


df = pd.DataFrame(iris_dataset.data, columns = iris_dataset.feature_names)
df['species'] = iris_dataset.target


# ### Viewing Dataset

# In[64]:


df


# ### Describing Dataset

# In[49]:


print("Keys of iris_dataset: \n{}".format(iris_dataset.keys()))


# In[7]:


print(iris_dataset['DESCR'] + "\n...")
print(iris_dataset['target_names'])


# In[51]:


print("Shape of data:",iris_dataset['data'].shape)


# ### Spliting The Dataset Into Training & Testing Data

# In[52]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
 iris_dataset['data'], iris_dataset['target'], random_state=0)


# In[53]:


print("X_train shape: {}".format(X_train.shape))
print("y_train shape: {}".format(y_train.shape))


# In[54]:


print("X_test shape: {}".format(X_test.shape))
print("y_test shape: {}".format(y_test.shape))


# In[72]:


# create dataframe from data in X_train
# label the columns using the strings in iris_dataset.feature_names
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
# create a scatter matrix from the dataframe, color by y_train
pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15),
                           marker='o', hist_kwds={'bins': 20}, s=50,
                           alpha=.8, cmap=mglearn.cm3)


# In[56]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=2)


# ### Making Predictions 

# In[38]:


knn.fit(X_train, y_train)


# In[39]:


X_new = np.array([[5.8, 2.7, 5.1, 1.9]])
print("X_new.shape:", X_new.shape)


# In[40]:


prediction = knn.predict(X_new)
print("Prediction:", prediction)
print("Predicted target name:",
       iris_dataset['target_names'][prediction])


# ### Model Evaluation

# In[41]:


y_pred = knn.predict(X_test)
print("Test set predictions:\n", y_pred)


# In[42]:


print("Test set score: {:.2f}".format(np.mean(y_pred == y_test)))


# In[43]:


print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))


# ### Summary & Outcome

# In[81]:


X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state=2)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))

