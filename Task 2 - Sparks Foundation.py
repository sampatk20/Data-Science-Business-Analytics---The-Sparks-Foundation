#!/usr/bin/env python
# coding: utf-8

# # Prediction Of Unsupervised ML

# In[9]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans


# In[10]:


df = pd.read_csv("iris.csv")
df


# In[11]:


df.describe()


# In[12]:


df.isnull().sum()


# In[13]:


df['Species'].value_counts()


# In[14]:


x = df.iloc[:, 1:-1]
x.head()


# # Finding optimum number of clusters

# In[15]:


cluster_num = list(range(1, 10))
inertia = []
for c in cluster_num:
    model = KMeans(n_clusters = c, init = 'k-means++', n_init = 15, max_iter = 100)
    model.fit(x)
    inertia.append(model.inertia_)


# In[16]:


plt.plot(cluster_num, inertia)
plt.title('The elbow method using inertia')
plt.xlabel('No of Cluster')
plt.ylabel('within-Cluster Sum of Squares')


# In[17]:


model = KMeans(n_clusters = 3, init = 'k-means++', n_init = 20, max_iter = 200).fit(x)


# In[18]:


x['cluster'] = model.labels_
x['cluster'].value_counts(sort = False)


# In[19]:


fig, ax = plt.subplots(figsize = (8,8))
color = ['red', 'blue', 'green']
label = list(range(len(np.unique(x['cluster']))))
for i in range(len(np.unique(x['cluster']))):
    plt.scatter(x.loc[x['cluster'] == i, 'SepalLengthCm'], 
                x.loc[x['cluster'] == i, 'SepalWidthCm'], 
               color = color[i], 
            label = 'cluster {}'.format(i))
plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], color = 'yellow', label = 'centroids')
plt.legend()


# In[ ]:





# In[ ]:




