#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
wine=pd.read_csv("wine.csv")
wine.head()


# In[3]:


wine.tail()


# In[4]:


wine.info()


# In[5]:


wine.describe()


# In[6]:


wine.shape


# In[7]:


wine.isnull().any(axis=1)


# In[8]:


wine.isnull().sum()


# In[9]:


from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale 


# In[10]:


wine=wine.iloc[:,1:]
wine.head()


# In[11]:


wine.shape


# In[12]:


#normalization
wine_norm=scale(wine)
wine_norm


# In[55]:


pca = PCA(n_components = 3)
pca_values = pca.fit_transform(wine_norm)


# In[56]:


#Variance
var = pca.explained_variance_ratio_
var


# In[57]:


pca.components_


# In[58]:


pca.components_[0]


# In[59]:


#Cumulative Variance
import numpy as np
cumvar = np.cumsum(np.round(var,decimals = 4)*100)
cumvar


# In[60]:


#Variance plot
plt.plot(cumvar,color="red")


# In[100]:


#plot between PCA1 and PCA2
x = pca_values[:,0]
y = pca_values[:,1]
z = pca_values[:2:3]
#color=["red","blue"]
plt.scatter(x,y)


# In[102]:


finalDf = pd.concat([pd.DataFrame(pca_values[:,0:2],columns=['pc1','pc2'],wine)], axis = 1)


# In[104]:


import seaborn as sns
sns.scatterplot(data=finalDf,x='pc1',y='pc2')


# In[63]:


# Clustering 
new_df = pd.DataFrame(pca_values[:,0:3])

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters = 3)
kmeans.fit(new_df)
kmeans.labels_


# # To find the no. of clusters using elbow curve for k-means

# In[64]:


import pandas as pd
import matplotlib.pylab as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist 
import numpy as np


# In[65]:


new_df = pd.DataFrame(pca_values[:,0:4])


# In[66]:


def norm(i):
    x=(i-(i.min())/(i.max()-(i.min())))
    return x
df_norm=norm(new_df.iloc[:,:])


# In[67]:


df_norm


# In[68]:


model=KMeans().fit(df_norm)


# In[69]:


model.labels_


# In[70]:


model.cluster_centers_


# In[71]:


model.inertia_


# In[72]:


model.n_iter_


# In[73]:


import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.cluster import KMeans


# In[74]:


kmeans_kwargs = {  "init": "random","n_init": 10,"max_iter": 300,"random_state": 42}
sse = []
for k in range(1, 15):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(df_norm)
    sse.append(kmeans.inertia_)


# In[75]:


plt.style.use("fivethirtyeight")
plt.plot(range(1, 15), sse)
plt.xticks(range(1, 15))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.show()


# In[76]:


kl = KneeLocator( range(1, 15), sse, curve="convex", direction="decreasing")
kl.elbow


# In[92]:


model1=KMeans(n_clusters=3).fit(df_norm)


# In[93]:


model1.labels_


# In[94]:


a=pd.Series(model1.labels_)


# In[95]:


wine['Cluster_kmeans']=a


# In[96]:


wine.head()


# In[97]:


import seaborn as sns
sns.pairplot(wine,hue="Cluster_kmeans")


# # To fond the no. of clusters using hierarchical clustering

# In[78]:


from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage 
import scipy.cluster.hierarchy as sch # for creating dendrogram 


# In[79]:


type(df_norm)


# In[80]:


help(linkage)
z = linkage(df_norm, method="complete",metric="euclidean")


# In[81]:


plt.figure(figsize=(15, 5))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Index')
plt.ylabel('Distance')
sch.dendrogram(
    z,
    leaf_rotation=0.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()


# In[84]:


h_complete	=	AgglomerativeClustering(n_clusters=3,linkage='complete',affinity = "euclidean").fit(df_norm) 


# In[85]:


h_complete.labels_


# In[86]:


cluster_labels=pd.Series(h_complete.labels_)


# In[87]:


wine['Cluster_hier']=cluster_labels


# In[88]:


wine.head()


# In[89]:


wine['Cluster_hier'].unique()


# In[90]:


wine['Cluster_hier'].value_counts()


# In[91]:


import seaborn as sns
sns.pairplot(wine,hue="Cluster_hier")


# In[ ]:




