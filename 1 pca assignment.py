# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 15:49:25 2023

@author: kailas
"""

BUSINESS OBJECTIVE:-
Perform Principal component analysis and perform clustering using first




###########################################################################
1]HIERACHICAL CLUSTERING

#Import Liabrary
import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pylab as plt
import scipy
from scipy import stats
import pylab

#Dataset

data=pd.read_csv("D:/data science assignment/Assignments/8.pca/wine.csv")

#EDA
data.info()
data.shape
data.head()
data.tail
data.describe()
#checking for null value
data.isna().sum()

#Normalization
def norm_func(i):
    x = (i - i.min()) / (i.max() - i.min())
    return(x)

#Normalized the DataFrame(df).Considering only numerical part of data
data1=norm_func(data.iloc[:,:])

#For Creating Dendrogram
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch

z=linkage(data1,method='complete')

#Dendrogram

plt.figure(figsize=(15,8));plt.title('Hierarchial clustering dendrogram');plt.xlabel('index');plt.ylabel('distance')
sch.dendrogram(z,
               leaf_rotation=0,
               leaf_font_size = 10 
)
plt.show()

#Now applying Agglomerative Clustering Choosing 5 Cluster.
from sklearn.cluster import AgglomerativeClustering

h=AgglomerativeClustering(n_clusters=5,linkage='complete')               
h.fit(data1)
h.labels_

# creating a new column and assigning it to new column
data['clust']=pd.Series(h.labels_) 

#Putting Clust column as 1st position
data=data.iloc[:,[14,13,12,11,10,9,8,7,6,5,4,3,2,1,0]]

#Creating a CSV file
data.to_csv('winehierarchy.csv',encoding='utf-8')
import os
os.getcwd()


###########################################################################################
2]K-MEANS CLUSTERING


          
#importLiabrary.
from sklearn.cluster import KMeans          

#datset
data=pd.read_csv("D:/data science assignment/Assignments/8.pca/wine.csv")          
          
#EDA
data.info()
data.shape
data.head()
data.tail
data.describe()
#checking for null value
data.isna().sum()

#Normalization
def norm_func(i):
    x = (i - i.min()) / (i.max() - i.min())
    return(x)

#Normalized the DataFrame(df).Considering only numerical part of data
data1=norm_func(data.iloc[:,:])

#Scree plot or Elbow curve
TWSS=[]
k= list(range(10,30))

for i in k:
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(data1)
    TWSS.append(kmeans.inertia_)
    
TWSS    


#Scree Plot(for Visulization)
plt.plot(k,TWSS,'ro-');plt.title('scree plot');plt.xlabel('no of clusters');plt.ylabel('total within sum of square')

#we Tune the model more and more time with different number of cluster and finally we choose 12 Clusters is optimum.

model=KMeans(n_clusters=12)
model.fit(data1)

model.labels_

data['clust']=pd.Series(model.labels_)# creating a new column and assigning it to new column 

data=data.iloc[:,[14,13,12,11,10,9,8,7,6,5,4,3,2,1,0]]#Putting Clust column as 1st position

#Creating a CSV file
data.to_csv('winekmeans.csv',encoding='utf-8')
import os
os.getcwd()


#############################################################################
3]PCA

            
#Import Liabrary
from sklearn.decomposition import PCA            
from sklearn.preprocessing import scale

#Datset

data=pd.read_csv("D:/data science assignment/Assignments/8.pca/wine.csv")

#EDA
data.info()
data.shape
data.head()
data.tail
data.describe()
#checking for null value
data.isna().sum()

#Nomalizing(only numerical part of data)
data1=scale(data.iloc[:,:])

#Apply the PCA model
pca = PCA(n_components=14)
pca_values=pca.fit_transform(data1)

#The amount of Variance that each pca explains is
var=pca.explained_variance_ratio_
var

#PCA weights
pca.components_
pca.components_[0]

#Commulative Varience

var1=np.cumsum(np.round(var,decimals=4) * 100)
var1

#PCA Score
pca_values

pca_data=pd.DataFrame(pca_values)#Convert array into DataFrame

#put the column name
pca_data.columns='comp0','comp1','comp2','comp3','comp4','comp5','comp6','comp7','comp8','comp9','comp10','comp11','comp12','comp13'


#We select First 5 column,beacuse it Captures 80% variance.. 
#Final we ready with pca values with First 5 pca values.
final = pca_data.iloc[:,0:5]

#Scatter diagram(for 1st two pca values visulaization)
ax = final.plot(x='comp0', y='comp1', kind='scatter',figsize=(12,8))



CONCLUSION == AT first there are 14 column,using PCA we reduce it to only 5 Columns,
and we see that using these 5 column we Cpature 80% information..so we succesfuly apply PCA. 
with reduce the dimension





#################################################################################

Now using this PCA values..do again HIEARCHICAL and KMEANS Clustering and identify optimum no of clusters...


4]HIERACHICAL CLUSTERING


       
using this 'final' dataframe.we are going to apply Hierarchial clustering.

#EDA
final.info()       
final.shape
final.head()
final.tail()
final.info()
final.isna().sum()

#Normalization
def norm_func(i):
    x = (i - i.min()) / (i.max() - i.min())
    return(x)

#Normalized the DataFrame(data).Considering only numerical part of data
final1=norm_func(final.iloc[:,:])


#For Creating Dendrogram
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch

z=linkage(final1,method='complete')

#Dendrogram
plt.figure(figsize=(15,8));plt.title('Hierarchial clustering dendrogram');plt.xlabel('index');plt.ylabel('distance')
sch.dendrogram(z,
               leaf_rotation=0,# rotates the x axis labels
               leaf_font_size = 10 # font size for the x axis labels
)
plt.show()

#Now applying Agglomerative Clustering Choosing 3 Cluster.
from sklearn.cluster import AgglomerativeClustering

h=AgglomerativeClustering(n_clusters=3,linkage='complete')
h.fit(final1)
h.labels_

final['clust']=pd.Series(h.labels_)# creating a new column and assigning it to new column 
data['clust']=pd.Series(h.labels_)

data=data.iloc[:,[14,13,12,11,10,9,8,7,6,5,4,3,2,1,0]]#Putting Clust column as 1st position

#Creating a CSV file
data.to_csv('winehierarchypca.csv',encoding='utf-8')
import os
os.getcwd()
          

    

##################################################################################
5]KMEANS CLUSTERING



using this 'final' dataframe.we are going to apply KMEANS clustering.

#EDA
final.info()       
final.shape
final.head()
final.tail()
final.info()
final.isna().sum()



#Normalization
def norm_func(i):
    x = (i - i.min()) / (i.max() - i.min())
    return(x)

#Normalized the DataFrame(data).Considering only numerical part of data
final1=norm_func(final.iloc[:,:])

#Scree plot or Elbow curve
TWSS=[]
k= list(range(2,9))

for i in k:
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(final1)
    TWSS.append(kmeans.inertia_)
    
TWSS    

#Scree Plot(for Visulization)
plt.plot(k,TWSS,'ro-');plt.title('scree plot');plt.xlabel('no of clusters');plt.ylabel('total within sum of square')

#we Tune the model more and more time with different number of cluster and finally we choose 3 Clusters is optimum.

model=KMeans(n_clusters=3)
model.fit(data1)

model.labels_

final['clust1'] =pd.Series(model.labels_)
data['clust']=pd.Series(model.labels_)# creating a new column and assigning it to new column 

data=data.iloc[:,[14,13,12,11,10,9,8,7,6,5,4,3,2,1,0]]#Putting Clust column as 1st position

#Creating a CSV file
data.to_csv('winekmeanspca.csv',encoding='utf-8')
import os
os.getcwd()




CONCLUSION= We observe in both HIEARCHICAL AND K-MEANS..
before PCA we need 5 and 12 clusters respectively...
and after doing PCA we only neeed 3 clusters in both....
