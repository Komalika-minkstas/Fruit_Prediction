
# coding: utf-8

# In[3]:

get_ipython().magic('matplotlib notebook')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

fruits=pd.read_table('fruit_data_with_colors.txt')


# In[4]:

fruits.head()


# In[5]:

fruits


# In[4]:

#goal is to make the classifier which can distinguish between fruits on the basis of this information
#for creating training and test sets scikit learn provides function train_test_split()


# In[6]:

#Cretate Train-Test split
X=fruits[['mass','width','height']]
y=fruits['fruit_label']

X_train,X_test,y_train,y_test=train_test_split(X, y, random_state=0)


# In[7]:

fruits.shape


# In[8]:

X_train.shape


# In[9]:

X_test.shape


# In[10]:

y_train.shape


# In[11]:

y_test.shape


# In[12]:

X_train


# In[13]:

X_test


# In[14]:

y_train


# In[15]:

y_test


# In[17]:

import pandas as pd
from matplotlib import cm
cmap=cm.get_cmap('gnuplot')
scatter=pd.scatter_matrix(X_train, c=y_train, marker='o', s=40, hist_kwds={'bins':15}, figsize=(12,12),cmap=cmap)


# In[18]:

#import required modules and load data file
get_ipython().magic('matplotlib notebook')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

fruits=pd.read_table('fruit_data_with_colors.txt')


# In[19]:

fruits.head()


# In[22]:

#defining a dictionary
lookup_fruit_name=dict(zip(fruits.fruit_label.unique(),fruits.fruit_name.unique()))
lookup_fruit_name


# In[23]:

#Create train-test split
X=fruits[['mass','width','height']]
y=fruits['fruit_label']

X_train, X_test, y_train, y_test=train_test_split(X, y, random_state=0)


# In[24]:

#Create classifier object
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=5)


# In[25]:

#Train the classifier(fit the estimator)using the training data
knn.fit(X_train, y_train)


# In[26]:

#estimate theaccuracy of the classifier on future data using the test data
knn.score(X_test, y_test)


# In[27]:

#Use the trained Knn classifier model to classify the new previously unseen objects
fruit_prediction=knn.predict([[20,4.3,5.5]])
lookup_fruit_name[fruit_prediction[0]]


# In[28]:

fruit_prediction=knn.predict([[100,6.3,8.5]])
lookup_fruit_name[fruit_prediction[0]]


# In[29]:

#Plot the decision boundaries of the K-NN classifier
from adspy_shared_utilities import plot_fruit_knn
plot_fruit_knn(X_train,y_train,5,'uniform')


# In[33]:

#How sensitive is k-NN classification accuracy to the choice of the 'k' paramete
k_range=range(1,20)
scores=[]
for k in k_range:
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    scores.append(knn.score(X_test, y_test))
plt.figure()
plt.xlabel('k')
plt.ylabel('accuracy')
plt.scatter(k_range,scores)
plt.xticks([0,5,10,15,20]);


# In[ ]:



