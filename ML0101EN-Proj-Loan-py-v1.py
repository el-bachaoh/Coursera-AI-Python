#!/usr/bin/env python
# coding: utf-8

# <a href="https://www.bigdatauniversity.com"><img src="https://ibm.box.com/shared/static/cw2c7r3o20w9zn8gkecaeyjhgw3xdgbj.png" width="400" align="center"></a>
# 
# <h1 align="center"><font size="5">Classification with Python</font></h1>

# In this notebook we try to practice all the classification algorithms that we learned in this course.
# 
# We load a dataset using Pandas library, and apply the following algorithms, and find the best one for this specific dataset by accuracy evaluation methods.
# 
# Lets first load required libraries:

# In[165]:


import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing
get_ipython().run_line_magic('matplotlib', 'inline')


# ### About dataset

# This dataset is about past loans. The __Loan_train.csv__ data set includes details of 346 customers whose loan are already paid off or defaulted. It includes following fields:
# 
# | Field          | Description                                                                           |
# |----------------|---------------------------------------------------------------------------------------|
# | Loan_status    | Whether a loan is paid off on in collection                                           |
# | Principal      | Basic principal loan amount at the                                                    |
# | Terms          | Origination terms which can be weekly (7 days), biweekly, and monthly payoff schedule |
# | Effective_date | When the loan got originated and took effects                                         |
# | Due_date       | Since it’s one-time payoff schedule, each loan has one single due date                |
# | Age            | Age of applicant                                                                      |
# | Education      | Education of applicant                                                                |
# | Gender         | The gender of applicant                                                               |

# Lets download the dataset

# In[5]:


get_ipython().system('wget -O loan_train.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_train.csv')


# ### Load Data From CSV File  

# In[166]:


df = pd.read_csv(r"C:\Users\el-bachaoh\Downloads\loan_train.csv")
df.head(50)
df.describe()


# In[167]:


df.shape


# ### Convert to date time object 

# In[168]:


df['due_date'] = pd.to_datetime(df['due_date'])
df['effective_date'] = pd.to_datetime(df['effective_date'])
df.head()


# # Data visualization and pre-processing
# 
# 

# Let’s see how many of each class is in our data set 

# In[169]:


df['loan_status'].value_counts()


# 260 people have paid off the loan on time while 86 have gone into collection 
# 

# Lets plot some columns to underestand data better:

# In[17]:


# notice: installing seaborn might takes a few minutes
get_ipython().system('conda install -c anaconda seaborn -y')
#Seaborn is available on my local machine


# In[170]:


import seaborn as sns

bins = np.linspace(df.Principal.min(), df.Principal.max(), 20)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'Principal', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()


# In[171]:


bins = np.linspace(df.age.min(), df.age.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set2", col_wrap=2)
g.map(plt.hist, 'age', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()


# # Pre-processing:  Feature selection/extraction

# ### Lets look at the day of the week people get the loan 

# In[172]:


df['dayofweek'] = df['effective_date'].dt.dayofweek
bins = np.linspace(df.dayofweek.min(), df.dayofweek.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'dayofweek', bins=bins, ec="w")
g.axes[-1].legend()
plt.show()


# We see that people who get the loan at the end of the week dont pay it off, so lets use Feature binarization to set a threshold values less then day 4 

# In[173]:


df['weekend'] = df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)
df.head()


# ## Convert Categorical features to numerical values

# Lets look at gender:

# In[174]:


df.groupby(['Gender'])['loan_status'].value_counts(normalize=True)


# 86 % of female pay there loans while only 73 % of males pay there loan
# 

# Lets convert male to 0 and female to 1:
# 

# In[175]:


df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)
df.head()


# ## One Hot Encoding  
# #### How about education?

# In[176]:


df.groupby(['education'])['loan_status'].value_counts(normalize=True)


# #### Feature befor One Hot Encoding

# In[177]:


df[['Principal','terms','age','Gender','education']].head()


# #### Use one hot encoding technique to conver categorical varables to binary variables and append them to the feature Data Frame 

# In[178]:


Feature = df[['Principal','terms','age','Gender','weekend']]
Feature = pd.concat([Feature,pd.get_dummies(df['education'])], axis=1)
Feature.drop(['Master or Above'], axis = 1,inplace=True)
Feature.head()


# ### Feature selection

# Lets defind feature sets, X:

# In[179]:


X = Feature
X[0:5]


# What are our lables?

# In[195]:


df['loan_status']=df['loan_status'].replace(to_replace=['PAIDOFF' , 'COLLECTION'],value=[1,0])
y = df['loan_status'].values
y[0:10]


# ## Normalize Data 

# Data Standardization give data zero mean and unit variance (technically should be done after train test split )

# In[181]:


X= preprocessing.StandardScaler().fit_transform(X)
X[0:5]


# # Classification 

# Now, it is your turn, use the training set to build an accurate model. Then use the test set to report the accuracy of the model
# You should use the following algorithm:
# - K Nearest Neighbor(KNN)
# - Decision Tree
# - Support Vector Machine
# - Logistic Regression
# 
# 
# 
# __ Notice:__ 
# - You can go above and change the pre-processing, feature selection, feature-extraction, and so on, to make a better model.
# - You should use either scikit-learn, Scipy or Numpy libraries for developing the classification algorithms.
# - You should include the code of the algorithm in the following cells.

# # K Nearest Neighbor(KNN)
# Notice: You should find the best k to build the model with the best accuracy.  
# **warning:** You should not use the __loan_test.csv__ for finding the best k, however, you can split your train_loan.csv into train and test to find the best __k__.

# In[197]:


from sklearn.neighbors import KNeighborsClassifier
k=2
neigh=KNeighborsClassifier(n_neighbors=k).fit(X,y)
neigh


# In[ ]:





# In[ ]:





# # Decision Tree

# In[204]:


from sklearn.tree import DecisionTreeClassifier
Tree = DecisionTreeClassifier(criterion="entropy", max_depth = 5)
Tree.fit(X,y)
predTree = Tree.predict(X_test)
print (predTree [0:5])
print (y_test [0:5])


# In[ ]:





# In[ ]:





# # Support Vector Machine

# In[206]:


from sklearn import svm
clf = svm.SVC(kernel='rbf')
clf.fit(X, y)


# In[208]:


SV_Pred = clf.predict(X_test)
SV_Pred [0:5]


# In[ ]:





# # Logistic Regression

# In[210]:


from sklearn.linear_model import LogisticRegression
LR = LogisticRegression(C=0.01, solver='newton-cg').fit(X,y)
LR


# In[211]:


ypredict=LR.predict(X_test)
ypredict


# In[217]:


from sklearn.linear_model import LogisticRegression
LRlib = LogisticRegression(C=0.01, solver='liblinear').fit(X,y)
ypredictLib=LR.predict(X_test)
ypredictLib


# # Model Evaluation using Test set

# In[183]:


from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss


# First, download and load the test set:

# In[ ]:


get_ipython().system('wget -O loan_test.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_test.csv')


# In[140]:


Feature_test[0:10]


# ### Load Test set for evaluation 

# In[192]:


#Importing the dataset and determining the vectors
test_df = pd.read_csv(r"C:\Users\el-bachaoh\Downloads\loan_test.csv")
test_df['due_date'] = pd.to_datetime(test_df['due_date'])
test_df['effective_date'] = pd.to_datetime(test_df['effective_date'])
test_df['dayofweek'] = test_df['effective_date'].dt.dayofweek
test_df['weekend'] = test_df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)
Feature_test = test_df[['Principal','terms','age','Gender','weekend']]
Feature_test = pd.concat([Feature_test,pd.get_dummies(test_df['education'])], axis=1)
Feature_test['Gender'].replace(to_replace=['male','female'], value=[0,1], inplace=True)
Feature_test.drop(['Master or Above'], axis = 1,inplace=True)
X_test=preprocessing.StandardScaler().fit_transform(Feature_test)
#test_df['loan_status']=test_df['loan_status'].replace(to_replace=['PAIDOFF' , 'COLLECTION'],value=[1,0])
test_df['loan_status'].replace(to_replace=['PAIDOFF','COLLECTION'], value=[1,0], inplace=True)
y_test = test_df['loan_status'].values
y_test[0:50]


# In[201]:


y_test[0:50]


# In[200]:


#using KNN model from earlier
yhat = neigh.predict(X_test)
yhat[0:50]


# In[199]:


#determining accuracy using Jaccard for KNN k=2
jScore=jaccard_score(y_test, yhat)
print("Jaccard smilarity score is " +str(jScore*100)+"%")


# In[218]:


#determining accuracy using f1_score for KNN k=2
f1s=f1_score(y_test, yhat)
print("F1 score is " +str(f1s*100)+"%")


# In[219]:


#determining accuracy using Jaccard & F1-score for DecisionTree
jScore=jaccard_score(y_test, predTree)
f1s=f1_score(y_test, predTree)
print("F1 score is " +str(f1s*100)+"%")
print("Jaccard smilarity score is " +str(jScore*100)+"%")


# In[220]:


#determining accuracy using Jaccard & F1-score for SVM
jScore=jaccard_score(y_test, SV_Pred)
f1s=f1_score(y_test, SV_Pred)
print("F1 score is " +str(f1s*100)+"%")
print("Jaccard smilarity score is " +str(jScore*100)+"%")


# In[213]:


#determining accuracy using Jaccard & F1-score for Logistic Regression
jScore=jaccard_score(y_test, ypredict)
f1s=f1_score(y_test, ypredict)
print("F1 score is " +str(jScore*100)+"% for newton-cg solver")
print("Jaccard smilarity score is " +str(jScore*100)+"% for newton-cg solver")


# In[222]:


#determining accuracy using Jaccard & F1-score for Logistic Regression
jScore=jaccard_score(y_test, ypredictLib)
f1s=f1_score(y_test, ypredictLib)
Lloss=log_loss(y_test, ypredictLib)
print("F1 score is " +str(f1s*100)+"% for liblinear solver")
print("Jaccard smilarity score is " +str(jScore*100)+"% for liblinear solver")
print("Jaccard smilarity score is " +str(Lloss)+"% for liblinear solver")


# # Report
# You should be able to report the accuracy of the built model using different evaluation metrics:

# | Algorithm          | Jaccard      | F1-score | LogLoss |
# |--------------------|--------------|----------|---------|
# | KNN                | 48.88%       | 65.67%   | NA      |
# | Decision Tree      | 68.75%       | 81.48%   | NA      |
# | SVM                | 72.22%       | 83.87%   | NA      |
# | LogisticRegression | 74.07        | 85.10    | 8.95    |

# <h2>Want to learn more?</h2>
# 
# IBM SPSS Modeler is a comprehensive analytics platform that has many machine learning algorithms. It has been designed to bring predictive intelligence to decisions made by individuals, by groups, by systems – by your enterprise as a whole. A free trial is available through this course, available here: <a href="http://cocl.us/ML0101EN-SPSSModeler">SPSS Modeler</a>
# 
# Also, you can use Watson Studio to run these notebooks faster with bigger datasets. Watson Studio is IBM's leading cloud solution for data scientists, built by data scientists. With Jupyter notebooks, RStudio, Apache Spark and popular libraries pre-packaged in the cloud, Watson Studio enables data scientists to collaborate on their projects without having to install anything. Join the fast-growing community of Watson Studio users today with a free account at <a href="https://cocl.us/ML0101EN_DSX">Watson Studio</a>
# 
# <h3>Thanks for completing this lesson!</h3>
# 
# <h4>Author:  <a href="https://ca.linkedin.com/in/saeedaghabozorgi">Saeed Aghabozorgi</a></h4>
# <p><a href="https://ca.linkedin.com/in/saeedaghabozorgi">Saeed Aghabozorgi</a>, PhD is a Data Scientist in IBM with a track record of developing enterprise level applications that substantially increases clients’ ability to turn data into actionable knowledge. He is a researcher in data mining field and expert in developing advanced analytic methods like machine learning and statistical modelling on large datasets.</p>
# 
# <hr>
# 
# <p>Copyright &copy; 2018 <a href="https://cocl.us/DX0108EN_CC">Cognitive Class</a>. This notebook and its source code are released under the terms of the <a href="https://bigdatauniversity.com/mit-license/">MIT License</a>.</p>
