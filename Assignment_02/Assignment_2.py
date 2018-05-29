# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 16:41:04 2017

@author: nvlas
"""
    
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns  
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.grid_search import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import model_selection
from sklearn.datasets import make_classification
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit



def main():
    
     data = pd.read_csv("mushrooms.csv")
#==============================================================================
#     print(data.head(6))
#     print("================================================")
#     print(data.isnull().sum())
#     print("=====================")
#     print(data['class'].unique())
#     print("=====================")
#     print(data.shape)
#==============================================================================
    
     labelencoder = LabelEncoder()
     for col in data.columns:
         data[col] = labelencoder.fit_transform(data[col])
        
    
 
     #print(data.head())
    
#==============================================================================
#      ax = sns.boxplot(x='class', y='stalk-color-above-ring',  data=data)
#      ax = sns.stripplot(x="class", y='stalk-color-above-ring',
#                    data=data, jitter=True,
#                    edgecolor="gray")
#      sns.plt.title("Class w.r.t stalkcolor above ring",fontsize=12)
#==============================================================================
    

     train_feature = data.iloc[:,1:23]
     test_feature = data.iloc[:, 0]
     
   #Heatmap  
#==============================================================================
#     data = pd.DataFrame(train_feature)
#     corrResult = data.corr()
#     sns.heatmap(corrResult)
#     plt.show()
#==============================================================================

#==============================================================================
#      # Build a classification task using 3 informative features
#      train_feature, test_feature = make_classification(n_samples=1000,
#                                 n_features=10,
#                                 n_informative=3,
#                                 n_redundant=0,
#                                 n_repeated=0,
#                                 n_classes=2,
#                                 random_state=0,
#                                 shuffle=False)
#      # Build a forest and compute the feature importance
#      forest = ExtraTreesClassifier(n_estimators=250, random_state=0)
#      forest.fit(train_feature, test_feature)
#      importances = forest.feature_importances_
#      for index in range(len(train_feature[0])):
#          print ("Importance of feature ", index, "is", importances[index])
#==============================================================================
     
     # Scale the data to be between -1 and 1
     scaler = StandardScaler()
     train_feature = scaler.fit_transform(train_feature)
     
     pca = PCA()
     pca.fit_transform(train_feature)
     covariance = pca.get_covariance()
     explained_variance=pca.explained_variance_
     print(explained_variance)
      
     
     # Splitting the data into training and testing dataset
     X_train, X_test, y_train, y_test = train_test_split(train_feature,test_feature,test_size=0.2,random_state=4)
     
     print("==============================================================")
     print("                     Logistic Regression                      ")
     print("==============================================================")
     
     # Logistic Regression
     logic = LogisticRegression()
     parameters_logic = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] ,
               'penalty':['l1','l2']
                    }
     logic_grid_search = GridSearchCV(logic, parameters_logic,cv=10)
     logic_grid_search.fit(X_train,y_train)
     
     # Positive class prediction probabilities
     y_prob = logic_grid_search.predict_proba(X_test)[:,1]   
     # Threshold the probabilities to give class predictions.
     y_pred = np.where(y_prob > 0.5, 1, 0)
     
     print("Logic Regresion result: ",logic_grid_search.score(X_test, y_pred),"%")
     print("Best parameters for this model are: ",logic_grid_search.best_params_)
     
     print("==============================================================")
     print("                        Naive Bayes                           ")
     print("==============================================================")
     
     # Gaussian Naive Bayes
     naive = GaussianNB()
     naive.fit(X_train, y_train)
     # Positive class prediction probabilities
     y_prob = naive.predict_proba(X_test)[:,1]   
     # Threshold the probabilities to give class predictions.
     y_pred = np.where(y_prob > 0.5, 1, 0) 
    
     print("Number of mislabeled points from %d points : %d" % (X_test.shape[0],(y_test!= y_pred).sum()))
     scores = cross_val_score(naive, train_feature, test_feature, cv=10, scoring='accuracy')
     print("Naive Bayes result: ",scores.mean(), "%")
     
     print("==============================================================")
     print("                 Support Vector Machine                       ")
     print("==============================================================")
     
     svm = SVC()
     parameters = {
             'C': [1, 10, 100,500, 1000], 'kernel': ['linear','rbf'],
             'C': [1, 10, 100,500, 1000], 'gamma': [1,0.1,0.01,0.001, 0.0001], 'kernel': ['rbf'],
             #'degree': [2,3,4,5,6] , 'C':[1,10,100,500,1000] , 'kernel':['poly']
     }
     
     randomize_search = RandomizedSearchCV(svm, parameters,cv=10,scoring='accuracy',n_iter=20)
     randomize_search.fit(X_train, y_train)
     print("Support Vector Machine best result: ",randomize_search.best_score_, "%")
     print("Best parameters for this model are: ",randomize_search.best_params_)
     
     print("==============================================================")
     print("                 Random Forest Classifier                     ")
     print("==============================================================")
     
     random_forest = RandomForestClassifier()
     random_forest.fit(X_train,y_train)
     # Positive class prediction probabilities
     y_prob = random_forest.predict_proba(X_test)[:,1]   
     # Threshold the probabilities to give class predictions.
     y_pred = np.where(y_prob > 0.5, 1, 0) 
     
     print(random_forest.score(X_test, y_pred))
     
     print("==============================================================")
     print("                     Decision Tree Classifier                 ")
     print("==============================================================")
     
     decision_tree = DecisionTreeClassifier()
 
 
     tuned_parameters = {'criterion': ['gini','entropy'], 'max_features': ["auto","sqrt","log2"],
                    'min_samples_leaf': range(1,100,1) , 'max_depth': range(1,50,1)
                   }
    
     decision_model= RandomizedSearchCV(decision_tree, tuned_parameters,cv=10,scoring='accuracy',n_iter=20,n_jobs= -1,random_state=5)
     decision_model.fit(X_train, y_train)
     print("Decision Tree Classifier result: ",decision_model.best_score_,"%")
     print("Best parameters for this model are: ",decision_model.best_params_)
     
main()

    
    
