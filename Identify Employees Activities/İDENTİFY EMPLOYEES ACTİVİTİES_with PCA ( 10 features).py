# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 02:13:36 2020

@author: Alper Soysal
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA

pca = PCA (n_components = 10)
#pca_result = pca.fit_transform(X_train)




#Task 1.1 
############################################################################################
X_train = np.loadtxt("X_train.txt")
X_train = pd.DataFrame(X_train) 
y_train = np.loadtxt("y_train.txt")
y_train= pd.DataFrame(y_train) 
X_train=X_train.astype('float64')

X_train = pca.fit_transform(X_train)
X_train = pd.DataFrame(X_train)


X_train_ytrain_Row_Quantity=len(X_train)




# Remove outliers from the information in the training set for WALKING and WALKING_UPSTAIRS 
#for all users in the training set.
#walking 1 , walkingupstair 2 

 #for to extract input walking  data from original train data.
X_Train_walk = pd.DataFrame()
      
for i in range(X_train_ytrain_Row_Quantity): 
       if y_train.iloc[i,0] == 1:
          a = X_train.iloc[i,:]
          a = a.to_frame()
          a = a.transpose()       
          X_Train_walk=pd.concat([X_Train_walk,a] , ignore_index=True)
          
          
#for to extract output walking  data from original train data.    
y_Train_walk = pd.DataFrame()      
for i in range(X_train_ytrain_Row_Quantity): 
       if y_train.iloc[i,0] == 1:
          a = y_train.iloc[i,:]
          a = a.to_frame()
          a = a.transpose()       
          y_Train_walk=pd.concat([y_Train_walk,a] , ignore_index=True)            
          
          


#for to extract input walking up data from original train data.
X_Train_walkup = pd.DataFrame()
#y_walkuptrain = pd.DataFrame()      

for i in range(X_train_ytrain_Row_Quantity): 
       if y_train.iloc[i,0] == 2:
          a = X_train.iloc[i,:]
          a = a.to_frame()
          a = a.transpose()       
          X_Train_walkup=pd.concat([X_Train_walkup,a] , ignore_index=True)
 
#for to extract output walking up data from original train data.         
y_Train_walkup = pd.DataFrame()      
for i in range(X_train_ytrain_Row_Quantity): 
       if y_train.iloc[i,0] == 2:
          a = y_train.iloc[i,:]
          a = a.to_frame()
          a = a.transpose()       
          y_Train_walkup=pd.concat([y_Train_walkup,a] , ignore_index=True)         
           
                   
                  
                   

#Identify outliers for walkup number of outlier = 215
clf = LocalOutlierFactor(contamination=0.20)
y_pred2 = clf.fit_predict(X_Train_walkup)
y_pred2=pd.DataFrame(y_pred2) 
y_pred2[0].value_counts()

#Delete outliers for walkup input and output row =  857 
#Data_Train_walkup  = pd.concat ([X_Train_walkup,y_pred2],  ignore_index=True,axis=1)
X_Train_walkup  = pd.concat ([X_Train_walkup,y_pred2],  ignore_index=True,axis=1)
X_Train_walkup_ypred2 = X_Train_walkup [ X_Train_walkup[10]  == 1 ]

y_Train_walkup  = pd.concat ([y_Train_walkup,y_pred2],  ignore_index=True,axis=1)
y_Train_walkup_ypred2 = y_Train_walkup [ y_Train_walkup[1]  == 1 ]




#Identify outliers for walk number of outlier : 184
clf = LocalOutlierFactor(contamination=0.15)
y_pred = clf.fit_predict(X_Train_walk)
y_pred=pd.DataFrame(y_pred) 
y_pred[0].value_counts()


#Delete outliers for walk row = 1042

X_Train_walk  = pd.concat ([X_Train_walk,y_pred],  ignore_index=True,axis=1)
X_Train_walk_ypred = X_Train_walk [ X_Train_walk[10]  == 1 ]

y_Train_walk  = pd.concat ([y_Train_walk,y_pred],  ignore_index=True,axis=1)
y_Train_walk_ypred = y_Train_walk [ y_Train_walk[1]  == 1 ]









# filter the output and input data( last column ). 
X_filtered_Walk_train = X_Train_walk_ypred.drop (columns = 10)
X_filtered_Walkup_train = X_Train_walkup_ypred2.drop (columns = 10)

y_filtered_Walk_train = y_Train_walk_ypred.drop (columns = 1)
y_filtered_Walkup_train = y_Train_walkup_ypred2.drop (columns = 1)


# create   X_train_for_Classifier for walking and walking up
X_train_for_Classifier = pd.DataFrame()

X_train_for_Classifier = pd.concat([X_train_for_Classifier,X_filtered_Walk_train] 
                                                 , ignore_index=True,axis=0)
X_train_for_Classifier = pd.concat([X_train_for_Classifier,X_filtered_Walkup_train] 
                                                  , ignore_index=True,axis=0)

# # create   y_train_for_Classifier for walking and walking up
y_train_for_Classifier = pd.DataFrame()
y_train_for_Classifier = pd.concat([y_train_for_Classifier,y_filtered_Walk_train] 
                                                 , ignore_index=True,axis=0)
y_train_for_Classifier = pd.concat([y_train_for_Classifier,y_filtered_Walkup_train] 
                                                 , ignore_index=True,axis=0)





################yoy###########################################
X_test = np.loadtxt("X_test.txt")# (2947,561)
X_test = pd.DataFrame(X_test) 
X_test=X_test.astype('float64')
y_test = np.loadtxt("y_test.txt")#(2947,1)
y_test= pd.DataFrame(y_test) 
y_test=y_test.astype('float64')
X_test_ytest_Row_Quantity=len(X_test.index)

X_test = pca.fit_transform(X_test)
X_test = pd.DataFrame(X_test)




# Remove outliers from the information in the testing set for WALKING and WALKING_UPSTAIRS 
#for all users in the test set.
#walking 1 , walkingupstair 2 

 #for to extract input walking  data from original test data.
X_Test_walk = pd.DataFrame()
      
for i in range(X_test_ytest_Row_Quantity): 
       if y_test.iloc[i,0] == 1:
          a = X_test.iloc[i,:]
          a = a.to_frame()
          a = a.transpose()       
          X_Test_walk=pd.concat([X_Test_walk,a] , ignore_index=True)#(496,561)
          
          
#for to extract output walking  data from original test data.    
y_Test_walk = pd.DataFrame()      
for i in range(X_test_ytest_Row_Quantity): 
       if y_test.iloc[i,0] == 1:
          a = y_test.iloc[i,:]
          a = a.to_frame()
          a = a.transpose()       
          y_Test_walk=pd.concat([y_Test_walk,a] , ignore_index=True)#(496,1)         
          
          


#for to extract input walking up data from original test data.
X_Test_walkup = pd.DataFrame()
#y_walkuptrain = pd.DataFrame()      

for i in range(X_test_ytest_Row_Quantity): 
       if y_test.iloc[i,0] == 2:
          a = X_test.iloc[i,:]
          a = a.to_frame()
          a = a.transpose()       
          X_Test_walkup=pd.concat([X_Test_walkup,a] , ignore_index=True)#(471,561)
 
#for to extract output walking up data from original test data.         
y_Test_walkup = pd.DataFrame()      
for i in range(X_test_ytest_Row_Quantity): 
       if y_test.iloc[i,0] == 2:
          a = y_test.iloc[i,:]
          a = a.to_frame()
          a = a.transpose()       
          y_Test_walkup=pd.concat([y_Test_walkup,a] , ignore_index=True)#(471,1)   
          

#Identify outliers for test walkup number of outlier = 94
clf = LocalOutlierFactor(contamination=0.20)
y_pred3 = clf.fit_predict(X_Test_walkup)
y_pred3=pd.DataFrame(y_pred3) 
y_pred3[0].value_counts()

#Delete outliers for test walkup input and output row =  377 
#Data_Train_walkup  = pd.concat ([X_Test_walkup,y_pred3],  ignore_index=True,axis=1)
X_Test_walkup  = pd.concat ([X_Test_walkup,y_pred3],  ignore_index=True,axis=1)
X_Test_walkup_ypred3 = X_Test_walkup [ X_Test_walkup[10]  == 1 ]

y_Test_walkup  = pd.concat ([y_Test_walkup,y_pred3],  ignore_index=True,axis=1)
y_Test_walkup_ypred3 = y_Test_walkup [ y_Test_walkup[1]  == 1 ]




#Identify outliers for test walk number of outlier : 75
clf = LocalOutlierFactor(contamination=0.15)
y_pred4 = clf.fit_predict(X_Test_walk)
y_pred4=pd.DataFrame(y_pred4) 
y_pred4[0].value_counts()


#Delete outliers for test walk row = 421

X_Test_walk  = pd.concat ([X_Test_walk,y_pred4],  ignore_index=True,axis=1)
X_Test_walk_ypred4 = X_Test_walk [ X_Test_walk[10]  == 1 ]

y_Test_walk  = pd.concat ([y_Test_walk,y_pred4],  ignore_index=True,axis=1)
y_Test_walk_ypred4 = y_Test_walk [ y_Test_walk[1]  == 1 ]



# filter the output and input data( last column ). 
X_filtered_Walk_test = X_Test_walk_ypred4.drop (columns = 10)
X_filtered_Walkup_test = X_Test_walkup_ypred3.drop (columns = 10)

y_filtered_Walk_test = y_Test_walk_ypred4.drop (columns = 1)
y_filtered_Walkup_test = y_Test_walkup_ypred3.drop (columns = 1)





#######################################################################################

#Task 1.2
########################################################################################





 #the piece of code that find the best hidden later size
#resultTrainbest = 0;
#for i in range(10,100,10):        
#        mlp = MLPClassifier(hidden_layer_sizes=(i,), activation='relu', max_iter=40, alpha=1e-4,
#                    solver='sgd', verbose=10, tol=1e-4, random_state=1,
#                    learning_rate_init=.1)
#        size = i 
#        mlp.fit(X_train_for_Classifier, y_train_for_Classifier)
#        resultTrain = mlp.score(X_train_for_Classifier, y_train_for_Classifier)
#        
#        if  resultTrain >= resultTrainbest :
#             resultTrainbest = resultTrain
#             print("Training set score: %f" % mlp.score(X_train_for_Classifier, y_train_for_Classifier))
#             sizebest = size
#          
#             
#print(sizebest)
#print(resultTrainbest)

# according to code best hidden layer size = 20 score = 0.991052



mlp = MLPClassifier(hidden_layer_sizes=(20,), activation='relu', max_iter=40, alpha=1e-4,
                 solver='sgd', verbose=10, tol=1e-4, random_state=1,
                    learning_rate_init=.1)
mlp.fit(X_train_for_Classifier, y_train_for_Classifier)
########################################################################################

#Task 1.3
##########################################################################################


print("Training set score data without outlier: %f" % mlp.score(X_train_for_Classifier, 
                                                                y_train_for_Classifier))

X_Train_walk_with_outlier = X_train[y_train.iloc[:,0]==1]
y_Train_walk_with_outlier = y_train[y_train.iloc[:,0]==1]

X_Train_walkup_with_outlier = X_train[y_train.iloc[:,0]==2]
y_Train_walkup_with_outlier = y_train[y_train.iloc[:,0]==2]



print("Training set score walking data with outlier: %f" % mlp.score(X_Train_walk_with_outlier
                                                                     ,y_Train_walk_with_outlier))
print("Training set score walkingup data with outlier: %f" % mlp.score(X_Train_walkup_with_outlier,
                                                                       y_Train_walkup_with_outlier))

##################################################################################################################################
#Task 1.4
#######################################################


print("test set score walking data without outlier: %f" % mlp.score(X_filtered_Walk_test,y_filtered_Walk_test))
print("test set score walkingup data without outlier: %f" % mlp.score(X_filtered_Walkup_test,y_filtered_Walkup_test))

###############################################################################################################################
#task 2.5                                                                       
#############################################################################################################################


X_Train_walkup_only_outliar = X_Train_walkup [ X_Train_walkup[10]  == -1 ]
y_Train_walkup_only_outliar = y_Train_walkup [ y_Train_walkup[1]  == -1 ]

X_Train_walkup_only_outliar = X_Train_walkup_only_outliar.drop(columns = 10)
y_Train_walkup_only_outliar = y_Train_walkup_only_outliar.drop(columns = 1)

n_of = mlp.predict(X_Train_walkup_only_outliar)
n_of = pd.DataFrame(n_of)
n_of[0].value_counts()
######################################################################################
# task 3.6

#Preparing data for barplot
l=n_of[n_of[0]==1]
o=n_of[n_of[0]==2]
l[0].value_counts()
o[0].value_counts()
l=int(l[0].value_counts())
o=int(o[0].value_counts())
#plot the results
import matplotlib.pyplot as plt

height = [l,o ]
bars = ('Walking', 'Walking upstairs')
y_pos = np.arange(len(bars))
 
# Create bars
plt.bar(y_pos, height, color=("red", "blue"))
plt.title("Task 2.5 graph of Missclassify samples ")
plt.ylabel('Quantity')

# Create names on the x-axis
plt.xticks(y_pos, bars)

# Show graphic
plt.show()


































