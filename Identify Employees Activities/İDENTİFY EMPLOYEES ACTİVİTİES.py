# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 19:07:08 2020

@author: Nagato
"""

## -*- coding: utf-8 -*-
#"""
#Created on Sun Mar 22 02:49:41 2020
#
#@author: Alper Soysal
#"""
import pandas as pd
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA






#Task 1.1 
############################################################################################

#loading data
X_train = np.loadtxt("X_train.txt")
X_train = pd.DataFrame(X_train) 
y_train = np.loadtxt("y_train.txt")
y_train= pd.DataFrame(y_train) 
X_train=X_train.astype('float64')
#X_test = np.loadtxt("X_test.txt")
#y_test = np.loadtxt("y_test.txt")
X_train_ytrain_Row_Quantity=len(X_train.index)

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

print("walking upstairs outliers: %f" % 215 )

#Delete outliers for walkup input and output row =  857 
#Data_Train_walkup  = pd.concat ([X_Train_walkup,y_pred2],  ignore_index=True,axis=1)
X_Train_walkup  = pd.concat ([X_Train_walkup,y_pred2],  ignore_index=True,axis=1)
X_Train_walkup_ypred2 = X_Train_walkup [ X_Train_walkup[561]  == 1 ]

y_Train_walkup  = pd.concat ([y_Train_walkup,y_pred2],  ignore_index=True,axis=1)
y_Train_walkup_ypred2 = y_Train_walkup [ y_Train_walkup[1]  == 1 ]




#Identify outliers for walk number of outlier : 184
clf = LocalOutlierFactor(contamination=0.15)
y_pred = clf.fit_predict(X_Train_walk)
y_pred=pd.DataFrame(y_pred) 
y_pred[0].value_counts()
print("walking  outliers: %f" % 184 )

#Delete outliers for walk row = 1042

X_Train_walk  = pd.concat ([X_Train_walk,y_pred],  ignore_index=True,axis=1)
X_Train_walk_ypred = X_Train_walk [ X_Train_walk[561]  == 1 ]

y_Train_walk  = pd.concat ([y_Train_walk,y_pred],  ignore_index=True,axis=1)
y_Train_walk_ypred = y_Train_walk [ y_Train_walk[1]  == 1 ]









# filter the output and input data( last column ). 
X_filtered_Walk_train = X_Train_walk_ypred.drop (columns = 561)
X_filtered_Walkup_train = X_Train_walkup_ypred2.drop (columns = 561)

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





# data modification for test
#load the data
X_test = np.loadtxt("X_test.txt")# (2947,561)
X_test = pd.DataFrame(X_test) 
X_test=X_test.astype('float64')
y_test = np.loadtxt("y_test.txt")#(2947,1)
y_test= pd.DataFrame(y_test) 
y_test=y_test.astype('float64')
X_test_ytest_Row_Quantity=len(X_test.index)
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
X_Test_walkup_ypred3 = X_Test_walkup [ X_Test_walkup[561]  == 1 ]

y_Test_walkup  = pd.concat ([y_Test_walkup,y_pred3],  ignore_index=True,axis=1)
y_Test_walkup_ypred3 = y_Test_walkup [ y_Test_walkup[1]  == 1 ]




#Identify outliers for test walk number of outlier : 75
clf = LocalOutlierFactor(contamination=0.15)
y_pred4 = clf.fit_predict(X_Test_walk)
y_pred4=pd.DataFrame(y_pred4) 
y_pred4[0].value_counts()


#Delete outliers for test walk row = 421

X_Test_walk  = pd.concat ([X_Test_walk,y_pred4],  ignore_index=True,axis=1)
X_Test_walk_ypred4 = X_Test_walk [ X_Test_walk[561]  == 1 ]

y_Test_walk  = pd.concat ([y_Test_walk,y_pred4],  ignore_index=True,axis=1)
y_Test_walk_ypred4 = y_Test_walk [ y_Test_walk[1]  == 1 ]



# filter the output and input data( last column ). 
X_filtered_Walk_test = X_Test_walk_ypred4.drop (columns = 561)
X_filtered_Walkup_test = X_Test_walkup_ypred3.drop (columns = 561)

y_filtered_Walk_test = y_Test_walk_ypred4.drop (columns = 1)
y_filtered_Walkup_test = y_Test_walkup_ypred3.drop (columns = 1)



# create   X_test_for_Classifier for walking and walking up
X_test_for_Classifier = pd.DataFrame()

X_test_for_Classifier = pd.concat([X_test_for_Classifier,X_filtered_Walk_test] 
                                                 , ignore_index=True,axis=0)
X_test_for_Classifier = pd.concat([X_test_for_Classifier,X_filtered_Walkup_test] 
                                                  , ignore_index=True,axis=0)

# # create   y_test_for_Classifier for walking and walking up
y_test_for_Classifier = pd.DataFrame()
y_test_for_Classifier = pd.concat([y_test_for_Classifier,y_filtered_Walk_test] 
                                                 , ignore_index=True,axis=0)
y_test_for_Classifier = pd.concat([y_test_for_Classifier,y_filtered_Walkup_test] 
                                                 , ignore_index=True,axis=0)


#######################################################################################

#Task 1.2
########################################################################################





# the piece of code that find the best hidden later size
#resultTrainbest = 0;
#for i in range(10,100,10):        
#        mlp = MLPClassifier(hidden_layer_sizes=(i,), activation='relu', max_iter=20, alpha=1e-4,
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

# according to code best hidden layer size = 80 score = 1



mlp = MLPClassifier(hidden_layer_sizes=(250,), activation='relu', max_iter=50, alpha=1e-4,
                 solver='sgd', verbose=10, tol=1e-4, random_state=1,
                    learning_rate_init=.1)
mlp.fit(X_train_for_Classifier, y_train_for_Classifier)
########################################################################################

#Task 1.3
##########################################################################################

# the score is so high because we delete the outlier and feed the model with this data.
print("Training set score data without outlier: %f" % mlp.score(X_train_for_Classifier, 
                                                                y_train_for_Classifier))
# creating train data with outlier
X_Train_walk_with_outlier = X_train[y_train.iloc[:,0]==1]
y_Train_walk_with_outlier = y_train[y_train.iloc[:,0]==1]

X_Train_walkup_with_outlier = X_train[y_train.iloc[:,0]==2]
y_Train_walkup_with_outlier = y_train[y_train.iloc[:,0]==2]


# creating test  data with outlier

X_test_walk_with_outlier = X_test[y_test.iloc[:,0]==1]
y_test_walk_with_outlier = y_test[y_test.iloc[:,0]==1]

X_test_walkup_with_outlier = X_test[y_test.iloc[:,0]==2]
y_test_walkup_with_outlier = y_test[y_test.iloc[:,0]==2]








# training test score os so high because we train the model with this data set.

print("Training set score walking data with outlier: %f" % mlp.score(X_Train_walk_with_outlier
                                                                     ,y_Train_walk_with_outlier))
print("Training set score walkingup data with outlier: %f" % mlp.score(X_Train_walkup_with_outlier,
                                                                       y_Train_walkup_with_outlier))



##################################################################################################################################
#Task 1.4
#######################################################

# compate with previous train set  score, test score is lower because we calculate the score with the data that is not know by the model.

print("test set score walking data without outlier: %f" % mlp.score(X_filtered_Walk_test,y_filtered_Walk_test))
print("test set score walkingup data without outlier: %f" % mlp.score(X_filtered_Walkup_test,y_filtered_Walkup_test))






###############################################################################################################################
#task 2.5                                                                       
#############################################################################################################################


X_Train_walkup_only_outliar = X_Train_walkup [ X_Train_walkup[561]  == -1 ]
y_Train_walkup_only_outliar = y_Train_walkup [ y_Train_walkup[1]  == -1 ]

X_Train_walkup_only_outliar = X_Train_walkup_only_outliar.drop(columns = 561)
y_Train_walkup_only_outliar = y_Train_walkup_only_outliar.drop(columns = 1)

n_of = mlp.predict(X_Train_walkup_only_outliar)
n_of = pd.DataFrame(n_of)
n_of[0].value_counts()

##################################################################################
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
####################################################################################

######################################################################################
# task 3.6
# in the another script

#############################################################################################
#task 4,7
################################################################################################

# loading subject data which include the number of participant
subject_train = np.loadtxt("subject_train.txt")# (2947,561)
subject_train = pd.DataFrame(subject_train) 
subject_train=subject_train.astype('float64')
subject_test = np.loadtxt("subject_test.txt")# (2947,561)
subject_test = pd.DataFrame(subject_test) 
subject_test=subject_test.astype('float64')


#we need as a input X_Train_walk_with_outlier 
#we need as a output walk subject train with outlier 


#extraxt subject data only for walk
subject_train_walk = subject_train[y_train.iloc[:,0]==1]
subject_test_walk = subject_test[y_test.iloc[:,0]==1]


#resultTrainbest = 0;
#for i in range(10,100,10):        
#        mlp = MLPClassifier(hidden_layer_sizes=(i,), activation='relu', max_iter=20, alpha=1e-4,
#                    solver='sgd', verbose=10, tol=1e-4, random_state=1,
#                    learning_rate_init=.1)
#        size = i 
#        mlp.fit(X_Train_walk_with_outlier, subject_train_walk)
#        resultTrain = mlp.score(X_Train_walk_with_outlier, subject_train_walk)
#        
#        if  resultTrain >= resultTrainbest :
#             resultTrainbest = resultTrain
#             print("Training set score: %f" % mlp.score(X_Train_walk_with_outlier, subject_train_walk))
#             sizebest = size
#          
#             
#print(sizebest)
#print(resultTrainbest)


# train the classifier # best hidden layer size = 50 score =1
mlp = MLPClassifier(hidden_layer_sizes=(50,), activation='relu', max_iter=40, alpha=1e-4,
                 solver='sgd', verbose=10, tol=1e-4, random_state=1,
                    learning_rate_init=.1)
mlp.fit(X_Train_walk_with_outlier, subject_train_walk)

# show the score = 0
#Test set score input walking data (X_test_walk_with_outlier)
# output subject walking test (subject_test_walk)
print("Test set score input walking data output subject data with outlier: %f" % mlp.score(X_Train_walk_with_outlier
                                                                     ,subject_train_walk))


print("Test set score input walking data output subject data with outlier: %f" % mlp.score(X_test_walk_with_outlier
                                                                     ,subject_test_walk))


#################################################################################
import seaborn as sns
t=mlp.predict(X_test_walk_with_outlier)
sns.regplot(t, subject_test_walk[0], fit_reg=False)
plt.title("Task 4.7 graph Identification of user by the way he walk ")
plt.ylabel('Users')
plt.xlabel('How they are predicted')

############################################################
#task 4.8
###############################################################################################

# create a data for svm
#############################################################################################start



#concatenate walk train and test data

X_data_for_svm = pd.DataFrame()

X_data_for_svm =pd.concat([X_data_for_svm,X_Train_walk_with_outlier], ignore_index=True,axis=0 )
X_data_for_svm =pd.concat([X_data_for_svm,X_test_walk_with_outlier], ignore_index=True,axis=0 )



#Create dataframe and concatenate subject walk train and subject walk test

y_data_for_svm=pd.DataFrame()

y_data_for_svm =pd.concat([y_data_for_svm,subject_train_walk ], ignore_index=True,axis=0 )
y_data_for_svm =pd.concat([y_data_for_svm,subject_test_walk], ignore_index=True,axis=0 )



# split the data for classfier as train and test (error code does not work properly)

X_train_for_svm, X_test_for_svm, y_train_for_svm  ,y_test_for_svm= train_test_split(X_data_for_svm, y_data_for_svm,
                                                                        test_size=0.30, train_size = 0.70 )
                                                                        




# best size. 2 hidden layer, size 40 and 40. score = 0.6663  learning_rate_init=.05
#resultTrainbest = 0;
#for i in range(10,50,10):
#    for j in range(10,50,10):
#        mlp = MLPClassifier(hidden_layer_sizes=(i,j), activation='relu', max_iter=40, alpha=1e-4,
#                    solver='sgd', verbose=10, tol=1e-4, random_state=1,
#                    learning_rate_init=.05)
#        size1 = i 
#        size2 = j
#        mlp.fit(X_train_for_svm, y_train_for_svm)
#        resultTrain = mlp.score(X_train_for_svm, y_train_for_svm)
#        if resultTrain >= resultTrainbest :
#             resultTrainbest = resultTrain
#             print("Training set score: %f" % mlp.score(X_train_for_svm, y_train_for_svm))
#             sizebest1 = size1
#             sizebest2 = size2
#             
#print(sizebest1)
#print(sizebest2)
#print(resultTrainbest)             




# train the classifier #2 hidden layer, size 40 and 40. score = 0.4622  learning_rate_init=.05
mlp = MLPClassifier(hidden_layer_sizes=(40,30), activation='relu', max_iter=40, alpha=1e-4,
                 solver='sgd', verbose=10, tol=1e-4, random_state=1,
                    learning_rate_init=.05)
mlp.fit(X_train_for_svm, y_train_for_svm)

print("Train set score : %f" % mlp.score(X_train_for_svm,y_train_for_svm))
print("Test set score : %f" % mlp.score(X_test_for_svm,y_test_for_svm))




# Plot 4.8####################################################

t1=mlp.predict(X_test_for_svm)
sns.regplot(t1, y_test_for_svm[0], fit_reg=False)
plt.title("Task 4.8 graph Identification of user by the way he walk ")
plt.ylabel('Users')
plt.xlabel('How they are predicted')
############################################################



################################################################################
#task 4.9
#####################################################################################3
# detect outlier
#Identify outliers number of outlier = 345
clf = LocalOutlierFactor(contamination=0.20)
y_pred_X_data_for_lstm = clf.fit_predict(X_data_for_svm)
y_pred_X_data_for_lstm=pd.DataFrame(y_pred_X_data_for_lstm) 
y_pred_X_data_for_lstm[0].value_counts()

#Delete outliers 

X_data_for_svm  = pd.concat ([X_data_for_svm,y_pred_X_data_for_lstm],  ignore_index=True,axis=1)
X_data_for_lstm_ypred = X_data_for_svm [ X_data_for_svm[561]  == 1 ]

              # input for mlp below
X_data_for_lstm_without_outlier = X_data_for_lstm_ypred.drop (columns = 561)

y_data_for_svm  = pd.concat ([y_data_for_svm,y_pred_X_data_for_lstm],  ignore_index=True,axis=1)
y_data_for_lstm_ypred = y_data_for_svm [ y_data_for_svm[1]  == 1 ]

# output for mlp below
y_data_for_lstm_without_outlier = y_data_for_lstm_ypred.drop (columns = 1)


#extract best 10 features witout outlier
pca = PCA (n_components = 10)

X_data_for_lstm_without_outlier = pca.fit_transform(X_data_for_lstm_without_outlier)
X_data_for_lstm_without_outlier = pd.DataFrame(X_data_for_lstm_without_outlier)

                                                 
#split the data for classfier as train and test 
X_train_for_lstm = X_data_for_lstm_without_outlier.head(round(len(X_data_for_lstm_without_outlier)*0.70))
y_train_for_lstm = y_data_for_lstm_without_outlier.head(round(len(y_data_for_lstm_without_outlier)*0.70))
X_test_for_lstm = X_data_for_lstm_without_outlier.head(round(len(X_data_for_lstm_without_outlier)*0.30))
y_test_for_lstm = y_data_for_lstm_without_outlier.head(round(len(y_data_for_lstm_without_outlier)*0.30))








# convert data into 3d. because lstm input shape needs a data 3D 
X_train_for_lstm = X_train_for_lstm.to_numpy()

X_train_for_lstm = X_train_for_lstm.reshape((X_train_for_lstm.shape[0], X_train_for_lstm.shape[1], 1))

X_test_for_lstm= X_test_for_lstm.to_numpy()

X_test_for_lstm = X_test_for_lstm.reshape((X_test_for_lstm.shape[0], X_test_for_lstm.shape[1], 1))






# mlp model


from tensorflow import keras


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(X_train_for_lstm.shape[1], 1)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(31, activation='softmax')
])
    
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train_for_lstm, y_train_for_lstm, epochs=20)
test_loss, test_acc = model.evaluate(X_test_for_lstm,  y_test_for_lstm, verbose=2)

print("test loss   : %f" % test_loss )
print("test  accuracy   : %f" % test_acc )




t1=model.predict(X_test_for_lstm)
#t1 = np.float64(t1)
#
#sns.regplot(t1[:,0], y_test_for_lstm[0], fit_reg=False)
#plt.title("Task 4.8 graph Identification of user by the way he walk ")
#plt.ylabel('Users')
#plt.xlabel('How they are predicted')
#


