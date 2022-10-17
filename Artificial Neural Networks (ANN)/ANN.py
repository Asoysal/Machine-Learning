# -*- coding: utf-8 -*-
"""
Read the production.csv file
Use the first 16 samples to train a MLPClassifier model with a single hidden layer
Use the las 4 samples to test the MLPClassifier model
You can train the model with different number of neurons in the hidden layer
Print the score values for the training and validation sets
Use the predict_proba function in the model to draw the decission boundary. Draw 2D plots for different values for the third feature (“time”). The following code may help:

"""
print(__doc__)

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neural_network import MLPClassifier
import numpy as np





data = pd.read_csv("production.csv")
DataX = data.drop(columns = "efficient")
DataY = data[ "efficient" ]
DataY = DataY.to_frame()

# create a training data : first 16 samples to train
train_data_size = 16
    
X_train = DataX.iloc[0:train_data_size,:]
y_train = DataY.iloc[0:train_data_size,:]
X_test = DataX.iloc[train_data_size:len(DataX),:]
y_test = DataY.iloc[train_data_size:len(DataX),:]  


#################################################################################################
# finding a best hidden layer size.
#resultTrainbest = 0;
#resulttestbest = 0;
#
#
#for i in range(1,100):        
#        mlp = MLPClassifier(hidden_layer_sizes=(i,), activation='relu', max_iter=20, alpha=1e-4,
#                    solver='sgd', verbose=10, tol=1e-4, random_state=1,
#                    learning_rate_init=.1)
#        size = i 
#        mlp.fit(X_train, y_train)
#        resultTrain = mlp.score(X_train, y_train)
#        resultTest = mlp.score(X_test , y_test)
#        if  resultTrain >= resultTrainbest and resultTest >= resulttestbest :
#             resultTrainbest = mlp.score(X_train, y_train)
#             resultTtestbest = mlp.score(X_test, y_test)
#             sizebest = size
#            
#             
#print(sizebest)
#
#print("Training set score: %f" % mlp.score(X_train, y_train))
#print("Test set score: %f" % mlp.score(X_test, y_test))    
#  ############################################################################################# 
  

# create a model . in the previous code piece, I test several size in order to find a best model
# according to result of the code, the best size is 89 in a single hidden layer and score is 1.
mlp = MLPClassifier(hidden_layer_sizes=(89,), activation='relu', max_iter=20, alpha=1e-4,
                    solver='sgd', verbose=10, tol=1e-4, random_state=1,
                    learning_rate_init=.1)
  
mlp.fit(X_train, y_train)   
print("Training set score: %f" % mlp.score(X_train, y_train))
print("Test set score: %f" % mlp.score(X_test, y_test))      


# plotting
DataY=DataY.astype('int')
Xneg = [[]]
Xpos = [[]]
for i in range(20):
    if DataY.iloc[i].all() == 0:
        Xneg=np.append(Xneg, DataX.iloc[i])
    else:
        Xpos=np.append(Xpos, DataX.iloc[i])

Xneg=np.reshape(Xneg,(int(len(Xneg)/3),3))
Xpos=np.reshape(Xpos,(int(len(Xpos)/3),3))



fig, axes = plt.subplots(1, 5, figsize=(10,3))
for time, ax in zip(range(10,20,2), axes.ravel()):
    ax.scatter(Xneg[:,0], Xneg[:,1], color='r')
    ax.scatter(Xpos[:,0], Xpos[:,1], color='b')
    xx1, yy1, zz1 = np.meshgrid(np.linspace(7, 24, 50), np.linspace(7, 24, 50), time)
    Z1 = mlp.predict_proba(np.c_[xx1.ravel(), yy1.ravel(), zz1.ravel()])
    Z1 = Z1[:,0].reshape(xx1.shape)
    ax.contour(xx1[:,:,0], yy1[:,:,0], Z1[:,:,0], levels=[0.5], linewidths=2, colors='green')
plt.show()



 