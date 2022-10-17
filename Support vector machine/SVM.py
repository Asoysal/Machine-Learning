# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
import pandas as pd
import warnings


warnings.filterwarnings("ignore")

#
#- Load the data in the file production.csv
#- Use SVM with different kernels to fit the data
#- Show the result for different feature vectors
#- Draw the decission boundary for a grid of points such as: 
#                 xx1, yy1, zz1 = np.meshgrid(np.linspace(7, 24, 50), np.linspace(7, 24, 50), 15)
#- Add some overlapping samples to the input files a see what happens

##############################################################################################
#normal Data
#data = pd.read_csv("production.csv")
#data = pd.read_csv("productionOverlap.csv")
#DataX = data.drop(columns = "efficient")
#DataY = data[ "efficient" ]
#DataY = DataY.to_frame()
#Xneg= DataX.iloc[0:6,:]
#Xpos= DataX.iloc[6:12,:]
###############################################################################################

###############################################################################################
# Overlapping data
data = pd.read_csv("productionOverlap.csv")
DataX = data.drop(columns = "efficient")
DataY = data[ "efficient" ]
DataY = DataY.to_frame()
# for Overlapping data
Xneg= DataX.iloc[0:11,:]
Xpos= DataX.iloc[11:19,:]
################################################################################################





linear_svc = svm.SVC(kernel='linear')
print(linear_svc.kernel)

poly_svc = svm.SVC(kernel='poly')
print(poly_svc.kernel)

linear_svc.fit(DataX.drop(columns = "input"), DataY)
print(linear_svc.predict([[1, 1]]))


poly_svc = svm.SVC(kernel='poly', degree= 3)
poly_svc.fit(DataX, DataY)










plt.figure(1)
plt.scatter(Xneg.iloc[:,0], Xneg.iloc[:,1], color='r')
plt.scatter(Xpos.iloc[:,0], Xpos.iloc[:,1], color='b')
plt.xlabel('x')
plt.ylabel('y')
xx1, yy1, zz1 = np.meshgrid(np.linspace(-1, 30, 50), np.linspace(-1, 30, 50), 15)
Z1 = poly_svc.decision_function(np.c_[xx1.ravel(), yy1.ravel(),zz1.ravel()])
Z1 = Z1.reshape(xx1.shape)
plt.contour(xx1[:,:,0], yy1[:,:,0],Z1[:,:,0],  levels=[0], linewidths=2, colors='green')
plt.show()





plt.figure(2)
plt.scatter(Xneg.iloc[:,0], Xneg.iloc[:,1], color='r')
plt.scatter(Xpos.iloc[:,0], Xpos.iloc[:,1], color='b')
plt.xlabel('x')
plt.ylabel('y')
xx1, yy1, zz1 = np.meshgrid(np.linspace(-1, 30, 50), np.linspace(-1, 30, 50), 15)
Z2 = linear_svc.decision_function(np.c_[xx1.ravel(), yy1.ravel()])
Z2 = Z2.reshape(xx1.shape)
plt.contour(xx1[:,:,0], yy1[:,:,0],Z2[:,:,0],  levels=[0], linewidths=2, colors='yellow')
plt.show()



















