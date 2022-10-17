# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 11:22:21 2020

@author: Alper Soysal
"""

#Detecting outliers micro-assignment

#Q1 Load the failure_in_one_month.csv file using pandas
#Use LocalOutlierFactor with different n_neighbors to detect outliers
#       Use fit_predict to classify the data samples. You can use the following sample code:
#               clf = LocalOutlierFactor(n_neighbors=5)
#              y_pred = clf.fit_predict(data.values)
#              colors = np.array(['#377eb8', '#ff7f00'])
#               plt.scatter(data.values[:, 0], data.values[:, 1], color=colors[(y_pred + 1) // 2])
#Use EllipticEnvelope with different contamination values to detect outliers
#       Plot the decision_function and explain results

import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
import numpy as np
import matplotlib.pyplot as plt
from sklearn.covariance import EllipticEnvelope


#Q1 Load the failure_in_one_month.csv file using pandas
###################################################################################################
data = pd.read_csv("failure_in_one_month.csv")

data["machine_temp"]=((data["machine_temp"]-data["machine_temp"].min()/
      data["machine_temp"].max()-data["machine_temp"].min()))*0.10

###################################################################################################

#Q2 Use LocalOutlierFactor with different n_neighbors to detect outliers
# Use fit_predict to classify the data samples

clf1a = LocalOutlierFactor(n_neighbors=5)
y_pred1a = clf1a.fit_predict(data.values)
colors = np.array(['#377eb8', '#ff7f00'])
plt.scatter(data.values[:, 0], data.values[:, 1], color=colors[(y_pred1a + 1) // 2])
print("Number of outlier ")
y_pred1a = pd.DataFrame(y_pred1a)
y_pred1a[0].value_counts()

clf1b = LocalOutlierFactor(n_neighbors=15)
y_pred1b = clf1b.fit_predict(data.values)
colors = np.array(['#377eb8', '#ff7f00'])
plt.scatter(data.values[:, 0], data.values[:, 1], color=colors[(y_pred1b + 1) // 2])
print("Number of outlier")
y_pred1b = pd.DataFrame(y_pred1b)
y_pred1b[0].value_counts()

clf1c = LocalOutlierFactor(n_neighbors=25)
y_pred1c = clf1c.fit_predict(data.values)
colors = np.array(['#377eb8', '#ff7f00'])
plt.scatter(data.values[:, 0], data.values[:, 1], color=colors[(y_pred1c + 1) // 2])
print("Number of outlier")
y_pred1c = pd.DataFrame(y_pred1c)
y_pred1c[0].value_counts()



#Use EllipticEnvelope with different contamination values to detect outliers

clf2 = EllipticEnvelope(support_fraction=0.8,contamination=0.10)
y_pred2 = clf2.fit_predict(data.values)
data= data.values


#Plot the decision_function and explain results

xx1, yy1 = np.meshgrid(np.linspace(-2, 16, 500), np.linspace(-1, 2, 500))
Z1 = clf2.decision_function(np.c_[xx1.ravel(), yy1.ravel()])
Z1 = Z1.reshape(xx1.shape)
plt.contour(xx1, yy1, Z1, levels=[0], linewidths=2, colors='red' )
plt.scatter(data[:, 0], data[:, 1], color='black')
plt.show()

#NOTE: Using eliptical Envelope with different contamination valuess.
# the volume of the  decision function is increased when the contamination values are higher 
#so  some points that which is outlier might be classify inside the decision function

#Test with different contamination values
#Contamination=0.20, decision function  mostly concentrate where the most point take place. it contain one outliers points 
#Contamination=0.38, decision function  completely concentrate where the most point take place. it does not contain outliers points 
#Contamination-0.10, decision function contain 2 outliers  point

