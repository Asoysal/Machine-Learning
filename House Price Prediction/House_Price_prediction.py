# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 18:51:01 2020

@author: Alper Soysal
"""

import pandas as pd
import numpy as np
#from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import SVR
import seaborn as sns
from math import sqrt
import matplotlib.pyplot as plt

from sklearn import linear_model,metrics


from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor


Submission = pd.read_csv("sample_submission.csv")

# Load the Data
House_data_train = pd.read_csv("train.csv")
House_data_test = pd.read_csv("test.csv")


## extract input trainig Data
X_train = House_data_train.drop(columns = "SalePrice")
#
## extract output trainig Data
y_train = House_data_train.SalePrice
y_train = y_train.reset_index(drop=True)
#
## extract input test Data
X_test = House_data_test

X_train = X_train.drop(columns = "Id")
X_test = X_test.drop(columns = "Id")

House_data=pd.concat([X_train,X_test] , axis = 0 ,  ignore_index=True)
 
House_data = House_data.assign(SalePrice = y_train) 

Num_of_missing_value_House_data = House_data.isnull().sum()


#######################################
# heat map for nana values
#Num_of_missing_value_House_data1 = House_data.isnull()
#a = House_data[Num_of_missing_value_House_data.index]
#b  = a.isnull()
#Num_of_missing_value_House_data = pd.DataFrame(Num_of_missing_value_House_data)
#sns.heatmap(b, yticklabels = False)
#######################################################################33
################################################################
# plot the table figure 2
# plot the feautures that has a nan values most
#total_rows = len(House_data.axes[0])
#Num_of_missing_value_House_data = Num_of_missing_value_House_data [ Num_of_missing_value_House_data[0]  > 0 ]
#plt.bar(Num_of_missing_value_House_data.index,Num_of_missing_value_House_data[0])
#
#plt.title("Total number of NaN value")
#plt.show()
#############################################################33


# drop the features which has mostly nan value
# print(House_data_train.isnull().sum())    examine nan value into the dataframe
#[ Alley , PoolQC ,Fence , MiscFeature ]
House_data = House_data.drop(columns = "Alley")
House_data = House_data.drop(columns = "PoolQC")
House_data = House_data.drop(columns = "Fence")
House_data = House_data.drop(columns = "MiscFeature")

# extract categorical and numeric value for transformation

House_data_numeric =  House_data.select_dtypes(np.number)
House_data_categorical = House_data.select_dtypes(np.object)


#############################################################
# plot the table figure 1
##a = ['Categorical' , 'Numeric']
##b = [len(House_data_numeric.axes[1]),len(House_data_categorical.axes[1])]
#plt.barh(type_of_data[0],type_of_data.index,height= 0.4)
#for index, value in enumerate(type_of_data[0]):
#    plt.text(value, index, str(value))
#
#plt.title("Type of the features")
#plt.show()
###############################################################



# Categorical encoding 
# 2 methods used : Label-Encoding and One-Hot-Encoder
# methods decided visually for each columns
# if the features includes name of street, devices etc One-Hot-Encoder is used,
# if the features comprises adjective like high low medium Label-Encoding is used.

Num_of_none_value_House_data_categorical = House_data_categorical.isnull().sum()

#################################################################
# plot the table figure 3
#Num_of_none_value_House_data_categorical = pd.DataFrame(Num_of_none_value_House_data_categorical)
#
#total_rows = len(House_data.axes[0])
#Num_of_none_value_House_data_categorical = Num_of_none_value_House_data_categorical[Num_of_none_value_House_data_categorical[0] != 0]
#
#Num_of_none_value_House_data_categorical.columns = ['The number of NaN values']
#cell_text = np.round(Num_of_none_value_House_data_categorical.values, 3)
#row_labels = Num_of_none_value_House_data_categorical.index
#col_labels = Num_of_none_value_House_data_categorical.columns
#
## kw = dict(cellColours=[["#EEECE1"] * len(data.iloc[0])] * len(data))  # to fill cells with color
#ytable = plt.table(cellText=cell_text, rowLabels=row_labels, colLabels=col_labels,cellLoc='center',  loc="center")
#plt.axis("off")
#plt.grid(False)
#plt.show()
###########################################################



# filling " MasVnrType BsmtQual:BsmtCond BsmtExposure: BsmtFinType1: : BsmtFinType2:
# FireplaceQu: GarageType:  GarageFinish: GarageQual: GarageCond: " with "None"



House_data_categorical[['MasVnrType','BsmtQual','BsmtCond','BsmtExposure',
                        'BsmtFinType1','BsmtFinType2','FireplaceQu','GarageType'
                        ,'GarageFinish','GarageQual','GarageCond']] = House_data_categorical[['MasVnrType','BsmtQual','BsmtCond'
                                                                  ,'BsmtExposure','BsmtFinType1','BsmtFinType2','FireplaceQu'
                                                                  ,'GarageType','GarageFinish','GarageQual',
                                                                  'GarageCond']].fillna('None')

# the rest of the features filled by the most frequent value.
# *MSZoning: *Utilities: Exterior1st  Exterior2nd Electrical  KitchenQual  Functional  SaleType   


House_data_categorical['MSZoning'] = House_data_categorical['MSZoning'].fillna(House_data_categorical['MSZoning'].value_counts().argmax())
House_data_categorical['Utilities'] = House_data_categorical['Utilities'].fillna(House_data_categorical['Utilities'].value_counts().argmax())
House_data_categorical['Exterior1st'] = House_data_categorical['Exterior1st'].fillna(House_data_categorical['Exterior1st'].value_counts().argmax())
House_data_categorical['Exterior2nd'] = House_data_categorical['Exterior2nd'].fillna(House_data_categorical['Exterior2nd'].value_counts().argmax())
House_data_categorical['Electrical'] = House_data_categorical['Electrical'].fillna(House_data_categorical['Electrical'].value_counts().argmax())
House_data_categorical['KitchenQual'] = House_data_categorical['KitchenQual'].fillna(House_data_categorical['KitchenQual'].value_counts().argmax())
House_data_categorical['Functional'] = House_data_categorical['Functional'].fillna(House_data_categorical['Functional'].value_counts().argmax())
House_data_categorical['SaleType'] = House_data_categorical['SaleType'].fillna(House_data_categorical['SaleType'].value_counts().argmax())





# dealing with nan value in the numeric house data
                       
Num_of_none_value_House_data_numeric = House_data_numeric.isnull().sum()            

#########################################################################
## plot the table figure 4
#Num_of_none_value_House_data_numeric = pd.DataFrame(Num_of_none_value_House_data_numeric)
#Num_of_none_value_House_data_numeric = Num_of_none_value_House_data_numeric.drop(index ="SalePrice")
#total_rows = len(House_data.axes[0])
#Num_of_none_value_House_data_numeric = Num_of_none_value_House_data_numeric[Num_of_none_value_House_data_numeric[0] != 0]
#
#Num_of_none_value_House_data_numeric.columns = ['The number of NaN values']
#cell_text = np.round(Num_of_none_value_House_data_numeric.values, 3)
#row_labels = Num_of_none_value_House_data_numeric.index
#col_labels = Num_of_none_value_House_data_numeric.columns
#
## kw = dict(cellColours=[["#EEECE1"] * len(data.iloc[0])] * len(data))  # to fill cells with color
#ytable = plt.table(cellText=cell_text, rowLabels=row_labels, colLabels=col_labels,cellLoc='center',  loc="center")
#plt.axis("off")
#plt.grid(False)
#plt.show()
############################################################################################

           

# filling the nan value in LotFrontage:  GarageYrBlt:

# find a mean value for columns LotFrontage
mean_LotFrontage = np.mean(House_data_numeric["LotFrontage"]) 

#fill nan value with mean value
House_data_numeric['LotFrontage'] = House_data_numeric['LotFrontage'].fillna(round(mean_LotFrontage))


  # find a mean value for columns GarageYrBlt                     
mean_GarageYrBlt = np.mean(House_data_numeric["GarageYrBlt"])
#fill nan value with mean value
House_data_numeric['GarageYrBlt'] = House_data_numeric['GarageYrBlt'].fillna(round(mean_GarageYrBlt))


# fill the rest wiht 0

House_data_numeric= House_data_numeric.fillna(0)

#House_data_numeric1 = House_data_numeric.dropna()
#
#Edited_House_data = pd.concat([House_data_numeric,House_data_categorical],axis=1 )  
#

#
#corr = House_data_numeric1.corr()
#print (corr['SalePrice'].sort_values(ascending=False)[:5], '\n')
#print (corr['SalePrice'].sort_values(ascending=False)[-5:])
#
#plt.scatter(x=House_data_numeric1['GrLivArea'], y=House_data_numeric1['SalePrice'],color = '#57595D')
#plt.ylabel('Sale Price')
#plt.xlabel('GrLivArea')
#
#m, b = np.polyfit(House_data_numeric1['GrLivArea'], House_data_numeric1['SalePrice'], 1)
#plt.plot(House_data_numeric1['GrLivArea'], m*House_data_numeric1['GrLivArea'] + b,color = 'r')
#
#plt.show()
#
#



#Edited_House_data = pd.concat([House_data_numeric,House_data_categorical],axis=1 )  

#import seaborn as sns
#
#corr = Edited_House_data.corr()
#ax = sns.heatmap(
#    corr, 
#    vmin=-1, vmax=1, center=0,
#    cmap=sns.diverging_palette(20, 220, n=200),
#    square=True
#)
#ax.set_xticklabels(
#    ax.get_xticklabels(),
#    rotation=45,
#    horizontalalignment='right'
#);
##
#House_data_categorical = pd.concat([House_data_categorical,House_data['SalePrice']],axis=1 )  
#
#quality_pivot = House_data_categorical.pivot_table(index='GarageQual',
#                  values='SalePrice', aggfunc=np.median)
#quality_pivot.plot(kind='bar', color='blue')
#plt.xlabel('Overall Quality')
#plt.ylabel('Median Sale Price')
#plt.xticks(rotation=0)
#plt.show()
#









# label encoding features : LotShape: LandContour: Utilities: LandSlope:  BldgType: HouseStyle: : 
#: ExterQual: ExterCond: BsmtQual: BsmtCond: BsmtExposure: BsmtFinType1: 
#BsmtFinType2: HeatingQC: Electrical: KitchenQual: 
#Functional: : GarageFinish: GarageQual: GarageCond: PavedDrive: 






# changing data type from object to category
House_data_categorical= House_data_categorical.astype('category')

####################################################################
# ploting for finding a features with low varinace
#for i in range(38):
#    a = House_data_categorical.iloc[:,i].value_counts()
#    plt.bar(a.index,a,height= 0.4)
#    plt.title(House_data_categorical.iloc[0,i])
#    plt.show()
#####################################################################



########################################################################
#figure 6
#a = ['Heating','RoofMatl','Condition2','Street','Utilities']
#b = House_data_categorical [a]
#for i in range(5):
#    c = b.iloc[:,i].value_counts()
#    plt.barh(c.index,c)
#    for index, value in enumerate(c):
#       plt.text(value, index, str(value))
#    plt.title(b.columns[i])
#    plt.show()
##############################################################

# drop the features which has a low variance. according the plot
#    Street: Utilities Condition2: RoofMatl: Heating: 

House_data_categorical = House_data_categorical.drop(['Heating','RoofMatl','Condition2','Street','Utilities'],axis=1)


Edited_House_data = pd.concat([House_data_categorical,House_data_numeric],axis=1 )  



#
#corr = House_data_numeric.iloc[:,range(20,37)].corr()
#ax = sns.heatmap(
#    corr, 
#    vmin=-1, vmax=1, center=0,
#    cmap=sns.diverging_palette(20, 220, n=200),
#    square=True
#)
#ax.set_xticklabels(
#    ax.get_xticklabels(),
#    rotation=45,
#    horizontalalignment='right'
#);



#one hot encoding: MSZoning:    LotConfig: Neighborhood: Condition1:  
#RoofStyle: Exterior1st: Exterior2nd: MasVnrType: Foundation: 
# CentralAir: GarageType: MiscFeature: 
#SaleType: SaleCondition: 


House_data_categorical = pd.get_dummies(House_data_categorical, columns=["MSZoning"])
House_data_categorical = pd.get_dummies(House_data_categorical, columns=["LotConfig"])
House_data_categorical = pd.get_dummies(House_data_categorical, columns=["Neighborhood"])
House_data_categorical = pd.get_dummies(House_data_categorical, columns=["Condition1"])
House_data_categorical = pd.get_dummies(House_data_categorical, columns=["RoofStyle"])
House_data_categorical = pd.get_dummies(House_data_categorical, columns=["Exterior1st"])
House_data_categorical = pd.get_dummies(House_data_categorical, columns=["Exterior2nd"])
House_data_categorical = pd.get_dummies(House_data_categorical, columns=["MasVnrType"])
House_data_categorical = pd.get_dummies(House_data_categorical, columns=["Foundation"])
House_data_categorical = pd.get_dummies(House_data_categorical, columns=["CentralAir"])
House_data_categorical = pd.get_dummies(House_data_categorical, columns=["GarageType"])
House_data_categorical = pd.get_dummies(House_data_categorical, columns=["SaleType"])
House_data_categorical = pd.get_dummies(House_data_categorical, columns=["SaleCondition"])

# label encoding
#X_train_categorical["LotShape","LandContour","Utilities","LandSlope","BldgType"
#,"HouseStyle","ExterQual","ExterCond","BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1"
#,"BsmtFinType2","HeatingQC","Electrical","KitchenQual","Functional","GarageFinish","GarageQual"
#,"GarageCond","PavedDrive"] = X_train_categorical["LotShape","LandContour","Utilities","LandSlope","BldgType"
#,"HouseStyle","ExterQual","ExterCond","BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1"
#,"BsmtFinType2","HeatingQC","Electrical","KitchenQual","Functional","GarageFinish","GarageQual"
#,"GarageCond","PavedDrive"].cat.codes 



House_data_categorical["LotShape"]= House_data_categorical["LotShape"].cat.codes 
House_data_categorical["LandContour"]= House_data_categorical["LandContour"].cat.codes 
House_data_categorical["FireplaceQu"]= House_data_categorical["FireplaceQu"].cat.codes 
House_data_categorical["LandSlope"]= House_data_categorical["LandSlope"].cat.codes 
House_data_categorical["BldgType"]= House_data_categorical["BldgType"].cat.codes 
House_data_categorical["HouseStyle"]= House_data_categorical["HouseStyle"].cat.codes 
House_data_categorical["ExterQual"]= House_data_categorical["ExterQual"].cat.codes 
House_data_categorical["ExterCond"]= House_data_categorical["ExterCond"].cat.codes 
House_data_categorical["BsmtQual"]= House_data_categorical["BsmtQual"].cat.codes 
House_data_categorical["BsmtCond"]= House_data_categorical["BsmtCond"].cat.codes 
House_data_categorical["BsmtExposure"]= House_data_categorical["BsmtExposure"].cat.codes 
House_data_categorical["BsmtFinType1"]= House_data_categorical["BsmtFinType1"].cat.codes 
House_data_categorical["BsmtFinType2"]= House_data_categorical["BsmtFinType2"].cat.codes 
House_data_categorical["HeatingQC"]= House_data_categorical["HeatingQC"].cat.codes 
House_data_categorical["Electrical"]= House_data_categorical["Electrical"].cat.codes 
House_data_categorical["KitchenQual"]= House_data_categorical["KitchenQual"].cat.codes 
House_data_categorical["Functional"]= House_data_categorical["Functional"].cat.codes 
House_data_categorical["GarageFinish"]= House_data_categorical["GarageFinish"].cat.codes 
House_data_categorical["GarageQual"]= House_data_categorical["GarageQual"].cat.codes 
House_data_categorical["GarageCond"]= House_data_categorical["GarageCond"].cat.codes 
House_data_categorical["PavedDrive"]= House_data_categorical["PavedDrive"].cat.codes 




#House_data_categorical = pd.concat([House_data_categorical,House_data['SalePrice']],axis=1 )  



#corrcat = House_data_categorical.corr()
#print (corrcat['SalePrice'].sort_values(ascending=False)[:5], '\n')
#print (corrcat['SalePrice'].sort_values(ascending=False)[-5:])
#
#
#condition_pivot = House_data_categorical.pivot_table(index='ExterQual', values='SalePrice', aggfunc=np.median)
#condition_pivot.plot(kind='bar', color='blue')
#plt.xlabel('Sale Condition')
#plt.ylabel('Median Sale Price')
#plt.xticks(rotation=0)
#plt.show()
#












# concatinate numerical features and encoded features
Edited_House_data = pd.concat([House_data_numeric,House_data_categorical],axis=1 )  













## split the data
X_train_Fmodel = Edited_House_data.loc[range(1460),:]

y_train_Fmodel = X_train_Fmodel['SalePrice']
X_train_Fmodel = X_train_Fmodel.drop(columns = "SalePrice")


X_Test_Fmodel = Edited_House_data.loc[Edited_House_data['SalePrice'] == 0]
X_Test_Fmodel = X_Test_Fmodel.drop(columns = "SalePrice")







#######################################################################################
### creating a data for outlier finding
# it is not worth it to use. result shows that it affects the result slightly
#X_train_LOF = X_train_numeric
#X_train_LOF=X_train_LOF.astype('int64')
#
##Identify outliers for training data  number of outlier = 219

#clf = LocalOutlierFactor(contamination=0.20)
#y_pred = clf.fit_predict(X_train_Fmodel)
#y_pred=pd.DataFrame(y_pred) 
#y_pred[0].value_counts()
#
#print("number of outliers: %f" % 219 )
#
##Delete outliers for X_train 
#X_train_Fmodel = X_train_Fmodel.assign(Outlier = y_pred) 
#X_train_without_outlier = X_train_Fmodel [ X_train_Fmodel["Outlier"]  == 1 ]
#X_train_without_outlier = X_train_without_outlier.reset_index(drop=True)
#X_train_without_outlier = X_train_without_outlier.drop(columns = "Outlier")
#
##Delete outliers for y_train 
#y_train = pd.DataFrame(y_train) 
#y_train = y_train.assign(Outlier = y_pred) 
#y_train_without_outlier = y_train [ y_train["Outlier"]  == 1 ]
#y_train_without_outlier = y_train_without_outlier.reset_index(drop=True)
#y_train_without_outlier = y_train_without_outlier.drop(columns = "Outlier")
#
#x_train_M,x_test_M,y_train_M,y_test_M = train_test_split(X_train_without_outlier,y_train_without_outlier,test_size=0.33,random_state=0)
#
#

################################################################################################


# data sampling
x_train_M,x_test_M,y_train_M,y_test_M = train_test_split(X_train_Fmodel,y_train_Fmodel,test_size=0.33,random_state=0)




###################################################################################
# Tryin Different Model

#xgb regressor
xgb =XGBRegressor( booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=0.6, gamma=0,
             importance_type='gain', learning_rate=0.01, max_delta_step=0,
             max_depth=4, min_child_weight=1.5, n_estimators=2400,
             n_jobs=1, nthread=None, objective='reg:linear',
             reg_alpha=0.6, reg_lambda=0.6, scale_pos_weight=1, 
             silent=None, subsample=0.8, verbosity=1)

xgb.fit(x_train_M, y_train_M)

Prediction_xgb = xgb.predict(x_test_M)

errors_model_xgb = [ metrics.mean_absolute_error(y_test_M, Prediction_xgb), (metrics.mean_squared_error(y_test_M, Prediction_xgb)),metrics.r2_score(y_test_M, Prediction_xgb)]

errors_model_xgb = pd.DataFrame(errors_model_xgb)
errors_model_xgb = errors_model_xgb.transpose()  


#
###features importance
#feat_importances = pd.Series(xgb.feature_importances_, index=x_test_M.columns)
#feat_importances.nlargest(20).plot(kind='barh')
###
#






print(' xgb mean absolute error test = ' + str((metrics.mean_absolute_error(y_test_M, Prediction_xgb))))

print(' xgb Root Mean Square Error test = ' + str(sqrt(metrics.mean_squared_error(y_test_M, Prediction_xgb))))

print(' xgb R-squared  Error test = ' + str((metrics.r2_score(y_test_M, Prediction_xgb))))




####################################################################
#lgbm regressor
lgbm = LGBMRegressor(objective='regression', 
                                       num_leaves=4,
                                       learning_rate=0.01, 
                                       n_estimators=12000, 
                                       max_bin=200, 
                                       bagging_fraction=0.75,
                                       bagging_freq=5, 
                                       bagging_seed=7,
                                       feature_fraction=0.4, 
                                       )
lgbm.fit(x_train_M, y_train_M,eval_metric='rmse')

Prediction_lgbm = lgbm.predict(x_test_M)



errors_model_lgbm = [metrics.mean_absolute_error(y_test_M, Prediction_lgbm),sqrt(metrics.mean_squared_error(y_test_M, Prediction_lgbm)),metrics.r2_score(y_test_M, Prediction_lgbm)]
errors_model_lgbm = pd.DataFrame(errors_model_lgbm)
errors_model_lgbm = errors_model_lgbm.transpose()  

print(' lgbm mean absolute error test = ' + str((metrics.mean_absolute_error(y_test_M, Prediction_lgbm))))


print(' lgbm Root Mean Square log Error test = ' + str(sqrt(metrics.mean_squared_error(y_test_M, Prediction_lgbm))))

print(' lgbm R-squared  Error test = ' + str((metrics.r2_score(y_test_M, Prediction_lgbm))))


#
###features importance
#feat_importances = pd.Series(lgbm.feature_importances_, index=x_test_M.columns)
#feat_importances.nlargest(20).plot(kind='barh')
###












##support vector regressor
#svr_lin = SVR(kernel='linear', C=10, gamma='auto')
#
#
#svr_lin.fit(x_train_M, y_train_M)
#
#Prediction_svr = svr_lin.predict(x_test_M)
#
#errors_model_svr_lin = [metrics.mean_absolute_error(y_test_M, Prediction_svr),sqrt(metrics.mean_squared_error(y_test_M, Prediction_svr)),metrics.r2_score(y_test_M, Prediction_svr)]
#errors_model_svr_lin = pd.DataFrame(errors_model_svr_lin)
#errors_model_svr_lin = errors_model_svr_lin.transpose()  
#print(' svr_lin mean absolute error test = ' + str(metrics.mean_absolute_error(y_test_M, Prediction_svr)))
#print(' svr_lin Root Mean Square Error test = ' + str(sqrt(metrics.mean_squared_error(y_test_M, Prediction_svr))))
#print(' svr_lin R-squared  Error test = ' + str((metrics.r2_score(y_test_M, Prediction_svr))))
#  
##plt.scatter(Prediction_lin, y_test_M, alpha=.75, color='b')
##plt.xlabel('Predicted Price')
##plt.ylabel('Actual Price')
##plt.title('Support Vector Regression')
##overlay = 'R^2 is: {}\nRMSE is: {}'.format(
##                    ridge_model.score(X_test, y_test),
##                    mean_squared_error(y_test, preds_ridge))
##plt.annotate(s=overlay,xy=(12.1,10.6),size='x-large')
##plt.show()
#


################################################################################
lr = linear_model.LinearRegression()
from sklearn.metrics import mean_squared_error

lr_model = lr.fit(x_train_M,y_train_M)

Prediction_lin = lr.predict(x_test_M)

errors_model_lin = [metrics.mean_absolute_error(y_test_M, Prediction_lin),sqrt(metrics.mean_squared_error(y_test_M, Prediction_lin)),metrics.r2_score(y_test_M, Prediction_lin)]
errors_model_lin = pd.DataFrame(errors_model_lin)
errors_model_lin = errors_model_lin.transpose()  

print(' LinearRegression mean absolute error test = ' + str(metrics.mean_absolute_error(y_test_M, Prediction_lin)))
print(' LinearRegression Root Mean Square Error test = ' + str(sqrt(metrics.mean_squared_error(y_test_M, Prediction_lin))))
print(' LinearRegression R-squared  Error test = ' + str((metrics.r2_score(y_test_M, Prediction_lin))))
  




#
#
#feat_importances = pd.Series(lr.coef_, index=x_test_M.columns)
#feat_importances.nlargest(20).plot(kind='barh')
###
#









############################################################################



errors_Models=pd.concat([errors_model_xgb,errors_model_lgbm,errors_model_lin,errors_model_svr_lin] , ignore_index=True)  
errors_Models.columns = ['mean absolute error ','Root Mean Square Error','R-squared  Error']
errors_Models.rename(index={0:'XGBRegressor',1:'LGBMRegressor ',2:'LinearRegression',3: 'Support Vector Regresssion'}, inplace=True)


 
########################################################
### plot for figure 7
#x_test_M_P = x_test_M.reset_index()
#y_test_M_P = y_test_M.reset_index()
#y_test_M_P = y_test_M_P.drop(columns = 'index')
#x_test_M_P = x_test_M_P.drop(columns = 'index')
#Prediction_xgb_P = pd.DataFrame(Prediction_lin)
#df=pd.DataFrame({'Id': range(480), 'Real Price': y_test_M_P.iloc[range(480),0], 'Predict Price': Prediction_xgb_P.iloc[range(480),0] })
## multiple line plot
## multiple line plot
#plt.plot( 'Id', 'Real Price', data=df, marker='o', markerfacecolor='blue', markersize=4, color='skyblue', linewidth=2)
#plt.plot( 'Id', 'Predict Price', data=df, marker='o',markerfacecolor='red',markersize=4, color='olive', linewidth=2)
#plt.legend()

#######################################################################################################33


# prediction for initial test data

Prediction_xgb_submission = xgb.predict(X_Test_Fmodel)

Prediction_xgb_submission = pd.DataFrame(Prediction_xgb_submission)

Submission = Submission.drop(columns = 'SalePrice')

Submission = Submission.assign(SalePrice = Prediction_xgb_submission) 





Submission.to_csv (r'C:\Users\Alper Soysal\Desktop\ export_dataframe.csv', index = False)