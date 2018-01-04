# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 12:27:08 2017

@author: Sujit Rokka Chhetri
"""


import numpy as np
import pandas as pd
from glob import glob
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score

#from sklearn.svm import LinearSVC
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import RandomizedPCA

from sklearn import clone
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              AdaBoostClassifier)
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

#%%Initialize Global Variables
featureParentPath='D:/GDrive/DT_Data/Sensor Positioning Data/Features/';
runs=3;
channel=10;

def getFilenames (runs, angle, channel):
   
    if runs<4:
        folderName=featureParentPath+'Run'+str(runs)+'_'+str(angle*30);
    else:
        folderName=featureParentPath+'Run'+str(runs);
        
    fileName=glob(folderName+'/Channel_'+str(channel)+'_*/');
    fileName1=fileName[0]+'timeFeatures.csv';
    fileNameLabel1=fileName[0]+'timeFeaturesLabel.csv';
    fileName2=fileName[0]+'frequencyCWTStastisticsFeatures.csv';
    fileNameLabel2=fileName[0]+'frequencyCWTStastisticsFeaturesLabel.csv';
    return fileName1, fileNameLabel1, fileName2, fileNameLabel2;

def getFilenamesCWT (runs, angle, channel):
   
    if runs<4:
        folderName=featureParentPath+'Run'+str(runs)+'_'+str(angle*30);
    else:
        folderName=featureParentPath+'Run'+str(runs);
        
    fileName=glob(folderName+'/Channel_'+str(channel)+'_*/');
    fileName1=fileName[0]+'frequencyCWTStastisticsFeatures.csv';
    fileNameLabel1=fileName[0]+'frequencyCWTStastisticsFeaturesLabel.csv';
    return fileName1, fileNameLabel1;

#%%Import the filenames based on the run, angle and channels and read the data
def getData (runs, channel):
        if runs<4:
            [fileName1, fileNameLabel1, fileName2, fileNameLabel2]= getFilenames(runs,0,channel);
            
            Xtr_time = pd.read_csv(fileName1);
            y1 = pd.read_csv(fileNameLabel1);
            Xtr_CWT = pd.read_csv(fileName2);
     
            Xtr0 = pd.concat([Xtr_time, Xtr_CWT], axis=1);
          
            y_Names=Xtr0.columns;
            
            [fileName1, fileNameLabel1, fileName2, fileNameLabel2]= getFilenames(runs,1,channel);
            
            Xtr_time = pd.read_csv(fileName1);
            y2 = pd.read_csv(fileNameLabel1);
             
             
            Xtr_CWT = pd.read_csv(fileName2);
           
            
            Xtr1 = pd.concat([Xtr_time, Xtr_CWT], axis=1);
           
        
            [fileName1, fileNameLabel1, fileName2, fileNameLabel2]= getFilenames(runs,2,channel);
            
            Xtr_time = pd.read_csv(fileName1);
            y3 = pd.read_csv(fileNameLabel1);
            
            Xtr_CWT = pd.read_csv(fileName2);
          
            
            Xtr2 = pd.concat([Xtr_time, Xtr_CWT], axis=1);
            
            Xtr = pd.concat([Xtr0, Xtr1, Xtr2], axis=0);
            y = pd.concat([y1, y2, y3], axis=0);
  
        else:
            [fileName1, fileNameLabel1, fileName2, fileNameLabel2]= getFilenames(runs,0,channel);
            Xtr_time = pd.read_csv(fileName1);
            y = pd.read_csv(fileNameLabel1);
            Xtr_CWT = pd.read_csv(fileName2);
            Xtr = pd.concat([Xtr_time, Xtr_CWT], axis=1);
            y_Names=Xtr.columns;
    
        return Xtr, y, y_Names;
    
def getDataCWT (runs, channel):
        if runs<4:
            [fileName1, fileNameLabel1]= getFilenamesCWT(runs,0,channel);
            
            Xtr0  = pd.read_csv(fileName1);
            y1 = pd.read_csv(fileNameLabel1);
            y_Names=Xtr0.columns;
            
            [fileName1, fileNameLabel1]= getFilenamesCWT(runs,1,channel);
            
            Xtr1 = pd.read_csv(fileName1);
            y2 = pd.read_csv(fileNameLabel1);
             
         
            [fileName1, fileNameLabel1]= getFilenamesCWT(runs,2,channel);
            
            Xtr2 = pd.read_csv(fileName1);
            y3 = pd.read_csv(fileNameLabel1);
           
            
            Xtr = pd.concat([Xtr0, Xtr1, Xtr2], axis=0);
            y = pd.concat([y1, y2, y3], axis=0);
  
        else:
            [fileName1, fileNameLabel1]= getFilenamesCWT(runs,0,channel);
            Xtr= pd.read_csv(fileName1);
            y = pd.read_csv(fileNameLabel1);
            y_Names=Xtr.columns;
    
        return Xtr, y, y_Names;
 
#%% Get the data for training from each channel and runs    

[Xtr, y, y_Names] = getDataCWT (runs, channel);
#%%
#fill NaN values with zeros
Xtr.replace([np.inf, -np.inf], np.nan);
Xtr.fillna(0, inplace=True);

#remove columns if all are zeros
Xtr = Xtr.loc[:, (Xtr != 0).any(axis=0)];
X = Xtr.as_matrix().astype(np.float)


#%%normalize the data
min_max_scaler = preprocessing.MinMaxScaler();
X = min_max_scaler.fit_transform(X);
#X =preprocessing.normalize(X, norm='l2')
#Shuffle The data
X, y = shuffle(X, y)  
   
#y = label_binarize(y, classes=[0, 1, 2,3])
#n_classes = y.shape[1]
#%%separate the training and the test data
offset = int(X.shape[0] * 0.8)
X_train, y_train = X[:offset], y[:offset]
X_test, y_test = X[offset:], y[offset:]

#%% Test the classifiers
n_estimators=100
n_iter=200
random_state = np.random.RandomState(0)

models = [DecisionTreeClassifier(max_depth=None),
          MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10, 4), random_state=1),
          SGDClassifier(loss="hinge", alpha=0.01, n_iter=n_iter, fit_intercept=True),
          svm.SVC(kernel='linear', probability=True,random_state=random_state),
          RandomForestClassifier(n_estimators=n_estimators),
          ExtraTreesClassifier(n_estimators=n_estimators),
          GaussianNB(),
          GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
          AdaBoostClassifier(DecisionTreeClassifier(max_depth=3),
                             n_estimators=n_estimators)]

#%%
for model in models:
        clf = clone(model)
        clf = model.fit(X_train, y_train)
        y_=clf.predict(X_test) 
        score = clf.score(X_test, y_test)
#        y_score = clf.decision_function(X_test)
        
        acc=accuracy_score(y_test, y_)
        
        print(model.__format__)
        print('Classifier Score')
        print(score)
        
        print('Accuracy Score')
        print(acc)
        
#        fpr = dict()
#        tpr = dict()
#        roc_auc = dict()
#        for i in range(n_classes):
#            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
#            roc_auc[i] = auc(fpr[i], tpr[i])
#        # Compute micro-average ROC curve and ROC area
#        fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
#        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
#        
#        plt.figure()
#        lw = 2
#        plt.plot(fpr[2], tpr[2], color='darkorange',
#                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
#        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
#        plt.xlim([0.0, 1.0])
#        plt.ylim([0.0, 1.05])
#        plt.xlabel('False Positive Rate')
#        plt.ylabel('True Positive Rate')
#        plt.title('Receiver operating characteristic example')
#        plt.legend(loc="lower right")
#        plt.show()
        
#     

























