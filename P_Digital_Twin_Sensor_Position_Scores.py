# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 17:31:07 2017

@author: AICPS
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 19:17:42 2017

@author: AICPS
"""

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
from pathlib import Path
from joblib import Parallel, delayed
import multiprocessing


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

#Initialize Global Variables
featureParentPath='D:/GDrive/DT_Data/Sensor Positioning Data/Features2/';
pathForScore = Path("D:/GDrive/DT_Data/Sensor Positioning Data/PositioningScores3.csv");
pathForFeatureRankingPics="D:/GDrive/DT_Data/Sensor Positioning Data/FeatureRanking3/";
#%%
def getFilenames (runs, angle, channel):
   
    if runs<5:
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
   
    if runs<5:
        folderName=featureParentPath+'Run'+str(runs)+'_'+str(angle*30);
    else:
        folderName=featureParentPath+'Run'+str(runs);
        
    fileName=glob(folderName+'/Channel_'+str(channel)+'_*/');
    fileName1=fileName[0]+'frequencyCWTStastisticsFeatures.csv';
    fileNameLabel1=fileName[0]+'frequencyCWTStastisticsFeaturesLabel.csv';
    return fileName1, fileNameLabel1;


def numberofChannels (*arg):
    if len(arg)==2:
        runs=arg[0];
        angle=arg[1];
    else:
        runs=arg[0];
            
    
    if runs<5:
        folderName=featureParentPath+'Run'+str(runs)+'_'+str(angle*30);
    else:
        folderName=featureParentPath+'Run'+str(runs);
        
    fileName=glob(folderName+'/Channel*/');
    numberofChannels=np.shape(fileName)[0];
    return numberofChannels;

#%%Import the filenames based on the run, angle and channels and read the data
def getData (runs, channel):
        if runs<5:
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
           
            if runs!=4:
                [fileName1, fileNameLabel1, fileName2, fileNameLabel2]= getFilenames(runs,2,channel);
                
                Xtr_time = pd.read_csv(fileName1);
                y3 = pd.read_csv(fileNameLabel1);
                
                Xtr_CWT = pd.read_csv(fileName2);
              
                
                Xtr2 = pd.concat([Xtr_time, Xtr_CWT], axis=1);
                
                Xtr = pd.concat([Xtr0, Xtr1, Xtr2], axis=0);
                y = pd.concat([y1, y2, y3], axis=0);
            else:
                Xtr = pd.concat([Xtr0, Xtr1], axis=0);
                y = pd.concat([y1, y2], axis=0);
  
        else:
            [fileName1, fileNameLabel1, fileName2, fileNameLabel2]= getFilenames(runs,0,channel);
            Xtr_time = pd.read_csv(fileName1);
            y = pd.read_csv(fileNameLabel1);
            Xtr_CWT = pd.read_csv(fileName2);
            Xtr = pd.concat([Xtr_time, Xtr_CWT], axis=1);
            y_Names=Xtr.columns;
    
        return Xtr, y, y_Names;
    
def getDataCWT (runs, channel):
        if runs<5:
            [fileName1, fileNameLabel1]= getFilenamesCWT(runs,0,channel);
            
#            Xtr0  = pd.read_csv(fileName1,usecols=np.array(range(1,59)),skip_blank_lines=True,engine='c',low_memory=False);
            Xtr0  = pd.read_csv(fileName1);
            y1 = pd.read_csv(fileNameLabel1);
            y_Names=Xtr0.columns;
            
            [fileName1, fileNameLabel1]= getFilenamesCWT(runs,1,channel,);
            
#            Xtr1 = pd.read_csv(fileName1,usecols=np.array(range(1,59)),skip_blank_lines=True,engine='c',low_memory=False);
            Xtr1 = pd.read_csv(fileName1);
            y2 = pd.read_csv(fileNameLabel1);
             
            if runs!=4:
                [fileName1, fileNameLabel1]= getFilenamesCWT(runs,2,channel);
                
#                Xtr2 = pd.read_csv(fileName1,usecols=np.array(range(1,59)),skip_blank_lines=True,engine='c',low_memory=False);
                Xtr2 = pd.read_csv(fileName1);
                y3 = pd.read_csv(fileNameLabel1);
               
                
                Xtr = pd.concat([Xtr0, Xtr1, Xtr2], axis=0);
                y = pd.concat([y1, y2, y3], axis=0);
            else:
                Xtr = pd.concat([Xtr0, Xtr1], axis=0);
                y = pd.concat([y1, y2], axis=0);
                
  
        else:
            [fileName1, fileNameLabel1]= getFilenamesCWT(runs,0,channel);
#            Xtr= pd.read_csv(fileName1,usecols=np.array(range(1,59)),skip_blank_lines=True,engine='c',low_memory=False);
            Xtr= pd.read_csv(fileName1);
            
            y = pd.read_csv(fileNameLabel1);
            y_Names=Xtr.columns;
    
        return Xtr, y, y_Names;
 
#%% Get the data for training from each channel and runs    
def getScores (runs, channel):

        [Xtr, y, y_Names] = getDataCWT (runs, channel);
        
        #fill NaN values with zeros
        Xtr.replace([np.finfo(np.float64).max, np.finfo(np.float64).min], np.nan);
        #Xtr.fillna(0, inplace=True);
        Xtr.fillna(Xtr.mean(),inplace=True);
        #%%remove columns if all are zeros
        Xtr = Xtr.loc[:, (Xtr != 0).any(axis=0)];
        #X = Xtr.as_matrix().astype(np.float)
        X=Xtr.values;
        X[np.isinf(X)]=X.mean();
        X[np.isneginf(X)]=X.mean();
        X[np.isnan(X)]=0;
        X = X[~np.all(X == 0, axis=1)];
        #%%
        #normalize the data
        min_max_scaler = preprocessing.MinMaxScaler();
        X = min_max_scaler.fit_transform(X);
        #X =preprocessing.normalize(X, norm='l2')
        #binalrize the class data
        #        y=pd.get_dummies(y);
        
        #Shuffle The data
        y=np.ravel(y);
        
        X, y = shuffle(X, y)  
           
        #y = label_binarize(y, classes=[0, 1, 2,3])
        #n_classes = y.shape[1]
        #%%
        #separate the training and the test data
        offset = int(X.shape[0] * 0.8)
        X_train, y_train = X[:offset], y[:offset]
        X_test, y_test = X[offset:], y[offset:]
        
        # Test the classifiers
        n_estimators=100;
        #        n_iter=200;
        #        max_depth=3;
        random_state = np.random.RandomState(0);
        n_jobs=10;
        
        models = [#DecisionTreeClassifier(max_depth=None),
        #                  MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10, 4), random_state=1),
        #                  SGDClassifier(loss="hinge", alpha=0.01, n_iter=n_iter, fit_intercept=True),
        #          svm.SVC(kernel='linear', probability=True,random_state=random_state),
        #                  RandomForestClassifier(n_estimators=n_estimators),
                  ExtraTreesClassifier(n_estimators=n_estimators,random_state=random_state, n_jobs= n_jobs)]
        #                  GaussianNB(),
        #                  GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
        #                  AdaBoostClassifier(DecisionTreeClassifier(max_depth=3),
        #                                     n_estimators=n_estimators)]
        #%%
        score=0;
        for model in models:
                clf = clone(model)
                print(model.__format__)
                clf = model.fit(X_train, y_train)
                score = clf.score(X_test, y_test)

                
                importances = clf.feature_importances_
                std = np.std([tree.feature_importances_ for tree in clf.estimators_],
                             axis=0)
                indices = np.argsort(importances)[::-1]
                
                # Print the feature ranking
                print("Feature ranking:")
                
                for f in range(X.shape[1]):
                    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
                
                # Plot the feature importances of the forest
                #range=X.shape[1]
                plt.figure()
                plt.title("Feature importances")
                plt.bar(range(X.shape[1]), importances[indices],
                       color="r", yerr=std[indices], align="center")
                plt.xticks(range(X.shape[1]), indices)
                plt.xlim([-1, 0.45*X.shape[1]])
                name=pathForFeatureRankingPics+'FeatureRanking_Run_'+str(runs)+'_Channel_'+str(channel)+'.png' ;
                plt.savefig(name,dpi=1200)
                plt.show()


        return score;         

 
#%%
#
#if pathForScore.is_file()==False:
#    np.savetxt(pathForScore, [], delimiter=',', header="Run,  Channel,  Score");
#else:
#    print('\nFile: ', pathForScore, ' already exists!')

#%%
for runs in range(5,10):
    if runs<3:
        channelNumbers=22;
    else:
        channelNumbers=23;
        
    for channel in range(1, 19):
        print('Run: ', runs)
        print('Channel: ', channel)
        score= getScores (runs, channel);
        f =open(pathForScore,'a');
        df = pd.DataFrame([[runs,channel,score]])
        df.to_csv(f,index = False,header= False);
        f.close();
        
        print('Score:',score,'\n')
  
#%%
#
#for runs in range(5,6):
#    if runs<3:
#        channelNumbers=22;
#    else:
#        channelNumbers=23;
#    for channel in range(10,11):
#         print('Now: Run->',runs,' Channel->',channel)
#         [Xtr, y, y_Names] = getDataCWT (runs, channel);
#         if np.shape(Xtr)[0] != np.shape(y)[0]:
#           print('Error! in: Run->',runs,' Channel->',channel,' X Shape:',np.shape(Xtr)[0],'Y Shape:',np.shape(y)[0])

#%%
#[Xtr, y, y_Names] = getDataCWT (5, 1);


#
#[fileName1, fileNameLabel1]= getFilenamesCWT(5,0,1);
#Xtr0  = pd.read_csv(fileName1,usecols=np.array(range(1,59)),skip_blank_lines=True,engine='c',low_memory=False);

















