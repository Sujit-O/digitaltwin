# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 12:27:08 2017

@author: Sujit Rokka Chhetri
"""

#%% Import all the libraries
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.utils import shuffle
import os
import argparse

#% Scikit modules
from sklearn import clone
from sklearn import ensemble
from sklearn.metrics import mean_squared_error


#%%Initialize Global Variables
featureParentPath='D:/GDrive/DT_Data/DAQ_Auto_Features/'
destinationFolder='D:/GDrive/DT_Data/DAQ_Auto_Features/Results'
KPI_fileName='D:/GDrive/DT_Data/DAQ_Auto_Features/KPI_Object_'
objectName = 'UM3_Corner_Wall_'
segment_Numbers=[2,7,8,13]#,8,13]
flowRates=[80,90,100,110,120]
segmentName='segments_Floor'
features=['timeFeatures.csv', 'frequencyFeatures.csv','STFTFeatures.csv']
#Use this when CWT feature extraction is complete!
#features=['timeFeatures.csv', 'frequencyFeatures.csv','STFTFeatures','CWTFeatures']  

# Function to acquire Data for training the models

#%% This function combines the data in the feature level
def combineFeatures(channel, dataread, dataFeature):
    print ('Combine Feature Called... \n')
    if 'Channel' in channel:
        temp=channel.split('_')
        if np.shape(temp)[0]==4:
            channel_name=temp[2]+'_'+temp[3]
        else:
            channel_name=temp[2]
    else:
        channel_name=channel
        
    dataread.columns=channel_name+'_'+dataread.columns
    if dataFeature.empty:
        dataFeature=dataread
    else:
        dataFeature=pd.concat([dataFeature, dataread], axis=1)   
    
    return dataFeature    

#%% This function combines the data in the channel level
def combineChannels(features, channel, segNum, objectFolderName,  
                    dataChannel, dataFeature):
    print ('Combine Channel Called... \n')
    for featureName in features:
        fileName = featureParentPath+objectFolderName+'/'+channel+'/'+segmentName+'/segment_'+str(segNum)+'/'+featureName
        dataread = pd.read_csv(fileName);
        dataFeature=combineFeatures(channel, dataread, dataFeature)
    
      
    if dataChannel.empty:
     dataChannel=dataFeature
    else:
     dataChannel=pd.concat([dataChannel,dataFeature], axis=1)    
     
    return dataChannel

#%% This function combines the data in the segment level
def combineSegNums(objectFolderName, segNum, KPI_values, 
                   KPI_columnIndex, dataSeg,  y_seg, dataChannel):
    print ('Combine Segment Called... \n')
    thickness_KPI=KPI_values.values[segNum][KPI_columnIndex]
    
    for channel in os.listdir(featureParentPath+objectFolderName):
        if not ('desktop' in channel):
            dataFeature=pd.DataFrame()
            dataChannel=combineChannels(features, channel, 
                                        segNum, objectFolderName, 
                                        dataChannel,dataFeature)
            
                 
    if dataSeg.empty:
        dataSeg=dataChannel
    else:
        dataSeg=pd.concat([dataSeg,dataChannel], axis=0)
    
    y_KPI=pd.DataFrame({'Y_KPI_Thickness_in_mm':np.repeat(thickness_KPI,dataChannel.shape[0])})
    #            print('dataLength: ', dataChannel.shape[0], ' KPI: ',thickness_KPI)
    
    if y_seg.empty:
        y_seg=y_KPI
    else:
        y_seg=pd.concat([y_seg,y_KPI], axis=0)
        
    return dataSeg, y_seg    
                
#%% This function combines the data in flow rate level and returns the data
def getXData(KPI_fileName,objectName,segment_Numbers, 
             flowRates, segmentName,features):  
    print ('Get Data Called... \n')
    data=pd.DataFrame()
    y_thickness=pd.DataFrame()
    y_flow=pd.DataFrame()
    
    for flow in flowRates:
        objectFolderName = objectName+ str(flow)+'p';
        fileNameKPI = KPI_fileName+str(flow)+'p.csv'
        KPI_values= pd.read_csv(fileNameKPI)
        if 'Floor' in segmentName:
            KPI_columnIndex=1
        elif 'Wall' in segmentName:
            KPI_columnIndex=2
        else:
            pass
        
        dataSeg=pd.DataFrame()
        y_seg=pd.DataFrame()
        
        for segNum in segment_Numbers:
            dataChannel=pd.DataFrame()
            dataSeg, y_seg= combineSegNums(objectFolderName,
                                           segNum, KPI_values,
                                           KPI_columnIndex, 
                                           dataSeg,  
                                           y_seg, dataChannel)
            
            
        if y_thickness.empty:
         y_thickness=y_seg
        else:
         y_thickness=pd.concat([y_thickness,y_seg], axis=0)  
        
        KPI_flow=pd.DataFrame({'Y_KPI_Flow(%)':np.repeat(flow,dataSeg.shape[0])})
        
        if y_flow.empty:
         y_flow=KPI_flow
        else:
         y_flow=pd.concat([y_flow,KPI_flow], axis=0)                 
    
        if data.empty:
         data=dataSeg
        else:
         data=pd.concat([data,dataSeg], axis=0)
   
      
    data = data.loc[:, (data != 0).any(axis=0)];                         
    return data, y_thickness, y_flow                    

#%% Read the Data for Training 
def parsingInit(*args):
    parser = argparse.ArgumentParser()
    parser.add_argument("n_estimators", type=int,
                        help="Enter the number of estimators")
    parser.add_argument("max_depth", type=int,
                        help="Enter the max depth for the boosting")
    parser.add_argument("min_samples_split", type=float,
                        help="Determine the min sampling rate")
    parser.add_argument("learning_rate", type=float,
                        help="Determine the learning rate")
    parser.add_argument("loss", type=str,
                        help="Enter the type of loss")
    
    n_estimators=500
    max_depth=3
    min_samples_split=2
    learning_rate=0.01
    loss='ls'
    
    if len(args)==1:
        n_estimators=args.n_estimators
        print ('No Arguements were passed \n',
               '1->n_estimators:',n_estimators,'\n',
               '2->max_depth Default:3 \n',
               '3->min_samples_split Default: 2 \n',
               '4->learning_rate Default: 0.01 \n',
               '5->loss Default: ls \n')
    elif len(args)==2:
        max_depth=args.max_depth
        print ('No Arguements were passed \n',
               '1->n_estimators Default: 500\n',
               '2->max_depth:', max_depth ,'\n',
               '3->min_samples_split Default: 2 \n',
               '4->learning_rate Default: 0.01 \n',
               '5->loss Default: ls \n')
    elif len(args)==3:
        min_samples_split=args.min_samples_split
        print ('No Arguements were passed \n',
               '1->n_estimators Default: 500 \n',
               '2->max_depth Default:3 \n',
               '3->min_samples_split:',min_samples_split, '\n',
               '4->learning_rate Default: 0.01 \n',
               '5->loss Default: ls \n')
    elif len(args)==4:
        learning_rate=args.learning_rate
        print ('No Arguements were passed \n',
               '1->n_estimators Default: 500 \n',
               '2->max_depth Default:3 \n',
               '3->min_samples_split Default: 2 \n',
               '4->learning_rate:',learning_rate ,'\n',
               '5->loss Default: ls \n')
        
    elif len(args)==5:    
        loss=args.loss
        print ('No Arguements were passed \n',
               '1->n_estimators Default: 500 \n',
               '2->max_depth Default:3 \n',
               '3->min_samples_split Default: 2 \n',
               '4->learning_rate Default: 0.01 \n',
               '5->loss:', loss ,'\n')
    else:
        print ('No Arguements were passed \n',
               '1->n_estimators Default: 500 \n',
               '2->max_depth Default:3 \n',
               '3->min_samples_split Default: 2 \n',
               '4->learning_rate Default: 0.01 \n',
               '5->loss Default: ls \n')
        
    return  n_estimators, max_depth,  min_samples_split, learning_rate,loss
        
    

def main(*args):
    print ('Main Function Called \n')
    n_estimators, max_depth,  min_samples_split, learning_rate,loss=parsingInit(*args)
    
    print ('Extracting Data... \n')
    Xtr,y_thic,y_flow=getXData(KPI_fileName,objectName,
                               segment_Numbers, flowRates, 
                               segmentName,features) 
    
    if not os.path.exists(destinationFolder):
        os.makedirs(destinationFolder)    
    
    
        
    #fill NaN values with zeros
    Xtr.replace([np.finfo(np.float64).max, np.finfo(np.float64).min], np.nan);
    #Xtr.fillna(0, inplace=True);
    #Xtr.fillna(Xtr.mean(),inplace=True);
    
    #Extract only the values
    y_T=y_thic.values
    y_F=y_flow.values
    X=Xtr.values
    X[np.isinf(X)]=X.mean()
    X[np.isneginf(X)]=X.mean()
    X[np.isnan(X)]=0
    X = X[~np.all(X == 0, axis=1)]
    #%%
    #normalize the data
    min_max_scaler = preprocessing.MinMaxScaler();
    X = min_max_scaler.fit_transform(X);
    
    #Shuffle The data
    y_T=np.ravel(y_T);
    y_F=np.ravel(y_F);
    
    X_T, y_T = shuffle(X, y_T) 
    
    X_F, y_F = shuffle(X, y_F)  
       
    
    #%%
    #separate the training and the test data for thickness versus emissions
    offset = int(X_T.shape[0] * 0.8)
    X_train_T, y_train_T = X_T[:offset], y_T[:offset]
    X_test_T, y_test_T = X_T[offset:], y_T[offset:]
    
    #separate the training and the test data for flow versus emissions
    offset = int(X_F.shape[0] * 0.8)
    X_train_F, y_train_F = X_F[:offset], y_F[:offset]
    X_test_F, y_test_F = X_F[offset:], y_F[offset:]
    
    
    # Test the classifiers
    
    
    paramsGBR = {'n_estimators': n_estimators, 'max_depth': max_depth, 
                 'min_samples_split': min_samples_split,
                 'learning_rate': learning_rate, 'loss': loss}
    
    model = ensemble.GradientBoostingRegressor(**paramsGBR)                      
              
    #%%
    
    clf_T = clone(model)
    clf_F = clone(model)
    
    clf_T = model.fit(X_train_T, y_train_T)
    mse_T = mean_squared_error(y_test_T, clf_T.predict(X_test_T))
    
    
    clf_F = model.fit(X_train_F, y_train_F)
    mse_F = mean_squared_error(y_test_F, clf_F.predict(X_test_F))
    
    print(model.__format__)
    print('Model MSE --> Flow Versus Emission: ',mse_F)
    print('Model MSE --> KPI Versus Emission: ',mse_T)
 

    


#%% Call the main function
if __name__ == '__main__':
    main()           
#%%
#
#if not os.path.exists(destinationFolder):
#    os.makedirs(destinationFolder)    
#
##directory=destinationCSVFolder+'/'+tdmsFolderName+'/data_'+str(directoryIndex);
#    
##fill NaN values with zeros
#Xtr.replace([np.finfo(np.float64).max, np.finfo(np.float64).min], np.nan);
##Xtr.fillna(0, inplace=True);
#Xtr.fillna(Xtr.mean(),inplace=True);
#
##Extract only the values
#y_T=y_thic.values
#y_F=y_flow.values
#X=Xtr.values
#X[np.isinf(X)]=X.mean()
#X[np.isneginf(X)]=X.mean()
#X[np.isnan(X)]=0
#X = X[~np.all(X == 0, axis=1)]
##%%
##normalize the data
#min_max_scaler = preprocessing.MinMaxScaler();
#X = min_max_scaler.fit_transform(X);
#
##Shuffle The data
#y_T=np.ravel(y_T);
#X_T, y_T = shuffle(X, y_T) 
#
#X_F, y_F = shuffle(X, y_F)  
#   
##y = label_binarize(y, classes=[0, 1, 2,3])
##n_classes = y.shape[1]
##%%
##separate the training and the test data for thickness versus emissions
#offset = int(X_T.shape[0] * 0.8)
#X_train_T, y_train_T = X_T[:offset], y_T[:offset]
#X_test_T, y_test_T = X_T[offset:], y_T[offset:]
#
##separate the training and the test data for flow versus emissions
#offset = int(X_F.shape[0] * 0.8)
#X_train_F, y_train_F = X_F[:offset], y_F[:offset]
#X_test_F, y_test_F = X_F[offset:], y_F[offset:]
#
#
## Test the classifiers
#n_estimators=500;
#min_samples_split= 2
#n_iter=200;
#learning_rate=0.01
#max_depth=3;
#random_state = np.random.RandomState(0);
#n_jobs=8
#n_neighbors=5
#rng = np.random.RandomState(1)
#
#paramsGBR = {'n_estimators': 500, 'max_depth': 3, 'min_samples_split': 2,
#          'learning_rate': 0.01, 'loss': 'ls'}
#
#models = [ensemble.GradientBoostingRegressor(**paramsGBR),
#          DecisionTreeRegressor(max_depth=3),
#          neighbors.KNeighborsRegressor(n_neighbors, weights='uniform'),
#          GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=5,
#                   param_grid={"C": [1e0, 1e1, 1e2, 1e3],
#                               "gamma": np.logspace(-2, 2, 5)}),
#          GridSearchCV(KernelRidge(kernel='rbf', gamma=0.1), cv=5,
#                  param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3],
#                              "gamma": np.logspace(-2, 2, 5)}),
#          AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),
#                          n_estimators=300, random_state=rng)
#          
#          ]
##%%
#score_T=0;
#score_F=0;
#
#for model in models:
#        clf_T = clone(model)
#        clf_F = clone(model)
#        
#        clf_T = model.fit(X_train_T, y_train_T)
#        mse_T = mean_squared_error(y_test_T, clf_T.predict(X_test_T))
#        
#        
#        clf_F = model.fit(X_train_F, y_train_F)
#        mse_F = mean_squared_error(y_test_F, clf_F.predict(X_test_F))
#        
#        print(model.__format__)
#        print('Model MSE --> Flow Versus Emission: ',mse_F)
#        print('Model MSE --> KPI Versus Emission: ',mse_T)
 


















       
#        importances_T = clf_T.feature_importances_
#        std_T = np.std([tree.feature_importances_ for tree in clf_T.estimators_],
#                     axis=0)
#        indices_T = np.argsort(importances_T)[::-1]
#        
#        # Print the feature ranking
#        print("Feature ranking_T:")
#        
#        for f in range(X_T.shape[1]):
#            print("%d. feature %d (%f)" % (f + 1, indices_T[f], importances_T[indices_T[f]]))
#        
#        # Plot the feature importances of the forest
#        #range=X.shape[1]
#        plt.figure()
#        plt.title("Feature importances_T")
#        plt.bar(range(X_T.shape[1]), importances_T[indices_T],
#               color="r", yerr=std_T[indices_T], align="center")
#        plt.xticks(range(X_T.shape[1]), indices_T)
#        plt.xlim([-1, 0.45*X_T.shape[1]])
##        name=pathForFeatureRankingPics+'FeatureRanking_Run_'+str(runs)+'_Channel_'+str(channel)+'.png' ;
##        plt.savefig(name,dpi=1200)
#        plt.show()
#        
#        importances_F = clf_F.feature_importances_
#        std_F = np.std([tree.feature_importances_ for tree in clf_F.estimators_],
#                     axis=0)
#        indices_F = np.argsort(importances_F)[::-1]
#        
#        # Print the feature ranking
#        print("Feature ranking_F:")
#        
#        for f in range(X_F.shape[1]):
#            print("%d. feature %d (%f)" % (f + 1, indices_F[f], importances_F[indices_F[f]]))
#        
#        # Plot the feature importances of the forest
#        #range=X.shape[1]
#        plt.figure()
#        plt.title("Feature importances_F")
#        plt.bar(range(X_F.shape[1]), importances_F[indices_F],
#               color="r", yerr=std_F[indices_F], align="center")
#        plt.xticks(range(X_F.shape[1]), indices_F)
#        plt.xlim([-1, 0.45*X_F.shape[1]])
##        name=pathForFeatureRankingPics+'FeatureRanking_Run_'+str(runs)+'_Channel_'+str(channel)+'.png' ;
##        plt.savefig(name,dpi=1200)
#        plt.show()



 
#%%
#
#if pathForScore.is_file()==False:
#    np.savetxt(pathForScore, [], delimiter=',', header="Run,  Channel,  Score");
#else:
#    print('\nFile: ', pathForScore, ' already exists!')

#%%
#for runs in range(5,10):
#    if runs<3:
#        channelNumbers=22;
#    else:
#        channelNumbers=23;
#        
#    for channel in range(1, 19):
#        print('Run: ', runs)
#        print('Channel: ', channel)
#        score= getScores (runs, channel);
#        f =open(pathForScore,'a');
#        df = pd.DataFrame([[runs,channel,score]])
#        df.to_csv(f,index = False,header= False);
#        f.close();
#        
#        print('Score:',score,'\n')
#

















