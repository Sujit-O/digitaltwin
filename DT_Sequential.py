# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 16:28:57 2017

@author: AICPS
"""


# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 12:27:08 2017

@author: Sujit Rokka Chhetri
Project: Siemens Digital Twin Prject Summer 2017

"""
#!/usr/bin/python

#%% Import all the libraries
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.utils import shuffle
import os
import argparse
import matplotlib.pyplot as plt
#% Scikit modules
from sklearn import clone
from sklearn import ensemble
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
#from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score
from sklearn.preprocessing import Imputer

#%%Initialize Global Variables
featureParentPath='D:/GDrive/DT_Data/DAQ_Auto_Features/'
destinationFolder='D:/GDrive/DT_Data/DAQ_Auto_Features/Results_Thickness_AllFeatures'
KPI_fileName='D:/GDrive/DT_Data/DAQ_Auto_Features/KPI_Object_'
objectName = 'UM3_Corner_Wall_'
segment_Numbers=[2,7,8,13]
#features=['STFTFeatures.csv','CWTFeatures.csv']
features=['timeFeatures.csv', 'frequencyFeatures.csv','STFTFeatures.csv','CWTFeatures.csv']

# Function to acquire Data for training the models

#%% This function combines the data in the feature level
def combineFeaturesSurf(channel, dataread, dataFeature):
#    print ('Combine Feature Called... \n')
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
def combineChannelsSurf(features, channel, segNum, objectFolderName,  
                    dataChannel, dataFeature,segmentName):
#    print ('Combine Channel Called... \n')
    for featureName in features:
        fileName = (featureParentPath+objectFolderName+'/'+channel+
                    '/'+segmentName+'/segment_'+str(segNum)+'/'+featureName)
        dataread = pd.read_csv(fileName);
        dataFeature=combineFeaturesSurf(channel, dataread, dataFeature)
    
      
    if dataChannel.empty:
     dataChannel=dataFeature
    else:
     dataChannel=pd.concat([dataChannel,dataFeature], axis=1)    
     
    return dataChannel

#%% This function combines the data in the segment level
def combineSegNumsSurf(objectFolderName, segNum, KPI_values, 
                   KPI_columnIndex, dataSeg,  y_seg,
                   y_seg_surf1,
                   y_seg_surf2,
                   dataChannel,segmentName,
                   KPI_values_surf1,
                   KPI_values_surf2):
#    print ('Combine Segment Called... \n')
    thickness_KPI=KPI_values.values[segNum][KPI_columnIndex]
    KPI_surf1=KPI_values_surf1.values[segNum][1]
    KPI_surf2=KPI_values_surf2.values[segNum][1]
       
    for channel in os.listdir(featureParentPath+objectFolderName):
        if not ('desktop' in channel):
            dataFeature=pd.DataFrame()
            dataChannel=combineChannelsSurf(features, channel, 
                                        segNum, objectFolderName, 
                                        dataChannel,dataFeature,segmentName)
            
                 
    if dataSeg.empty:
        dataSeg=dataChannel
    else:
        dataSeg=pd.concat([dataSeg,dataChannel], axis=0)
    
    y_KPI=pd.DataFrame({'Y_KPI_Thickness_in_mm':
        np.repeat(thickness_KPI, dataChannel.shape[0])})
    y_KPI_surf1=pd.DataFrame({'Y_KPI_Surface_Dispersion':
        np.repeat(KPI_surf1, dataChannel.shape[0])})
    y_KPI_surf2=pd.DataFrame({'Y_KPI_Surface_Dispersion':
        np.repeat(KPI_surf2, dataChannel.shape[0])})
       
        
    if y_seg.empty:
        y_seg=y_KPI
        y_seg_surf1=y_KPI_surf1
        y_seg_surf2=y_KPI_surf2
                
    else:
        y_seg=pd.concat([y_seg,y_KPI], axis=0)
        y_seg_surf1=pd.concat([y_seg_surf1,y_KPI_surf1], axis=0)
        y_seg_surf2=pd.concat([y_seg_surf2,y_KPI_surf2], axis=0)
        
        
    return dataSeg, y_seg, y_seg_surf1, y_seg_surf2
                
#%% This function combines the data in flow rate level and returns the data
def getXDataSurf(KPI_fileName,KPI_fileName_surf,objectName,segment_Numbers, 
             flowRates, segmentName,features):  
#    print ('Get Data Called... \n')
    data=pd.DataFrame()
    y_thickness=pd.DataFrame()
    y_flow=pd.DataFrame()
    y_surf1=pd.DataFrame()
    y_surf2=pd.DataFrame()
    
    
    for flow in flowRates:
        objectFolderName = objectName+ str(flow)+'p';
        
        fileNameKPI = KPI_fileName+str(flow)+'p.csv'
        if 'Floor' in segmentName:
            fileNameKPI_surf1 = KPI_fileName_surf+objectName+str(flow)+'p/KPI/1_directionality.csv'
            fileNameKPI_surf2 = KPI_fileName_surf+objectName+str(flow)+'p/KPI/4_directionality.csv'
        elif 'Wall' in segmentName:
            fileNameKPI_surf1 = KPI_fileName_surf+objectName+str(flow)+'p/KPI/3_directionality.csv'
            fileNameKPI_surf2 = KPI_fileName_surf+objectName+str(flow)+'p/KPI/2_directionality.csv'    
        else:
            print('Segment Name does not match!')
            return
        
        KPI_values= pd.read_csv(fileNameKPI)
        KPI_values_surf1= pd.read_csv(fileNameKPI_surf1)
        KPI_values_surf2= pd.read_csv(fileNameKPI_surf2)
       
        
        if 'Floor' in segmentName:
            KPI_columnIndex=1
        elif 'Wall' in segmentName:
            KPI_columnIndex=2
        else:
            pass
        
        dataSeg=pd.DataFrame()
        y_seg=pd.DataFrame()
        y_seg_surf1=pd.DataFrame()
        y_seg_surf2=pd.DataFrame()
        
        
        for segNum in segment_Numbers:
            dataChannel=pd.DataFrame()
            
            (dataSeg, y_seg,y_seg_surf1,
            y_seg_surf2) = combineSegNumsSurf(objectFolderName,
                                           segNum, KPI_values,
                                           KPI_columnIndex, 
                                           dataSeg,  
                                           y_seg, 
                                           y_seg_surf1,
                                           y_seg_surf2,
                                           dataChannel,segmentName,
                                           KPI_values_surf1,
                                           KPI_values_surf2)
            
            
        if y_thickness.empty:
            y_thickness=y_seg
            y_surf1=y_seg_surf1
            y_surf2=y_seg_surf2
            
        else:
            y_thickness=pd.concat([y_thickness,y_seg], axis=0) 
            y_surf1=pd.concat([y_surf1,y_seg_surf1], axis=0)
            y_surf2=pd.concat([y_surf2,y_seg_surf2], axis=0)
            
                    
        KPI_flow=pd.DataFrame({'Y_KPI_Flow(%)':np.repeat(flow,
                               dataSeg.shape[0])})
        
        if y_flow.empty:
         y_flow=KPI_flow
        else:
         y_flow=pd.concat([y_flow,KPI_flow], axis=0)                 
    
        if data.empty:
         data=dataSeg
        else:
         data=pd.concat([data,dataSeg], axis=0)
   
            
    return data, y_thickness, y_flow , y_surf1, y_surf2                   


#%% This function combines the data in the feature level



def combineFeaturesDim(channel, dataread, dataFeature):
#    print ('Combine Feature Called... \n')
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
def combineChannelsDim(features, channel, segNum, objectFolderName,  
                    dataChannel, dataFeature,segmentName):
#    print ('Combine Channel Called... \n')
    for featureName in features:
        fileName = (featureParentPath+objectFolderName+'/'+channel+
                    '/'+segmentName+'/segment_'+str(segNum)+'/'+featureName)
        dataread = pd.read_csv(fileName);
        dataFeature=combineFeaturesDim(channel, dataread, dataFeature)
    
      
    if dataChannel.empty:
     dataChannel=dataFeature
    else:
     dataChannel=pd.concat([dataChannel,dataFeature], axis=1)    
     
    return dataChannel

#%% This function combines the data in the segment level
def combineSegNumsDim(objectFolderName, segNum, KPI_values, 
                   KPI_columnIndex, dataSeg,  y_seg, dataChannel,segmentName):
#    print ('Combine Segment Called... \n')
    thickness_KPI=KPI_values.values[segNum][KPI_columnIndex]
    
    for channel in os.listdir(featureParentPath+objectFolderName):
        if not ('desktop' in channel):
            dataFeature=pd.DataFrame()
            dataChannel=combineChannelsDim(features, channel, 
                                        segNum, objectFolderName, 
                                        dataChannel,dataFeature,segmentName)
            
                 
    if dataSeg.empty:
        dataSeg=dataChannel
    else:
        dataSeg=pd.concat([dataSeg,dataChannel], axis=0)
    
    y_KPI=pd.DataFrame({'Y_KPI_Thickness_in_mm':
        np.repeat(thickness_KPI, dataChannel.shape[0])})
        
    if y_seg.empty:
        y_seg=y_KPI
    else:
        y_seg=pd.concat([y_seg,y_KPI], axis=0)
        
    return dataSeg, y_seg    
                
#%% This function combines the data in flow rate level and returns the data
def getXDataDim(KPI_fileName,objectName,segment_Numbers, 
             flowRates, segmentName,features):  
#    print ('Get Data Called... \n')
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
            dataSeg, y_seg= combineSegNumsDim(objectFolderName,
                                           segNum, KPI_values,
                                           KPI_columnIndex, 
                                           dataSeg,  
                                           y_seg, dataChannel,segmentName)
            
            
        if y_thickness.empty:
         y_thickness=y_seg
        else:
         y_thickness=pd.concat([y_thickness,y_seg], axis=0)  
        
        KPI_flow=pd.DataFrame({'Y_KPI_Flow(%)':np.repeat(flow,
                               dataSeg.shape[0])})
        
        if y_flow.empty:
         y_flow=KPI_flow
        else:
         y_flow=pd.concat([y_flow,KPI_flow], axis=0)                 
    
        if data.empty:
         data=dataSeg
        else:
         data=pd.concat([data,dataSeg], axis=0)
   
            
    return data, y_thickness, y_flow                    

#%% Read the Data for Training 
def parsingInit():
    parser = argparse.ArgumentParser()
    parser.add_argument("-ne","--n_estimators", type=int, nargs='?',
                        default=1000,
                        help="Enter the number of estimators")
    parser.add_argument("-md","--max_depth", type=int,nargs='?',
                        default=2,
                        help="Enter the max depth for the boosting")
    parser.add_argument("-ms","--min_samples_split", type=int,nargs='?', 
                        default=2,
                        help="Determine the min sampling rate")
    parser.add_argument("-lr","--learning_rate", type=float, nargs='?',
                        default=0.01,
                        help="Determine the learning rate")
    parser.add_argument("-loss","--loss", type=str, nargs='?',default='ls',
                        help="Enter the type of loss")
    parser.add_argument("-start","--trainGroupStart", type=int, nargs='?',
                        default=80,
                        help="Train Group Starting Flowrate")
    parser.add_argument("-stop","--trainGroupStop", type=int,nargs='?',
                        default=120,
                        help="Train Group Stopping Flowrate")
    parser.add_argument("-testGroup","--testGroup", type=int,nargs='?', 
                        default=130,
                        help="Test Group Emissions")
    parser.add_argument("-surf","--testSurface", type=str, nargs='?', 
                        default='segments_Floor',
                        help="Test Surface")
    
    args = parser.parse_args()
    
    print ('Arguements:\n',
               '1-> n_estimators     : ', args.n_estimators ,'\n',
               '2-> max_depth        : ', args.max_depth ,'\n',
               '3-> min_samples_split: ', args.min_samples_split ,'\n',
               '4-> learning_rate    : ', args.learning_rate,'\n',
               '5-> loss             : ', args.loss,'\n',
               '6-> trainGroupStart  : ', args.trainGroupStart,'\n',
               '7-> trainGroupStop   : ', args.trainGroupStop,'\n',
               '8-> testGroup        : ', args.testGroup,'\n',
               '9-> testSurface      : ', args.testSurface,'\n')
        
    return  (args.n_estimators, args.max_depth, 
            args.min_samples_split, args.learning_rate,
            args.loss,args.trainGroupStart, 
            args.trainGroupStop,  args.testGroup, args.testSurface)
        
    
#%%
def heldout_score(clf, X_test, y_test,n_estimators):
    score = np.zeros((n_estimators,), dtype=np.float64)
    for i, y_pred in enumerate(clf.staged_decision_function(X_test)):
        score[i] = clf.loss_(y_test, y_pred)
    return score

#%%    
def crossValidation(cv_clf_T,n_splits,n_estimators,X_train_T,y_train_T):
    cv = KFold(n_splits=n_splits)
    val_scores_T = np.zeros((n_estimators,), dtype=np.float64)
    for train, test in cv.split(X_train_T, y_train_T):
        cv_clf_T.fit(X_train_T[train], y_train_T[train])
        val_scores_T += heldout_score(cv_clf_T, X_train_T[test], 
                                      y_train_T[test],n_estimators)
    val_scores_T /= n_splits
    return val_scores_T

#%%  
def maxDepthCheck(paramsGBR,X_train_T, y_train_T,X_test_T,y_test_T):
    params=paramsGBR
    test_score = np.zeros((paramsGBR['max_depth'],), dtype=np.float64)
    train_score = np.zeros((paramsGBR['max_depth'],), dtype=np.float64)
    
    for i,depth in enumerate(range(1,paramsGBR['max_depth']+1)):
        params['max_depth'] =depth 
        model = ensemble.GradientBoostingRegressor(**params)                      
        clf_T = clone(model)
        clf_T = model.fit(X_train_T, y_train_T)
        y_pred= clf_T.predict(X_test_T)
        test_score[i] = clf_T.loss_(y_test_T, y_pred)
        y_pred_Train= clf_T.predict(X_train_T)
        train_score[i] = clf_T.loss_(y_train_T, y_pred_Train)
    
    plt.figure()
    plt.plot(train_score ,'b-', label='Training Set Deviance')
    plt.plot(test_score, 'r-',  label='Test Set Deviance')
    plt.xlabel('Max Depths')
    plt.ylabel('Deviance')
    plt.show()

#%%  
def minSplitCheck(paramsGBR,X_train_T, y_train_T,X_test_T,y_test_T):
    params=paramsGBR
    test_score = np.zeros((paramsGBR['min_samples_split'],), dtype=np.float64)
    train_score = np.zeros((paramsGBR['min_samples_split'],), dtype=np.float64)
    
    for i,split in enumerate(range(2,paramsGBR['min_samples_split']+2)):
        params['min_samples_split'] = split
        model = ensemble.GradientBoostingRegressor(**params)                      
        clf_T = clone(model)
        clf_T = model.fit(X_train_T, y_train_T)
        y_pred= clf_T.predict(X_test_T)
        test_score[i] = clf_T.loss_(y_test_T, y_pred)
        y_pred_Train= clf_T.predict(X_train_T)
        train_score[i] = clf_T.loss_(y_train_T, y_pred_Train)
        
    plt.figure()
    plt.plot(train_score ,'b-', label='Training Set Deviance')
    plt.plot(test_score, 'r-',  label='Test Set Deviance')
    plt.xlabel('min samples of split')
    plt.ylabel('Deviance')
    plt.show()
    
#%%
def preProcess(Xtr,y_thic):
    print ('\t Inside Processing Function... ') 
    X=Xtr.values
    y_T=y_thic.values
#    y_T=np.float32(y_T)
#    X[X<=np.finfo(np.float32).min]=np.nan
#    X[X>=np.finfo(np.float32).max]=np.nan
#    X=np.float32(X)
    X[np.isinf(X)]=0
    X[np.isneginf(X)]=0
    X[np.isnan(X)]=0
    
#    X = X[~np.all(X == 0, axis=1)]
    y_T=np.ravel(y_T);
#    y_T[y_T<=np.finfo(np.float32).min]=np.nan
#    y_T[y_T>=np.finfo(np.float32).max]=np.nan
    y_T[np.isinf(y_T)]=0
    y_T[np.isneginf(y_T)]=0
    y_T[np.isnan(y_T)]=0

    
    if np.isnan(X).any():
        print('\t NaN values found in X')
    if ~np.isfinite(X).all():
        print('\t Infinite values found in X')    
#    if (X<=np.finfo(np.float32).min).any():
#        print('\t Values less than float32 found in X')
#    if (X>=np.finfo(np.float32).max).any():
#        print('\t Values more than float32 found in X') 
    if (X==0).any():
        print('\t Zero Values found in X')    
    
    if np.isnan(y_T).any():
        print('\t NaN values found in y')
    if ~np.isfinite(y_T).all():
        print('\t Infinite values found in y')    
#    if (y_T<=np.finfo(np.float32).min).any():
#        print('\t Values less than float32 found in y')
#    if (y_T>=np.finfo(np.float32).max).any():
#        print('\t Values more than float32 found in y') 
    if (y_T==0).any():
        print('\t Zero Values found in X')     
        
    print('\t Finished Processing \n')
    return X, y_T


#%%
def normalizeData(X,y):
    print ('Normalizing the Data... \n')  
    
    min_max_scaler = preprocessing.MinMaxScaler();
    X = min_max_scaler.fit_transform(X);
    
    return X, y

#%%
def splitData(X_T,y_T):    
    #separate the training and the test data for thickness versus emissions
    print ('Splitting the Data... \n') 
    offset = int(X_T.shape[0] * 0.75)
    X_train_T, y_train_T = X_T[:offset], y_T[:offset]
    X_test_T, y_test_T = X_T[offset:], y_T[offset:]
    
    return X_train_T, y_train_T, X_test_T, y_test_T

#%%    
def featureImportance(clf,feature_names,filename_featureImportance):
    
    feature_importance = clf.feature_importances_
    # make importances relative to max importance
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    sorted_idx=sorted_idx[::-1]
    sorted_idx=sorted_idx[0:25]
    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.barh(pos, feature_importance[sorted_idx], align='center')
    plt.yticks(pos, feature_names[sorted_idx])
    plt.xlabel('Relative Feature Importance')
    plt.title('Feature Importance')
#    name=destinationFolder+'/FeatureRanking_'+textDescription+'.pdf' ;
    plt.savefig(filename_featureImportance,bbox_inches='tight',dpi=1200)
#    plt.show()    
    
    
#%%    
def DT_Sequential(start, stop,  testGroup,  segmentName, age, surf):
    print ('\n----------Start-----------\n')
#    (n_estimators, 
#     max_depth,  
#     min_samples_split, 
#     learning_rate,
#     loss, 
#     start, 
#     stop, 
#     testGroup, 
#     segmentName) = parsingInit()
    n_estimators =1000
    max_depth  = 2
    min_samples_split =2
    learning_rate=0.01
    loss ='ls'
    
    
    if age:
        filename_featureImportance= destinationFolder+'/FeatureRanking_'+str(testGroup)+'_initialRankings_'+segmentName+'.pdf'
        fileName_ErrorCSV=destinationFolder+'/FeatureRanking_'+str(testGroup)+'_'+segmentName+'.csv'
        filename_featureImportance2= destinationFolder+'/FeatureRanking_'+str(testGroup)+'_'+segmentName+'.csv'
    else:
        fileName_ErrorCSV=destinationFolder+'/FeatureRanking_'+str(testGroup)+'_'+segmentName+'.csv'
    
    flowRates_Train=np.array([i for i in range(start,stop+10,10)])
    
    flowRates_Test=np.array([i for i in range(testGroup,testGroup+10,10)])
    
    flowRates_reTrain= np.append(flowRates_Train, flowRates_Test)
    
    #The 160 flow rate data is corrupted!!
    #TODO: recollect the data
    flowRates_Train=np.delete(flowRates_Train,np.where(flowRates_Train==160))
    flowRates_Test=np.delete(flowRates_Test,np.where(flowRates_Test==160))
    flowRates_reTrain=np.delete(flowRates_reTrain,np.where(flowRates_reTrain==160))
    
    print('Train: ',flowRates_Train)
    print('Test: ',flowRates_Test)
    print('reTrain: ',flowRates_reTrain)
    
    print ('1. Extracting Data... ')
    #Train Data
    X_Train,y_thic_Train,y_flow_Train=getXData(KPI_fileName,objectName,
                               segment_Numbers, flowRates_Train, 
                               segmentName,features) 
    featureNames=X_Train.columns
      
    #Test Data
    X_Test,y_thic_Test,y_flow_Test=getXData(KPI_fileName,objectName,
                               segment_Numbers, flowRates_Test, 
                               segmentName,features) 
        
    #ReTrain Data
    X_reTrain,y_thic_reTrain,y_flow_reTrain=getXData(KPI_fileName,objectName,
                               segment_Numbers, flowRates_reTrain, 
                               segmentName,features) 
    
     
    #%% Preprocessing Data converting to float32 and removing NaN
    print ('2. Preprocessing Data...')
    imp1 = Imputer(missing_values='NaN', strategy='mean', axis=0)
#    imp2 = Imputer(missing_values=0, strategy='mean', axis=0)
    
    X_Train, y_thic_Train = preProcess(X_Train,y_thic_Train)
    X_Train=imp1.fit_transform(X_Train)
    
    X_Test,y_thic_Test= preProcess(X_Test,y_thic_Test)
    X_Test=imp1.fit_transform(X_Test)
    
    X_reTrain, y_thic_reTrain = preProcess(X_reTrain,y_thic_reTrain)
    X_reTrain=imp1.fit_transform(X_reTrain)
    
    #%%
    if not os.path.exists(destinationFolder):
        os.makedirs(destinationFolder)    
      
    paramsGBR = {'n_estimators': n_estimators, 'max_depth': max_depth, 
                 'min_samples_split': min_samples_split,
                 'learning_rate': learning_rate, 'loss': loss}
    
    model = ensemble.GradientBoostingRegressor(**paramsGBR)                      
    
    clf_Tr = clone(model)
   
    #%%
    print ('3. Building Model with all the Samples...')
    X_Train, y_thic_Train = shuffle(X_Train,y_thic_Train)
    
    print('\t Shape Train: ', X_Train.shape)
    print('\t DataType Train: ', X_Train.dtype)
    
    print('\t Shape Train: ', y_thic_Train.shape)
    print('\t DataType Train: ', y_thic_Train.dtype)
    

    min_max_scaler_Train_X = preprocessing.MinMaxScaler().fit(X_Train);
    scaler_Train_X = preprocessing.StandardScaler().fit(X_Train)
   
    X_Tr=scaler_Train_X.transform(X_Train)
    X_Tr=min_max_scaler_Train_X.transform(X_Tr)
    
    
    clf_Tr = model.fit(X_Tr, y_thic_Train)
    
   
    #%%
    print ('4. Results for Training:')
    y_pred1=clf_Tr.predict(X_Tr)
    featureImportance(clf_Tr, featureNames,filename_featureImportance)
    
    mse_Test = mean_squared_error(y_thic_Train, y_pred1)
    mae_Test=mean_absolute_error(y_thic_Train, y_pred1)
    medae_Test=median_absolute_error(y_thic_Train, y_pred1)
    r2_Test=r2_score(y_thic_Train, y_pred1)
    exvs_Test=explained_variance_score(y_thic_Train,y_pred1)
    
  
    print ('\t Mean Squared Error      :', mse_Test )
    print ('\t Mean Absolute Error     :', mae_Test )
    print ('\t Median Absolute Error   :', medae_Test )
    print ('\t R2 Score                :', r2_Test )
    print ('\t Explained Variance Score:', exvs_Test )
    
    #%%
    print ('\n5. Processing emissions Signals for Group ',flowRates_Test,' ...')
    X_Test,y_thic_Test= shuffle(X_Test,y_thic_Test)
    
    print('\t Shape Test: ', X_Test.shape)
    print('\t DataType Train: ', X_Test.dtype)
    
    print('\t Shape y Test: ', y_thic_Test.shape)
    print('\t DataType y Test: ', y_thic_Test.dtype)
    
    print ('6. Transforming emissions Signals for Group ',flowRates_Test,' ...')    
    X_Te=scaler_Train_X.transform(X_Test)
    X_Te=min_max_scaler_Train_X.transform(X_Te)
        
    print('\t Shape X_Te: ', X_Te.shape)
    print('\t DataType X_te: ', X_Te.dtype)
  
    
    print ('7. Predicting KPI for Signals for Group ',flowRates_Test,' ...')
    y_pred_Te=clf_Tr.predict(X_Te)

    print ('8. Results for Predicting KPI for Signals for Group ',flowRates_Test,' ...')
    mse_Test = mean_squared_error(y_thic_Test, y_pred_Te)
    mae_Test=mean_absolute_error(y_thic_Test, y_pred_Te)
    medae_Test=median_absolute_error(y_thic_Test, y_pred_Te)
    r2_Test=r2_score(y_thic_Test, y_pred_Te)
    exvs_Test=explained_variance_score(y_thic_Test, y_pred_Te)
    
  
    print ('\t Mean Squared Error      :', mse_Test )
    print ('\t Mean Absolute Error     :', mae_Test )
    print ('\t Median Absolute Error   :', medae_Test )
    print ('\t R2 Score                :', r2_Test )
    print ('\t Explained Variance Score:', exvs_Test )
    
    
    print ('9. Saving Results',fileName_ErrorCSV,' ...')
    np.savetxt(fileName_ErrorCSV, [[mse_Test,
                              mae_Test,
                              medae_Test,
                              r2_Test, 
                              exvs_Test]], 
        delimiter=',',header='Mean Squared Error, Mean Absolute Error, Median Absolute Error,R2 Score, Explained Variance Score',comments='')
    
    print ('10. Retraining the Model with new emission Signal...')
    X_reTrain, y_thic_reTrain = shuffle(X_reTrain,y_thic_reTrain)
    
    print('\t Shape reTrain: ', y_thic_reTrain.shape)
    print('\t DataType reTrain: ', y_thic_reTrain.dtype)
      
    print('\t Shape y reTrain: ', y_thic_Test.shape)
    print('\t DataType y reTrain: ', y_thic_Test.dtype)
    
    min_max_scaler_Train_X2 = preprocessing.MinMaxScaler().fit(X_reTrain);
    scaler_Train_X2 = preprocessing.StandardScaler().fit(X_reTrain)
  
    X_reTr=scaler_Train_X2.transform(X_reTrain)
    X_reTr=min_max_scaler_Train_X2.transform(X_reTr)

    print('\t Shape X_reTr: ', X_reTr.shape)
    print('\t DataType X_reTr: ', X_reTr.dtype)
       
    
    X_Te=scaler_Train_X.transform(X_Test)
    X_Te=min_max_scaler_Train_X.transform(X_Te)

    print('\t Shape X_Te: ', X_Te.shape)
    print('\t DataType X_Te: ', X_Te.dtype)
   
    clf_reTr = model.fit(X_reTr, y_thic_reTrain)
    print ('11. New Results with emission signals Incorporated:')
    y_pred_Te=clf_reTr.predict(X_Te)
    mse_Test = mean_squared_error(y_thic_Test, y_pred_Te)
    mae_Test=mean_absolute_error(y_thic_Test, y_pred_Te)
    medae_Test=median_absolute_error(y_thic_Test, y_pred_Te)
    r2_Test=r2_score(y_thic_Test, y_pred_Te)
    exvs_Test=explained_variance_score(y_thic_Test, y_pred_Te)
    
    print ('\t Mean Squared Error      :', mse_Test )
    print ('\t Mean Absolute Error     :', mae_Test)
    print ('\t Median Absolute Error   :', medae_Test)
    print ('\t R2 Score                :', r2_Test )
    print ('\t Explained Variance Score:', exvs_Test )
    
    print ('12. Saving the new Results',fileName_ErrorCSV,' ...')
    f =open(fileName_ErrorCSV,'a');
    df = pd.DataFrame([[mse_Test, mae_Test,medae_Test,r2_Test, exvs_Test]])
    df.to_csv(f,index = False,header= False);
    f.close();
    featureImportance(clf_reTr, featureNames,filename_featureImportance2)

    
    print ('-----------:Finished!:--------------- \n')


