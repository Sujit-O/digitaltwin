# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 12:16:00 2017

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
KPI_fileName='D:/GDrive/DT_Data/DAQ_Auto_Features/KPI_Object_'
KPI_fileName_surf='D:/GDrive/DT_Data/DAQ_Auto/'
objectName = 'UM3_Corner_Wall_'
segment_Numbers=[2,7,8,13]
#features=['CWTFeatures.csv']
features=['timeFeatures.csv', 'frequencyFeatures.csv','STFTFeatures.csv','CWTFeatures.csv']

# Function to acquire Data for training the models

#%% This function combines the data in the feature level
def combineFeatures(channel, dataread, dataFeature):
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
def combineChannels(features, channel, segNum, objectFolderName,  
                    dataChannel, dataFeature,segmentName):
#    print ('Combine Channel Called... \n')
    for featureName in features:
        fileName = (featureParentPath+objectFolderName+'/'+channel+
                    '/'+segmentName+'/segment_'+str(segNum)+'/'+featureName)
        dataread = pd.read_csv(fileName);
        dataFeature=combineFeatures(channel, dataread, dataFeature)
    
      
    if dataChannel.empty:
     dataChannel=dataFeature
    else:
     dataChannel=pd.concat([dataChannel,dataFeature], axis=1)    
     
    return dataChannel

#%% This function combines the data in the segment level
def combineSegNums(objectFolderName, segNum, KPI_values, 
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
            dataChannel=combineChannels(features, channel, 
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
def getXData(KPI_fileName,KPI_fileName_surf,objectName,segment_Numbers, 
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
            y_seg_surf2) = combineSegNums(objectFolderName,
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
#    print ('\t Inside Processing Function... ') 
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

    
#    if np.isnan(X).any():
#        print('\t NaN values found in X')
#    if ~np.isfinite(X).all():
#        print('\t Infinite values found in X')    
##    if (X<=np.finfo(np.float32).min).any():
##        print('\t Values less than float32 found in X')
##    if (X>=np.finfo(np.float32).max).any():
##        print('\t Values more than float32 found in X') 
#    if (X==0).any():
#        print('\t Zero Values found in X')    
#    
#    if np.isnan(y_T).any():
#        print('\t NaN values found in y')
#    if ~np.isfinite(y_T).all():
#        print('\t Infinite values found in y')    
##    if (y_T<=np.finfo(np.float32).min).any():
##        print('\t Values less than float32 found in y')
##    if (y_T>=np.finfo(np.float32).max).any():
##        print('\t Values more than float32 found in y') 
#    if (y_T==0).any():
#        print('\t Zero Values found in X')     
#        
#    print('\t Finished Processing \n')
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
def featureImportance(clf,feature_names,fileName):
    
    feature_importance = clf.feature_importances_
    # make importances relative to max importance
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    sorted_idx=sorted_idx[::-1]
    sorted_idx=sorted_idx[0:25]
    plt.figure()
    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.barh(pos, feature_importance[sorted_idx], align='center')
    plt.yticks(pos, feature_names[sorted_idx])
    plt.xlabel('Relative Feature Importance')
    plt.title('Feature Importance')
#    name=destinationFolder+'/Age_FeatureRanking_'+textDescription+'.pdf' ;
    plt.savefig(fileName,bbox_inches='tight',dpi=600)
#    plt.show()   
    
#%%    
def DT_surface_wall_back(start, stop,  testGroup,  segmentName,agingTest):
#start=80
#stop=90
#testGroup=90
#segmentName='segments_Floor'
#agingTest=True
    print ('\n----------Start-----------')
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
    
    if agingTest:
        nameStore='_Aging_Test_allFeatures_surface_wall_back'
    else:
        nameStore='_Normal_Test_allFeatures'
        
    if 'Floor' in segmentName:
        name1='Surface_Floor1_Top'
#        name2='Surface_Floor4_Bottom'
#        segment_Numbers_Top1=[7,13]
        
        
    elif 'Wall' in segmentName:
        name1='Surface_Wall3_Back'
#        name2='Surface_Wall2_Front'
    else:
        print('Invalid Segment Names')
     
    destinationFolder='D:/GDrive/DT_Data/DAQ_Auto_Features/Results_Surface'+nameStore
    
    if not os.path.exists(destinationFolder):
        os.makedirs(destinationFolder)   
        
    filename1=destinationFolder+'/Original_'+name1+'_Start_'+str(start)+'_Stop_'+str(stop)+'_TestG_'+str(testGroup)+'.pdf'
#    filename2=destinationFolder+'/Original_'+name2+'_Start_'+str(start)+'_Stop_'+str(stop)+'_TestG_'+str(testGroup)+'.pdf'
    
    fileNamecsv1=destinationFolder+'/'+name1+'_Start_'+str(start)+'_Stop_'+str(stop)+'_TestG_'+str(testGroup)+'.csv'
#    fileNamecsv2=destinationFolder+'/'+name2+'_Start_'+str(start)+'_Stop_'+str(stop)+'_TestG_'+str(testGroup)+'.csv'
    
    filename1_reTr=destinationFolder+'/Final_'+name1+'_Start_'+str(start)+'_Stop_'+str(stop)+'_TestG_'+str(testGroup)+'.pdf'
#    filename2_reTr=destinationFolder+'/Final_'+name2+'_Start_'+str(start)+'_Stop_'+str(stop)+'_TestG_'+str(testGroup)+'.pdf'
      
    
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
    #%%
    print ('1. Extracting Data... \n')
    #Train Data
    (X_Train,y_thic_Train,
    y_flow_Train,y_surf1_Train,
    y_surf2_Train) =  getXData(KPI_fileName,KPI_fileName_surf,objectName,
                               segment_Numbers, flowRates_Train, 
                               segmentName,features) 
    featureNames=X_Train.columns
    #Test Data
    (X_Test,y_thic_Test,y_flow_Test,
    y_surf1_Test,
    y_surf2_Test) = getXData(KPI_fileName,KPI_fileName_surf,objectName,
                               segment_Numbers, flowRates_Test, 
                               segmentName,features) 
    #ReTrain Data
    (X_reTrain,y_thic_reTrain,
    y_flow_reTrain,y_surf1_reTrain,
    y_surf2_reTrain)   = getXData(KPI_fileName,KPI_fileName_surf,objectName,
                               segment_Numbers, flowRates_reTrain, 
                               segmentName,features) 
    
     
     #%%
    paramsGBR = {'n_estimators': n_estimators, 'max_depth': max_depth, 
                 'min_samples_split': min_samples_split,
                 'learning_rate': learning_rate, 'loss': loss}
    
    model = ensemble.GradientBoostingRegressor(**paramsGBR)                      
    
    #    clf_Tr1 = clone(model)
    #    clf_Tr2 = clone(model)
       
    #%%
    
    print ('2. Preprocessing Data...')
    imp1 = Imputer(missing_values='NaN', strategy='mean', axis=0)
    #    X_Train, y_thic_Train = preProcess(X_Train,y_thic_Train)
    X_Train1, y_surf1_Train = preProcess(X_Train,y_surf1_Train)
#    X_Train2, y_surf2_Train = preProcess(X_Train,y_surf2_Train)
    
    
    X_Train1, y_surf1_Train=shuffle(X_Train1, y_surf1_Train)
#    X_Train2, y_surf2_Train=shuffle(X_Train2, y_surf2_Train)
    
    X_Train1=imp1.fit_transform(X_Train1)
#    X_Train2=imp1.fit_transform(X_Train2)
    
    X_Test1,y_surf1_Test= preProcess(X_Test,y_surf1_Test)
#    X_Test2,y_surf2_Test= preProcess(X_Test,y_surf2_Test)
    
    X_Test1,y_surf1_Test=shuffle(X_Test1,y_surf1_Test)
#    X_Test2,y_surf2_Test=shuffle(X_Test2,y_surf2_Test)
    
    X_Test1=imp1.fit_transform(X_Test1)
#    X_Test2=imp1.fit_transform(X_Test2)
    
        
    min_max_scaler_Train_X1 = preprocessing.MinMaxScaler().fit(X_Train1);
    scaler_Train_X1 = preprocessing.StandardScaler().fit(X_Train1)
    X_Tr1=min_max_scaler_Train_X1.transform(X_Train1)
    X_Tr1=scaler_Train_X1.transform(X_Tr1)
    
#    min_max_scaler_Train_X2 = preprocessing.MinMaxScaler().fit(X_Train2);
#    scaler_Train_X2 = preprocessing.StandardScaler().fit(X_Train2)
#    X_Tr2=min_max_scaler_Train_X2.transform(X_Train2)
#    X_Tr2=scaler_Train_X2.transform(X_Tr2)
#    
    X_Te1=min_max_scaler_Train_X1.transform(X_Test1)
    X_Te1=scaler_Train_X1.transform(X_Te1)
#    
#    X_Te2=min_max_scaler_Train_X2.transform(X_Test2)
#    X_Te2=scaler_Train_X2.transform(X_Te2)
    
    X_reTrain1, y_surf1_reTrain = preProcess(X_reTrain,y_surf1_reTrain)
#    X_reTrain2, y_surf2_reTrain = preProcess(X_reTrain,y_surf2_reTrain)
    
    X_reTrain1, y_surf1_reTrain=shuffle(X_reTrain1, y_surf1_reTrain)
#    X_reTrain2, y_surf2_reTrain=shuffle(X_reTrain2, y_surf2_reTrain)
    
    X_reTrain1=imp1.fit_transform(X_reTrain1)
#    X_reTrain2=imp1.fit_transform(X_reTrain2)
    
    #%%
    print ('3. Building Model with all the Samples...')
    X_Tr1, y_surf1_Train=shuffle(X_Tr1, y_surf1_Train)
#    X_Tr2, y_surf2_Train = shuffle(X_Tr2, y_surf2_Train)
    
    clf_Tr1 = model.fit(X_Tr1, y_surf1_Train)
#    clf_Tr2 = model.fit(X_Tr2, y_surf2_Train)
       
    print ('4. Saving results of Training...')
    featureImportance(clf_Tr1, featureNames, filename1)
#    featureImportance(clf_Tr2, featureNames, filename2)
        
    #%%
    print ('5. Predicting for Group: ',flowRates_Test,' ...')
    y_pred_Te1=clf_Tr1.predict(X_Te1)
#    y_pred_Te2=clf_Tr2.predict(X_Te2)
    
    
    mse_Test1 = mean_squared_error(y_surf1_Test, y_pred_Te1)
    mae_Test1=mean_absolute_error(y_surf1_Test, y_pred_Te1)
    medae_Test1=median_absolute_error(y_surf1_Test, y_pred_Te1)
    r2_Test1=r2_score(y_surf1_Test, y_pred_Te1)
    exvs_Test1=explained_variance_score(y_surf1_Test, y_pred_Te1)
    
#    mse_Test2 = mean_squared_error(y_surf2_Test, y_pred_Te2)
#    mae_Test2=mean_absolute_error(y_surf2_Test, y_pred_Te2)
#    medae_Test2=median_absolute_error(y_surf2_Test, y_pred_Te2)
#    r2_Test2=r2_score(y_surf2_Test, y_pred_Te2)
#    exvs_Test2=explained_variance_score(y_surf2_Test, y_pred_Te2)
#    
    
    print ('6. Results for testing Group:',flowRates_Test,':')
    print ('\t Mean Squared Errors      :', mse_Test1)
    print ('\t Mean Absolute Error     :', mae_Test1 )
    print ('\t Median Absolute Error   :', medae_Test1)
    print ('\t R2 Score                :', r2_Test1)
    print ('\t Explained Variance Score:', exvs_Test1 )
    
    print ('7. Saving Results for testing Group:',flowRates_Test,':')
    np.savetxt(fileNamecsv1, [[mse_Test1,
                              mae_Test1,
                              medae_Test1,
                              r2_Test1, 
                              exvs_Test1]], 
        delimiter=',',header='Mean Squared Error, Mean Absolute Error, Median Absolute Error,R2 Score, Explained Variance Score',comments='')
#    np.savetxt(fileNamecsv2, [[mse_Test2,
#                              mae_Test2,
#                              medae_Test2,
#                              r2_Test2, 
#                              exvs_Test2]], 
#        delimiter=',',header='Mean Squared Error, Mean Absolute Error, Median Absolute Error,R2 Score, Explained Variance Score',comments='')
    #%%
    print ('8. Retraining the Model with new emission Signal...')
    
    min_max_scaler_Train_X1 = preprocessing.MinMaxScaler().fit(X_reTrain1);
    scaler_Train_X1 = preprocessing.StandardScaler().fit(X_reTrain1)
    X_reTr1=min_max_scaler_Train_X1.transform(X_reTrain1)
    X_reTr1=scaler_Train_X1.transform(X_reTr1)
    
#    min_max_scaler_Train_X2 = preprocessing.MinMaxScaler().fit(X_reTrain2);
#    scaler_Train_X2 = preprocessing.StandardScaler().fit(X_reTrain2)
#    X_reTr2=min_max_scaler_Train_X2.transform(X_reTrain2)
#    X_reTr2=scaler_Train_X2.transform(X_reTr2)
#    
    
    X_Te1=min_max_scaler_Train_X1.transform(X_Test1)
    X_Te1=scaler_Train_X1.transform(X_Te1)
    
#    X_Te2=min_max_scaler_Train_X2.transform(X_Test2)
#    X_Te2=scaler_Train_X2.transform(X_Te2)
    
    X_reTr1, y_surf1_reTrain=shuffle(X_reTr1, y_surf1_reTrain)
#    X_reTr2, y_surf2_reTrain=shuffle(X_reTr2, y_surf2_reTrain)
    
    clf_reTr1 = model.fit(X_reTr1, y_surf1_reTrain)
#    clf_reTr2 = model.fit(X_reTr2, y_surf2_reTrain)
    
    print ('8. new Results after training with recent emissions:')
    
    y_pred_Te1=clf_reTr1.predict(X_Te1)
    mse_Test1 = mean_squared_error(y_surf1_Test, y_pred_Te1)
    mae_Test1=mean_absolute_error(y_surf1_Test, y_pred_Te1)
    medae_Test1=median_absolute_error(y_surf1_Test, y_pred_Te1)
    r2_Test1=r2_score(y_surf1_Test, y_pred_Te1)
    exvs_Test1=explained_variance_score(y_surf1_Test, y_pred_Te1)
    
#    y_pred_Te2=clf_reTr2.predict(X_Te2)
#    mse_Test2 = mean_squared_error(y_surf2_Test, y_pred_Te2)
#    mae_Test2=mean_absolute_error(y_surf2_Test, y_pred_Te2)
#    medae_Test2=median_absolute_error(y_surf2_Test, y_pred_Te2)
#    r2_Test2=r2_score(y_surf2_Test, y_pred_Te2)
#    exvs_Test2=explained_variance_score(y_surf2_Test, y_pred_Te2)
    
    print ('\t Mean Squared Error      :', mse_Test1 )
    print ('\t Mean Absolute Error     :', mae_Test1)
    print ('\t Median Absolute Error   :', medae_Test1)
    print ('\t R2 Score                :', r2_Test1)
    print ('\t Explained Variance Score:', exvs_Test1)
    
    
    
    print ('9. Saving the new Results after training with recent emissions...')
    f =open(fileNamecsv1,'a');
    df = pd.DataFrame([[mse_Test1, mae_Test1,medae_Test1,r2_Test1, exvs_Test1]])
    df.to_csv(f,index = False,header= False);
    f.close();
    featureImportance(clf_reTr1, featureNames,filename1_reTr)
    
#    f =open(fileNamecsv2,'a');
#    df = pd.DataFrame([[mse_Test2, mae_Test2,medae_Test2,r2_Test2, exvs_Test2]])
#    df.to_csv(f,index = False,header= False);
#    f.close();
#    featureImportance(clf_reTr2, featureNames,filename2_reTr)
    
    print ('-----------:Finished!:--------------- \n')