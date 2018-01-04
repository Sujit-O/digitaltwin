
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


#%%Initialize Global Variables
featureParentPath='D:/GDrive/DT_Data/DAQ_Auto_Features/'
destinationFolder='D:/GDrive/DT_Data/DAQ_Auto_Features/Results_Thickness'
KPI_fileName='D:/GDrive/DT_Data/DAQ_Auto_Features/KPI_Object_'
objectName = 'UM3_Corner_Wall_'
segment_Numbers=[2,7,8,13]
features=['timeFeatures.csv', 'frequencyFeatures.csv','STFTFeatures.csv']
#Use this when CWT feature extraction is complete!
#features=['timeFeatures.csv', 'frequencyFeatures.csv',
#'STFTFeatures','CWTFeatures']  

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
                   KPI_columnIndex, dataSeg,  y_seg, dataChannel,segmentName):
#    print ('Combine Segment Called... \n')
    thickness_KPI=KPI_values.values[segNum][KPI_columnIndex]
    
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
        
    if y_seg.empty:
        y_seg=y_KPI
    else:
        y_seg=pd.concat([y_seg,y_KPI], axis=0)
        
    return dataSeg, y_seg    
                
#%% This function combines the data in flow rate level and returns the data
def getXData(KPI_fileName,objectName,segment_Numbers, 
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
            dataSeg, y_seg= combineSegNums(objectFolderName,
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
    print ('Processing Data... \n') 
#    Xtr.fillna(0,inplace=True);
#    Xtr.values[np.isnan(Xtr.values)]=0
#    Xtr.values[np.isinf(Xtr.values)]=0
#    Xtr.values[np.isneginf(Xtr.values)]=0
#    Xtr.values = Xtr.values[~np.all(Xtr.values == 0, axis=1)]
    
#    Xtr.replace([np.finfo(np.float64).max, np.finfo(np.float64).min], np.nan);
#    Xtr.dropna(axis=0, how='any')
#    Xtr.dropna(axis=1, how='any')
#    Xtr=Xtr.fillna(Xtr.mean(),inplace=True)
#    
#    Xtr = Xtr.loc[:, (Xtr != 0).any(axis=0)];
#    Xtr = Xtr[(Xtr.T != 0).any()]
    #fill NaN values with zeros
#    Xtr.replace([np.finfo(np.float64).max, np.finfo(np.float64).min], 0);
    #Xtr.fillna(Xtr.mean(),inplace=True);
    y_T=y_thic.values
    X=Xtr.values
    X[np.isinf(X)]=X.mean()
    X[np.isneginf(X)]=X.mean()
    X[np.isnan(X)]=0
    X = X[~np.all(X == 0, axis=1)]
    y_T=np.ravel(y_T);
    X, y_T = shuffle(X, y_T) 
    return X, y_T

#%%
#def preProcessTwoTestData(X1,y1,X2,y2):
#    print ('Processing Data... \n')    
#    frames = [X1,X2]
#    result = pd.concat(frames, keys=['x', 'y'])
#    
#    #fill NaN values with zeros
#    result.replace([np.finfo(np.float64).max, np.finfo(np.float64).min], np.nan);
#    #Xtr.fillna(Xtr.mean(),inplace=True);
#    
#    X=result.values
#    X[np.isinf(X)]=X.mean()
#    X[np.isneginf(X)]=X.mean()
#    X[np.isnan(X)]=0
#    X = X[~np.all(X == 0, axis=1)]
#    y_T=np.ravel(y_T);
#    X, y_T = shuffle(X, y_T) 
#    return X, y_T


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
def featureImportance(clf,feature_names,textDescription):
    
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
    name=destinationFolder+'/FeatureRanking_'+textDescription+'.pdf' ;
    plt.savefig(name,bbox_inches='tight',dpi=1200)
#    plt.show()    
    
    
#%%    
def DT_main():
    print ('\n----------Start-----------\n')
    (n_estimators, 
     max_depth,  
     min_samples_split, 
     learning_rate,
     loss, 
     start, 
     stop, 
     testGroup, 
     segmentName) = parsingInit()
    
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
    
    print ('Extracting Data... \n')
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
    
    if not os.path.exists(destinationFolder):
        os.makedirs(destinationFolder)    
      
    paramsGBR = {'n_estimators': n_estimators, 'max_depth': max_depth, 
                 'min_samples_split': min_samples_split,
                 'learning_rate': learning_rate, 'loss': loss}
    
    model = ensemble.GradientBoostingRegressor(**paramsGBR)                      
    
    clf_Tr = clone(model)
   
    #%%
    print ('Building Model with all the Samples...\n')
    X_Train, y_thic_Train = preProcess(X_Train,y_thic_Train)
    
    
    min_max_scaler_Train_X = preprocessing.MinMaxScaler().fit(X_Train);
    scaler_Train_X = preprocessing.StandardScaler().fit(X_Train)
   
    
    X_Tr=min_max_scaler_Train_X.transform(X_Train)
    X_Tr=scaler_Train_X.transform(X_Tr)
#    print ('Shape of Training X: ',np.shape(X_Tr),' ...\n')
    
    clf_Tr = model.fit(X_Tr, y_thic_Train)
    print ('Results:\n')
    featureImportance(clf_Tr, featureNames,str(testGroup)+'_initialRankings_'+segmentName)
    
    #%%
    print ('Processing emissions Signals for Group ',flowRates_Test,' ...\n')
    X_Test,y_thic_Test= preProcess(X_Test,y_thic_Test)
    
#    print ('Shape of Testing X: ',np.shape(X_Test),' ...\n')
    X_Te=min_max_scaler_Train_X.transform(X_Test)
    X_Te=scaler_Train_X.transform(X_Te)
    
    print ('Predicting for Group ',flowRates_Test,' ...\n')
    y_pred_Te=clf_Tr.predict(X_Te)
    mse_Test = mean_squared_error(y_thic_Test, y_pred_Te)
    mae_Test=mean_absolute_error(y_thic_Test, y_pred_Te)
#    msle_Test=mean_squared_log_error(y_thic_Test, y_pred_Te)
    medae_Test=median_absolute_error(y_thic_Test, y_pred_Te)
    r2_Test=r2_score(y_thic_Test, y_pred_Te)
    exvs_Test=explained_variance_score(y_thic_Test, y_pred_Te)
    
    print ('Results:\n')
    print ('Mean Squared Error      :', mse_Test ,'\n')
    print ('Mean Absolute Error     :', mae_Test ,'\n')
#    print ('Mean squared Log Error  :', msle_Test ,'\n')
    print ('Median Absolute Error   :', medae_Test ,'\n')
    print ('R2 Score                :', r2_Test ,'\n')
    print ('Explained Variance Score:', exvs_Test ,'\n')
    
    fileNamecsv=destinationFolder+'/FeatureRanking_'+str(testGroup)+'_'+segmentName+'.csv'
    np.savetxt(fileNamecsv, [[mse_Test,
                              mae_Test,
                              medae_Test,
                              r2_Test, 
                              exvs_Test]], 
        delimiter=',',header='Mean Squared Error, Mean Absolute Error, Median Absolute Error,R2 Score, Explained Variance Score',comments='')
    print ('Retraining the Model with new emission Signal...\n')
    X_reTrain, y_thic_reTrain = preProcess(X_reTrain,y_thic_reTrain)
    
    
    min_max_scaler_Train_X = preprocessing.MinMaxScaler().fit(X_reTrain);
    scaler_Train_X = preprocessing.StandardScaler().fit(X_reTrain)
   
    
    X_reTr=min_max_scaler_Train_X.transform(X_reTrain)
    X_reTr=scaler_Train_X.transform(X_reTr)
    
    X_Te=min_max_scaler_Train_X.transform(X_Test)
    X_Te=scaler_Train_X.transform(X_Te)
#    print ('Shape of Training X: ',np.shape(X_reTr),' ...\n')
    
    clf_reTr = model.fit(X_reTr, y_thic_reTrain)
    print ('Results New:\n')
    y_pred_Te=clf_reTr.predict(X_Te)
    mse_Test = mean_squared_error(y_thic_Test, y_pred_Te)
    mae_Test=mean_absolute_error(y_thic_Test, y_pred_Te)
#    msle_Test=mean_squared_log_error(y_thic_Test, y_pred_Te)
    medae_Test=median_absolute_error(y_thic_Test, y_pred_Te)
    r2_Test=r2_score(y_thic_Test, y_pred_Te)
    exvs_Test=explained_variance_score(y_thic_Test, y_pred_Te)
    
    print ('Results:\n')
    print ('Mean Squared Error      :', mse_Test ,'\n')
    print ('Mean Absolute Error     :', mae_Test ,'\n')
#    print ('Mean squared Log Error  :', msle_Test ,'\n')
    print ('Median Absolute Error   :', medae_Test ,'\n')
    print ('R2 Score                :', r2_Test ,'\n')
    print ('Explained Variance Score:', exvs_Test ,'\n')
    
    f =open(fileNamecsv,'a');
    df = pd.DataFrame([[mse_Test, mae_Test,medae_Test,r2_Test, exvs_Test]])
    df.to_csv(f,index = False,header= False);
    f.close();
    featureImportance(clf_reTr, featureNames,str(testGroup)+'_reTrainedRankings_'+segmentName)

    
    print ('-----------:Finished!:--------------- \n')


#%% Call the main function
if __name__ == '__main__':
    DT_main()
