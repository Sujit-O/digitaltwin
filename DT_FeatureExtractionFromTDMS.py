# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 15:24:17 2017

@author: AICPS
"""

from nptdms import TdmsFile
import numpy as np
import pandas as pd
from glob import glob
import os
from pathlib import Path


originTDMSFolder = 'D:/GDrive/DT_Data/DAQ_Auto';
destinationCSVFolder = 'D:/GDrive/DT_Data/DAQ_Auto_TDMStoCSV';
tdmsFolderName = 'UM3_Corner_Wall_130p';

tdmsFolderNameFull=glob(originTDMSFolder+'/'+tdmsFolderName);


channelNames=['Mic_1',
              'Mic_2',
              'Mic_3',
              'Mic_4',
              'Current',
              'Vib_x2',
              'Vib_y2',
              'Vib_z2',
              'Vib_x1',
              'Vib_y1',
              'Vib_z1',
              'Vib_x0',
              'Vib_y0',
              'Vib_z0',
              'Temperature',
              'Humidity',
              'Mag_x0',
              'Mag_y0',
              'Mag_z0',
              'Mag_x1',
              'Mag_y1',
              'Mag_z1',
              'Mag_x2',
              'Mag_y2',
              'Mag_z2' ];

tdmsFiles=[]
#for f in listdir(tdmsFolderName[0]):
#        if re.search('.tdms', f):
#            tdmsFiles+= [f]
for file in os.listdir(tdmsFolderNameFull[0]):
    if file.endswith(".tdms"):
          tdmsFiles+= [file]
#%% Check if the destination foldername exists
destinationFolderName=destinationCSVFolder+'/'+tdmsFolderName;

if not os.path.exists(destinationFolderName):
    os.makedirs(destinationFolderName)    
         
#%%
directoryIndex=1;
for file in tdmsFiles:          
    tdms_file = TdmsFile(tdmsFolderNameFull[0]+'/'+file)
    
    directory=destinationCSVFolder+'/'+tdmsFolderName+'/data_'+str(directoryIndex);
    directoryIndex+=1;
    
    if not os.path.exists(directory):
        os.makedirs(directory) 
    
    for channelName in channelNames:
        channel = tdms_file.object('data',channelName)
        data=channel.data
        startTime=channel.property('wf_start_time')
        samplingIncrement=channel.property('wf_increment')
        
        np.savetxt(directory+'/'+channelName+'.csv', data, delimiter=',')
        np.savetxt(directory+'/timingMetaData.csv', [startTime.hour,startTime.minute,startTime.second,startTime.microsecond, samplingIncrement], delimiter=',')
        print(file+' : '+channelName)
#    channel = tdms_file.object('data','Mic_1')
#    data=channel.data;
#
#    np.savetxt('testdata.csv', data, delimiter=',')
    
#data = channel.data
#time = channel.time_track()