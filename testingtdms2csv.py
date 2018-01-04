# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 17:10:23 2017

@author: AICPS
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 16:17:54 2017

@author: Sujit R. Chhetri
"""

from nptdms import TdmsFile
import numpy as np
import os

originTDMSFolder = 'D:/GDrive/DT_Data/DAQ_Auto/data';
destinationCSVFolder = 'D:/GDrive/DT_Data/DAQ_Auto_TDMStoCSV';

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

#def tdms2csv(tdmsFolderName):
tdmsFolderName = 'UM3_Corner_Wall_'+str(160)+'p';
tdmsFiles=[]
tdmsFolderNameFull=originTDMSFolder+'/'+tdmsFolderName;

if not os.path.exists(tdmsFolderNameFull):
    exit()

for file in os.listdir(tdmsFolderNameFull):
    if file.endswith(".tdms"):
          tdmsFiles+= [file]
#%% Check if the destination foldername exists
destinationFolderName=destinationCSVFolder+'/'+tdmsFolderName;

if not os.path.exists(destinationFolderName):
    os.makedirs(destinationFolderName)    
         
#%%
directoryIndex=1;
for file in tdmsFiles: 
    print(file)         
    tdms_file = TdmsFile(tdmsFolderNameFull+'/'+file)
    
    directory=destinationCSVFolder+'/'+tdmsFolderName+'/data_'+str(directoryIndex);
    directoryIndex+=1;
    
    if not os.path.exists(directory):
        os.makedirs(directory) 
    
    for channelName in channelNames:
        channel = tdms_file.object('data',channelName)
        data=channel.data
        s=channel.property('wf_start_time')
        samplingIncrement=channel.property('wf_increment')
        
        np.savetxt(directory+'/'+channelName+'.csv', data, delimiter=',')
        np.savetxt(directory+'/timingMetaData.csv', [[s.hour,s.minute,s.second,s.microsecond, samplingIncrement]], delimiter=',')
            
