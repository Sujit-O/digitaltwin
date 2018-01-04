# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 15:08:38 2017

@author: AICPS
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 17:26:42 2017

@author: AICPS
"""

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

originTDMSFolder = 'D:/GDrive/DT_Data/DAQ_Auto';
destinationCSVFolder = 'D:/GDrive/DT_Data/DAQ_Auto_TDMStoCSV';
tdmsFolderName = 'UM3_Corner_Wall_'+str(200)+'p';

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

#def tdms2csvTimingMetaData(tdmsFolderName):
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
        startTime=channel.property('wf_start_time')
        samplingIncrement=channel.property('wf_increment')
            #np.savetxt(directory+'/timingMetaData.csv', [startTime.hour,startTime.minute,startTime.second,startTime.microsecond, samplingIncrement], delimiter=',')
            
#def main():
#    tdms2csvTimingMetaData(tdmsFolderName)
#    
#if __name__ == '__main__':
#    main()    