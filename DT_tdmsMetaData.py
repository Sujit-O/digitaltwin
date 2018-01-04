from nptdms import TdmsFile
import numpy as np
from glob import glob
from os import listdir
import pandas as pd


originTDMSFolder = 'D:/GDrive/DT_Data/DAQ_Auto';
destinationCSVFolder = 'D:/GDrive/DT_Data/DAQ_Auto_Features';
tdmsFolderName = 'UM3_Corner_Wall_130p';

tdmsFolderName=glob(originTDMSFolder+'/'+tdmsFolderName);

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
for file in listdir(tdmsFolderName[0]):
    if file.endswith(".tdms"):
          tdmsFiles+= [file]
            
#%%
for file in tdmsFiles:          
    tdms_file = TdmsFile(tdmsFolderName[0]+'/'+file)
    for channelName in channelNames:
        channel = tdms_file.object('data',channelName)
        data=channel.data
        startTime=channel.property('wf_start_time')
        samplingIncrement=channel.property('wf_increment')
        
        
        destinationFolderName=glob(destinationCSVFolder+'/'+tdmsFolderName[0]+'/'+tdms_file);
        
        np.savetxt('testdata.csv', data, delimiter=',')
        print(file+' : '+channelName)
#    channel = tdms_file.object('data','Mic_1')
#    data=channel.data;
#
#    np.savetxt('testdata.csv', data, delimiter=',')
    
#data = channel.data
#time = channel.time_track()