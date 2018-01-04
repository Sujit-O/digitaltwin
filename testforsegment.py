# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 16:17:54 2017

@author: Sujit R. Chhetri
"""

from nptdms import TdmsFile
import numpy as np
import os
import pandas as pd
from dateutil import parser
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import math
from sympy.geometry import Point,  Segment, Polygon

originTDMSFolder = 'D:/GDrive/DT_Data/DAQ_Auto';
destinationCSVFolder = 'D:/GDrive/DT_Data/DAQ_Auto_TDMStoCSV';
tdmsFolderName = 'UM3_Corner_Wall_'+str(160)+'p/data';
numberBaseSegments=64;
numberSideSegments=64;


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

              
class gcodeParser:
    def __init__(self, gcodeString):
        self.G=np.NAN
        self.M=np.NAN
        self.F=np.NAN
        self.X=np.NAN
        self.Y=np.NAN
        self.Z=np.NAN
        self.E=np.NAN
        self.ALL=0
        try:
            if np.isnan(gcodeString):
                return
        except:
            temp=0;
            data_1=gcodeString.split(";")
            data_2=data_1[0].split()
            for data in data_2:
                
                if 'G' in data:
                    self.G=float(data.split("G")[1])
                    temp+=1;
                elif 'M' in data:
                    self.M=float(data.split("M")[1])
                    temp+=1;
                elif 'F' in data:
                    self.F=float(data.split("F")[1])
                    temp+=1;
                elif 'X' in data:
                    self.X=float(data.split("X")[1])
                    temp+=1;
                elif 'Y' in data:
                    self.Y=float(data.split("Y")[1])    
                    temp+=1;
                elif 'Z' in data:
                    self.Z=float(data.split("Z")[1])    
                    temp+=1;
                elif 'E' in data:
                    self.E=float(data.split("E")[1])    
                    temp+=1;
                else:
                    pass
                self.ALL=temp
                
class lineSegment:
      def __init__(self,timeStart,timeStop, X1,Y1,Z1,X2,Y2,Z2,layer,OorI):
        self.timeStart=timeStart
        self.timeStop=timeStop
        self.X1=X1
        self.Y1=Y1
        self.X2=X2
        self.Y2=Y2
        self.Z1=Z1
        self.Z2=Z2
        self.layer=layer
        self.OorI=OorI  #true for O outskirt, false for I infill
        
    
#def tdms2csvTimeSegmentation(tdmsFolderName):
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
timingData = pd.read_csv(tdmsFolderNameFull+'/Timing.csv')
pcTime=parser.parse(timingData.PCTime[0])
lineSegmentList=[]


previousTime=0;
previousGcodeX=0;
previousGcodeY=0
previousGcodeZ=0;
layer=0;
OorI=True

consecutiveG=0;
startPrint=False;
for index, gcodeString in enumerate(timingData.GCode):
    gcode=gcodeParser(gcodeString)
    
    if not (startPrint):
     
        # Ignore the initial data E becomes negative before printing begins
        if gcode.E<0 or gcode.M==107:
            startPrint=True;
            continue
        else:
            pass
        
    else:
        
        if gcode.G==1 and (not (np.isnan(gcode.X)) or not(np.isnan(gcode.Y)) or not(np.isnan(gcode.Y))):
            
            
            consecutiveG=0
            if np.isnan(gcode.Z):
                gcode.Z=previousGcodeZ
            
            
            lineSegmentList.append(lineSegment(previousTime,timingData.PCTime[index],previousGcodeX,previousGcodeY, previousGcodeZ, gcode.X,gcode.Y,gcode.Z,layer,OorI))
            previousGcodeX=gcode.X;
            previousGcodeY=gcode.Y;
            previousGcodeZ=gcode.Z;
            
        elif gcode.G==0:
            consecutiveG+=1;
            
            if OorI and consecutiveG>=2:
                OorI=False
                    
            
            previousTime=timingData.PCTime[index]
            
            if not (np.isnan(gcode.X)):
                previousGcodeX=gcode.X;
                
            if not (np.isnan(gcode.Y)):
                previousGcodeY=gcode.Y;
                
            if not (np.isnan(gcode.Z)): 
                previousGcodeZ=gcode.Z
                layer+=1
                OorI=True
                consecutiveG=0;
                
            
        else:
            pass
 

fig = plt.figure()
#ax=fig.gca(projection='3d')
ax = fig.add_subplot(111, projection='3d')      


for coordinate in lineSegmentList:
         ax.plot([coordinate.X1,coordinate.X2], [coordinate.Y1,coordinate.Y2], [coordinate.Z1,coordinate.Z2])
#        ax.plot_wireframe([coordinate.X1,coordinate.X2], [coordinate.Y1,coordinate.Y2], [coordinate.Z1,coordinate.Z2],rstride=5, cstride=5)

plt.show()            
#plt.ion()
fig = plt.figure()
#ax=fig.gca(projection='3d')
ax = fig.add_subplot(111, projection='3d')      


for coordinate in lineSegmentList:
    if coordinate.layer<2 and coordinate.OorI==False:
        ax.plot([coordinate.X1,coordinate.X2], [coordinate.Y1,coordinate.Y2], [coordinate.Z1,coordinate.Z2])
#        ax.plot_wireframe([coordinate.X1,coordinate.X2], [coordinate.Y1,coordinate.Y2], [coordinate.Z1,coordinate.Z2],rstride=5, cstride=5)

plt.show()        

#%% Start segmenting the GCode based on sections




smallSquareX=[[] for i in range(numberBaseSegments)]
smallSquareY=[[] for i in range(numberBaseSegments)]


smallSquareSideX=[[] for i in range(numberBaseSegments)]
smallSquareSideY=[[] for i in range(numberBaseSegments)]
smallSquareSideZ=[[] for i in range(numberBaseSegments)]


#%% The Area for the Base of the object!

infillSquareX=[lineSegmentList[11].X2, lineSegmentList[8].X2,
              lineSegmentList[9].X2, lineSegmentList[11].X1,lineSegmentList[11].X2]
              
infillSquareY=[lineSegmentList[11].Y2, lineSegmentList[8].Y2,
              lineSegmentList[9].Y2,lineSegmentList[11].Y1,lineSegmentList[11].Y2]

fig = plt.figure()
#ax=fig.gca(projection='3d')
ax3 = fig.add_subplot(111, projection='3d')      


for i in range (4):
        ax3.plot([infillSquareX[i],infillSquareX[i+1]], [infillSquareY[i],infillSquareY[i+1]] )


plt.show()  


midinSide=int(math.sqrt(numberBaseSegments)-1)
numberofBaseMidPoints=int(midinSide*4)


  
numberSeg=0      
def areaSegments(numberBaseSegments, infillSquareX, infillSquareY,numberSeg):
#    print(numberBaseSegments)
#    print(numberSeg)  
    if int(numberBaseSegments==1):
        smallSquareX[int(numberSeg)]=infillSquareX
        smallSquareY[int(numberSeg)]=infillSquareY
        numberSeg+=1;
        return numberSeg
    
    else:
        squareX=[[] for i in range(4)]
        squareY=[[] for i in range(4)]
        midX=[]
        midY=[]
        for j in range(4):
                midX.append((infillSquareX[j]+infillSquareX[j+1])/2)
                midY.append((infillSquareY[j]+infillSquareY[j+1])/2)
        centerX=(midX[0]+midX[2])/2
        centerY=(midY[0]+midY[2])/2
        
        squareX[0].append(infillSquareX[0])
        squareX[0].append(midX[0])
        squareX[0].append(centerX)
        squareX[0].append(midX[3])
        squareX[0].append(infillSquareX[0])
        
        squareY[0].append(infillSquareY[0])
        squareY[0].append(midY[0])
        squareY[0].append(centerY)
        squareY[0].append(midY[3])
        squareY[0].append(infillSquareY[0])
        
        squareX[1].append(midX[0])
        squareX[1].append(infillSquareX[1])
        squareX[1].append(midX[1])
        squareX[1].append(centerX)
        squareX[1].append(midX[0])
        
        squareY[1].append(midY[0])
        squareY[1].append(infillSquareY[1])
        squareY[1].append(midY[1])
        squareY[1].append(centerY)
        squareY[1].append(midY[0])
        
        squareX[2].append(centerX)
        squareX[2].append(midX[1])
        squareX[2].append(infillSquareX[2])
        squareX[2].append(midX[2])
        squareX[2].append(centerX)
        
        squareY[2].append(centerY)
        squareY[2].append(midY[1])
        squareY[2].append(infillSquareY[2])
        squareY[2].append(midY[2])
        squareY[2].append(centerY)
        
        squareX[3].append(midX[3])
        squareX[3].append(centerX)
        squareX[3].append(midX[2])
        squareX[3].append(infillSquareX[3])
        squareX[3].append(midX[3])
        
        squareY[3].append(midY[3])
        squareY[3].append(centerY)
        squareY[3].append(midY[2])
        squareY[3].append(infillSquareY[3])
        squareY[3].append(midY[3])
        
        numberSeg=areaSegments(numberBaseSegments/4, squareX[0], squareY[0], numberSeg)
        
        numberSeg=areaSegments(numberBaseSegments/4, squareX[1], squareY[1], numberSeg)
       
        numberSeg=areaSegments(numberBaseSegments/4, squareX[2], squareY[2], numberSeg)
      
        numberSeg=areaSegments(numberBaseSegments/4, squareX[3], squareY[3], numberSeg)
        
#        print(numberSeg) 
        return numberSeg
        
numberSeg= areaSegments(numberBaseSegments, infillSquareX, infillSquareY,numberSeg)

fig = plt.figure()
#ax=fig.gca(projection='3d')
ax = fig.add_subplot(111, projection='3d')      


for j in range(numberBaseSegments):
    coordinateX=smallSquareX[j]
    coordinateY=smallSquareY[j]
    for i in range (4):
        ax.plot([coordinateX[i],coordinateX[i+1]], [coordinateY[i],coordinateY[i+1]] ,[lineSegmentList[0].Z1,lineSegmentList[0].Z2],color='b',linewidth=2.0)
    ax.text((coordinateX[0]+coordinateX[2])/2 ,(coordinateY[0]+coordinateY[2])/2, lineSegmentList[0].Z1,  '%s' % (str(j)), size=15, zorder=1,  color='r')     

plt.show()   


##print(numberSeg)       
# #plt.ion()
fig = plt.figure()
#ax=fig.gca(projection='3d')
ax = fig.add_subplot(111, projection='3d')      

for coordinate in lineSegmentList:
    if coordinate.layer<2 and coordinate.OorI==False:
        ax.plot([coordinate.X1,coordinate.X2], [coordinate.Y1,coordinate.Y2], [coordinate.Z1,coordinate.Z2])
        
for j in range(numberBaseSegments):
    coordinateX=smallSquareX[j]
    coordinateY=smallSquareY[j]
    for i in range (4):
        ax.plot([coordinateX[i],coordinateX[i+1]], [coordinateY[i],coordinateY[i+1]] ,[lineSegmentList[0].Z1,lineSegmentList[0].Z2],color='b',linewidth=2.0)


plt.show()         
  
#%%
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')      


for coordinate in lineSegmentList:
    if coordinate.layer>2 and coordinate.OorI==True:
        ax.plot([coordinate.X1,coordinate.X2], [coordinate.Y1,coordinate.Y2], [coordinate.Z1,coordinate.Z2])
        
plt.show() 

#%%  Segmentation for the wall The Area for the Side Wall!
#
infillSquareX=([lineSegmentList[11].X2, lineSegmentList[-1].X2, lineSegmentList[-1].X1,
              lineSegmentList[11].X1, lineSegmentList[11].X2])
              
infillSquareY=([lineSegmentList[11].Y2, lineSegmentList[-1].Y2, lineSegmentList[-1].Y1,
              lineSegmentList[11].Y1, lineSegmentList[11].Y2])

infillSquareZ=([lineSegmentList[11].Z2, lineSegmentList[-1].Z2, lineSegmentList[-1].Z1,
              lineSegmentList[11].Z1, lineSegmentList[11].Z2])
# The old area considered the base part as well!
    
#
fig = plt.figure()
#ax=fig.gca(projection='3d')
ax = fig.add_subplot(111, projection='3d')      


for i in range (4):
        ax.plot([infillSquareX[i],infillSquareX[i+1]], [infillSquareY[i],infillSquareY[i+1]],[infillSquareZ[i],infillSquareZ[i+1]] )


plt.show()  

numberSeg=0      
def areaSegmentsSide(numberSideSegments, infillSquareX, infillSquareY,infillSquareZ,numberSeg):
#    print(numberBaseSegments)
#    print(numberSeg)  
    if int(numberSideSegments==1):
        smallSquareSideX[int(numberSeg)]=infillSquareX
        smallSquareSideY[int(numberSeg)]=infillSquareY
        smallSquareSideZ[int(numberSeg)]=infillSquareZ
        numberSeg+=1;
        return numberSeg
    
    else:
        squareX=[[] for i in range(4)]
        squareY=[[] for i in range(4)]
        squareZ=[[] for i in range(4)]
        midX=[]
        midY=[]
        midZ=[]
        for j in range(4):
                midX.append((infillSquareX[j]+infillSquareX[j+1])/2)
                midY.append((infillSquareY[j]+infillSquareY[j+1])/2)
                midZ.append((infillSquareZ[j]+infillSquareZ[j+1])/2)
        centerX=(midX[0]+midX[2])/2
        centerY=(midY[0]+midY[2])/2
        centerZ=(midZ[0]+midZ[2])/2
        
        squareX[0].append(infillSquareX[0])
        squareX[0].append(midX[0])
        squareX[0].append(centerX)
        squareX[0].append(midX[3])
        squareX[0].append(infillSquareX[0])
        
        squareY[0].append(infillSquareY[0])
        squareY[0].append(midY[0])
        squareY[0].append(centerY)
        squareY[0].append(midY[3])
        squareY[0].append(infillSquareY[0])
        
        squareZ[0].append(infillSquareZ[0])
        squareZ[0].append(midZ[0])
        squareZ[0].append(centerZ)
        squareZ[0].append(midZ[3])
        squareZ[0].append(infillSquareZ[0])
        
        squareX[1].append(midX[0])
        squareX[1].append(infillSquareX[1])
        squareX[1].append(midX[1])
        squareX[1].append(centerX)
        squareX[1].append(midX[0])
        
        squareY[1].append(midY[0])
        squareY[1].append(infillSquareY[1])
        squareY[1].append(midY[1])
        squareY[1].append(centerY)
        squareY[1].append(midY[0])
        
        squareZ[1].append(midZ[0])
        squareZ[1].append(infillSquareZ[1])
        squareZ[1].append(midZ[1])
        squareZ[1].append(centerZ)
        squareZ[1].append(midZ[0])
        
        squareX[2].append(centerX)
        squareX[2].append(midX[1])
        squareX[2].append(infillSquareX[2])
        squareX[2].append(midX[2])
        squareX[2].append(centerX)
        
        squareY[2].append(centerY)
        squareY[2].append(midY[1])
        squareY[2].append(infillSquareY[2])
        squareY[2].append(midY[2])
        squareY[2].append(centerY)
        
        squareZ[2].append(centerZ)
        squareZ[2].append(midZ[1])
        squareZ[2].append(infillSquareZ[2])
        squareZ[2].append(midZ[2])
        squareZ[2].append(centerZ)
        
        squareX[3].append(midX[3])
        squareX[3].append(centerX)
        squareX[3].append(midX[2])
        squareX[3].append(infillSquareX[3])
        squareX[3].append(midX[3])
        
        squareY[3].append(midY[3])
        squareY[3].append(centerY)
        squareY[3].append(midY[2])
        squareY[3].append(infillSquareY[3])
        squareY[3].append(midY[3])
        
        squareZ[3].append(midZ[3])
        squareZ[3].append(centerZ)
        squareZ[3].append(midZ[2])
        squareZ[3].append(infillSquareZ[3])
        squareZ[3].append(midZ[3])
        
        numberSeg=areaSegmentsSide(numberSideSegments/4, squareX[0], squareY[0], squareZ[0],numberSeg)
        
        numberSeg=areaSegmentsSide(numberSideSegments/4, squareX[1], squareY[1],squareZ[1], numberSeg)
       
        numberSeg=areaSegmentsSide(numberSideSegments/4, squareX[2], squareY[2], squareZ[2],numberSeg)
      
        numberSeg=areaSegmentsSide(numberSideSegments/4, squareX[3], squareY[3],squareZ[3], numberSeg)
        
#        print(numberSeg) 
        return numberSeg
        
numberSeg= areaSegmentsSide(numberBaseSegments, infillSquareX, infillSquareY,infillSquareZ, numberSeg)
     
fig = plt.figure()
#ax=fig.gca(projection='3d')
ax = fig.add_subplot(111, projection='3d')      


for j in range(numberSideSegments):
    coordinateX=smallSquareSideX[j]
    coordinateY=smallSquareSideY[j]
    coordinateZ=smallSquareSideZ[j]
    for i in range (4):
        ax.plot([coordinateX[i],coordinateX[i+1]], [coordinateY[i],coordinateY[i+1]] ,[coordinateZ[i],coordinateZ[i+1]],color='b',linewidth=2.0)
    ax.text((coordinateX[0]+coordinateX[2])/2 ,(coordinateY[0]+coordinateY[2])/2, (coordinateZ[0]+coordinateZ[2])/2,  '%s' % (str(j)), size=15, zorder=1,  color='r')     

plt.show()   


##print(numberSeg)       
# #plt.ion()
fig = plt.figure()
#ax=fig.gca(projection='3d')
ax = fig.add_subplot(111, projection='3d')      

for coordinate in lineSegmentList:
     ax.plot([coordinate.X1,coordinate.X2], [coordinate.Y1,coordinate.Y2], [coordinate.Z1,coordinate.Z2])
        
for j in range(numberSideSegments):
    coordinateX=smallSquareSideX[j]
    coordinateY=smallSquareSideY[j]
    coordinateZ=smallSquareSideZ[j]
    for i in range (4):
        ax.plot([coordinateX[i],coordinateX[i+1]], [coordinateY[i],coordinateY[i+1]] ,[coordinateZ[i],coordinateZ[i+1]],color='b',linewidth=2.0)


plt.show()

fig = plt.figure()
#ax=fig.gca(projection='3d')
ax = fig.add_subplot(111, projection='3d')      

for coordinate in lineSegmentList:
     ax.plot([coordinate.X1,coordinate.X2], [coordinate.Y1,coordinate.Y2], [coordinate.Z1,coordinate.Z2])
        
for j in range(numberSideSegments):
    coordinateX=smallSquareSideX[j]
    coordinateY=smallSquareSideY[j]
    coordinateZ=smallSquareSideZ[j]
    for i in range (4):
        ax.plot([coordinateX[i],coordinateX[i+1]], [coordinateY[i],coordinateY[i+1]] ,[coordinateZ[i],coordinateZ[i+1]],color='b',linewidth=2.0)

for j in range(numberBaseSegments):
    coordinateX=smallSquareX[j]
    coordinateY=smallSquareY[j]
    for i in range (4):
        ax.plot([coordinateX[i],coordinateX[i+1]], [coordinateY[i],coordinateY[i+1]] ,[lineSegmentList[0].Z1,lineSegmentList[0].Z2],color='k',linewidth=2.0)

plt.show()
 

#%% outskirt dimension Segments

lIndex=[[] for i in range(int(math.sqrt(numberBaseSegments)))]
bIndex=[[] for i in range(int(math.sqrt(numberBaseSegments)))]



if numberBaseSegments==4:
    lIndex[0].append(3)
    lIndex[0].append(0)
    
    lIndex[1].append(2)
    lIndex[1].append(1)
    
    bIndex[0].append(3)
    bIndex[0].append(2)
    
    bIndex[1].append(0)
    bIndex[1].append(1)
    
elif numberBaseSegments==16:
    lIndex[0].append(15)
    lIndex[0].append(0)
    
    lIndex[1].append(14)
    lIndex[1].append(1)
    
    lIndex[2].append(11)
    lIndex[2].append(4)
    
    lIndex[3].append(10)
    lIndex[3].append(5)
    
    bIndex[0].append(15)
    bIndex[0].append(10)
    
    bIndex[1].append(12)
    bIndex[1].append(9)
    
    bIndex[2].append(3)
    bIndex[2].append(6)
    
    bIndex[3].append(0)
    bIndex[3].append(5)
    
    
    
    
elif numberBaseSegments==64:
    lIndex[0].append(63)
    lIndex[0].append(0)
    
    lIndex[1].append(62)
    lIndex[1].append(1)
    
    lIndex[2].append(59)
    lIndex[2].append(4)
    
    lIndex[3].append(5)
    lIndex[3].append(58)
    
    lIndex[4].append(47)
    lIndex[4].append(16)
    
    lIndex[5].append(46)
    lIndex[5].append(17)
    
    lIndex[6].append(43)
    lIndex[6].append(20)
    
    lIndex[7].append(42)
    lIndex[7].append(21)
    
    
    bIndex[0].append(63)
    bIndex[0].append(42)
    
    bIndex[1].append(60)
    bIndex[1].append(41)
    
    bIndex[2].append(51)
    bIndex[2].append(38)
    
    bIndex[3].append(48)
    bIndex[3].append(37)
    
    bIndex[4].append(15)
    bIndex[4].append(26)
    
    bIndex[5].append(12)
    bIndex[5].append(25)
    
    bIndex[6].append(3)
    bIndex[6].append(22)
    
    bIndex[7].append(0)
    bIndex[7].append(21)
    
    
else:
    pass


lengthOutskirtAreaSegmentsX=[[] for i in range(int(math.sqrt(numberBaseSegments)))]
lengthOutskirtAreaSegmentsY=[[] for i in range(int(math.sqrt(numberBaseSegments)))]

breadthOutskirtAreaSegmentsX=[[] for i in range(int(math.sqrt(numberBaseSegments)))]
breadthOutskirtAreaSegmentsY=[[] for i in range(int(math.sqrt(numberBaseSegments)))]

for i in range(int(math.sqrt(numberBaseSegments))):
    
    lengthOutskirtAreaSegmentsX[i].append(smallSquareX[lIndex[i][0]])
    lengthOutskirtAreaSegmentsX[i].append(smallSquareX[lIndex[i][1]])
    
    lengthOutskirtAreaSegmentsY[i].append(smallSquareY[lIndex[i][0]])
    lengthOutskirtAreaSegmentsY[i].append(smallSquareY[lIndex[i][1]])
    
    breadthOutskirtAreaSegmentsX[i].append(smallSquareX[bIndex[i][0]])
    breadthOutskirtAreaSegmentsX[i].append(smallSquareX[bIndex[i][1]])
    
    breadthOutskirtAreaSegmentsY[i].append(smallSquareY[bIndex[i][0]])
    breadthOutskirtAreaSegmentsY[i].append(smallSquareY[bIndex[i][1]])


#%%
fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')      

for coordinate in lineSegmentList:
     ax.plot([coordinate.X1,coordinate.X2], [coordinate.Y1,coordinate.Y2], [coordinate.Z1,coordinate.Z2])
        

for j in range(int(math.sqrt(numberBaseSegments))):
    
    
        squareX=lengthOutskirtAreaSegmentsX[j]
        squareY =lengthOutskirtAreaSegmentsY[j]
        for k in range(2):
            
            coordinateX=squareX[k]
            coordinateY=squareY[k]
            for i in range (4):
                ax.plot([coordinateX[i],coordinateX[i+1]], [coordinateY[i],coordinateY[i+1]] ,[lineSegmentList[0].Z1,lineSegmentList[0].Z2],color='k',linewidth=2.0)

plt.show()
    
#%%
fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')      

for coordinate in lineSegmentList:
     ax.plot([coordinate.X1,coordinate.X2], [coordinate.Y1,coordinate.Y2], [coordinate.Z1,coordinate.Z2])

for j in range(int(math.sqrt(numberBaseSegments))):
    
    
        squareX=breadthOutskirtAreaSegmentsX[j]
        squareY =breadthOutskirtAreaSegmentsY[j]
        for k in range(2):
            
            coordinateX=squareX[k]
            coordinateY=squareY[k]
            for i in range (4):
                ax.plot([coordinateX[i],coordinateX[i+1]], [coordinateY[i],coordinateY[i+1]] ,[lineSegmentList[0].Z1,lineSegmentList[0].Z2],color='k',linewidth=2.0)

plt.show()


#%% Wall outskirts
lengthSideAreaSegmentsX=[[] for i in range(int(math.sqrt(numberSideSegments)))]
lengthSideAreaSegmentsY=[[] for i in range(int(math.sqrt(numberSideSegments)))]
lengthSideAreaSegmentsZ=[[] for i in range(int(math.sqrt(numberSideSegments)))]

breadthSideAreaSegmentsX=[[] for i in range(int(math.sqrt(numberSideSegments)))]
breadthSideAreaSegmentsY=[[] for i in range(int(math.sqrt(numberSideSegments)))]
breadthSideAreaSegmentsZ=[[] for i in range(int(math.sqrt(numberSideSegments)))]

for i in range(int(math.sqrt(numberSideSegments))):
    
    lengthSideAreaSegmentsX[i].append(smallSquareSideX[lIndex[i][0]])
    lengthSideAreaSegmentsX[i].append(smallSquareSideX[lIndex[i][1]])
    
    lengthSideAreaSegmentsY[i].append(smallSquareSideY[lIndex[i][0]])
    lengthSideAreaSegmentsY[i].append(smallSquareSideY[lIndex[i][1]])
    
    lengthSideAreaSegmentsZ[i].append(smallSquareSideZ[lIndex[i][0]])
    lengthSideAreaSegmentsZ[i].append(smallSquareSideZ[lIndex[i][1]])
    
    
    breadthSideAreaSegmentsX[i].append(smallSquareSideX[bIndex[i][0]])
    breadthSideAreaSegmentsX[i].append(smallSquareSideX[bIndex[i][1]])
    
    breadthSideAreaSegmentsY[i].append(smallSquareSideY[bIndex[i][0]])
    breadthSideAreaSegmentsY[i].append(smallSquareSideY[bIndex[i][1]])

    breadthSideAreaSegmentsZ[i].append(smallSquareSideZ[bIndex[i][0]])
    breadthSideAreaSegmentsZ[i].append(smallSquareSideZ[bIndex[i][1]])


#%%
fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')      

for coordinate in lineSegmentList:
     ax.plot([coordinate.X1,coordinate.X2], [coordinate.Y1,coordinate.Y2], [coordinate.Z1,coordinate.Z2])

for j in range(int(math.sqrt(numberBaseSegments))):
    
    
        squareX=lengthSideAreaSegmentsX[j]
        squareY =lengthSideAreaSegmentsY[j]
        squareZ =lengthSideAreaSegmentsZ[j]
        for k in range(2):
            
            coordinateX=squareX[k]
            coordinateY=squareY[k]
            coordinateZ=squareZ[k]
            for i in range (4):
                ax.plot([coordinateX[i],coordinateX[i+1]], [coordinateY[i],coordinateY[i+1]] ,[coordinateZ[i],coordinateZ[i+1]],color='k',linewidth=2.0)

plt.show()
#%%

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')      

for coordinate in lineSegmentList:
     ax.plot([coordinate.X1,coordinate.X2], [coordinate.Y1,coordinate.Y2], [coordinate.Z1,coordinate.Z2])



for j in range(int(math.sqrt(numberBaseSegments))):
    
    
        squareX=breadthSideAreaSegmentsX[j]
        squareY =breadthSideAreaSegmentsY[j]
        squareZ =breadthSideAreaSegmentsZ[j]
        for k in range(2):
            
            coordinateX=squareX[k]
            coordinateY=squareY[k]
            coordinateZ=squareZ[k]
            for i in range (4):
                ax.plot([coordinateX[i],coordinateX[i+1]], [coordinateY[i],coordinateY[i+1]] ,[coordinateZ[i],coordinateZ[i+1]],color='k',linewidth=2.0)

plt.show()

    
#%%
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')      

for coordinate in lineSegmentList:
     ax.plot([coordinate.X1,coordinate.X2], [coordinate.Y1,coordinate.Y2], [coordinate.Z1,coordinate.Z2])

for j in range(int(math.sqrt(numberBaseSegments))):
    
    
        squareX=lengthOutskirtAreaSegmentsX[j]
        squareY =lengthOutskirtAreaSegmentsY[j]
        for k in range(2):
            
            coordinateX=squareX[k]
            coordinateY=squareY[k]
            for i in range (4):
                ax.plot([coordinateX[i],coordinateX[i+1]], [coordinateY[i],coordinateY[i+1]] ,[lineSegmentList[0].Z1,lineSegmentList[0].Z2],color='k',linewidth=2.0)


for j in range(int(math.sqrt(numberBaseSegments))):
    
    
        squareX=breadthOutskirtAreaSegmentsX[j]
        squareY =breadthOutskirtAreaSegmentsY[j]
        for k in range(2):
            
            coordinateX=squareX[k]
            coordinateY=squareY[k]
            for i in range (4):
                ax.plot([coordinateX[i],coordinateX[i+1]], [coordinateY[i],coordinateY[i+1]] ,[lineSegmentList[0].Z1,lineSegmentList[0].Z2],color='k',linewidth=2.0)

for j in range(int(math.sqrt(numberBaseSegments))):
    
    
        squareX=lengthSideAreaSegmentsX[j]
        squareY =lengthSideAreaSegmentsY[j]
        squareZ =lengthSideAreaSegmentsZ[j]
        for k in range(2):
            
            coordinateX=squareX[k]
            coordinateY=squareY[k]
            coordinateZ=squareZ[k]
            for i in range (4):
                ax.plot([coordinateX[i],coordinateX[i+1]], [coordinateY[i],coordinateY[i+1]] ,[coordinateZ[i],coordinateZ[i+1]],color='k',linewidth=2.0)
                
for j in range(int(math.sqrt(numberBaseSegments))):
    
    
        squareX=breadthSideAreaSegmentsX[j]
        squareY =breadthSideAreaSegmentsY[j]
        squareZ =breadthSideAreaSegmentsZ[j]
        for k in range(2):
            
            coordinateX=squareX[k]
            coordinateY=squareY[k]
            coordinateZ=squareZ[k]
            for i in range (4):
                ax.plot([coordinateX[i],coordinateX[i+1]], [coordinateY[i],coordinateY[i+1]] ,[coordinateZ[i],coordinateZ[i+1]],color='k',linewidth=2.0)

plt.show()


#%% Map the Segments and find their corresponding Timing!

#These list hold the outskirt segments for the side wall
lengthSideAreaSegmentsX
lengthSideAreaSegmentsY
lengthSideAreaSegmentsZ

breadthSideAreaSegmentsX
breadthSideAreaSegmentsY
breadthSideAreaSegmentsZ


# These list hold the outskirt segments for the base region
lengthOutskirtAreaSegmentsX
lengthOutskirtAreaSegmentsY

breadthOutskirtAreaSegmentsX
breadthOutskirtAreaSegmentsY

# This represent the total number of area segents for base and side wall
numberBaseSegments
numberSideSegments


#These are all small square segments in the base
smallSquareX
smallSquareY

#These are the small square segments in the side wall with constant X
smallSquareSideX
smallSquareSideY
smallSquareSideZ


#This holds all the linesegment in the G-code with following detail
#timeStart=timeStart
#timeStop=timeStop
#X1 
#Y1
#X2
#Y2
#Z1
#Z2
#layer number 1 and 2 for base.
#OorI  #true for O outskirt, false for I infill
lineSegmentList

    
    
def intersectionsInsideSquareBASE(lineSegmentT, xCoordinates, yCoordinates):
    p1=Point(xCoordinates[0],yCoordinates[0])
    p2=Point(xCoordinates[1],yCoordinates[1])
    p3=Point(xCoordinates[2],yCoordinates[2])
    p4=Point(xCoordinates[3],yCoordinates[3])
     
    lp1=Point(lineSegmentT.X1,lineSegmentT.Y1)
    lp2=Point(lineSegmentT.X2,lineSegmentT.Y2)
    
    gcodeSegment=Segment(lp1,lp2)
    square= Polygon(p1,p2,p3,p4)
    
    insideSquare = False
    
    startTime = parser.parse(lineSegmentT.timeStart)
    stopTime = parser.parse(lineSegmentT.timeStop)
    
    startTime1 = startTime
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')      
       
    ax.plot([lineSegmentT.X1,lineSegmentT.X2], [lineSegmentT.Y1,lineSegmentT.Y2], color='r',linewidth=5.0)
    
    for i in range (4):
      ax.plot([xCoordinates[i],xCoordinates[i+1]], [ yCoordinates[i], yCoordinates[i+1]] ,color='k',linewidth=1.0)
    
    
    
    interSections=gcodeSegment.intersection(square)
    
    print('intersection size: ', np.size(interSections))
    print('Intersection results', interSections)
    insideSquare = False
    print('Square Encloses: ', lp1, square.encloses(lp1))
    print('Square Encloses: ', lp2, square.encloses(lp2))
    
    d=lp1.distance(lp2)
    timewidth=stopTime-startTime
        
    if np.size(interSections)==4:
        insideSquare = True
        
        d1=lp1.distance(interSections[0])
        d2=lp1.distance(interSections[1])
        
        xt1,yt1=interSections[0]
        xt2,yt2=interSections[1]
        ax.plot([xt1,xt2], [yt1, yt2] ,color='g',linewidth=5.0)
        if d1<d2:
            startTime = startTime1 + timewidth*float(d1/d)
            stopTime = startTime1 + timewidth*float(d2/d)
            
        else:
           startTime = startTime1 + timewidth*float(d2/d)
           stopTime  = startTime1 + timewidth*float(d1/d)
           
    elif np.size(interSections)==2:
        if square.encloses(lp1):
              insideSquare = True
              d1=lp1.distance(interSections[0])
              xt1,yt1=lp1
              xt2,yt2=interSections[0]
              ax.plot([xt1,xt2], [yt1, yt2] ,color='g',linewidth=5.0)
              stopTime = startTime1 + timewidth*float(d1/d)
              
        elif square.encloses(lp2):
              insideSquare = True
              d1=lp1.distance(interSections[0])
              xt1,yt1=lp2
              xt2,yt2=interSections[0]
              ax.plot([xt1,xt2], [yt1, yt2] ,color='g',linewidth=5.0)
              startTime = startTime1 + timewidth*float(d1/d)   
              
        elif ('Segment' in str(type(interSections[0]))):  
              insideSquare = True
              xt1,yt1=interSections[0].p1
              xt2,yt2=interSections[0].p2
              ax.plot([xt1,xt2], [yt1, yt2] ,color='g',linewidth=5.0)
              
              d1=lp1.distance(interSections[0].p1)
              d2=lp1.distance(interSections[0].p2)
                
              lengthtointersection=lp1.distance(interSections[1])
              
              if interSections[0].contains(lp2):
                 startTime = startTime1 + timewidth*float(lengthtointersection/d)    
              else:
                 stopTime = startTime1 + timewidth*float(lengthtointersection/d)     
  
        else:   
             insideSquare = False
             
    elif np.size(interSections)==1:
          if ('Segment' in str(type(interSections[0]))):  
              insideSquare = True
              xt1,yt1=interSections[0].p1
              xt2,yt2=interSections[0].p2
              ax.plot([xt1,xt2], [yt1, yt2] ,color='g',linewidth=5.0)

          
    elif np.size(interSections)==3: 
          insideSquare = True
          for val in interSections:         
              if ('Segment' in str(type(val))):  
                   xt1,yt1=val.p1
                   xt2,yt2=val.p2
                   ax.plot([xt1,xt2], [yt1, yt2] ,color='g',linewidth=5.0)
                   
                   d1=lp1.distance(val.p1)
                   d2=lp1.distance(val.p2)
                   
                   if d1<d2:
                       startTime = startTime1 + timewidth*float(d1/d)
                       stopTime = startTime1 + timewidth*float(d2/d)    
                   else:
                       startTime = startTime1 + timewidth*float(d2/d)
                       stopTime = startTime1 + timewidth*float(d1/d)  

              
    elif np.size(interSections)==0: 
         if square.encloses(lp1) and square.encloses(lp2):
             insideSquare = True
             xt1,yt1=lp1
             xt2,yt2=lp2
             ax.plot([xt1,xt2], [yt1, yt2] ,color='g',linewidth=5.0)
        
    else:
        insideSquare = False
          
    plt.show()

    return  insideSquare, startTime, stopTime    
#%%
def intersectionsInsideSquareSIDE(lineSegmentT, xCoordinates, yCoordinates,zCoordinates):
    p1=Point(yCoordinates[0],zCoordinates[0])
    p2=Point(yCoordinates[1],zCoordinates[1])
    p3=Point(yCoordinates[2],zCoordinates[2])
    p4=Point(yCoordinates[3],zCoordinates[3])
     
    lp1=Point(lineSegmentT.Y1,lineSegmentT.Z1)
    lp2=Point(lineSegmentT.Y2,lineSegmentT.Z2)
    
    gcodeSegment=Segment(lp1,lp2)
    square= Polygon(p1,p2,p3,p4)
    
    insideSquare = False
    
    startTime = parser.parse(lineSegmentT.timeStart)
    stopTime = parser.parse(lineSegmentT.timeStop)
    
    startTime1 = startTime
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')      
       
    ax.plot([lineSegmentT.X1,lineSegmentT.X2], [lineSegmentT.Y1,lineSegmentT.Y2],[lineSegmentT.Z1,lineSegmentT.Z2], color='r',linewidth=5.0)
    
    for i in range (4):
      ax.plot([xCoordinates[i],xCoordinates[i+1]], [ yCoordinates[i], yCoordinates[i+1]] ,[zCoordinates[i], zCoordinates[i+1]],color='k',linewidth=1.0)
    
    
    
    interSections=gcodeSegment.intersection(square)
    
    print('intersection size: ', np.size(interSections))
    print('Intersection results', interSections)
    insideSquare = False
    print('Square Encloses: ', lp1, square.encloses(lp1))
    print('Square Encloses: ', lp2, square.encloses(lp2))
    
    d=lp1.distance(lp2)
    timewidth=stopTime-startTime
        
    if np.size(interSections)==4:
        insideSquare = True
        
        d1=lp1.distance(interSections[0])
        d2=lp1.distance(interSections[1])
        
        xt1,yt1=interSections[0]
        xt2,yt2=interSections[1]
        ax.plot([lineSegmentT.X1,lineSegmentT.X2],[xt1,xt2], [yt1, yt2] ,color='g',linewidth=5.0)
        if d1<d2:
            startTime = startTime1 + timewidth*float(d1/d)
            stopTime = startTime1 + timewidth*float(d2/d)
            
        else:
           startTime = startTime1 + timewidth*float(d2/d)
           stopTime  = startTime1 + timewidth*float(d1/d)
           
    elif np.size(interSections)==2:
        if square.encloses(lp1):
              insideSquare = True
              d1=lp1.distance(interSections[0])
              xt1,yt1=lp1
              xt2,yt2=interSections[0]
              ax.plot([lineSegmentT.X1,lineSegmentT.X2],[xt1,xt2], [yt1, yt2] ,color='g',linewidth=5.0)
              stopTime = startTime1 + timewidth*float(d1/d)
              
        elif square.encloses(lp2):
              insideSquare = True
              d1=lp1.distance(interSections[0])
              xt1,yt1=lp2
              xt2,yt2=interSections[0]
              ax.plot([lineSegmentT.X1,lineSegmentT.X2],[xt1,xt2], [yt1, yt2] ,color='g',linewidth=5.0)
              startTime = startTime1 + timewidth*float(d1/d)   
              
        elif ('Segment' in str(type(interSections[0]))):  
              insideSquare = True
              xt1,yt1=interSections[0].p1
              xt2,yt2=interSections[0].p2
              ax.plot([lineSegmentT.X1,lineSegmentT.X2],[xt1,xt2], [yt1, yt2] ,color='g',linewidth=5.0)
              
              d1=lp1.distance(interSections[0].p1)
              d2=lp1.distance(interSections[0].p2)
                
              lengthtointersection=lp1.distance(interSections[1])
              
              if interSections[0].contains(lp2):
                 startTime = startTime1 + timewidth*float(lengthtointersection/d)    
              else:
                 stopTime = startTime1 + timewidth*float(lengthtointersection/d)     
  
        else:   
             insideSquare = False
             
    elif np.size(interSections)==1:
          if ('Segment' in str(type(interSections[0]))):  
              insideSquare = True
              xt1,yt1=interSections[0].p1
              xt2,yt2=interSections[0].p2
              ax.plot([lineSegmentT.X1,lineSegmentT.X2],[xt1,xt2], [yt1, yt2] ,color='g',linewidth=5.0)

          
    elif np.size(interSections)==3: 
          insideSquare = True
          for val in interSections:         
              if ('Segment' in str(type(val))):  
                   xt1,yt1=val.p1
                   xt2,yt2=val.p2
                   ax.plot([lineSegmentT.X1,lineSegmentT.X2],[xt1,xt2], [yt1, yt2] ,color='g',linewidth=5.0)
                   
                   d1=lp1.distance(val.p1)
                   d2=lp1.distance(val.p2)
                   
                   if d1<d2:
                       startTime = startTime1 + timewidth*float(d1/d)
                       stopTime = startTime1 + timewidth*float(d2/d)    
                   else:
                       startTime = startTime1 + timewidth*float(d2/d)
                       stopTime = startTime1 + timewidth*float(d1/d)  

              
    elif np.size(interSections)==0: 
         if square.encloses(lp1) and square.encloses(lp2):
             insideSquare = True
             xt1,yt1=lp1
             xt2,yt2=lp2
             ax.plot([lineSegmentT.X1,lineSegmentT.X2],[xt1,xt2], [yt1, yt2] ,color='g',linewidth=5.0)
        
    else:
        insideSquare = False
          
    plt.show()

    return  insideSquare, startTime, stopTime      

#%%  Data Structure to store the start stop time and the other metadata for segments
class segmentTimes:
      def __init__(self,timeStart,timeStop):
        self.timeStart=timeStart
        self.timeStop=timeStop
        


class areaSegment:
      def __init__(self,segmentName, segmentNumber, segmentTimes):
        self.segmentTimes=segmentTimes
        self.segmentName= segmentName
        self.segmentNumber=segmentNumber
         
#%% Lets see for each Small Squares in the Base what lines will lie on it!
segmentName='BaseSquares';

timesForBaseSquares=[]

for i in range (numberBaseSegments):
    segmentTimeBag=[]
    xCoordinates=smallSquareX[i]
    yCoordinates=smallSquareY[i]
    print('current Segment: ', i )
    for index, lineSegmentT in enumerate(lineSegmentList):
        if lineSegmentT.layer<=2:
            print('lineSegment number: ', index )
            insideSquare, startTime, stopTime = intersectionsInsideSquareBASE(lineSegmentT,xCoordinates, yCoordinates)
            if insideSquare:
                segmentTimeBag.append(segmentTimes(startTime,stopTime))
            else:
                pass
        else:
             continue
    timesForBaseSquares.append(areaSegment(segmentName,i,segmentTimeBag))
#%%   Lets see for each Small Squares in the Side what lines will lie on it!   
segmentName='SideSquares';

timesForSideSquares=[]

for i in range (numberBaseSegments):
    segmentTimeBag=[]
    xCoordinates=smallSquareSideX[i]
    yCoordinates=smallSquareSideY[i]
    zCoordinates=smallSquareSideZ[i]
    print('current Segment: ', i )
    for index, lineSegmentT in enumerate(lineSegmentList):
        if lineSegmentT.layer>2:
            print('lineSegment number: ', index )
            insideSquare, startTime, stopTime = intersectionsInsideSquareSIDE(lineSegmentT,xCoordinates, yCoordinates, zCoordinates)
            if insideSquare:
                segmentTimeBag.append(segmentTimes(startTime,stopTime))
            else:
                pass
        else:
             continue
    timesForSideSquares.append(areaSegment(segmentName,i,segmentTimeBag))          
#%%   Lets see for each Small Squares in the Base Length what lines will lie on it!   

segmentName='BaseLengthSquares';

timesForBaseLengthSquares=[]

for i in range (int(math.sqrt(numberBaseSegments))):
    segmentTimeBag=[]
    for j in range(2):
        xCoordinates=lengthOutskirtAreaSegmentsX[i][j]
        yCoordinates=lengthOutskirtAreaSegmentsY[i][j]
        print('current Segment: ', i )
        for index, lineSegmentT in enumerate(lineSegmentList):
            if lineSegmentT.layer<=2:
                print('lineSegment number: ', index )
                insideSquare, startTime, stopTime = intersectionsInsideSquareBASE(lineSegmentT,xCoordinates, yCoordinates)
                if insideSquare:
                    segmentTimeBag.append(segmentTimes(startTime,stopTime))
                else:
                    pass
            else:
                 continue
    timesForBaseLengthSquares.append(areaSegment(segmentName,i,segmentTimeBag))
#%%   Lets see for each Small Squares in the Base breadth what lines will lie on it!   

segmentName='BaseBreadthSquares';

timesForBaseBreadthSquares=[]

 
for i in range (int(math.sqrt(numberBaseSegments))):
    segmentTimeBag=[]
    for j in range(2):
        xCoordinates=breadthOutskirtAreaSegmentsX[i][j]
        yCoordinates=breadthOutskirtAreaSegmentsY[i][j]
        print('current Segment: ', i )
        for index, lineSegmentT in enumerate(lineSegmentList):
            if lineSegmentT.layer<=2:
                print('lineSegment number: ', index )
                insideSquare, startTime, stopTime = intersectionsInsideSquareBASE(lineSegmentT,xCoordinates, yCoordinates)
                if insideSquare:
                    segmentTimeBag.append(segmentTimes(startTime,stopTime))
                else:
                    pass
            else:
                 continue
    timesForBaseBreadthSquares.append(areaSegment(segmentName,i,segmentTimeBag)) 

#%%   Lets see for each Small Squares in the Side Lengthwhat lines will lie on it!   
segmentName='SideLengthSquares';

timesForSideLengthSquares=[]



for i in range (int(math.sqrt(numberBaseSegments))):
    segmentTimeBag=[]
    for j in range(2):
        xCoordinates=lengthSideAreaSegmentsX[i][j]
        yCoordinates=lengthSideAreaSegmentsY[i][j]
        zCoordinates=lengthSideAreaSegmentsZ[i][j]
        
        print('current Segment: ', i )
        for index, lineSegmentT in enumerate(lineSegmentList):
            if lineSegmentT.layer>2:
                print('lineSegment number: ', index )
                insideSquare, startTime, stopTime = intersectionsInsideSquareSIDE(lineSegmentT,xCoordinates, yCoordinates, zCoordinates)
                if insideSquare:
                    segmentTimeBag.append(segmentTimes(startTime,stopTime))
                else:
                    pass
            else:
                 continue
    timesForSideLengthSquares.append(areaSegment(segmentName,i,segmentTimeBag))      
#%%   Lets see for each Small Squares in the Side Breadth what lines will lie on it!   
segmentName='SideBreadthSquares';

timesForSideBreadthSquares=[]

breadthSideAreaSegmentsY
breadthSideAreaSegmentsZ
for i in range (int(math.sqrt(numberBaseSegments))):
    segmentTimeBag=[]
    for j in range(2):
        xCoordinates=breadthSideAreaSegmentsX[i][j]
        yCoordinates=breadthSideAreaSegmentsY[i][j]
        zCoordinates=breadthSideAreaSegmentsZ[i][j]
        print('current Segment: ', i )
        for index, lineSegmentT in enumerate(lineSegmentList):
            if lineSegmentT.layer>2:
                
                print('lineSegment number: ', index )
                insideSquare, startTime, stopTime = intersectionsInsideSquareSIDE(lineSegmentT,xCoordinates, yCoordinates, zCoordinates)
                if insideSquare:
                    segmentTimeBag.append(segmentTimes(startTime,stopTime))
                else:
                    pass
                        
            elif lineSegmentT.layer<=2 and lineSegmentT.OorI==True:
              
                print('lineSegment number: ', index )
                insideSquare, startTime, stopTime = intersectionsInsideSquareSIDE(lineSegmentT,xCoordinates, yCoordinates, zCoordinates)
                if insideSquare:
                    segmentTimeBag.append(segmentTimes(startTime,stopTime))
                else:
                    pass
            else:
                 continue
    timesForSideBreadthSquares.append(areaSegment(segmentName,i,segmentTimeBag))       

#%%

