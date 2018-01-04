
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 16:17:54 2017

@author: Sujit R. Chhetri
"""


from tdms2csv import tdms2csv
from DT_getTimingBasedonSegments import DT_getTimingBasedonSegments

#%%
def main():

    for i in [160]:#range (20,210,10):
        tdmsFolderNames = 'UM3_Corner_Wall_'+str(i)+'p';
        print(tdmsFolderNames)
        tdms2csv(tdmsFolderNames) # Add this for next iteration
        DT_getTimingBasedonSegments(tdmsFolderNames)
        
#        DT_getTimingBasedonSegments_withFigureOutput(tdmsFolderNames)


#%%    
if __name__ == '__main__':
    main()    
    
    