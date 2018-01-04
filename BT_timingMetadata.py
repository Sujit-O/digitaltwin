# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 15:07:40 2017

@author: AICPS
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 16:17:54 2017

@author: Sujit R. Chhetri
"""

#from multiprocessing import Pool
from tdms2csvTimingMetaData import tdms2csvTimingMetaData

#%%
def main():
#     tdmsFolderNames ='UM3_Corner_Wall_'+str(100)+'p';
#     tdms2csv(tdmsFolderNames)
    for i in range (200,210,10):
        tdmsFolderNames = 'UM3_Corner_Wall_'+str(i)+'p';
        print(tdmsFolderNames)
        tdms2csvTimingMetaData(tdmsFolderNames)
#    p = Pool(4)   
#    print(p.map(tdms2csv, tdmsFolderNames[0:5]));

#%%    
if __name__ == '__main__':
    main()    