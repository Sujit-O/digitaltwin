# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 16:17:54 2017

@author: Sujit R. Chhetri
"""

#from multiprocessing import Pool
from tdms2csv import tdms2csv

#%%
def main():
#     tdmsFolderNames ='UM3_Corner_Wall_'+str(100)+'p';
#     tdms2csv(tdmsFolderNames)
    #TODO 70p remaining
    for i in [160]:#range (10,210,10):
        tdmsFolderName = 'UM3_Corner_Wall_'+str(i)+'p';
        print(tdmsFolderName)
        tdms2csv(tdmsFolderName)
#    p = Pool(4)   
#    print(p.map(tdms2csv, tdmsFolderNames[0:5]));

#%%    
if __name__ == '__main__':
    main()    