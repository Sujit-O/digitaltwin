import numpy as np
import pandas as pd


df1 = pd.DataFrame({'A': [0, 0, 0, 0],
                    'B': [0, 4, 7, 9],
                    'C': [0, 4, 1, 3],
                    'D': [0, 2, 1, 8]},
                     index=[0, 1, 2, 3])


df2 = pd.DataFrame({'A': [0, 7, 1, np.NaN],
                    'B': [0, 4, 12, np.NaN],
                    'C': [0, 7, 0, np.NaN],
                    'D': [0, 0, 0, 0]},
                     index=[0, 1, 2, 3])



frames = [df1, df2]
result = pd.concat(frames, keys=['x', 'y'])
 
result.replace([np.finfo(np.float64).max, np.finfo(np.float64).min], np.nan);
    #Xtr.fillna(Xtr.mean(),inplace=True);
df1.fillna(0,inplace=True);    
print(df1) 
df1= df1.loc[:, (df1 != 0).any(axis=0)];  
print(df1) 
df1 = df1[(df1.T != 0).any()]
print(df1)  



df2.fillna(0,inplace=True);
print(df2) 
df2= df2.loc[:, (df2 != 0).any(axis=0)];  
print(df2) 
df2 = df2[(df2.T != 0).any()]
print(df2) 


