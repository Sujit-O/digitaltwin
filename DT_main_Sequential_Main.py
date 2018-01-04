
from DT_main_Sequential import DT_main_Sequential

#%%
def main():
      for i in range(120,150,10):
          DT_main_Sequential(80, i, i+10,  'segments_Wall')
          DT_main_Sequential(80, i, i+10,  'segments_Floor')
      for i in range(170,230,10):
          DT_main_Sequential(80, i, i+10,  'segments_Wall')
          DT_main_Sequential(80, i, i+10,  'segments_Floor')    
      for i in range(20,80,10):
          DT_main_Sequential(i+10, 120, i,  'segments_Wall')
          DT_main_Sequential(i+10, 120, i,  'segments_Floor')
        
#    for i in range(120,230,10):
#         if (i==160):
#             continue
#         print(i)
#         DT_main_Sequential(80, i, i+10,  'segments_Floor')
##    print('break')
#    for j in range(120,230,10):
#         if (j==160):
#             continue
#         print(j)
#         DT_main_Sequential(80, i, i+10,  'segments_Wall')     
##    print('break')
#    for i in range(30,90,10):
#        print(i)
#        DT_main_Sequential(i, 120, i-10,  'segments_Wall')  
##    print('break')    
#    for j in range(30,90,10):
#        print(j)
#        DT_main_Sequential(i, 120, i-10,  'segments_Floor')               
#         

#%%    
if __name__ == '__main__':
    main()    