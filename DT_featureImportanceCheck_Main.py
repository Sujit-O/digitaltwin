

from DT_featureImportanceCheck import DT_featureImportanceCheck

start=80
stop=[120,130]

testGroup = 200

for i,stop in enumerate(stop):
    DT_featureImportanceCheck(start, stop,  testGroup,  'segments_Floor', True)
    