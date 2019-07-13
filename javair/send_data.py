import requests
import pandas as pd
import sys
import time

endpoint = "http://localhost:8050/update/"
if len(sys.argv)!=3 or float(sys.argv[2])<=0:
    print('Usage:',sys.argv[0],'<filename> <delay>')
    exit()

delay=float(sys.argv[2])
df=pd.read_csv(sys.argv[1],index_col=0)
for idx in range(df.shape[0]):
    j=df[idx:idx+1].to_json()
    requests.post(endpoint, json=j)
    print(idx)
    time.sleep(delay)


