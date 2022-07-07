# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 13:35:46 2022

@author: Stagista01
"""

from tiingo import TiingoClient
import pandas as pd
import matplotlib.pyplot as plt
import quantstats as qs
import numpy as np
import itertools
import pickle
import time
from datetime import datetime
oggi=datetime.today().strftime('%Y-%m-%d')

config = {}
config['session']= True
config['api_key'] = "your_key"
client = TiingoClient(config)

universe_1=['SPY', 'EFA', 'DBC', 'SHY', 'EEM', 'AGG', 'VNQ']
universe_2= ['SPY', 'QQQ', 'VNQ', 'REM', 'IEF', 'TLT', 'TIP', 'VGK', 'EWJ', 'SCZ', 'EEM', 'RWX', 'BWX', 'DBC', 'GLD', 'SHY']
universe_3= ['XLB' ,'XLV', 'XLP', 'XLY', 'XLE', 'XLF', 'XLI', 'XLK', 'XLU', 'XLRE',"SHY"]
universe_4=['VTI', 'VEA', 'VWO', 'SHY', 'BND', 'GSG', 'VNQ', 'VFISX']
universe_l=list(set(universe_1+universe_2+universe_3+universe_4))

try:
    df=pd.read_parquet(oggi+'_database.parquet', engine='fastparquet')
except:
    print('connesione')
    df = client.get_dataframe(universe_l, frequency="monthly", metric_name="adjClose", startDate="1998-01-01").dropna()
    df.index = pd.to_datetime(df.index)
    df.index = df.index + pd.offsets.MonthEnd(0) 
    df.to_parquet(oggi+'1.parquet', engine='fastparquet')
    
#data
df = df[universe4].dropna()
cash='VFISX'
universe1 = universe.copy() 
universe1.remove(cash)

df1 = df[universe1].shift()
df_rate = (df.pct_change()).applymap(lambda x:0.00001 if x==0 else x)#pct change al mese
mom =(((12*df1.pct_change(1))+(4*df1.pct_change(3))+(2*df1.pct_change(6))+df1.pct_change(12))/4).dropna(how = 'all') #media momento 13612

#%%

def faa(df, df1, mnt_m, mnt_v, mnt_c, mnt_s, w_r, w_v, w_c, a, capital, cash, universe1, mom, df_rate):
   
    df_r = mom if mnt_m==1 else df1.pct_change(mnt_m)#momentum    
    rank_r = df_r.rank(axis=1, ascending=False) #rank momentum
    df3 = df_r.copy().applymap(lambda x:(np.nan) if x<0 else x).applymap(lambda x:1 if np.isnan(x)==False else x).fillna(0) #check for positive
    rank_v = df1.rolling(mnt_v).var().rank(axis=1) #rank variance
    df_c = (df1.rolling(mnt_c).corr().mean(axis=1).unstack())[universe1] #correlation
    rank_c = df_c.rank(axis=1) #rank correlation
    l_rank= ((w_c*rank_c+w_r*rank_r+w_v*rank_v)*df3).applymap(lambda x:(np.nan) if x==0 else x).rank(method='first', axis=1).dropna(how='all') #L rank 
    l_rank = l_rank.applymap(lambda x:1 if x<=a else 0) #select your asset
    l_rank[cash]= a-l_rank.sum(axis=1) #cash part
    portf = (l_rank* df_rate).dropna(how='all').applymap(lambda x:x+1 if x!=0 else x) #portfolio
    portf[cash] = (portf[cash]+(l_rank[cash]-1)).apply(lambda x:0 if x<0 else x)  #normailze cash part
    ricavi = pd.DataFrame(((portf.sum(axis=1)/a).cumprod())*capital, columns=['faa']) #calculate the revenue
    
    #weighted (MINCOR)
    df_c1 = ((1-df_c)*l_rank[universe1]).dropna(how='all') #normalize correlation
    initial_w= df_c1.div((df_c1.sum(axis=1)), axis=0) #initial weight
    initial_w1=((1/(df1.rolling(mnt_s).std()))*initial_w).dropna(how='all') #multiply by risk party multiplier
    final_w= initial_w1.div((initial_w1.sum(axis=1)), axis=0) #re-weighting with new values
    final_w[cash]=l_rank[cash]/a  #add cash part
    final_w[universe1] = final_w[universe1].multiply((1-final_w[cash]), axis=0) #re-weight with cash
    portf1 = ((df_rate.applymap(lambda x:x+1 if x!=0 else x))*final_w).dropna(how='all') #multiply weight by rate of change
    ricavi1 = pd.DataFrame(((portf1.sum(axis=1)).cumprod())*capital, columns=['faaW']) #calculate the revenue
   
    #log
    log1 = l_rank.copy().applymap(lambda x:1 if x!=1 and x!=0 else x) 
    log2= ((log1*log1.columns).agg(lambda x: ', '.join(map(str, filter(None,x))), axis=1)).str.split(', ', 3, expand=True) #name asset
    log3 = (log1*df).dropna().round(2).replace(0, None).agg(lambda x: ' , '.join(map(str, filter(None,x))), axis=1).str.split(', ', 3, expand=True) #price asset
    log4 = final_w.round(2).replace(0, None).agg(lambda x: ' , '.join(map(str, filter(None,x))), axis=1).str.split(', ', 3, expand=True).round(2) #weight asset
    log = log2 + ', price= '+ log3 + ', weight='+log4
    
    return ricavi, ricavi1, log

#%%
#r = faa(df, df1, 1, 4, 6, 6, 1, 0.49, 0.49, 3, 100000, cash, universe1, mom, df_rate)

#parameters
a= [1,2,3]
mnt_m = [4,6,12,1]
mnt_v = [4,6,12]
mnt_c = [4,6,12]
mnt_s = [4,6,12]
z = [a,mnt_m,mnt_v,mnt_c,mnt_s]
param=list(itertools.product(*z))

#loop to calculate all possible option
start_time = time.time()
R={}
L = {}
for x in param:
    dfplot = pd.DataFrame()
    r = faa(df, df1, x[1], x[2], x[3], x[4], 1, 0.49, 0.49, x[0], 100000, cash, universe1, mom, df_rate)
    dfplot['FAA ' + str(x)] = r[0]
    dfplot['FAAw ' + str(x)] = r[1]
    dfplot = dfplot.fillna(method = 'bfill')
    R[str(x)]=dfplot
    L[str(x)]= r[2]
print("--- %s seconds ---" % (time.time() - start_time))

#loop to calculate CAGR
G=[]
for x in R:
    df_cagr = pd.DataFrame()
    df_cagr['CAGR-'] = R[x].iloc[-1].div(R[x].iloc[0]).pow(1./((len(R[x].index)/12) - 1)).sub(1)*100 
    dd=((R[x].expanding().max()-R[x])/R[x].expanding().max()*-100).round(1)
    df_cagr['DD%-'] =dd.min()*-1
    #df_cagr['ROA-'] =(R[x].iloc[-1]/-dd.min()).round(0)
    df_cagr['ROA-'] = (df_cagr['CAGR-']/df_cagr['DD%-'])*100
    G.append(df_cagr)
h=pd.concat(G)
h=h.sort_values("ROA-",ascending=False) #rank based on ROA




