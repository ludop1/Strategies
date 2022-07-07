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
config['api_key'] = "fa31589f77e2137356f53177655f97c430eb2a11"
client = TiingoClient(config)

#universe1=['SPY', 'EFA', 'DBC', 'SHY', 'EEM', 'AGG', 'VNQ']
#universe2= ['SPY', 'QQQ', 'VNQ', 'REM', 'IEF', 'TLT', 'TIP', 'VGK', 'EWJ', 'SCZ', 'EEM', 'RWX', 'BWX', 'DBC', 'GLD', 'SHY']
#universe= ['XLB' ,'XLV', 'XLP', 'XLY', 'XLE', 'XLF', 'XLI', 'XLK', 'XLU', 'XLRE',"SHY"]
#universe=list(set(universe1+universe2+universe3+universe4+universe5))
universe=['VTI', 'VEA', 'VWO', 'SHY', 'BND', 'GSG', 'VNQ', 'VFISX']
try:
    df=pd.read_parquet(oggi+'_database.parquet', engine='fastparquet')
except:
    print('connesione')
    df = client.get_dataframe(universe, frequency="monthly", metric_name="adjClose", startDate="1998-01-01").dropna()
    df.index = pd.to_datetime(df.index)
    df.index = df.index + pd.offsets.MonthEnd(0) 
    df.to_parquet(oggi+'1.parquet', engine='fastparquet')
#data
df = df[universe].dropna()
cash='VFISX'
universe1 = universe.copy() 
universe1.remove(cash)

df1 = df[universe1].shift()
df_rate = (df.pct_change()).applymap(lambda x:0.00001 if x==0 else x)#pct change al mese
mom =(((12*df1.pct_change(1))+(4*df1.pct_change(3))+(2*df1.pct_change(6))+df1.pct_change(12))/4).dropna(how = 'all') #media momento 13612

#%%

def faa(df, df1, mnt_m, mnt_v, mnt_c, mnt_s, w_r, w_v, w_c, a, capital, cash, universe1, mom, df_rate):
    
    #ri
    df_r = mom if mnt_m==1 else df1.pct_change(mnt_m)#momentum    
    rank_r = df_r.rank(axis=1, ascending=False) #rank momentum
    df3 = df_r.copy().applymap(lambda x:(np.nan) if x<0 else x).applymap(lambda x:1 if np.isnan(x)==False else x).fillna(0) #controllo quali sono positivi
    rank_v = df1.rolling(mnt_v).var().rank(axis=1)# rank variance
    df_c = (df1.rolling(mnt_c).corr().mean(axis=1).unstack())[universe1]#correlation
    rank_c = df_c.rank(axis=1)#rank correlation
    #due modi: vecchio modo = no_cash(l_rank*df3), nuovo modo = l_rank(df3*rank)
    l_rank= ((w_c*rank_c+w_r*rank_r+w_v*rank_v)*df3).applymap(lambda x:(np.nan) if x==0 else x).rank(method='first', axis=1).dropna(how='all')#rank senza contare i negativi
    l_rank = l_rank.applymap(lambda x:1 if x<=a else 0) #faccio la select di quelli necessari
    l_rank[cash]= a-l_rank.sum(axis=1) #trovo la parte di cash
    portf = (l_rank* df_rate).dropna(how='all').applymap(lambda x:x+1 if x!=0 else x) #calcolo il portfolio
    portf[cash] = (portf[cash]+(l_rank[cash]-1)).apply(lambda x:0 if x<0 else x)  #normalizzo la parte di cash
    ricavi = pd.DataFrame(((portf.sum(axis=1)/a).cumprod())*capital, columns=['faa']) #calcolo i ricavi
    
    #weighted
    df_c1 = ((1-df_c)*l_rank[universe1]).dropna(how='all') #normalizzo la correlazione
    initial_w= df_c1.div((df_c1.sum(axis=1)), axis=0) #peso iniziale
    initial_w1=((1/(df1.rolling(mnt_s).std()))*initial_w).dropna(how='all') #moltiplico per risk parity multiplier
    final_w= initial_w1.div((initial_w1.sum(axis=1)), axis=0) #ripeso con nuovi valori
    final_w[cash]=l_rank[cash]/a # aggiungo la parte cash pesata
    final_w[universe1] = final_w[universe1].multiply((1-final_w[cash]), axis=0) #ripeso il resto a seconda del cash
    portf1 = ((df_rate.applymap(lambda x:x+1 if x!=0 else x))*final_w).dropna(how='all') #moltiplico il peso per il pct change
    ricavi1 = pd.DataFrame(((portf1.sum(axis=1)).cumprod())*capital, columns=['faaW']) #calcolo i ricavi  
    #log
    log1 = l_rank.copy().applymap(lambda x:1 if x!=1 and x!=0 else x)
    log2= ((log1*log1.columns).agg(lambda x: ', '.join(map(str, filter(None,x))), axis=1)).str.split(', ', 3, expand=True) #nome asset
    log3 = (log1*df).dropna().round(2).replace(0, None).agg(lambda x: ' , '.join(map(str, filter(None,x))), axis=1).str.split(', ', 3, expand=True) #prezzo asset
    log4 = final_w.round(2).replace(0, None).agg(lambda x: ' , '.join(map(str, filter(None,x))), axis=1).str.split(', ', 3, expand=True).round(2) #peso asset
    log = log2 + ', price= '+ log3 + ', weight='+log4
    
    return ricavi, ricavi1, log

#%%
r = faa(df, df1, 1, 4, 6, 6, 1, 0.49, 0.49, 3, 100000, cash, universe1, mom, df_rate)


asst= [1,2,3]
mnt_m = [4,6,12,1]
mnt_v = [4,6,12]
mnt_c = [4,6,12]
mnt_s = [4,6,12]
z = [asst,mnt_m,mnt_v,mnt_c,mnt_s]
param=list(itertools.product(*z))

#

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
h=h.sort_values("ROA-",ascending=False)

plot = h[0:5]
plot.index = plot.index.str.split(' ', 1, expand=True)

plot_df = pd.DataFrame()
for x in plot.index:
    a=x[1]
    b=R.get(a)
    plot_df[x]= b[x[0]+' '+x[1]]

plot_df.plot()

D = {}
D['G']= G
D['h']= h
D['L']= L
D['R']= R

with open(r'C:\Users\Stagista01\Desktop\strategies\fAA_update.pickle', 'wb') as f:
    obj = pickle.dump(D,f)


#%%

'''
 a= [1,2,3][2]
 mnt_m = [4,6,12,1][3]
 mnt_v = [4,6,12][0]
 mnt_c = [4,6,12][0]
 mnt_s = [4,6,12][1]
 capital = 100000
 w_r=1
 w_v= 0.49
 w_c= 0.49

    a= [1,2,3][2]
    mnt_m = [4,6,12,1][0]
    mnt_v = [4,6,12][0]
    mnt_c = [4,6,12][0]
    mnt_s = [4,6,12][0]
    capital = 100000
    w_r =1
    w_v =0.49
    w_c =0.49
def rebalace_annually(universe,capital,df_rend):
    df_rend=df_rend[list(universe.keys())]#select the portfolio ticker
    data={}#create a dict to store every year
    for year in sorted(set(df_rend.index.year)):#set is not sorted
        data[year]=((1+df_rend[df_rend.index.year==year]).cumprod()-1)*capital+capital#make a productory
        data[year]*= data[year].columns.map(universe)/100#multyply for weights
        capital=data[year].iloc[-1].sum()#update capital
    df_strategy=pd.concat(data.values())#rebuild the equity
    return df_strategy.sum(axis=1)#create a portfolio


strategy={}
#strategy["All Weather"]={"DBC":7.5,"GLD":7.5,"IEF":15,"SPY":30,"TLT":40}
strategy["Benchmark 60/40"]={"SPY":60,"IEF":40}
#strategy["Benchmark SPY"]={"SPY":100}
#strategy["Golden Butterfly"]={"GLD":20,"IWN":20,"SHY":20,"SPY":20,"TLT":20}
#strategy["Meb Faber’s Ivy"]={"DBC":20,"EFA":20,"IEF":20,"SPY":20,"VNQ":20}
#strategy["Permanent Portfolio CASH"]={"GLD":25,"SPY":25,"TLT":25,"CASH":25}
#strategy["Permanent Portfolio SHY"]={"GLD":25,"SPY":25,"TLT":25,"SHY":25}
ticker=set().union(*(d.keys() for d in strategy.values()))#extract the ticker and remove the duplicates

df = client.get_dataframe(["SPY","IEF"], frequency="monthly", metric_name="adjClose", startDate="1998-01-01").dropna()
df_rend=df.pct_change().dropna()#calc returns
df_rend.index = pd.to_datetime(df_rend.index)#convert index from string to datetime
equity = {k: rebalace_annually(v,capital,df_rend) for k, v in strategy.items()}#calc the strategy
df_n=pd.concat(equity,axis=1).dropna()#create a dataframe comparable
'''



'''
(dfplot/dfplot.iloc[0]*100).plot()
plt.show()
'''


