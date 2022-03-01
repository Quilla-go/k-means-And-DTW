# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 15:19:36 2022

@author: lenovo
"""
import os
import pandas as pd
import numpy as np

#作图
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm  #字体管理器
from pylab import *
import seaborn as sns
from matplotlib import rcParams

plt.rcParams['font.sans-serif'] = ['SimHei']  # 防止中文显示不出来
plt.rcParams['axes.unicode_minus'] = False  # 防止坐标轴符号显示不出来
plt.rcParams['figure.dpi'] = 300  

# os.chdir(r"E:/Meituan/")
# allcounty=pd.read_csv("./data_new/allcounty_1002+od100.csv",index_col=0) 
# cq=allcounty[allcounty["county_name"]=="重庆市"]
# cq=cq.reset_index(drop=True)
# #前期数据处理

# #转化时间 
# cq["start_hour"]=[int(i[11:13])+1 for i in cq["start_time1"]]
# cq["end_hour"]=[int(i[11:13])+1 for i in cq["end_time1"]]

# #选出工作日+使用量每周大于5的user？
# daytype={}
# daytype["weekday"]=['2021-03-01', '2021-03-02', '2021-03-03', '2021-03-04', '2021-03-05','2021-03-08', '2021-03-09', '2021-03-10',
#        '2021-03-11', '2021-03-12']
# daytype["Weekend"]=['2021-03-06', '2021-03-07','2021-03-13', '2021-03-14']

# cq["daytype"]=['Weekend' if i in daytype["Weekend"] else "Weekday" for i in cq["pt_dt"]]

# cq=cq[['user_id', 'age', 'gender',
#        'order_id', 'bike_id', 'start_lng',
#        'start_lat','end_lng', 'end_lat', 'hour', 'start_time1',
#        'end_time1', 'distance', 'count', 'dura',
#        'agetitle','start', 'end', 'start_hour', 'daytype']]

# cq.to_csv("./data_new/cq0222.csv")

'''
Data clean for NMF 
'''
os.chdir(r"E:/Meituan/data_new/")
cq=pd.read_csv("./cq0222.csv",index_col=0)


workcq=cq[cq["daytype"]=="Weekday"]
mcq=workcq[workcq["gender"]==1]
fcq=workcq[workcq["gender"]==2]

'''

#删掉出行count小于10的，保证每天至少到过一次？
startid=workcq.groupby(["start"],as_index=False).sum()  #14587
startid=startid.rename(columns={"start": "location"})
startid=startid[['location','count']] 
#startid1=startid[startid["count"]>=10]                  #6448 

endid=workcq.groupby(["end"],as_index=False).sum()  #9971
endid=endid.rename(columns={"end": "location"})    
endid=endid[['location','count']] 
#endid1=endid[endid["count"]>=10]                    #4948     

new=pd.concat([startid,endid],axis=0)
new1=new.groupby(["location"],as_index=False).sum()  #15611
#new1=new1[new1["count"]>=10]                         #7336
new2=new1[new1["count"]>=20]                         #5598
location=list(new2["location"])  #得到需要的location?

#得到矩阵 5598*24的矩阵
'''

'''
#选出有location list的坐标网格,邻接表转邻接矩阵，转换数据形式成为矩阵
每天平均流量为5以上的网格，count大于50时，有2959个网格
每天平均流量为10以上的网格，count大于100，有2055个网格
'''
def tomatrix(workcq,endid,count):
    #workcq=workcq[["start","start_hour","end","end_hour",'count']] 
    endid2=endid[endid["count"]>=count] 
    locationtest=list(endid2["location"])  
    # #方式一
    # M=workcq.groupby(["end","end_hour"],as_index=False).agg({"count":np.sum})
    # Mt=M[M["end"].isin(locationtest)]
    # table=pd.pivot_table(Mt,index=['end'],columns=['end_hour'],values=['count'],fill_value=0)
    
    # #方式二 直接透视
    test=pd.pivot_table(workcq,index=['end'],columns=['end_hour'],values=['count']
                        ,aggfunc=[np.sum],fill_value=0)
    test=test[test.index.isin(locationtest)]
    mx=test.values
    return test,mx

#table.to_csv("./travel_pattern/cq_end_up50_matrix.csv")
'''
非负矩阵分解
'''
from sklearn.decomposition import NMF
import numpy as np
import matplotlib.pyplot as plt
import warnings


def nmf_matrix(table,n_components):
    # 特征矩阵
    RATE_MATRIX = table.values
    nmf_model = NMF(n_components=n_components)  
    param = nmf_model.fit_transform(RATE_MATRIX)
    basism = nmf_model.components_

    return basism, param

table,mat=tomatrix(fcq,endid,count=20)
basism,param =nmf_matrix(table,n_components=5)

basis=pd.DataFrame(basism.T)  
basis.columns=["Type "+str(i+1) for i in basis.columns]
basis["grid"]=basis.index+1
timeline=range(1,25)

'''
                                     作图
'''

for i in range(len(basis.columns)-1):
    # plt.figure(dpi=100)  #加上后，线图无法叠在一起,所以作图时，在最开始的地方设置好
    ax1 = sns.lineplot(x="grid", y="Type "+str(i+1),data=basis)
plt.xticks(timeline,fontsize=8.5,rotation=90,fontweight='bold')
plt.xlabel('Time',fontsize=10)



