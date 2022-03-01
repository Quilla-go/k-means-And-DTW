# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 15:19:36 2022

@author: lenovo
"""
import os
import pandas as pd
import numpy as np

#NMF
from sklearn.decomposition import NMF
import numpy as np
import matplotlib.pyplot as plt
import warning

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
plt.rcParams['figure.figsize'] = (12.0, 8.0)


################################################################################

os.chdir(r"E:/Meituan/")
allcounty=pd.read_csv("./data_new/allcounty_1002+od100.csv",index_col=0) 
daytype={}
daytype["weekday"]=['2021-03-01', '2021-03-02', '2021-03-03', '2021-03-04', '2021-03-05','2021-03-08', '2021-03-09', '2021-03-10',
        '2021-03-11', '2021-03-12']
daytype["Weekend"]=['2021-03-06', '2021-03-07','2021-03-13', '2021-03-14']

allcounty["daytype"]=['Weekend' if i in daytype["Weekend"] else "Weekday" for i in allcounty["pt_dt"]]
allcounty["start_hour"]=[int(i[11:13])+1 for i in allcounty["start_time1"]]
allcounty["end_hour"]=[int(i[11:13])+1 for i in allcounty["end_time1"]]

workcounty=allcounty[allcounty["daytype"]=="Weekday"]
workcounty=workcounty[['county_name', 'age', 'gender','distance', 'count', 'dura','start', 'end', 'start_hour', 'end_hour']]

county_name=list(allcounty["county_name"].unique())
timeline=range(1,25)
le=len(county_name)

'''
下次把county_name整体转拼音，见天气数据爬取

'''
from xpinyin import Pinyin
p = Pinyin()
county_name_pin=[p.get_pinyin(i) for i in county_name]

county_name["county_name_pin_"]=test



def tomatrix(workcq,count):
    endid=workcq.groupby(["end"],as_index=False).sum()  #9971  
    endid2=endid[endid["count"]>=count] 
    locationtest=list(endid2["end"])  
    table=pd.pivot_table(workcq,index=['end'],columns=['end_hour'],values=['count']
                        ,aggfunc=[np.sum],fill_value=0)
    table=table[table.index.isin(locationtest)]
    return table

def nmf_matrix(table,n_components):
    RATE_MATRIX = table.values
    nmf_model = NMF(n_components=n_components)  # 设有2个主题
    param = nmf_model.fit_transform(RATE_MATRIX)
    basism = nmf_model.components_
    return basism, param

for n in range(3,7):
    for j in range(le):
        #j=0
        #print(i)
        workcq=workcounty[workcounty['county_name']==county_name[j]]
        #cq=cq.reset_index(drop=True)
        #mcq=workcq[workcq["gender"]==1]
        #fcq=workcq[workcq["gender"]==2]
        table=tomatrix(workcq,count=20)
        
    
        basism,param =nmf_matrix(table,n_components=n)
        basis=pd.DataFrame(basism.T)  
        basis.columns=["Type "+str(i+1) for i in basis.columns]
        basis["grid"]=basis.index+1
        
        
        plt.figure(figsize=(24,12), dpi=300)
        plt.figure(1)
        plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=0.2,hspace=0.5)
        ax1 = plt.subplot(7,9,j+1)
        for z in range(len(basis.columns)-1):
            ax1 = sns.lineplot(x="grid", y="Type "+str(z+1),data=basis)
        plt.xticks(timeline,fontsize=6,rotation=90,fontweight='bold')
        plt.xlabel(county_name[j],fontsize=10)
        plt.ylabel('  ',fontsize=8)
    plt.savefig("./graph/travel pattern/allcounty_n{}.jpg".format(n))
    plt.clf() #需要重新更新画布，否则会出现同一张画布上绘制多张图片
        


    
    
    
    

        
   
    



