# -*- coding: utf-8 -*-
"""
Created on Sun Feb 27 23:39:11 2022

@author: lenovo
"""
import os
import pandas as pd
import numpy as np
import numpy.matlib 
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm  #字体管理器
from pylab import *
import seaborn as sns
mpl.rcParams['font.sans-serif'] = ['SimHei']


'''

for K-means

'''
os.chdir(r"E:/Meituan/")
cq=pd.read_csv("./data_new/cq0222.csv",index_col=0)
workcq=cq[cq["daytype"]=="Weekday"]

user=workcq.groupby(["user_id"]).sum()  #268157人使用ebike
ax = sns.kdeplot(user["count"], shade=True, color="r")

regular_user2=user[user["count"]>=20]  #大于20，平均一天2次以上的，有2303人
regular_user3=user[user["count"]>=30]  #大于20，平均一天3次以上的，有253人
regular_user3=regular_user3.reset_index()

#以编号为18的这个做实验
regular_user3[regular_user3.index==18]["user_id"] #16bdec96294306536c477e519699077b
topuser=workcq[workcq["user_id"]=="16bdec96294306536c477e519699077b"]


#需要把经纬度搞一下等差数列


sln=np.linspace(min(topuser["start_lng"]),max(topuser["start_lng"]),50)



#先看下全部的情况
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

#先直接用K-means聚类试试  (或许确实用bikeid比较合适？)

u = np.linspace(0, 2 * np.pi, 100) 
v = np.linspace(0, np.pi, 100) 
x = 10 * np.outer(np.cos(u), np.sin(v))

sln=topuser["start_lng"]
sla=topuser["start_lat"]

start=zip(sln,sla)