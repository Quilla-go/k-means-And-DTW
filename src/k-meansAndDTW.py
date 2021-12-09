import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import math
import random
import numpy as np
from pandas import read_csv as pdreadcsv
from pandas import DataFrame as pddataframe


# 展示一下如何使用plot绘图。
def get_draw():
    x = np.linspace(0, 50, 100)
    ts1 = pd.Series(3.1 * np.sin(x / 1.5) + 3.5)
    ts2 = pd.Series(2.2 * np.sin(x / 3.5 + 2.4) + 3.2)
    ts3 = pd.Series(0.04 * x + 3.0)
    ts1.plot()
    ts2.plot()
    ts3.plot()
    plt.ylim(-2, 10)
    plt.legend(['ts1', 'ts2', 'ts3'])
    plt.show()

    def euclid_dist(t1, t2):
        return math.sqrt(sum((t1 - t2)**2))

    print(euclid_dist(ts1, ts2))  # 26.959216038
    print(euclid_dist(ts1, ts3))  # 23.1892491903


# 1   数据提取
def get_wbcdata(filename):
    df = pdreadcsv(filename)
    workdata = df[["num", "days", "wbc"]]

    workdata_len = len(workdata["num"])
    days = [0] * (workdata_len)
    WBC = [0] * (workdata_len)
    Numb = [0] * (workdata_len)
    for i in range(workdata_len):
        Numb[i] = workdata["num"][i]
        days[i] = workdata["days"][i]
        WBC[i] = workdata["wbc"][i]
    s = int(len(WBC) / 30)
    WBCData = np.mat(WBC).reshape(s, 30)
    WBCData = np.array(WBCData)
    return WBCData, Numb, days, WBC
    # print(WBCData)


def draw_wbcdata(filename):
    WBCData, _, _, _ = get_wbcdata(filename)

    for every_wbcdata in WBCData:
        plt.plot(every_wbcdata)
        plt.legend(['WBC'])
    plt.show()


# 2  定义相似距离
# DTW距离，时间复杂度为两个时间序列长度相乘
def DTWDistance(s1, s2):
    DTW = {}
    s1_len = len(s1)
    s2_len = len(s2)
    for i in range(s1_len):
        DTW[(i, -1)] = float('inf')
    for i in range(s2_len):
        DTW[(-1, i)] = float('inf')
    DTW[(-1, -1)] = 0
    for i in range(s1_len):
        for j in range(s2_len):
            dist = (s1[i] - s2[j])**2
            DTW[(i, j)] = dist + min(DTW[(i - 1, j)], DTW[(i, j - 1)],
                                     DTW[(i - 1, j - 1)])
    return math.sqrt(DTW[len(s1) - 1, len(s2) - 1])


# DTW_W距离, 优化后的算法，只检测前W个窗口的值
def DTWDistance_W(s1, s2, w):
    DTW = {}
    s1_len = len(s1)
    s2_len = len(s2)
    w = max(w, abs(s1_len - s2_len))
    for i in range(-1, s1_len):
        for j in range(-1, s2_len):
            DTW[(i, j)] = float('inf')
    DTW[(-1, -1)] = 0
    for i in range(s1_len):
        for j in range(max(0, i - w), min(s2_len, i + w)):
            dist = (s1[i] - s2[j])**2
            DTW[(i, j)] = dist + min(DTW[(i - 1, j)], DTW[(i, j - 1)],
                                     DTW[(i - 1, j - 1)])
    return math.sqrt(DTW[s1_len - 1, s2_len - 1])


# Another way to speed things up is to use the LB Keogh lower
# bound of dynamic time warping
def LB_Keogh(s1, s2, r):
    LB_sum = 0
    for ind, i in enumerate(s1):
        # print(s2)
        lower_bound = min(s2[(ind - r if ind - r >= 0 else 0):(ind + r)])
        upper_bound = max(s2[(ind - r if ind - r >= 0 else 0):(ind + r)])
        if i >= upper_bound:
            LB_sum = LB_sum + (i - upper_bound)**2
        elif i < lower_bound:
            LB_sum = LB_sum + (i - lower_bound)**2
    return math.sqrt(LB_sum)


# 3  定义K-means算法
# num_clust分类的数量，
def k_means_clust(data, num_clust, num_iter, w=5):
    # 步骤一: 初始化均值点
    centroids = random.sample(list(data), num_clust)
    counter = 0
    for n in range(num_iter):
        counter += 1

        assignments = {}  # 存储类别0，1，2等类号和所包含的类的号码
        # 遍历每一个样本点 i ,因为本题与之前有所不同，多了ind的编码
        for ind, i in enumerate(data):
            min_dist = float('inf')  # 最近距离，初始定一个较大的值
            closest_clust = None  # closest_clust：最近的均值点编号
            # 步骤二: 寻找最近的均值点
            for c_ind, j in enumerate(centroids):  # 每个点和中心点的距离，共有num_clust个值
                if LB_Keogh(i, j, 3) < min_dist:  # 循环去找最小的那个
                    cur_dist = DTWDistance_W(i, j, w)
                    if cur_dist < min_dist:  # 找到了ind点距离c_ind最近
                        min_dist = cur_dist
                        closest_clust = c_ind
            # 步骤三: 更新 ind 所属簇
            # print(closest_clust)
            if closest_clust in assignments:
                assignments[closest_clust].append(ind)
            else:
                assignments[closest_clust] = []
                assignments[closest_clust].append(ind)
        # 步骤四: 更新簇的均值点
        for key in assignments:
            clust_sum = 0
            for k in assignments[key]:
                clust_sum = clust_sum + data[k]
            centroids[key] = [m / len(assignments[key]) for m in clust_sum]
    # 返回聚类中心值，和聚类的所有点的数组序号
    return centroids, assignments


def main(filename):
    num_clust = 2  # 定义需要分类的数量
    WBCData, Numb, days, WBC = get_wbcdata(filename)

    centroids, assignments = k_means_clust(WBCData, num_clust, 800, 3)

    for i in range(num_clust):
        s = []
        WBC01 = []
        days01 = []
        for j, indj in enumerate(assignments[i]):  # 画出各分类点的坐标
            s.append(int(Numb[indj * 30]))
            WBC01 = np.hstack((WBC01, WBC[30 * indj:30 * indj + 30]))
            days01 = np.hstack((days01, days[0:30]))
        plt.title('%s' % s)
        plt.plot(centroids[i], lw=4)
        plt.scatter(days01, WBC01)
    plt.show()


if __name__ == '__main__':
    # get_draw()
    filename = r"./20200315.csv"
    # draw_wbcdata(filename)
    # get_wbcdata(filename)
    # get_Numbdata(filename)
    main(filename)
