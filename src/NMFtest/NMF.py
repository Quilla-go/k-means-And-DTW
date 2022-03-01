# -*- coding: utf-8 -*-
"""
Created on Sun Feb 27 23:05:05 2022

@author: lenovo
"""

from sklearn.decomposition import NMF
import numpy as np
import matplotlib.pyplot as plt
import warnings
# warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['SimHei']  # 防止中文显示不出来
plt.rcParams['axes.unicode_minus'] = False  # 防止坐标轴符号显示不出来


def get_datas(n_components):
    # 10部电影
    item = [
        '希特勒回来了', '死侍', '房间', '龙虾', '大空头',
        '极盗者', '裁缝', '八恶人', '实习生', '间谍之桥',
    ]

    # 15个用户
    user = ['五柳君', '帕格尼六', '木村静香', 'WTF', 'airyyouth',
            '橙子c', '秋月白', 'clavin_kong', 'olit', 'You_某人',
            '凛冬将至', 'Rusty', '噢！你看！', 'Aron', 'ErDong Chen']

    # 15×5的特征矩阵
    RATE_MATRIX = np.array(
        [[5, 5, 3, 0, 5, 5, 4, 3, 2, 1, 4, 1, 3, 4, 5],
         [5, 0, 4, 0, 4, 4, 3, 2, 1, 2, 4, 4, 3, 4, 0],
         [0, 3, 0, 5, 4, 5, 0, 4, 4, 5, 3, 0, 0, 0, 0],
         [5, 4, 3, 3, 5, 5, 0, 1, 1, 3, 4, 5, 0, 2, 4],
         [5, 4, 3, 3, 5, 5, 3, 3, 3, 4, 5, 0, 5, 2, 4],
         [5, 4, 2, 2, 0, 5, 3, 3, 3, 4, 4, 4, 5, 2, 5],
         [5, 4, 3, 3, 2, 0, 0, 0, 0, 0, 0, 0, 2, 1, 0],
         [5, 4, 3, 3, 2, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
         [5, 4, 3, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2],
         [5, 4, 3, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]]
    )

    nmf_model = NMF(n_components=n_components)  # 设有2个主题
    item_dis = nmf_model.fit_transform(RATE_MATRIX)
    user_dis = nmf_model.components_

    print('用户的主题分布：')
    print(user_dis)
    print('电影的主题分布：')
    print(item_dis)
    return item, user, user_dis, item_dis, RATE_MATRIX


def get_item_dis(item, item_dis):
    plt.plot(item_dis[:, 0], item_dis[:, 1], 'ro')
    plt.draw()  # 直接画出矩阵，只打了点，下面对图plt1进行一些设置

    plt.xlim((-1, 3))
    plt.ylim((-1, 3))
    plt.title(u'the distribution of items (NMF)')  # 设置图的标题

    # count = 1
    zipitem = zip(item, item_dis)  # 把电影标题和电影的坐标联系在一起

    for item in zipitem:
        item_name = item[0]
        data = item[1]
        plt.text(data[0], data[1], item_name,
                horizontalalignment='center',
                verticalalignment='top')

    plt.show()


def get_user_dis(user, user_dis):
    user_dis = user_dis.T  # 转置用户分布矩阵
    plt.plot(user_dis[:, 0], user_dis[:, 1], 'ro')
    plt.xlim((-1, 3))
    plt.ylim((-1, 3))
    plt.title(u'the distribution of user (NMF)')

    zipuser = zip(user, user_dis)  # 把电影标题和电影的坐标联系在一起
    for user in zipuser:
        user_name = user[0]
        data = user[1]
        plt.text(data[0], data[1], user_name,
                horizontalalignment='center',
                verticalalignment='top')

    plt.show()


def recomd_item(RATE_MATRIX, item_dis, user_dis, rec_user):
    filter_matrix = RATE_MATRIX < 1e-8
    rec_mat = np.dot(item_dis, user_dis)
    print('重建矩阵，并过滤掉已经评分的物品：')
    rec_filter_mat = (filter_matrix * rec_mat).T
    print(rec_filter_mat)

    rec_userid = user.index(rec_user)  # 推荐用户ID
    rec_list = rec_filter_mat[rec_userid, :]  # 推荐用户的电影列表

    print('推荐用户[%s]的电影：' % rec_user)
    print(np.nonzero(rec_list))


if __name__ == '__main__':
    item, user, user_dis, item_dis, RATE_MATRIX = get_datas(n_components=4)
    get_item_dis(item, item_dis)
    get_user_dis(user, user_dis)
    recomd_item(RATE_MATRIX, item_dis, user_dis, rec_user='凛冬将至')
