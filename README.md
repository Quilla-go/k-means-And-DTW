# 项目介绍

![Alt](https://repobeats.axiom.co/api/embed/1b81d2577f36a09f53a1f9216390ff64eeed8116.svg "Repobeats analytics image")



## 涉及的问题

1、当面对带有时间序列的多组数据进行聚类问题时。

2、首先尝试：提取时间序列的统计学特征值，例如最大值，最小值等。

然后利目前常用的算法根据提取的特征进行分类，例如`Naive Bayes`, `SVMs`，`KNN` 等。

但是效果并不好。

3、然后可以尝试基于`K-means`的无监督形式分类。

这种分类方式基于两个数据的距离进行分类。

需要先定义好`距离`的概念，因为是时间序列数据，考虑使用动态时间规整（Dynamic Time Warping，`DTW`）。

这部分介绍见文件夹`doc`里的原理说明文档。

本代码库，主要讲解聚类算法，包括`kmean`、 `DTW`、 `NMF`、 `infomap`等算法。

\<END>
