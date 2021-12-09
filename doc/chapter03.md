## k-means-And-DTW原理及算法介绍 Chapter02

[toc]



### 四、结果

定义了分成两类的情形，可以根据num_clust 的值进行灵活的调整，等于2是的分类和图示情况如下：

WBC01：[6774, 7193, 8070, 8108, 8195, 2020006799, 2020007003, 2020007251, 2020007420, 2020007636, 2020007718, 2020007928, 2020007934, 2020008022, 2020008196, 2020008239, 2020008302, 2020008354, 2020008418, 2020008513, 2020008535, 2020008737, 2020008890, 2020008909, 2020009042, 2020009043, 2020009050, 2020009201, 2020009213, 2020009289, 2020009420, 2020009557]

WBC02：[2020007250, 2020007388, 2020007389, 2020007422, 2020007625, 2020007703, 2020007927, 2020009049, 2020009158, 2020009284, 2020009580]

说明：
代码训练过程中，一定要注意数据类型，比如`matrix`和`ndarray`，虽然打印的时候都是`（45，30）`，但是再训练的时候，稍加不注意，就会导致乱七八糟的问题，需要排查好久。

本文的数据和代码，请登录：my github，进行下载。若是对您有用，请不吝给颗星。

具体请看博文：https://www.cnblogs.com/yifanrensheng/p/12501238.html