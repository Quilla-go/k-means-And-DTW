# K-Means算法、非负矩阵分解(NMF)与图像压缩

[原文地址](http://sofasofa.io/tutorials/image_compression/index.php)

`K-Means`算法是最基础的聚类算法、也是最常用的机器学习算法之一。 本教程中，我们利用`K-Means`对图像中的像素点进行聚类，然后用每个像素所在的簇的中心点来代替每个像素的真实值，从而达到图像压缩的目的。

非负矩阵分解(Non-negative Matrix Factorization, NMF)是一种对非负矩阵进行低维近似逼近的常见方法，同样也能达到图像压缩的目的。

预计学习用时：30分钟。

本教程基于`Python 3.5`。

原创者：SofaSofa TeamM | 修改校对：SofaSofa TeamC |

## 0. 前言

`K Means`算法比`NMF`算法慢很多，尤其是当聚类数较大时，所以实验时请耐心等待。此外，由于两者重建图像的原理不同，所以两者的视觉也相差很大，`k Means`牺牲了颜色的个数而保留了边界和形状，而`NMF`牺牲了形状以及边界却尽量保留颜色。整个实验过程中会产生一些有趣风格的图像，注意留意哦！

完整的代码在第4节。

## 1. 图像的读取
本教程以台湾省台北市最高建筑“台北101”大厦的夜景图为例，该图的分辨率为600 * 800。

点击[这里](http://sofasofa.io/tutorials/image_compression/taibei101.jpg)下载实验图片。

我们需要`skimage`、`numpy`、`matplotlib.pyplot`这三个库来实现图像的读取以及显示。

```python
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
```

利用`io.imread`直接读取图像文件，并存入`np.ndarray`类型的变量d。注意：为了图像在`plt`中无色差地显示，请一定要将`d`中的元素转成0到1的浮点数。

```python
d = io.imread('taibei101.jpg')
d = np.array(d, dtype=np.float64) / 255
```

`d`是三维`array`，第一个维度代表行数，第二个维度代表列数，第三个维度代表该图像是`RGB`图像，每个像素点由三个数值组成。

```
d.shape
(600, 800, 3)
```

调用`plt.imshow`函数便可以显示图像。

```python
plt.axis('off')
plt.imshow(d); plt.show()
```

![](http://sofasofa.io/tutorials/image_compression/taibei101_fake.jpg)

## 2. K Means 压缩图像

在我们的例子当中，每个像素点都是一个数据；每个数据拥有三个特征，分别代表R(红色)，G(绿色)，B(蓝色)。 `K Means`是一种常见的聚类方法，其思想就是让“距离”近的数据点分在一类。这里的“距离”就是指两个像素点`RBG`差值的平方和的根号。


$$Dist(P1,P2)=∥P1−P2∥_2$$

`K Means`压缩图像的原理是，用每个聚类(cluster)的中心点(center)来代替聚类中所有像素原本的颜色，所以压缩后的图像只保留了K个颜色。

假如这个`600×800`的图像中每个像素点的颜色都不一样，那么我们需要
`800×600×3=1440000`
个数来表示这个图像。对于`K Means`压缩之后的图像，我们只需要
`800×600×1+K×3`
个数来表示。`800×600×1`是因为每个像素点需要用一个数来表示其归属的簇，`K×3`是因为我们需要记录K个中心点的`RGB`数值。所以经过`K Means`压缩后，我们只需要三分之一左右的数就可以表示图像。

下面的函数`KMeansImage(d, n_colors)`就可以用来生成`n_colors`个颜色构成的图像。

```python
from sklearn.cluster import KMeans
def KMeansImage(d, n_colors):
    w, h, c = d.shape
    dd = np.reshape(d, (w * h, c))
    km = KMeans(n_clusters=n_colors)
    km.fit(dd)
    labels = km.predict(dd)
    centers = km.cluster_centers_
    new_img = d.copy()
    for i in range(w):
        for j in range(h):
            ij = i * h + j
            new_img[i][j] = centers[labels[ij]]
    return {'new_image': new_img, 'center_colors': centers}
```

运行以上函数，我们可以看看在不同的`K`的取值之下，图像压缩的效果。

```python
plt.figure(figsize=(12, 9))
plt.imshow(d); plt.axis('off')
plt.show()
for i in [2, 3, 5, 10, 30]:
    print('Number of clusters:', i)
    out = KMeansImage(d, i)
    centers, new_image = out['center_colors'], out['new_image']
    plt.figure(figsize=(12, 1))
    plt.imshow([centers]); plt.axis('off')
    plt.show()
    
    plt.figure(figsize=(12, 9))
    plt.imshow(new_image); plt.axis('off')
    plt.show()
```

`Number of clusters: 2`

![](http://sofasofa.io/tutorials/image_compression/km_cb2.png)

![](http://sofasofa.io/tutorials/image_compression/km_2.png)


`Number of clusters: 3`

![](http://sofasofa.io/tutorials/image_compression/km_cb3.png)
![](http://sofasofa.io/tutorials/image_compression/km_3.png)



Number of clusters: 5

![](http://sofasofa.io/tutorials/image_compression/km_cb5.png)
![](http://sofasofa.io/tutorials/image_compression/km_5.png)



Number of clusters: 10

![](http://sofasofa.io/tutorials/image_compression/km_cb10.png)
![](http://sofasofa.io/tutorials/image_compression/km_10.png)



Number of clusters: 30

![](http://sofasofa.io/tutorials/image_compression/km_cb30.png)
![](http://sofasofa.io/tutorials/image_compression/km_30.png)

3. NMF压缩图像
非负矩阵分解(NMF)是一个常用的对矩阵进行填充、平滑的算法。一个著名的案例就是`Netflix`利用`NMF`来填充“用户-电影”矩阵。这里，我们将对三个颜色逐一进行矩阵分解。

$$d_{Red}=P_{Red}Q_{Red}$$

这里$$d_{Red}$$是`600×800`的矩阵，$$P_{Red}$$是`600×K`的非负矩阵，$$Q_{Red}$$是K×800的非负矩阵，其中K的矩阵分解中成分的个数。通常成分个数越多，拟合效果越好。

假如这个600×800的图像中每个像素点的颜色都不一样，那么我们需要
`800×600×3=1440000`
个数来表示这个图像。

对于`NMF`压缩之后的图像，我们只需要
`600×K×3+K×800×3`
个数来表示。在`K`较小时，比如100以内，`NMF`的压缩效率明显优于`K Means`。

下面，我们就来看看NMF压缩后的图像的视觉效果。

下面的函数`NMFImage(d, num_components)`就可以用来生成根据`NMF`压缩得到的图像。

```python
from sklearn.decomposition import NMF
def NMFImage(d, num_components):
    w, h, c = d.shape
    new_img = d.copy()
    for i in range(c):
        nmf = NMF(n_components=num_components)
        P = nmf.fit_transform(d[:, :, i])
        Q = nmf.components_
        new_img[:, : ,i] = np.clip(P @ Q, 0, 1)
    return {'new_image': new_img}
```

运行以上函数，我们可以看看在不同的成分个数取值之下，图像压缩的效果。

```python
plt.figure(figsize=(12, 9))
plt.imshow(d); plt.axis('off')
plt.show()
for i in [1, 2, 3]:
    print('Number of components:', i)
    out = NMFImage(d, i)
    new_image = out['new_image']
    plt.figure(figsize=(12, 9))
    plt.imshow(new_image); plt.axis('off')
    plt.show()
```

Number of components: 1

![]()
![](http://sofasofa.io/tutorials/image_compression/nmf_1.jpg)

Number of components: 2

![]()
![](http://sofasofa.io/tutorials/image_compression/nmf_2.jpg)

Number of components: 3

![]()
![](http://sofasofa.io/tutorials/image_compression/nmf_3.jpg)

Number of components: 5

![]()
![](http://sofasofa.io/tutorials/image_compression/nmf_5.jpg)

Number of components: 10

![]()
![](http://sofasofa.io/tutorials/image_compression/nmf_10.jpg)

Number of components: 20

![]()
![](http://sofasofa.io/tutorials/image_compression/nmf_20.jpg)

Number of components: 80

![]()
![](http://sofasofa.io/tutorials/image_compression/nmf_80.jpg)

Number of components: 150

![]()
![](http://sofasofa.io/tutorials/image_compression/nmf_150.jpg)

4. 完整代码
点击这里下载实验图片。

```python
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF

# 利用K Means进行压缩
def KMeansImage(d, n_colors):
    w, h, c = d.shape
    dd = np.reshape(d, (w * h, c))
    km = KMeans(n_clusters=n_colors)
    km.fit(dd)
    labels = km.predict(dd)
    centers = km.cluster_centers_
    new_img = d.copy()
    for i in range(w):
        for j in range(h):
            ij = i * h + j
            new_img[i][j] = centers[labels[ij]]
    return {'new_image': new_img, 'center_colors': centers}

# 利用NMF进行压缩
def NMFImage(d, n_components):
    w, h, c = d.shape
    new_img = d.copy()
    for i in range(c):
        nmf = NMF(n_components=n_components)
        P = nmf.fit_transform(d[:, :, i])
        Q = nmf.components_
        new_img[:, : ,i] = np.clip(P @ Q, 0, 1)
    return {'new_image': new_img}

# 查看K Means在不同聚类个数下的视觉效果
plt.figure(figsize=(12, 9))
plt.imshow(d); plt.axis('off')
plt.show()
for i in [2, 3, 5, 10, 20, 30]:
    print('Number of clusters:', i)
    out = KMeansImage(d, i)
    centers, new_image = out['center_colors'], out['new_image']
    plt.figure(figsize=(12, 1))
    plt.imshow([centers]); plt.axis('off')
    plt.show()

    plt.figure(figsize=(12, 9))
    plt.imshow(new_image); plt.axis('off')
    plt.show()

# 查看NMF在不同成分个数下的视觉效果
plt.figure(figsize=(12, 9))
plt.imshow(d); plt.axis('off')
plt.show()
for i in [1, 2, 3, 5, 10, 20, 30, 50, 80, 150]:
    print('Number of components:', i)
    out = NMFImage(d, i)
    new_image = out['new_image']
    plt.figure(figsize=(12, 9))
    plt.imshow(new_image); plt.axis('off')
    plt.show()
```

\<END>
