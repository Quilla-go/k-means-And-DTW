import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF


# 利用K Means进行压缩
def KMeansImage(d, n_colors):
    w, h, c = d.shape
    new_img = d.copy()

    dd = np.reshape(d, (w * h, c))
    km = KMeans(n_clusters=n_colors)
    km.fit(dd)
    labels = km.predict(dd)
    centers = km.cluster_centers_

    for i in range(w):
        for j in range(h):
            ij = i * h + j
            new_img[i][j] = centers[labels[ij]]
    return {'new_image': new_img, 'center_colors': centers}


# 利用NMF进行压缩
def NMFImage(d, n_components):
    _, _, c = d.shape
    new_img = d.copy()
    for i in range(c):
        nmf = NMF(n_components=n_components)
        P = nmf.fit_transform(d[:, :, i])
        Q = nmf.components_
        new_img[:, :, i] = np.clip(P @ Q, 0, 1)
    return {'new_image': new_img}


# 查看K Means在不同聚类个数下的视觉效果
def drawkmeans():
    d = io.imread('taibei101.jpg')
    d = np.array(d, dtype=np.float64) / 255
    # for i in [2, 3, 5, 10, 20, 30]:
    for i in [2, 3]:
        print('Number of clusters:', i)
        out = KMeansImage(d, i)
        centers, new_image = out['center_colors'], out['new_image']
        plt.figure(figsize=(12, 3))
        plt.imshow([centers])
        plt.axis('off')
        plt.title(u"Number of clusters: %d" % i)
        plt.show()

        plt.figure(figsize=(12, 9))
        plt.imshow(new_image)
        plt.axis('off')
        plt.title(u"Number of clusters: %d" % i)
        plt.show()


# 查看NMF在不同成分个数下的视觉效果
def drawNMF():
    d = io.imread('taibei101.jpg')
    d = np.array(d, dtype=np.float64) / 255
    for i in [1, 2, 3, 5, 10, 20, 30, 50, 80, 150]:
        print('Number of components:', i)
        out = NMFImage(d, i)
        new_image = out['new_image']
        plt.figure(figsize=(12, 9))
        plt.imshow(new_image)
        plt.axis('off')
        plt.show()


def main():
    d = io.imread('taibei101.jpg')
    # d = np.array(d, dtype=np.float64) / 255
    # d是三维array，第一个维度代表行数，
    # 第二个维度代表列数，
    # 第三个维度代表该图像是RGB图像，每个像素点由三个数值组成。

    print(d.shape)  # (600, 800, 3)
    print(d)

    # 调用plt.imshow函数便可以显示图像。
    # plt.figure(figsize=(12, 9))
    # plt.axis('off')
    # plt.imshow(d)
    # plt.show()


if __name__ == '__main__':
    main()
    # drawkmeans()
    # drawNMF()
