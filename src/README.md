## 一、数据处理

实例数据在`data`文件夹里，是一个`excel`表格，`CSV`格式。

具体是`43`个患者，连续`30`天的白细胞计数的测量结果。所以共`1290`行数据。

数据样式是：

| num  | days | WBC  |
| ---- | ---- | ---- |
| ---- | ---- | ---- |

分别代表：患者编号，哪天，以及当天测量结果。

`python`代码在文件夹`src`。为了便于调试，同时提供 `python` 脚本 和 `jupyterlab ipynb` 文件。

算法原理见`doc/chapter01.md`和`doc/chapter02.md`，具体示例的实现说明见`doc/chapter03.md`。

## 二、算法

`DTW`距离代码在脚本源文件中是：

```python
# `DTW`距离，时间复杂度为两个时间序列长度相乘
def DTWDistance(s1, s2):
```

但是这个计算全量代价太大，时间复杂度比较高，于是进行优化，只检测前`W`个窗口的值。
优化后的`DTW`距离在文件中是：

```python
# DTW_W距离, 优化后的算法，只检测前W个窗口的值
def DTWDistance_W(s1, s2, w):
```

之后的计算就用优化后的`DTWDistance_W()`函数。


\<END>
