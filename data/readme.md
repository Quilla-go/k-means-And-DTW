`DTW`距离代码在脚本源文件中是：

```python
# `DTW`距离，时间复杂度为两个时间序列长度相乘
def DTWDistance(s1, s2):
```

但是这个计算全量代价太大，时间复杂度比较高，于是进行优化，只检测前W个窗口的值。优化后的`DTW`距离在文件中是：

```python
# DTW_W距离, 优化后的算法，只检测前W个窗口的值
def DTWDistance_W(s1, s2, w):
```

之后的计算就用优化后的``DTW`Distance_W()`函数。