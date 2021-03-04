# 百度飞桨领航团零基础图像分类速成营 课程总结2
![paddlepaddle](https://paddlepaddle-org-cn.cdn.bcebos.com/paddle-site-front/favicon-128.png  "百度logo")

[课程链接](https://aistudio.baidu.com/aistudio/course/introduce/11939)	`https://aistudio.baidu.com/aistudio/course/introduce/11939`  
[飞桨官网](https://www.paddlepaddle.org.cn/)	`https://www.paddlepaddle.org.cn/`   
[推荐学习网站](https://www.runoob.com/python3/python3-tutorial.html)	`https://www.runoob.com/python3/python3-tutorial.html`  

****
## 目录
* [图像处理的概念与基本操作](#图像处理的概念与基本操作)
* [OpenCV库进阶操作](#OpenCV库进阶操作)
* [图像分类任务概念导入](#图像分类任务概念导入)
* [PaddleClas数据增强代码解析](#PaddleClas数据增强代码解析)
* [作业：Python编程基础](#作业Python编程基础)


# 课节2：图像处理入门基础（一）

## 图像处理的概念与基本操作
```python
 # 引入依赖包
%matplotlib inline
import numpy as np
import cv2
import matplotlib.pyplot as plt
import paddle
from PIL import Image
```
```python
# 引入依赖包
%matplotlib inline
import numpy as np
import cv2
import matplotlib.pyplot as plt
import paddle
from PIL import Image
```

```python
# 加载一张手写数字的灰度图片
# 从Paddle2.0内置数据集中加载手写数字数据集，本文第3章会进一步说明
from paddle.vision.datasets import MNIST
# 选择测试集
mnist = MNIST(mode='test')
# 遍历手写数字的测试集
for i in range(len(mnist)):
    # 取出第一张图片
    if i == 0:
        sample = mnist[i]
        # 打印第一张图片的形状和标签
        print('图片的形状',sample[0].size, '图片的标签',sample[1])
        print('MNIST数据集第一条：',mnist[0])
        print('MNIST数据集第一条第一项：',sample[0],'\n','MNIST数据集第一条第二项：',sample[1])

```
输出：
```python
图片的形状 (28, 28) 图片的标签 [7]
MNIST数据集第一条： (<PIL.Image.Image image mode=L size=28x28 at 0x7F1676CBBA50>, array([7]))
MNIST数据集第一条第一项： <PIL.Image.Image image mode=L size=28x28 at 0x7F1676C6C890> 
 MNIST数据集第一条第二项： [7]
```
```python
# 查看测试集第一个数字
plt.imshow(mnist[0][0])
print('手写数字是：', mnist[0][1])
```
灰度值与光学三原色（RGB）：红、绿、蓝(靛蓝)。光学三原色混合后，组成像素点的显示颜色，三原色同时相加为白色，白色属于无色系（黑白灰）中的一种。
```python
img = Image.open('lena.jpg')

# 使用PIL分离颜色通道
r,g,b = img.split()

# 将图片转为矩阵表示
np.array(img)

# 获取第一个通道转的灰度图
img.getchannel(0)

# 获取第二个通道转的灰度图
img.getchannel(1)

# 获取第三个通道转的灰度图
img.getchannel(2)

# 将矩阵保存成文本，数字格式为整数
np.savetxt('lena-r.txt', r, fmt='%4d')
np.savetxt('lena-g.txt', g, fmt='%4d')
np.savetxt('lena-b.txt', b, fmt='%4d')

```

## OpenCV库进阶操作

## 图像分类任务概念导入

## PaddleClas数据增强代码解析

## 作业一：

