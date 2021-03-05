# 百度飞桨领航团零基础图像分类速成营 课程总结3
![paddlepaddle](https://paddlepaddle-org-cn.cdn.bcebos.com/paddle-site-front/favicon-128.png  "百度logo")

[课程链接](https://aistudio.baidu.com/aistudio/course/introduce/11939)	`https://aistudio.baidu.com/aistudio/course/introduce/11939`  
[飞桨官网](https://www.paddlepaddle.org.cn/)	`https://www.paddlepaddle.org.cn/`   
[推荐学习网站](https://www.runoob.com/python3/python3-tutorial.html)	`https://www.runoob.com/python3/python3-tutorial.html`  

****
## 目录
* [基础理论](#基础理论)
* [建模实战](#建模实战)
* [作业](#作业)
## 参考资料
* [学习目录](https://aistudio.baidu.com/aistudio/projectdetail/1354419)


# 课节3：图像分类基础

## 基础理论
* 图像识别面临的挑战：  语义鸿沟：图像的底层视觉特性和高层语义概念之间的鸿沟
* 图像识别基本框架：
  测量空间（表象空间，现实世界）→`特征表示 `→特征空间→`特征匹配 `→类别空间（概念空间，理念世界）
  
  
1. 建立模型：神经元、网络结构、激活函数。输入层、隐藏层、输出层。
2. 损失函数：常用损失函数：平方误差、交叉熵
3. 参数学习：梯度下降法 权重weights,偏置bias
## 建模实战
### 线性回归
[线性回归](https://aistudio.baidu.com/aistudio/projectdetail/1322247)

### SoftMax分类器
[SoftMax分类器](https://aistudio.baidu.com/aistudio/projectdetail/1323298)

### 多层感知机模型
[多层感知机模型](https://aistudio.baidu.com/aistudio/projectdetail/1323886)

### 卷积网络LeNet-5
[卷积网络LeNet-5](https://aistudio.baidu.com/aistudio/projectdetail/1329509)



## 作业
作业要求：

1. 补全网络代码，并运行手写数字识别项目。以出现最后的图片和预测结果为准。（65分）
2. 保留原本的multilayer_perceptron网络定义（自己补全完的），自己定义一个卷积网络并运行成功。以出现最后的图片和预测结果为准（45分）

        首先导入必要的包

        numpy---------->python第三方库，用于进行科学计算

        PIL------------> Python Image Library,python第三方图像处理库

        matplotlib----->python的绘图库 pyplot:matplotlib的绘图框架

        os------------->提供了丰富的方法来处理文件和目录
```python
#导入需要的包
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import paddle
print("本教程基于Paddle的版本号为："+paddle.__version__)
```
### 1.准备数据

1. 数据集介绍：MNIST数据集包含60000个训练集和10000测试数据集。分为图片和标签，图片是28*28的像素矩阵，标签为0~9共10个数字。

2. transform函数是定义了一个归一化标准化的标准

3. train_dataset和test_dataset：paddle.vision.datasets.MNIST()中的mode='train'和mode='test'分别用于获取mnist训练集和测试集
```python
#导入数据集Compose的作用是将用于数据集预处理的接口以列表的方式进行组合。
#导入数据集Normalize的作用是图像归一化处理，支持两种方式： 1. 用统一的均值和标准差值对图像的每个通道进行归一化处理； 2. 对每个通道指定不同的均值和标准差值进行归一化处理。
from paddle.vision.transforms import Compose, Normalize
transform = Compose([Normalize(mean=[127.5],std=[127.5],data_format='CHW')])
# 使用transform对数据集做归一化
print('下载并加载训练数据')
train_dataset = paddle.vision.datasets.MNIST(mode='train', transform=transform)
test_dataset = paddle.vision.datasets.MNIST(mode='test', transform=transform)
print('加载完成')
```
### 2.网络配置
以下的代码判断就是定义一个简单的多层感知器，一共有三层，两个大小为100的隐层和一个大小为10的输出层，因为MNIST数据集是手写0到9的灰度图像，类别有10个，所以最后的输出大小是10。最后输出层的激活函数是Softmax，所以最后的输出层相当于一个分类器。加上一个输入层的话，多层感知器的结构是：输入层-->>隐层-->>隐层-->>输出层。
```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```
