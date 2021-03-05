# 百度飞桨领航团零基础图像分类速成营 课程总结3
![paddlepaddle](https://paddlepaddle-org-cn.cdn.bcebos.com/paddle-site-front/favicon-128.png  "百度logo")

[课程链接](https://aistudio.baidu.com/aistudio/course/introduce/11939)	`https://aistudio.baidu.com/aistudio/course/introduce/11939`  
[飞桨官网](https://www.paddlepaddle.org.cn/)	`https://www.paddlepaddle.org.cn/`   
[推荐学习网站](https://www.runoob.com/python3/python3-tutorial.html)	`https://www.runoob.com/python3/python3-tutorial.html`  

****
## 目录
* [感知机](#感知机)
* [基础理论](#基础理论)
* [建模实战](#建模实战)
* [作业](#作业)
## 参考资料
* [学习目录](https://aistudio.baidu.com/aistudio/projectdetail/1354419)


# 课节3：图像分类基础
## 感知机
[感知机](https://aistudio.baidu.com/aistudio/projectdetail/1615953)

感知机（perceptron）是二分类的线性分类模型，属于监督学习算法。输入为实例的特征向量，输出为实例的类别（取+1和-1）。

感知机对应于输入空间中将实例划分为两类的分离超平面。感知机旨在求出该超平面，为求得超平面导入了基于误分类的损失函数，利用梯度下降法 对损失函数进行最优化（最优化）。

感知机的学习算法具有简单而易于实现的优点，分为原始形式和对偶形式。感知机预测是用学习得到的感知机模型对新的实例进行预测的，因此属于判别模型。

感知机由Rosenblatt于1957年提出的，是神经网络和支持向量机的基础。

### 定义

假设输入空间(特征向量)为X=[x1,x2,x3,x......]，输出空间为Y = [1,-1]。

输入 = X

表示实例的特征向量，对应于输入空间的点；

输出 = Y

表示示例的类别。

由输入空间到输出空间的函数为

$f(x)=sign(w^Tx+b)$

称为感知机。其中，参数w叫做权值向量(weight)，b称为偏置(bias)。表示$w^T$和x的点积

$\mathbf{w}^{T} \mathbf{x} =w_1*x_1+w_2*x_2+w_3*x_3+...+w_n*x_n$ 
，$\mathbf{w} = [w_1, w_2,...,w_n]^{T}$
，$\mathbf{x} = [x_1, x_2,...,x_n]^{T}$

sign为符号函数，即

$sign(A)=\left\{\begin{matrix}+1，A \geq 0\\-1，A<0\end{matrix}\right.$

感知机算法就是要找到一个超平面将我们的数据分为两部分。

超平面就是维度比我们当前维度空间小一个维度的空间
```python
#引入必要的包
import paddle
print("本教程使用的paddle版本为：" + paddle.__version__)
import numpy as np
import matplotlib.pyplot as plt
```
```python
# 数据准备
# 机数种子，每次生成的随机数相同
np.random.seed(0)
num=100

#生成数据集x1,x2,y0/1
#随机生成100个x1
# numpy.random.normal(loc=0.0, scale=1.0, size=None) normal正态分布，loc均值、scale标准差、size输出大小
x1=np.random.normal(6,1,size=(num))
#随机生成100个x2
x2=np.random.normal(3,1,size=(num))
#生成100个y
y=np.ones(num)
#将生成好的点放入到一个分类中
# class1.shape:(3, 100)
class1=np.array([x1,x2,y])

#接下来生成第二类点，原理跟第一类一样
x1=np.random.normal(3,1,size=(num))
x2=np.random.normal(6,1,size=(num))
y=np.ones(num)*(-1)
class2=np.array([x1,x2,y])

# 转置成(100, 3)
class1=class1.T
class2=class2.T

#将两类数据都放到一个变量里面，(200, 3)
all_data = np.concatenate((class1,class2))

#将数据打乱
np.random.shuffle(all_data)

#截取出坐标数据
train_data_x=all_data[:150,:2]
#截取出标签数据
train_data_y=all_data[:150,-1].reshape(150,1)

#将数据转化为tensor形式
x_data = paddle.to_tensor(train_data_x.astype('float32'))
y_data = paddle.to_tensor(train_data_y.astype('float32'))
```
```python
# 模型配置
linear = paddle.nn.Linear(in_features=2, out_features=1)
mse_loss = paddle.nn.MSELoss()
sgd_optimizer = paddle.optimizer.SGD(learning_rate=0.001, parameters = linear.parameters())

# 模型训练
total_epoch = 50000
#构建训练过程
for i in range(total_epoch):
    
    y_predict = linear(x_data)
    #获取loss
    loss = mse_loss(y_predict, y_data)
    #反向传播
    loss.backward()
    sgd_optimizer.step()
    sgd_optimizer.clear_grad()
    #w1
    w1_after_opt = linear.weight.numpy()[0].item()
    #w2
    w2_after_opt = linear.weight.numpy()[1].item()
    #b
    b_after_opt = linear.bias.numpy().item()
    #每1000次输出一次数据
    if i%1000 == 0:
        print("epoch {} loss {}".format(i, loss.numpy()))
        print("w1 after optimize: {}".format(w1_after_opt))
        print("w2 after optimize: {}".format(w2_after_opt))
        print("b after optimize: {}".format(b_after_opt))
print("finished training， loss {}".format(loss.numpy()))
```
```python
x=np.arange(10)

# 画线的公式（公式的推导课件没说，等到将来碰到了仔细研究吧，个人认为下面y1才是对的）
y1=-(w1_after_opt * x + b_after_opt) / w2_after_opt
y=-(w1_after_opt/w2_after_opt) *x + b_after_opt

plt.subplot(121);plt.plot(x,y);plt.title('in the reference');plt.scatter(class1[:,0],class1[:,1]);plt.scatter(class2[:,0],class2[:,1],marker='*')
plt.subplot(122);plt.plot(x,y1);plt.title('personnally think right');plt.scatter(class1[:,0],class1[:,1]);plt.scatter(class2[:,0],class2[:,1],marker='*')
```

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


```python

```
```python

```

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

补全代码，参考课件，直接补全multilayer_perceptron类。
```python
import paddle
import paddle.nn.functional as F
from paddle.vision.transforms import ToTensor
```

```python
# 定义多层感知器 
#动态图定义多层感知器
class multilayer_perceptron(paddle.nn.Layer):
    def __init__(self):
        super(multilayer_perceptron,self).__init__()
        #请在这里补全网络代码
        self.flatten=paddle.nn.Flatten()
        self.hidden=paddle.nn.Linear(in_features=784,out_features=128)
        self.output=paddle.nn.Linear(in_features=128,out_features=10)

    def forward(self, x):
        #请在这里补全传播过程的代码
        x=self.flatten(x)
        x=self.hidden(x) #经过隐藏层
        x=F.relu(x) #经过激活层
        x=self.output(x)
        return x
    
model=paddle.Model(multilayer_perceptron())

model.prepare(paddle.optimizer.Adam(parameters=model.parameters()),
              paddle.nn.CrossEntropyLoss(),
              paddle.metric.Accuracy())

model.fit(train_dataset,
          epochs=5,
          batch_size=64,
          verbose=1)

model.evaluate(test_dataset,verbose=1)
```
运行出结果：{'loss': [0.0003590036], 'acc': 0.9684}

接下来完成作业要求2，自己定义一个模型，训练、预测、评估。

LeNet-5模型;
```python
import paddle.nn as nn

class LeNet(nn.Layer):
    """
    继承paddle.nn.Layer定义网络结构
    """

    def __init__(self, num_classes=10):
        """
        初始化函数
        """
        super(LeNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2D(in_channels=1, out_channels=6, kernel_size=3, stride=1, padding=1),  # 第一层卷积
            nn.ReLU(), # 激活函数
            nn.MaxPool2D(kernel_size=2, stride=2),  # 最大池化，下采样
            nn.Conv2D(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0), # 第二层卷积
            nn.ReLU(), # 激活函数
            nn.MaxPool2D(kernel_size=2, stride=2) # 最大池化，下采样
        )

        self.fc = nn.Sequential(
            nn.Linear(400, 120),  # 全连接
            nn.Linear(120, 84),   # 全连接
            nn.Linear(84, num_classes) # 输出层
        )

    def forward(self, inputs):
        """
        前向计算
        """
        y = self.features(inputs)
        y = paddle.flatten(y, 1)
        out = self.fc(y)

        return out
```

```python
from paddle.metric import Accuracy

# 用Model封装模型
model = paddle.Model(LeNet())   

# 定义损失函数
optim = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())

# 配置模型
model.prepare(optim,paddle.nn.CrossEntropyLoss(),paddle.metric.Accuracy())

# 训练保存并验证模型
model.fit(train_dataset,test_dataset,epochs=2,batch_size=64,save_dir='multilayer_perceptron',verbose=1)
```

```python
#获取测试集的第一个图片
test_data0, test_label_0 = test_dataset[0][0],test_dataset[0][1]
test_data0 = test_data0.reshape([28,28])
plt.figure(figsize=(2,2))
#展示测试集中的第一个图片
print(plt.imshow(test_data0, cmap=plt.cm.binary))
print('test_data0 的标签为: ' + str(test_label_0))
#模型预测
result = model.predict(test_dataset, batch_size=1)
#打印模型预测的结果
print('test_data0 预测的数值为：%d' % np.argsort(result[0][0])[0][-1])
```
预测的是数字“7”
```python
model.evaluate(test_dataset,verbose=1)
```
评估输出：{'loss': [6.807082e-05], 'acc': 0.9837}

