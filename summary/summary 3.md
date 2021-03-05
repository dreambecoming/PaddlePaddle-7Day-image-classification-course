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
1. 表达式

当样本数为$n$，特征数为$k$时，线性回归模型的表达式为：

$\mathbf{\hat{y}}=\mathbf{Xw}+b$
   
其中，线性回归模型的输出形状大小为$\mathbf{\hat{y}}\in \mathbb{R}^{n \times 1}$，样本特征形状大小为$\mathbf{X}\in \mathbb{R}^{n \times k}$，权重形状大小为$\mathbf{w}\in \mathbb{R}^{k \times 1}$，偏置项为$b\in\mathbb{R}^{1}$，设模型的参数为$\mathbf{\theta}= \left [ \mathbf{w},b\right ]^{T}$，加偏置项$b \in \mathbb{R}^{1}$采用了广播机制。

为什么模型需要加截距项呢？因为如果模型不添加截距项，估计出来的模型将一定会通过原点，即在$\mathbf{x}$取值为0时，$\mathbf{y}$的估计值也是0。为了消除这个模型设定偏误，我们在模型中添加截距项，这使得模型估计既有可能通过原点，也有可能不通过原点，提升了模型的适用范围。

2. 均方损失函数

$L$可以看成是以$\mathbf{w},b$为参数的函数。二次函数求导会使得项式前乘以2，为了使得导数表达式更简洁，通常会在二次误差项前面乘以1/2。

单个样本$i$的损失： $L^{i}\left ( \mathbf{w},\mathbf{b}\right )=\frac{1}{2}\left ( \hat{y}^{i}-y^{i}\right )^{2}$

全部样本损失：$L \left ( \mathbf{w},\mathbf{b}\right )=\frac{1}{2n}\left ( \hat{y}^{i}-y^{i}\right )^{2}$

在模型训练过程中，我们最终的目标是要找到一组参数$\left ( \mathbf{w^{*}},\mathbf{b^{*}}\right )$，使得训练数据集上的全部样本损失尽可能小。

$\mathbf{w^{*}},\mathbf{b^{*}} = argmin L \left ( \mathbf{w},\mathbf{b} \right )$

损失函数又可以写成矢量表达式：

$L \left ( \mathbf{\theta} \right )=\frac{1}{2n} \left ( \mathbf{\hat{y}}-\mathbf{y} \right )^{T}\left ( \mathbf{\hat{y}}-\mathbf{y} \right )$

其中，线性回归模型的输出形状大小为$\mathbf{\hat{y}}\in \mathbb{R}^{n \times 1}$，样本标签的形状大小为$\mathbf{y}\in \mathbb{R}^{n \times 1}$，样本数量为$n$.

3. 解析优化与随机梯度下降

* 解析方法，它适用于损失函数形式较为简单的场景。

$L \left ( \theta \right )=\frac{1}{2n} \left ( \mathbf{\hat{y}}-\mathbf{y} \right )^{T}\left ( \mathbf{\hat{y}}-\mathbf{y} \right ) = \frac{1}{2n} \left ( \mathbf{Xw}+b-\mathbf{y} \right )^{T}\left ( \mathbf{Xw}+b-\mathbf{y} \right )$

在样本特征矩阵$\mathbf{X}$上增加一个全为1的列，于是新的样本特征矩阵为$\tilde{X} \in \mathbb{R}^{n \times (k+1)}$，上式可以简化为：

$L \left ( \theta \right )=\frac{1}{2n} \left ( \mathbf{\tilde{X} \theta}-\mathbf{y} \right )^{T}\left ( \mathbf{\tilde{X} \theta}-\mathbf{y} \right )$

其中，$\mathbf{\theta} =\left [ \mathbf{w},\mathbf{b} \right ]^{T}$.

将上式对$\mathbf{\theta}$求梯度，得到：

$\bigtriangledown _{\mathbf{\theta}}L(\mathbf{\theta})=\frac{1}{n}\mathbf{\tilde{X}}^{T}(\mathbf{\tilde{X} \theta}-\mathbf{y})$

令上式为零，解出$\mathbf{\theta}$为：

$\hat{\mathbf{\theta} }=\left ( \mathbf{\tilde{X}}^{T}\mathbf{\tilde{X}}\right )^{-1}\mathbf{\tilde{X}}^{T}\mathbf{y}$

* 梯度下降

通常有4个步骤：一是选择模型参数值。如果是首轮迭代，可以采用随机方式选取初始值；如果是非首轮迭代，可以选择上一轮迭代更新的参数值。二是在训练数据集中选取一批样本组成小批量集合（Batch Set），小批量中的样本个数通常是固定的，用$m$来代表小批量中样本的个数。三是把模型参数初始值与小批量中的样本数据，都代入模型，$n$替换成$m$，得到损失函数值。损失函数以$(\mathbf{w}, b)$为参数，把损失函数分别对$(\mathbf{w}, b)$参数求偏导数。四是用求出的三个偏导数与预先设定的一个正数（学习率）相乘作为本轮迭代中的减少量。

$w_{1}\leftarrow w_{1}-\frac{\lambda }{m}\sum_{i=1}^{m}\frac{\partial L^{i}(w_{1},w_{2},b)}{\partial w_{1}}$

$w_{2}\leftarrow w_{2}-\frac{\lambda }{m}\sum_{i=1}^{m}\frac{\partial L^{i}(w_{1},w_{2},b)}{\partial w_{2}}$

$b\leftarrow b-\frac{\lambda }{m}\sum_{i=1}^{m}\frac{\partial L^{i}(w_{1},w_{2},b)}{\partial b}$

其中，批量大小$m$和学习率$\lambda$是超参数，并不是通过模型训练得出，需要根据经验来提前设定。
上述过程的矢量计算表达式为：

$\theta \leftarrow \theta-\frac{\lambda }{m}\sum_{i=1}^{m}\bigtriangledown _{\mathbf{\theta}}L^{i}(\mathbf{\theta})$

模型训练完成后，将得到参数$\mathbf{\hat{\theta}}$，$\mathbf{\hat{\theta}}$参数可以看成真实${\mathbf{\theta}}$的最佳估计。接下来把$\mathbf{\hat{\theta}}$参数代入线性回归模型，待预测样本的输入特征分别乘以回归系数（权重）后加和即可得到输出，该输出便是预测值。

* 截距项的功能等将来碰上了应用再研究。
* 其中数学推导，将来会进行手工复现的。

### SoftMax分类器
[SoftMax分类器](https://aistudio.baidu.com/aistudio/projectdetail/1323298)

SoftMax分类器是根据输入特征来对离散型输出概率做出预测的模型，适用于多分类预测问题。

1. 模型表达式

以手写数字分类为例构建一个分类器，根据手写数字的光学识别图片来预测这张图片所包含的数字。每个图片样本的宽与高均为28像素，28乘以28得到784，可以把这784个像素看作为输入特征，用$x_{1},x_{2},...,x_{784}$表示。每个样本的标签值是0~9十个数字中的一个。与线性回归模型不同的是，SoftMax分类器最终输出的是样本在每个类别上的概率，用$y_{1},y_{2},...,y_{10}$表示。
模型共有784个输入特征和10个输出类别，每个输入特征与输出类别全连接，所以权重包含7840个标量。这些权重可看作为行数为784、列数为10的矩阵$\mathbf{w}$。偏置项可看作为行数为1、列数为10的矩阵$\mathbf{b}$。

$\mathbf{w} \in \mathbb{R}^{784 \times 10}$

$\mathbf{b} \in \mathbb{R}^{1 \times 10}$

对于第$i$个样本$\mathbf{x}^{i} \in \mathbb{R}^{1 \times 784}$，输入特征与权重的线性加权求和表示为：

$\mathbf{o}^{i}=\mathbf{x}^{i} \mathbf{w} + \mathbf{b}$

$\mathbf{o}^{i}=\left [ o_{1}^{i}, o_{2}^{i},..., o_{10}^{i}\right ]$

其中，$\mathbf{o} \in \mathbb{R}^{n \times 10}$

接下来，进行SoftMax运算，把经过线性加权后的输出值转变成概率分布。

$softmax(o_{q})=\frac{exp(o_{q})}{\sum_{q=1}^{10}exp(o_{q})}$

其中，$q$为最终输出的类别个数，在手写数字分类任务中，类别数为10。从上式中可看出，每个样本的预测值在十个类别（数字0~9）概率之和都等于1.

2. 信息量

信息论创始人Shannon在“通讯的数学理论”一文中指出“信息是用来消除随机不确定性的东西”。信息量衡量的是某个具体事件发生所带来的信息，信息量大小就是这个信息消除不确定性的程度。

3. 信息熵

信息熵是所有可能发生事件所带来的信息量的期望。信息熵越大，代表事物越具不确定性。设$X$是一个离散型随机变量，类别个数为$q$，信息熵表示为：

$E(\mathbf{X})=-\sum_{i=1}^{q}P(x_{i})ln(P(x_{i})), \mathbf{X}=x_{1},x_{2},...,x_{q}$

4. 交叉熵

交叉熵主要用于衡量估计值与真实值之间的差距。

$E(y^{i},\hat{y}^{i})=-\sum_{j=1}^{q}y_{j}^{i}ln(\hat{y}_{j}^{i})$

其中，$y^{i} \in \mathbb{R}^{q}$为真实值，$y_{j}^{i}$是$\mathbf{y}^{i}$中的元素（取值为0或1），$j=1,...,q$。$\hat{y}^{i} \in \mathbb{R}^{q}$是预测值，是模型预测在各类别上的概率分布。

```python
import paddle

train_dataset=paddle.vision.datasets.MNIST(mode="train", backend="cv2") #训练数据集
test_dataset=paddle.vision.datasets.MNIST(mode="test", backend="cv2") #测试数据集

linear=paddle.nn.Sequential(
        paddle.nn.Flatten(),#将[1,28,28]形状的图片数据改变形状为[1,784]
        paddle.nn.Linear(784,10)
        )
#利用paddlepaddle2的高阶功能，可以大幅减少训练和测试的代码量
model=paddle.Model(linear)
model.prepare(paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters()),
              paddle.nn.CrossEntropyLoss(), #交叉熵损失函数。线性模型+该损失函数，即softmax分类器。
              paddle.metric.Accuracy(topk=(1,2)))
model.fit(train_dataset, epochs=2, batch_size=64, verbose=1)

model.evaluate(test_dataset,batch_size=64,verbose=1)
```

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

