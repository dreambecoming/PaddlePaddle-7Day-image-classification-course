# 百度飞桨领航团零基础图像分类速成营 课程总结4
![paddlepaddle](https://paddlepaddle-org-cn.cdn.bcebos.com/paddle-site-front/favicon-128.png  "百度logo")

[课程链接](https://aistudio.baidu.com/aistudio/course/introduce/11939)	`https://aistudio.baidu.com/aistudio/course/introduce/11939`  
[飞桨官网](https://www.paddlepaddle.org.cn/)	`https://www.paddlepaddle.org.cn/`   
[推荐学习网站](https://www.runoob.com/python3/python3-tutorial.html)	`https://www.runoob.com/python3/python3-tutorial.html`  

****
## 目录
* [前置知识](#前置知识)
* [基础理论](#基础理论)
* [建模实战](#建模实战)
* [作业](#作业)
## 参考资料
* [关于LeNet的前世今生](https://www.jiqizhixin.com/graph/technologies/6c9baf12-1a32-4c53-8217-8c9f69bd011b)

* [常用网络结构](https://www.jiqizhixin.com/articles/2020-05-06-16)

* [GoogleNet解析](https://www.itread01.com/content/1544969366.html)

* [你必须要知道CNN模型：ResNet](https://zhuanlan.zhihu.com/p/31852747)

* [Alexnet论文](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)

* [GoogleNet论文](https://arxiv.org/pdf/1409.4842v1.pdf)

* [Resnet论文](https://arxiv.org/pdf/1512.03385.pdf)


# 课节4：卷积神经网络基础：蝴蝶图像识别分类
## 前置知识
### 矩阵
1. 矩阵加法运算 
	I2 = I1 + b
    
	1.1 如果I1和b维度一致时，结果为两个矩阵对应项相加。

	1.2 如果I1和b维度不一致，但是满足一定条件，亦可进行加法计算。

	假设I1的矩阵形状为shape_I1=（h,w,c）,那么在b的矩阵形状shape_b为shape_I1的某个切片相等时，I1和b矩阵可以进行“广播”相加。

2. 矩阵和数乘法运算
	
    I3 = a * I1
    
    I1各个元素分别乘以a
    
3. 矩阵乘法运算
	
	I2 = I1 * A
    
    1. 二维矩阵
	I1的r行向量与A的c列向量的点积作为I3的（r,c）元素
    
    2. 多维矩阵
    所得结果的后两维的shape与二维矩阵的计算方式一致。不同的是高维度（3位以上）的尺寸大小的计算方式-取较大的尺寸。作为多维矩阵，数据还是保持在后两维中，高维度只是数据的排列方式的定义。本质上，高维矩阵的乘法还是二维矩阵之间的乘法，再加上排列方法的保持，当高维尺寸不同时，只要可以“广播”，就依然可以计算。
    
    譬如：
    2.1 I1的shape为 (c,n,s),A的shape为（c，s，m）时，I2的shape为（c,n,m).
    结果shape的计算过程：I1的后两维shape为（n,s), A的后两维shape为（s,m),类似于二维矩阵乘法所得结果的后两维shape为（n,m), 高维度的尺寸取c。
    
    2.2 I1的shape为 (n,c,h,s),A的shape为（1,1，s，w）时，I2的shape为（n,c,h,w).
    结果shape的计算过程：I1的后两维shape为（h,s), A的后两维shape为（s,m),类似于二维矩阵乘法所得结果的后两维shape为（h,w), 高维度的尺寸取（n,c）和（1,1）中尺寸较大值（n,c）。
```python
I1 = [[j*4+i for i in range(4)] for j in range(4)]
I1 = np.array(I1,dtype='float32')
A = np.random.uniform(0,1,(4,4)) # numpy.random.uniform(low,high,size) 随机数

## 二维矩阵乘法 1
I2 = np.matmul(I1,A)
## 二维矩阵乘法 2
I3 = np.zeros_like(I2)
for i in range(4):
    for j in range(4):
        I3[i,j] = np.dot(I1[i,:],A[:,j])

value = {}
value['I1'] = I1
value['A'] = A
value['I2'] = I2
value['I3'] = I3
print("I1={I1},\nA={A},\nI2={I2}\n".format(**value))
print("I1={I1},\nA={A},\nI3={I2}\n".format(**value))
print("I2 ==I3: {}\n".format(np.allclose(I2,I3)))
```
```python
## 高维矩阵乘法
## (3,100,9) , (3,9,99)
I1 = np.random.uniform(0,1,(3,100,9))
A = np.random.uniform(0,1,(3,9,99))
I2 = np.matmul(I1,A)
value = {}
value['I1'] = I1
value['A'] = A
value['I2'] = I2
#print("I1={I1},\na={a},\nI2={I2}\n".format(**value))
print("I1 shape={},A shape={},I2 shape={}".format(I1.shape,A.shape,I2.shape))

```
```python
## 高维矩阵乘法
## (100,3,100,9) , (1,3,9,99)
I1 = np.random.uniform(0,1,(100,3,100,9))
A = np.random.uniform(0,1,(1,3,9,99))
I2 = np.matmul(I1,A)
value = {}
value['I1'] = I1
value['A'] = A
value['I2'] = I2
#print("I1={I1},\na={a},\nI2={I2}\n".format(**value))
print("I1 shape={},A shape={},I2 shape={}".format(I1.shape,A.shape,I2.shape))
```
### 算子
传统计算机视觉利用算子，以及算子集对进行图像处理，从而获得特征图像，CNN中的卷积核即是与之相对应的概念。不同是，前者需要算法开发者手动设计，而后者通过训练数据的驱动来自动选择，而且通常情况下CNN网络中的卷积核数量远超过手动设计的。

Sobel算子
![](https://ai-studio-static-online.cdn.bcebos.com/5dcab1a62a764dc783cef4a8da3706d6ec59c432868d4a45b8d9b272b8ce2791)


Laplace算子
![](https://ai-studio-static-online.cdn.bcebos.com/c6bef46b534e4af5bde59f6ed07726d1810e87e3defa44a9a20c7c59f526bb95)

### 卷积
该过程可理解为算子（核）窗口在图像上沿指定维度滑动，同时将矢量点积值作为对应输出矩阵元素的值。

###批归一化
批归一化（Batch Normalization）

### 感受野
感受野（Receptive Field）的定义是卷积神经网络每一层输出的特征图（feature map）上每个像素点在原始图像上映射的区域大小，这里的原始图像是指网络的输入图像，是经过预处理（如resize，warp，crop）后的图像。


## 基础理论
卷积神经网络（Convolutional Neural Networks, CNN）

特性：
 * 局部连接
 * 权重共享
 * 下采样

网络结构：卷积层 → 池化层 → 全连接层 → 输出层：soft-max

卷积核（kernel）：input image → kernel → feature map
	image size= wi x hi  
	kernel size= wk × hk  
	stride=1 stride为卷积窗口移动的步长
	feature map size = wf × hf , 其中 wf = (wi-wk)/stride + 1，hf =(hi-hk)/stride + 1
	
## 建模实战

* 卷积网络模型AlexNet htps://aistudio.baidu.com/aistudio/projectdetail/1332790

AlexNet模型由Hinton和Alex Krizhevsky开发，是2012年lmageNet挑战赛冠军模型。相比于LeNet模型，AlexNet的神经网络层数更多，其中包含ReLU激活层，并且在全连接层引入Dropout机制防止过拟合。
* 并联卷积神经网络-GoogleNet htps:/aistudio.baidu.com/aistudio/proiectdetai/1340883

GoogLeNet模型是由谷歌（Google）团队开发出来的卷积神经网络，它是2014年lmageNet挑战赛的冠军模型。相比于AlexNet模型，GoogLeNet模型的网络结构更深，共包括87层。尽管模型结构变得更复杂，但参数量更少了。
* 残差网络-ResNet httes:/aistudio.baidu.com/aistudio/proiectdetail/1342659

残差网络（ResNet）模型是由何凯明开发，它是2015年ImageNet ILSVRC-2015分类挑战赛的冠军模型。
ResNet模型引入残差模块，它能够有效地消除由于模型层数增加而导致的梯度弥散或梯度爆炸问题。

## 作业
作业说明：

1. 能够跑通项目得到结果，得到底分60分；
2. 通过修改模型、数据增强等方法，使得模型在测试数据集上的准确度达到85%及以上的，在底分上再得到加分30分；
3. 通过修改模型、数据增强等方法，使得模型在测试数据集上的准确度达到90%及以上的，直接得到满分100分。

项目：
### 1. 蝴蝶识别分类任务概述

人工智能技术的应用领域日趋广泛，新的智能应用层出不穷。本项目将利用人工智能技术来对蝴蝶图像进行分类，需要能对蝴蝶的类别、属性进行细粒度的识别分类。相关研究工作者能够根据采集到的蝴蝶图片，快速识别图中蝴蝶的种类。期望能够有助于提升蝴蝶识别工作的效率和精度。

#### 2. 创建项目和挂载数据

数据集都来源于网络公开数据（和鲸社区）。图片中所涉及的蝴蝶总共有9个属，20个物种，文件genus.txt中描述了9个属名，species.txt描述了20个物种名。

在创建项目时，可以为该项目挂载Butterfly20蝴蝶数据集，即便项目重启，该挂载的数据集也不会被自动清除。具体方法如下：首先采用notebook方式构建项目，项目创建框中的最下方有个数据集选项，选择“+添加数据集”。然后，弹出搜索框，在关键词栏目输入“bufferfly20”，便能够查询到该数据集。最后，选中该数据集，可以自动在项目中挂载该数据集了。

需要注意的是，每次重新打开该项目，data文件夹下除了挂载的数据集，其他文件都将被删除。

被挂载的数据集会自动出现在data目录之下，通常是压缩包的形式。在data/data63004目录，其中有两个压缩文件，分别是Butterfly20.zip和Butterfly20_test.zip。也可以利用下载功能把数据集下载到本地进行训练。

### 3. 初探蝴蝶数据集

我们看看蝴蝶图像数据长什么样子？

首先，解压缩数据。类以下几个步骤：

第一步，把当前路径转换到data目录，可以使用命令!cd data。在AI studio nootbook中可以使用Linux命令，需要在命令的最前面加上英文的感叹号(!)。用&&可以连接两个命令。用\号可以换行写代码。需要注意的是，每次重新打开该项目，data文件夹下除了挂载的数据集，其他文件都会被清空。因此，如果把数据保存在data目录中，每次重新启动项目时，都需要解压缩一下。如果想省事持久化保存，可以把数据保存在work目录下。

实际上，!加某命令的模式，等价于python中的get_ipython().system('某命令')模式。

第二步，利用unzip命令，把压缩包解压到当前路径。unzip的-q参数代表执行时不显示任何信息。unzip的-o参数代表不必先询问用户，unzip执行后覆盖原有的文件。两个参数合起来，可以写为-qo。

第三步，用rm命令可以把一些文件夹给删掉，比如，__MACOSX文件夹
```python
!cd data &&\
unzip -qo data63004/Butterfly20_test.zip &&\
unzip -qo data63004/Butterfly20.zip &&\
rm -r __MACOSX
```
### 4. 数据准备

数据准备过程包括以下两个重点步骤：

一是建立样本数据读取路径与样本标签之间的关系。

二是构造读取器与数据预处理。可以写个自定义数据读取器，它继承于PaddlePaddle2.0的dataset类，在__getitem__方法中把自定义的预处理方法加载进去。
```python
#以下代码用于建立样本数据读取路径与样本标签之间的关系
import os
import random

data_list = [] #用个列表保存每个样本的读取路径、标签

#由于属种名称本身是字符串，而输入模型的是数字。需要构造一个字典，把某个数字代表该属种名称。键是属种名称，值是整数。
label_list=[]
with open("/home/aistudio/data/species.txt") as f:
    for line in f:
        a,b = line.strip("\n").split(" ")
        label_list.append([b, int(a)-1]) # label_list=[蝴蝶类名，类编号]
label_dic = dict(label_list)

#获取Butterfly20目录下的所有子目录名称，保存进一个列表之中
class_list = os.listdir("/home/aistudio/data/Butterfly20") # 返回文件夹包含子文件名的列表
class_list.remove('.DS_Store') #删掉列表中名为.DS_Store的元素，因为.DS_Store并没有样本。

for each in class_list:
    for f in os.listdir("/home/aistudio/data/Butterfly20/"+each):# each 是 子文件夹名，蝴蝶类
        data_list.append(["/home/aistudio/data/Butterfly20/"+each+'/'+f,label_dic[each]])# f 是 子文件夹里面每张图片名，具体蝴蝶图像名

# data_list=[图片路径，类编号]
# print(data_list)
# 按文件顺序读取，可能造成很多属种图片存在序列相关，用random.shuffle方法把样本顺序彻底打乱。
random.shuffle(data_list)

#打印前十个，可以看出data_list列表中的每个元素是[样本读取路径, 样本标签]。
print(data_list[0:10])

#打印样本数量，一共有1866个样本。
print("样本数量是：{}".format(len(data_list)))
```

```python
#以下代码用于构造读取器与数据预处理
#首先需要导入相关的模块
import paddle
from paddle.vision.transforms import Compose, ColorJitter, Resize,Transpose, Normalize
import cv2
import numpy as np
from PIL import Image
from paddle.io import Dataset
import paddle.vision.transforms as T
#自定义的数据预处理函数，输入原始图像，输出处理后的图像，可以借用paddle.vision.transforms的数据处理功能
def preprocess(img):
    transform = Compose([
        Resize(size=(224, 224)), #把数据长宽像素调成224*224
        T.RandomHorizontalFlip(0.5),         #水平翻转
        T.RandomRotation(15),                #随机反转角度范围
        T.RandomVerticalFlip(0.15),          #垂直翻转
        T.RandomRotation(15),                #按指定角度范围随机旋转图像
        Normalize(mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], data_format='HWC'), #标准化
        Transpose(), #原始数据形状维度是HWC格式，经过Transpose，转换为CHW格式
        ])
    img = transform(img).astype("float32")
    return img

#自定义数据读取器
class Reader(Dataset):
    def __init__(self, data, is_val=False):
        super().__init__()
        #在初始化阶段，把数据集划分训练集和测试集。由于在读取前样本已经被打乱顺序，取20%的样本作为测试集，80%的样本作为训练集。
        self.samples = data[-int(len(data)*0.2):] if is_val else data[:-int(len(data)*0.2)]

    def __getitem__(self, idx):
        #处理图像
        img_path = self.samples[idx][0] #得到某样本的路径
        img = Image.open(img_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = preprocess(img) #数据预处理--这里仅包括简单数据预处理，没有用到数据增强

        #处理标签
        label = self.samples[idx][1] #得到某样本的标签
        label = np.array([label], dtype="int64") #把标签数据类型转成int64
        return img, label

    def __len__(self):
        #返回每个Epoch中图片数量
        return len(self.samples)

#生成训练数据集实例
train_dataset = Reader(data_list, is_val=False)

#生成测试数据集实例
eval_dataset = Reader(data_list, is_val=True)

#打印一个训练样本
#print(train_dataset[1136][0])
print(train_dataset[1136][0].shape)
print(train_dataset[1136][1])
```
### 5. 建立模型

为了提升探索速度，建议首先选用比较成熟的基础模型，看看基础模型所能够达到的准确度。之后再试试模型融合，准确度是否有提升。最后可以试试自己独创模型。

为简便，这里直接采用101层的残差网络ResNet，并且采用预训练模式。为什么要采用预训练模型呢？因为通常模型参数采用随机初始化，而预训练模型参数初始值是一个比较确定的值。这个参数初始值是经历了大量任务训练而得来的，比如用CIFAR图像识别任务来训练模型，得到的参数。虽然蝴蝶识别任务和CIFAR图像识别任务是不同的，但可能存在某些机器视觉上的共性。用预训练模型可能能够较快地得到比较好的准确度。

在PaddlePaddle2.0中，使用预训练模型只需要设定模型参数pretained=True。值得注意的是，预训练模型得出的结果类别是1000维度，要用个线性变换，把类别转化为20维度。
```python
#定义模型
class MyNet(paddle.nn.Layer):
    def __init__(self):
        super(MyNet,self).__init__()
        self.layer=paddle.vision.models.resnet101(pretrained=True)
        self.dropout=paddle.nn.Dropout(p=0.5)
        self.fc = paddle.nn.Linear(1000, 20)
    #网络的前向计算过程
    def forward(self,x):
        x=self.layer(x)
        x=self.dropout(x)
        x=self.fc(x)
        return x
```
### 6. 应用高阶API训练模型

一是定义输入数据形状大小和数据类型。

二是实例化模型。如果要用高阶API，需要用Paddle.Model()对模型进行封装，如model = paddle.Model(model,inputs=input_define,labels=label_define)。

三是定义优化器。这个使用Adam优化器，学习率设置为0.0001，优化器中的学习率(learning_rate)参数很重要。要是训练过程中得到的准确率呈震荡状态，忽大忽小，可以试试进一步把学习率调低。

四是准备模型。这里用到高阶API，model.prepare()。

五是训练模型。这里用到高阶API，model.fit()。参数意义详见下述代码注释。
```python
#定义输入
input_define = paddle.static.InputSpec(shape=[-1,3,224,224], dtype="float32", name="img")
label_define = paddle.static.InputSpec(shape=[-1,1], dtype="int64", name="label")

#实例化网络对象并定义优化器等训练逻辑
model = MyNet()
model = paddle.Model(model,inputs=input_define,labels=label_define) #用Paddle.Model()对模型进行封装
optimizer = paddle.optimizer.Adam(learning_rate=0.00005, parameters=model.parameters())
#上述优化器中的学习率(learning_rate)参数很重要。要是训练过程中得到的准确率呈震荡状态，忽大忽小，可以试试进一步把学习率调低。

model.prepare(optimizer=optimizer, #指定优化器
              loss=paddle.nn.CrossEntropyLoss(), #指定损失函数
              metrics=paddle.metric.Accuracy()) #指定评估方法

model.fit(train_data=train_dataset,     #训练数据集
          eval_data=eval_dataset,         #测试数据集
          batch_size=128,                  #一个批次的样本数量
          epochs=100,                      #迭代轮次
          save_dir="/home/aistudio/lup", #把模型参数、优化器参数保存至自定义的文件夹
          save_freq=20,                    #设定每隔多少个epoch保存模型参数及优化器参数
          log_freq=100                     #打印日志的频率
)
```

```python
model.evaluate(eval_dataset,verbose=1)
```
### 7. 应用已经训练好的模型进行预测

如果是要参加建模比赛，通常赛事组织方会提供待预测的数据集，我们需要利用自己构建的模型，来对待预测数据集合中的数据标签进行预测。也就是说，我们其实并不知道到其真实标签是什么，只有比赛的组织方知道真实标签，我们的模型预测结果越接近真实结果，那么分数也就越高。

预测流程分为以下几个步骤：

一是构建数据读取器。因为预测数据集没有标签，该读取器写法和训练数据读取器不一样，建议重新写一个类，继承于Dataset基类。

二是实例化模型。如果要用高阶API，需要用Paddle.Model()对模型进行封装，如paddle.Model(MyNet(),inputs=input_define)，由于是预测模型，所以仅设定输入数据格式就好了。

三是读取刚刚训练好的参数。这个保存在/home/aistudio/work目录之下，如果指定的是final则是最后一轮训练后的结果。可以指定其他轮次的结果，比如model.load('/home/aistudio/work/30')，这里用到了高阶API，model.load()

四是准备模型。这里用到高阶API，model.prepare()。

五是读取待预测集合中的数据，利用已经训练好的模型进行预测。

六是结果保存。
```python
class InferDataset(Dataset):
    def __init__(self, img_path=None):
        """
        数据读取Reader(推理)
        :param img_path: 推理单张图片
        """
        super().__init__()
        if img_path:
            self.img_paths = [img_path]
        else:
            raise Exception("请指定需要预测对应图片路径")

    def __getitem__(self, index):
        # 获取图像路径
        img_path = self.img_paths[index]
        # 使用Pillow来读取图像数据并转成Numpy格式
        img = Image.open(img_path)
        if img.mode != 'RGB': 
            img = img.convert('RGB') 
        img = preprocess(img) #数据预处理--这里仅包括简单数据预处理，没有用到数据增强
        return img

    def __len__(self):
        return len(self.img_paths)

#实例化推理模型
model = paddle.Model(MyNet(),inputs=input_define)

#读取刚刚训练好的参数
model.load('/home/aistudio/lup/final')

#准备模型
model.prepare()

#得到待预测数据集中每个图像的读取路径
infer_list=[]
with open("/home/aistudio/data/testpath.txt") as file_pred:
    for line in file_pred:
        infer_list.append("/home/aistudio/data/"+line.strip())

#模型预测结果通常是个数，需要获得其对应的文字标签。这里需要建立一个字典。
def get_label_dict2():
    label_list2=[]
    with open("/home/aistudio/data/species.txt") as filess:
        for line in filess:
            a,b = line.strip("\n").split(" ")
            label_list2.append([int(a)-1, b])# [类编号，蝴蝶类名称]
    label_dic2 = dict(label_list2)
    return label_dic2

label_dict2 = get_label_dict2()
#print(label_dict2)

#利用训练好的模型进行预测
results=[]
for infer_path in infer_list:
    infer_data = InferDataset(infer_path)
    result = model.predict(test_data=infer_data)[0] #关键代码，实现预测功能
    result = paddle.to_tensor(result)
    result = np.argmax(result.numpy()) #获得最大值所在的序号
    results.append("{}".format(label_dict2[result])) #查找该序号所对应的标签名字

#把结果保存起来
with open("work/result.txt", "w") as f:
    for r in results:
        f.write("{}\n".format(r))
```
