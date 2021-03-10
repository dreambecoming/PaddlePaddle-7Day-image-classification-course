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

**{'loss': [0.14068076], 'acc': 0.9235924932975871}**
```python
!cd data &&\
unzip -qo data63004/Butterfly20_test.zip &&\
unzip -qo data63004/Butterfly20.zip &&\
rm -r __MACOSX
```


```python
import os
import random

import cv2
import numpy as np
from matplotlib import pyplot as plt
```


```python
#以下代码用于建立样本数据读取路径与样本标签之间的关系

data_list = [] #用个列表保存每个样本的读取路径、标签

#由于属种名称本身是字符串，而输入模型的是数字。需要构造一个字典，把某个数字代表该属种名称。键是属种名称，值是整数。
label_list=[]
with open("/home/aistudio/data/species.txt") as f:
    for line in f:
        a,b = line.strip("\n").split(" ")
        label_list.append([b, int(a)-1])
label_dic = dict(label_list)
# print(label_dic)

#获取Butterfly20目录下的所有子目录名称，保存进一个列表之中
class_list = os.listdir("/home/aistudio/data/Butterfly20")
class_list.remove('.DS_Store') #删掉列表中名为.DS_Store的元素，因为.DS_Store并没有样本。
# print(class_list)

for each in class_list:
    #print(each)
    for f in os.listdir("/home/aistudio/data/Butterfly20/"+each):
        #print(f)
        filename = "/home/aistudio/data/Butterfly20/"+each+'/'+f
        img = cv2.imread(filename)
        dst = cv2.flip(img,1)  #水平翻转图片
        #plt.imshow(dst)
        cv2.imwrite("/home/aistudio/data/Butterfly20/"+each+'/new_'+f, dst) #将翻转的图片加入数据集
```


```python
for each in class_list:
    for f in os.listdir("/home/aistudio/data/Butterfly20/"+each):# each 是 子文件夹名，蝴蝶类
        data_list.append(["/home/aistudio/data/Butterfly20/"+each+'/'+f,label_dic[each]])# f 是 子文件夹里面每张图片名，具体蝴蝶图像名

#按文件顺序读取，可能造成很多属种图片存在序列相关，用random.shuffle方法把样本顺序彻底打乱。
random.shuffle(data_list)

#打印前十个，可以看出data_list列表中的每个元素是[样本读取路径, 样本标签]。
# print(data_list[0:10])

#打印样本数量
print("样本数量是：{}".format(len(data_list)))
```

    样本数量是：3732



```python
#以下代码用于构造读取器与数据预处理
#首先需要导入相关的模块
import paddle
from paddle.vision.transforms import Compose, ColorJitter, Resize,Transpose, Normalize,ColorJitter
import cv2
import numpy as np
from PIL import Image
from paddle.io import Dataset
import paddle.vision.transforms as T
#自定义的数据预处理函数，输入原始图像，输出处理后的图像，可以借用paddle.vision.transforms的数据处理功能
def preprocess(img):
    transform = Compose([
        Resize(size=(224, 224)), #把数据长宽像素调成224*224
        T.ColorJitter(0.125,0.4,0.4,0.08),                                                   # 保持亮度、对比度、饱和度、色调等在测试集上一致
        # T.BrightnessTransform(0.4),                                                        # 只对亮度调整做调整
        T.RandomHorizontalFlip(224),                                                         # 水平翻转
        T.RandomRotation(90),                                                                # 随机反转角度范围
        T.RandomVerticalFlip(224),                                                           # 垂直翻转
        T.RandomRotation(90),        
        Normalize(mean=[0,0,0], std=[255, 255, 255], data_format='HWC'), #标准化
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

    (3, 224, 224)
    [2]

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
```python
#定义输入
input_define = paddle.static.InputSpec(shape=[-1,3,224,224], dtype="float32", name="img")
label_define = paddle.static.InputSpec(shape=[-1,1], dtype="int64", name="label")

#实例化网络对象并定义优化器等训练逻辑
model = MyNet()
model = paddle.Model(model,inputs=input_define,labels=label_define) #用Paddle.Model()对模型进行封装
optimizer = paddle.optimizer.Adam(learning_rate=3e-4, parameters=model.parameters())
#上述优化器中的学习率(learning_rate)参数很重要。要是训练过程中得到的准确率呈震荡状态，忽大忽小，可以试试进一步把学习率调低。

model.prepare(optimizer=optimizer, #指定优化器
              loss=paddle.nn.CrossEntropyLoss(), #指定损失函数
              metrics=paddle.metric.Accuracy()) #指定评估方法
```

    2021-03-10 13:42:46,934 - INFO - unique_endpoints {''}
    2021-03-10 13:42:46,936 - INFO - Downloading resnet101.pdparams from https://paddle-hapi.bj.bcebos.com/models/resnet101.pdparams
    100%|██████████| 263160/263160 [00:03<00:00, 66956.99it/s]
    2021-03-10 13:42:51,169 - INFO - File /home/aistudio/.cache/paddle/hapi/weights/resnet101.pdparams md5 checking...


```python
model.fit(train_data=train_dataset,     #训练数据集
          eval_data=eval_dataset,         #测试数据集
          batch_size=64,                  #一个批次的样本数量
          epochs=10,                      #迭代轮次
          shuffle=True,
          save_dir="/home/aistudio/lup", #把模型参数、优化器参数保存至自定义的文件夹
          save_freq=20,                    #设定每隔多少个epoch保存模型参数及优化器参数
          log_freq=100,
          verbose=1                    #打印日志的频率
)
```

    The loss value printed in the log is the current step, and the metric is the average value of previous step.
    Epoch 1/10


    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/layers/utils.py:77: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated, and in 3.8 it will stop working
      return (isinstance(seq, collections.Sequence) and
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/nn/layer/norm.py:648: UserWarning: When training, we now always track global mean and variance.
      "When training, we now always track global mean and variance.")


    step 47/47 [==============================] - loss: 1.2684 - acc: 0.4799 - 976ms/step
    save checkpoint at /home/aistudio/lup/0
    Eval begin...
    The loss value printed in the log is the current batch, and the metric is the average value of previous step.
    step 12/12 [==============================] - loss: 1.1635 - acc: 0.7225 - 851ms/step
    Eval samples: 746
    Epoch 2/10
    step 47/47 [==============================] - loss: 0.5241 - acc: 0.7545 - 972ms/step
    Eval begin...
    The loss value printed in the log is the current batch, and the metric is the average value of previous step.
    step 12/12 [==============================] - loss: 0.7176 - acc: 0.7855 - 877ms/step
    Eval samples: 746
    Epoch 3/10
    step 47/47 [==============================] - loss: 0.5882 - acc: 0.8088 - 982ms/step
    Eval begin...
    The loss value printed in the log is the current batch, and the metric is the average value of previous step.
    step 12/12 [==============================] - loss: 0.8078 - acc: 0.8137 - 884ms/step
    Eval samples: 746
    Epoch 4/10
    step 47/47 [==============================] - loss: 0.4881 - acc: 0.8486 - 976ms/step
    Eval begin...
    The loss value printed in the log is the current batch, and the metric is the average value of previous step.
    step 12/12 [==============================] - loss: 0.6811 - acc: 0.8633 - 877ms/step
    Eval samples: 746
    Epoch 5/10
    step 47/47 [==============================] - loss: 0.2994 - acc: 0.8583 - 985ms/step
    Eval begin...
    The loss value printed in the log is the current batch, and the metric is the average value of previous step.
    step 12/12 [==============================] - loss: 0.7759 - acc: 0.8056 - 881ms/step
    Eval samples: 746
    Epoch 6/10
    step 47/47 [==============================] - loss: 0.5039 - acc: 0.8784 - 986ms/step
    Eval begin...
    The loss value printed in the log is the current batch, and the metric is the average value of previous step.
    step 12/12 [==============================] - loss: 0.7076 - acc: 0.8499 - 884ms/step
    Eval samples: 746
    Epoch 7/10
    step 47/47 [==============================] - loss: 0.4930 - acc: 0.8868 - 972ms/step
    Eval begin...
    The loss value printed in the log is the current batch, and the metric is the average value of previous step.
    step 12/12 [==============================] - loss: 0.2629 - acc: 0.8727 - 878ms/step
    Eval samples: 746
    Epoch 8/10
    step 47/47 [==============================] - loss: 0.2420 - acc: 0.9099 - 989ms/step
    Eval begin...
    The loss value printed in the log is the current batch, and the metric is the average value of previous step.
    step 12/12 [==============================] - loss: 0.3926 - acc: 0.8968 - 879ms/step
    Eval samples: 746
    Epoch 9/10
    step 47/47 [==============================] - loss: 0.2041 - acc: 0.9273 - 982ms/step
    Eval begin...
    The loss value printed in the log is the current batch, and the metric is the average value of previous step.
    step 12/12 [==============================] - loss: 0.7978 - acc: 0.8820 - 885ms/step
    Eval samples: 746
    Epoch 10/10
    step 47/47 [==============================] - loss: 0.0587 - acc: 0.9233 - 976ms/step
    Eval begin...
    The loss value printed in the log is the current batch, and the metric is the average value of previous step.
    step 12/12 [==============================] - loss: 0.1458 - acc: 0.8954 - 879ms/step
    Eval samples: 746
    save checkpoint at /home/aistudio/lup/final


```python
# 模型评价
model.evaluate(eval_dataset,verbose=1)
```

    Eval begin...
    The loss value printed in the log is the current batch, and the metric is the average value of previous step.
    step 746/746 [==============================] - loss: 0.1407 - acc: 0.9236 - 59ms/step         
    Eval samples: 746

    {'loss': [0.14068076], 'acc': 0.9235924932975871}


```python
# 模型预测
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

总结：
* 参考课件，通过cv2.flip操作生成图像加入样本，增加样本数量。
* 改进的地方还有很多，以后再进一步。
