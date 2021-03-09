# 百度飞桨领航团零基础图像分类速成营 课程总结5
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



# 课节4：卷积神经网络基础：蝴蝶图像识别分类
## 前置知识

图像分类比赛的一般解题流程：
  1. 数据EDA （Pandas、Matplotlib）
  2. 数据预处理 （OpenCV、PIL、Pandas、Numpy、Scikit-Learn）
  3. 根据赛题任务定义好读取方法，即Dataset和Dataloader（PaddlePaddle2.0）
  4. 选择一个图像分类模型进行训练 （PaddlePaddle2.0）
  5. 对测试集进行测试并提交结果（PaddlePaddle2.0、Pandas）

图像分类竞赛常见难点：
  1. 类别不均衡
  2. One-Shot和Few-Shot分类
  3. 细粒度分类

一、EDA（Exploratory Data Analysis）与数据预处理
* EDA
&emsp;&emsp;探索性数据分析（Exploratory Data Analysis，简称EDA），是指对已有的数据（原始数据）进行分析探索，通过作图、制表、方程拟合、计算特征量等手段探索数据的结构和规律的一种数据分析方法。一般来说，我们最初接触到数据的时候往往是毫无头绪的，不知道如何下手，这时候探索性数据分析就非常有效。

&emsp;&emsp;对于图像分类任务，我们通常首先应该统计出每个类别的数量，查看训练集的数据分布情况。通过数据分布情况分析赛题，形成解题思路。

* 数据预处理

Compose实现将用于数据集预处理的接口以列表的方式进行组合。
```python
# 定义数据预处理
data_transforms = T.Compose([
    T.Resize(size=(32, 32)),
    T.Transpose(),    # HWC -> CHW
    T.Normalize(      # 归一化
        mean=[0, 0, 0],        # 均值
        std=[255, 255, 255],   # 标准差
        to_rgb=True)    
])
```
图像标准化与归一化

&emsp;&emsp;最常见的对图像预处理方法有两种，一种叫做图像标准化处理，另外一种方法叫做归一化处理。数据的标准化是指将数据按照比例缩放,使之落入一个特定的区间。
将数据通过去均值，实现中心化。处理后的数据呈正态分布,即均值为零。数据归一化是数据标准化的一种典型做法,即将数据统一映射到[0,1]区间上。

作用:
1. 有利于初始化的进行
2. 避免给梯度数值的更新带来数值问题
3. 有利于学习率数值的调整
4. 加快寻找最优解速度

<center><p>
  
   标准化

  ![](https://ai-studio-static-online.cdn.bcebos.com/0c290c7b103c41b09f118f977e7e1fe26871a6416e314dc6ae1e534548701114)
 
  
   归一化

  ![](https://ai-studio-static-online.cdn.bcebos.com/b6133a31796a4ccc964e62b4acb1b5b907d9c5d86e244acd9495126b4849ced1)
  
  </p></center>
```python
import numpy as np
from PIL import Image
from paddle.vision.transforms import Normalize

normalize_std = Normalize(mean=[127.5, 127.5, 127.5],
                        std=[127.5, 127.5, 127.5],
                        data_format='HWC')
# np.random.rand(d0,d1,d2……dn) 返回一个或一组服从“0~1”均匀分布的随机样本值。随机样本取值范围是[0,1)，不包括1。 
fake_img = Image.fromarray((np.random.rand(300, 320, 3) * 255.).astype(np.uint8))
# np.random.rand(300, 320, 3) * 255 决定了 每个像素范围是[0, 255]
# 输入的每个channel做 ( [0, 255] - mean（127.5) ) / std（127.5）= [-1, 1] 的运算，所以这一句的实际结果是将[0，255]的张量归一化到[-1, 1]上
fake_img = normalize_std(fake_img)
# print(fake_img.shape)
print(fake_img)
```

```python
import numpy as np
from PIL import Image
from paddle.vision.transforms import Normalize

normalize = Normalize(mean=[0, 0, 0],
                        std=[255, 255, 255],
                        data_format='HWC')

fake_img = Image.fromarray((np.random.rand(300, 320, 3) * 255.).astype(np.uint8))
# np.random.rand(300, 320, 3) * 255 决定了 每个像素范围是[0, 255]
# 输入的每个channel做 ( [0, 255] - mean（0) ) / std（255）= [0, 1] 的运算，所以这一句的实际结果是将[0，255]的张量归一化到[0, 1]上
fake_img = normalize(fake_img)
# print(fake_img.shape)
print(fake_img)
```
数据集划分
```python
# 读取数据

train_images = pd.read_csv('data/data71799/lemon_lesson/train_images.csv', usecols=['id','class_num'])

# 划分训练集和校验集
all_size = len(train_images)
print(all_size)
train_size = int(all_size * 0.8)
train_image_path_list = train_images[:train_size]
val_image_path_list = train_images[train_size:]

print(len(train_image_path_list))
print(len(val_image_path_list))
```

```python
# 构建Dataset
class MyDataset(paddle.io.Dataset):
    """
    步骤一：继承paddle.io.Dataset类
    """
    def __init__(self, train_list, val_list, mode='train'):
        """
        步骤二：实现构造函数，定义数据读取方式
        """
        super(MyDataset, self).__init__()
        self.data = []
        # 借助pandas读取csv文件
        self.train_images = train_list
        self.test_images = val_list
        if mode == 'train':
            # 读train_images.csv中的数据
            for row in self.train_images.itertuples():
                self.data.append(['data/data71799/lemon_lesson/train_images/'+getattr(row, 'id'), getattr(row, 'class_num')])
        else:
            # 读test_images.csv中的数据
            for row in self.test_images.itertuples():
                self.data.append(['data/data71799/lemon_lesson/train_images/'+getattr(row, 'id'), getattr(row, 'class_num')])

    def load_img(self, image_path):
        # 实际使用时使用Pillow相关库进行图片读取即可，这里我们对数据先做个模拟
        image = Image.open(image_path).convert('RGB')

        return image

    def __getitem__(self, index):
        """
        步骤三：实现__getitem__方法，定义指定index时如何获取数据，并返回单条数据（训练数据，对应的标签）
        """
        image = self.load_img(self.data[index][0])
        label = self.data[index][1]

        return data_transforms(image), np.array(label, dtype='int64')

    def __len__(self):
        """
        步骤四：实现__len__方法，返回数据集总数目
        """
        return len(self.data)
```
数据加载器定义
```python
#train_loader
train_dataset = MyDataset(train_list=train_image_path_list, val_list=val_image_path_list, mode='train')
train_loader = paddle.io.DataLoader(train_dataset, places=paddle.CPUPlace(), batch_size=128, shuffle=True, num_workers=0)

#val_loader
val_dataset =MyDataset(train_list=train_image_path_list, val_list=val_image_path_list, mode='test')
val_loader = paddle.io.DataLoader(val_dataset, places=paddle.CPUPlace(), batch_size=128, shuffle=True, num_workers=0)
```

```python
print('=============train dataset=============')
for image, label in train_dataset:
    print('image shape: {}, label: {}'.format(image.shape, label))
    break

for batch_id, data in enumerate(train_loader()):
    x_data = data[0]
    y_data = data[1]
    print(x_data)
    print(y_data)
    break

```
二、Baseline选择

&emsp;&emsp;理想情况中，模型越大拟合能力越强，图像尺寸越大，保留的信息也越多。在实际情况中模型越复杂训练时间越长，图像输入尺寸越大训练时间也越长。
比赛开始优先使用最简单的模型（如ResNet），快速跑完整个训练和预测流程；分类模型的选择需要根据任务复杂度来进行选择，并不是精度越高的模型越适合比赛。
在实际的比赛中我们可以逐步增加图像的尺寸，比如先在64 * 64的尺寸下让模型收敛，进而将模型在128 * 128的尺寸下训练，进而到224 * 224的尺寸情况下，这种方法可以加速模型的收敛速度。

Baseline应遵循以下几点原则：
1. 复杂度低，代码结构简单。
2. Loss收敛正确，评价指标（metric）出现相应提升（如accuracy/AUC之类的）
3. 迭代快速，没有很复杂（Fancy）的模型结构/Loss function/图像预处理方法之类的
4. 编写正确并简单的测试脚本，能够提交submission后获得正确的分数

模型组网方式

&emsp;&emsp;对于组网方式，飞桨框架统一支持 Sequential 或 SubClass 的方式进行模型的组建。我们根据实际的使用场景，来选择最合适的组网方式。如针对顺序的线性网络结构我们可以直接使用 Sequential ，相比于 SubClass ，Sequential 可以快速的完成组网。 如果是一些比较复杂的网络结构，我们可以使用 SubClass 定义的方式来进行模型代码编写，在 init 构造函数中进行 Layer 的声明，在 forward 中使用声明的 Layer 变量进行前向计算。通过这种方式，我们可以组建更灵活的网络结构。

使用 SubClass 进行组网
```python
#定义卷积神经网络
class MyNet(paddle.nn.Layer):
    def __init__(self, num_classes=4):
        super(MyNet, self).__init__()
        self.conv1 = paddle.nn.Conv2D(in_channels=3, out_channels=32, kernel_size=(3, 3), stride=1, padding = 1)
        # self.pool1 = paddle.nn.MaxPool2D(kernel_size=2, stride=2)

        self.conv2 = paddle.nn.Conv2D(in_channels=32, out_channels=64, kernel_size=(3,3),  stride=2, padding = 0)
        # self.pool2 = paddle.nn.MaxPool2D(kernel_size=2, stride=2)

        self.conv3 = paddle.nn.Conv2D(in_channels=64, out_channels=64, kernel_size=(3,3), stride=2, padding = 0)

        self.conv4 = paddle.nn.Conv2D(in_channels=64, out_channels=64, kernel_size=(3,3), stride=2, padding = 1)

        self.flatten = paddle.nn.Flatten()
        self.linear1 = paddle.nn.Linear(in_features=1024, out_features=64)
        self.linear2 = paddle.nn.Linear(in_features=64, out_features=num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        # x = self.pool1(x)
        # print(x.shape)
        x = self.conv2(x)
        x = F.relu(x)
        # x = self.pool2(x)
        # print(x.shape)

        x = self.conv3(x)
        x = F.relu(x)
        # print(x.shape)
        
        x = self.conv4(x)
        x = F.relu(x)
        # print(x.shape)

        x = self.flatten(x)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x
```

使用 Sequential 进行组网
```python
# Sequential形式组网
MyNet = nn.Sequential(
    nn.Conv2D(in_channels=3, out_channels=32, kernel_size=(3, 3), stride=1, padding = 1),
    nn.ReLU(),
    nn.Conv2D(in_channels=32, out_channels=64, kernel_size=(3,3),  stride=2, padding = 0),
    nn.ReLU(),
    nn.Conv2D(in_channels=64, out_channels=64, kernel_size=(3,3), stride=2, padding = 0),
    nn.ReLU(),
    nn.Conv2D(in_channels=64, out_channels=64, kernel_size=(3,3), stride=2, padding = 1),
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(in_features=50176, out_features=64),
    nn.ReLU(),
    nn.Linear(in_features=64, out_features=4)
)
```
```python
# 模型封装
model = paddle.Model(MyNet())
```
网络结构可视化

通过summary打印网络的基础结构和参数信息。
```python
model.summary((1, 3, 32, 32))
```
特征图尺寸计算

![](https://ai-studio-static-online.cdn.bcebos.com/a0e6774cd7174937b0d65f7d9026eebe59c00e4cff294e2e85534d0a237ae4e3)
```python
# 定义优化器
optim = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())
```
```python
# 配置模型
model.prepare(
    optim,
    paddle.nn.CrossEntropyLoss(),
    Accuracy()
    )
```
扩展知识点：训练过程可视化

&emsp;&emsp;然后我们调用VisualDL工具，在命令行中输入： `visualdl --logdir ./visualdl_log_dir --port 8080`,打开浏览器，输入网址 http://127.0.0.1:8080 就可以在浏览器中看到相关的训练信息，具体如下：![](https://ai-studio-static-online.cdn.bcebos.com/7c29ce3c73e24b649a45f93580f78b5223b9d0d63b1143b1a3df9160eabb7615)

调参，训练，记录曲线，分析结果。
```python
# 调用飞桨框架的VisualDL模块，保存信息到目录中。
# callback = paddle.callbacks.VisualDL(log_dir='visualdl_log_dir')

from visualdl import LogReader, LogWriter

args={
    'logdir':'./vdl',
    'file_name':'vdlrecords.model.log',
    'iters':0,
}

# 配置visualdl
write = LogWriter(logdir=args['logdir'], file_name=args['file_name'])
#iters 初始化为0
iters = args['iters'] 

#自定义Callback
class Callbk(paddle.callbacks.Callback):
    def __init__(self, write, iters=0):
        self.write = write
        self.iters = iters

    def on_train_batch_end(self, step, logs):

        self.iters += 1

        #记录loss
        self.write.add_scalar(tag="loss",step=self.iters,value=logs['loss'][0])
        #记录 accuracy
        self.write.add_scalar(tag="acc",step=self.iters,value=logs['acc'])
```
```python
# 模型训练与评估
model.fit(train_loader,
        val_loader,
        log_freq=1,
        epochs=5,
        callbacks=Callbk(write=write, iters=iters),
        verbose=1,
        )
```
```python
# 保存模型参数
# model.save('Hapi_MyCNN')  # save for training
model.save('Hapi_MyCNN1', False)  # save for inference
```
三、模型预测
```python
import os, time
import matplotlib.pyplot as plt
import paddle
from PIL import Image
import numpy as np

def load_image(img_path):
    '''
    预测图片预处理
    '''
    img = Image.open(img_path).convert('RGB')
    plt.imshow(img)          #根据数组绘制图像
    plt.show()               #显示图像
    
    #resize
    img = img.resize((32, 32), Image.BILINEAR) #Image.BILINEAR双线性插值
    img = np.array(img).astype('float32')

    # HWC to CHW 
    img = img.transpose((2, 0, 1))
    
    #Normalize
    img = img / 255         #像素值归一化
    # mean = [0.31169346, 0.25506335, 0.12432463]   
    # std = [0.34042713, 0.29819837, 0.1375536]
    # img[0] = (img[0] - mean[0]) / std[0]
    # img[1] = (img[1] - mean[1]) / std[1]
    # img[2] = (img[2] - mean[2]) / std[2]
    
    return img

def infer_img(path, model_file_path, use_gpu):
    '''
    模型预测
    '''
    paddle.set_device('gpu:0') if use_gpu else paddle.set_device('cpu')
    model = paddle.jit.load(model_file_path)
    model.eval() #训练模式

    #对预测图片进行预处理
    infer_imgs = []
    infer_imgs.append(load_image(path))
    infer_imgs = np.array(infer_imgs)
    label_list = ['0:優良', '1:良', '2:加工品', '3:規格外']

    for i in range(len(infer_imgs)):
        data = infer_imgs[i]
        dy_x_data = np.array(data).astype('float32')
        dy_x_data = dy_x_data[np.newaxis,:, : ,:]
        img = paddle.to_tensor(dy_x_data)
        out = model(img)

        print(out[0])
        print(paddle.nn.functional.softmax(out)[0]) # 若模型中已经包含softmax则不用此行代码。

        lab = np.argmax(out.numpy())  #argmax():返回最大数的索引
        print("样本: {},被预测为:{}".format(path, label_list[lab]))

    print("*********************************************")
```

```python

```

```python

```

```python

```

v
```python

```
