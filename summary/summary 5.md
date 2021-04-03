# 百度飞桨领航团零基础图像分类速成营 课程总结5
![paddlepaddle](https://paddlepaddle-org-cn.cdn.bcebos.com/paddle-site-front/favicon-128.png  "百度logo")

[课程链接](https://aistudio.baidu.com/aistudio/course/introduce/11939)	`https://aistudio.baidu.com/aistudio/course/introduce/11939`  
[飞桨官网](https://www.paddlepaddle.org.cn/)	`https://www.paddlepaddle.org.cn/`   
[推荐学习网站](https://www.runoob.com/python3/python3-tutorial.html)	`https://www.runoob.com/python3/python3-tutorial.html`  

****
## 目录
* [竞赛全流程](#竞赛全流程)
* [调参](#调参)
* [PaddleClas](#PaddleClas)
* [作业](#作业)

# 课节5：图像分类竞赛全流程实战
## 竞赛全流程

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

### 一、EDA（Exploratory Data Analysis）与数据预处理
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
### 二、Baseline选择

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
### 三、模型预测
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
image_path = []

for root, dirs, files in os.walk('work/'):
    # 遍历work/文件夹内图片
    for f in files:
        image_path.append(os.path.join(root, f))

for i in range(len(image_path)):
    infer_img(path=image_path[i], use_gpu=True, model_file_path="Hapi_MyCNN")
    # time.sleep(0.5) #防止输出错乱
    break
```
baseline选择技巧：
* 模型：复杂度小的模型可以快速迭代。
* optimizer：推荐Adam，或者SGD
* Loss Function: 多分类Cross entropy;
* metric：以比赛的评估指标为准。
* 数据增强：数据增强其实可为空，或者只有一个HorizontalFlip即可。
* 图像分辨率：初始最好就用小图，如224*224之类的。

## 调参
###  数据处理部分
label shuffling

&emsp;&emsp;首先对原始的图像列表，按照标签顺序进行排序；
然后计算每个类别的样本数量，并得到样本最多的那个类别的样本数。
根据这个最多的样本数，对每类都产生一个随机排列的列表；
然后用每个类别的列表中的数对各自类别的样本数求余，得到一个索引值，从该类的图像中提取图像，生成该类的图像随机列表；
然后把所有类别的随机列表连在一起，做个Random Shuffling，得到最后的图像列表，用这个列表进行训练。


![](https://ai-studio-static-online.cdn.bcebos.com/36dc85bab2c84602a04dec6fe7db3914a96c66546a7b4bb68f61b8eb77387c35)
```python
# labelshuffling

def labelShuffling(dataFrame, groupByName = 'class_num'):

    groupDataFrame = dataFrame.groupby(by=[groupByName])
    labels = groupDataFrame.size()
    print("length of label is ", len(labels))
    maxNum = max(labels)
    lst = pd.DataFrame()
    for i in range(len(labels)):
        print("Processing label  :", i)
        tmpGroupBy = groupDataFrame.get_group(i)
        createdShuffleLabels = np.random.permutation(np.array(range(maxNum))) % labels[i]
        print("Num of the label is : ", labels[i])
        lst=lst.append(tmpGroupBy.iloc[createdShuffleLabels], ignore_index=True)
        print("Done")
    # lst.to_csv('test1.csv', index=False)
    return lst
```
```python
from sklearn.utils import shuffle

# 读取数据
train_images = pd.read_csv('data/data71799/lemon_lesson/train_images.csv', usecols=['id','class_num'])

# 读取数据

df = labelShuffling(train_images)
df = shuffle(df)

image_path_list = df['id'].values
label_list = df['class_num'].values
label_list = paddle.to_tensor(label_list, dtype='int64')
label_list = paddle.nn.functional.one_hot(label_list, num_classes=4)

# 划分训练集和校验集
all_size = len(image_path_list)
train_size = int(all_size * 0.8)
train_image_path_list = image_path_list[:train_size]
train_label_list = label_list[:train_size]
val_image_path_list = image_path_list[train_size:]
val_label_list = label_list[train_size:]
```
图像扩增

&emsp;&emsp;为了获得更多数据，我们只需要对现有数据集进行微小改动。例如翻转剪裁等操作。对图像进行微小改动，模型就会认为这些是不同的图像。常用的有两种数据增广方法：
第一个方法称为离线扩充。对于相对较小的数据集，此方法是首选。
第二个方法称为在线增强，或即时增强。对于较大的数据集，此方法是首选。

飞桨2.0中的预处理方法：

&emsp;&emsp;在图像分类任务中常见的数据增强有翻转、旋转、随机裁剪、颜色噪音、平移等，具体的数据增强方法要根据具体任务来选择，要根据具体数据的特定来选择。对于不同的比赛来说数据扩增方法一定要反复尝试，会很大程度上影响模型精度。
```python
import numpy as np
from PIL import Image
from paddle.vision.transforms import RandomHorizontalFlip

transform = RandomHorizontalFlip(224)

fake_img = Image.fromarray((np.random.rand(300, 320, 3) * 255.).astype(np.uint8))

fake_img = transform(fake_img)
print(fake_img.size)
```
```python
# 定义数据预处理
data_transforms = T.Compose([
    T.Resize(size=(224, 224)),
    T.RandomHorizontalFlip(224),
    T.RandomVerticalFlip(224),
    T.Transpose(),    # HWC -> CHW
    T.Normalize(
        mean=[0, 0, 0],        # 归一化
        std=[255, 255, 255],
        to_rgb=True)    
])
```
```python
# 构建Dataset
class MyDataset(paddle.io.Dataset):
    """
    步骤一：继承paddle.io.Dataset类
    """
    def __init__(self, train_img_list, val_img_list,train_label_list,val_label_list, mode='train'):
        """
        步骤二：实现构造函数，定义数据读取方式，划分训练和测试数据集
        """
        super(MyDataset, self).__init__()
        self.img = []
        self.label = []
        # 借助pandas读csv的库
        self.train_images = train_img_list
        self.test_images = val_img_list
        self.train_label = train_label_list
        self.test_label = val_label_list
        if mode == 'train':
            # 读train_images的数据
            for img,la in zip(self.train_images, self.train_label):
                self.img.append('data/data71799/lemon_lesson/train_images/'+img)
                self.label.append(la)
        else:
            # 读test_images的数据
            for img,la in zip(self.train_images, self.train_label):
                self.img.append('data/data71799/lemon_lesson/train_images/'+img)
                self.label.append(la)

    def load_img(self, image_path):
        # 实际使用时使用Pillow相关库进行图片读取即可，这里我们对数据先做个模拟
        image = Image.open(image_path).convert('RGB')
        return image

    def __getitem__(self, index):
        """
        步骤三：实现__getitem__方法，定义指定index时如何获取数据，并返回单条数据（训练数据，对应的标签）
        """
        image = self.load_img(self.img[index])
        label = self.label[index]
        # label = paddle.to_tensor(label)
        
        return data_transforms(image), paddle.nn.functional.label_smooth(label)

    def __len__(self):
        """
        步骤四：实现__len__方法，返回数据集总数目
        """
        return len(self.img)
```
```python
#train_loader
train_dataset = MyDataset(train_img_list=train_image_path_list, val_img_list=val_image_path_list, train_label_list=train_label_list, val_label_list=val_label_list, mode='train')
train_loader = paddle.io.DataLoader(train_dataset, places=paddle.CPUPlace(), batch_size=32, shuffle=True, num_workers=0)

#val_loader
val_dataset = MyDataset(train_img_list=train_image_path_list, val_img_list=val_image_path_list, train_label_list=train_label_list, val_label_list=val_label_list, mode='test')
val_loader = paddle.io.DataLoader(train_dataset, places=paddle.CPUPlace(), batch_size=32, shuffle=True, num_workers=0)
```
### 模型训练部分

* 标签平滑（LSR）

&emsp;&emsp;在分类问题中，一般最后一层是全连接层，然后对应one-hot编码，这种编码方式和通过降低交叉熵损失来调整参数的方式结合起来，会有一些问题。这种方式鼓励模型对不同类别的输出分数差异非常大，或者说模型过分相信他的判断，但是由于人工标注信息可能会出现一些错误。模型对标签的过分相信会导致过拟合。
标签平滑可以有效解决该问题，它的具体思想是降低我们对于标签的信任，例如我们可以将损失的目标值从1稍微降到0.9，或者将从0稍微升到0.1。总的来说，标签平滑是一种通过在标签y中加入噪声，实现对模型约束，降低模型过拟合程度的一种正则化方法。
[论文地址](https://arxiv.org/abs/1512.00567)  [飞桨2.0API地址](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/functional/common/label_smooth_cn.html#label-smooth)

<center><p>


  $\tilde{y_k} = (1 - \epsilon) * y_k + \epsilon * \mu_k$


  </p></center>
  
  
其中 1−ϵ 和 ϵ 分别是权重，$\tilde{y_k}$是平滑后的标签，通常 μ 使用均匀分布。

独热编码

One-Hot编码是分类变量作为二进制向量的表示。这首先要求将分类值映射到整数值。然后，每个整数值被表示为二进制向量，除了整数的索引之外，它都是零值，它被标记为1。

离散特征的编码分为两种情况：

1. 离散特征的取值之间没有大小的意义，比如color：[red,blue],那么就使用one-hot编码
1. 离散特征的取值有大小的意义，比如size:[X,XL,XXL],那么就使用数值的映射{X:1,XL:2,XXL:3}，标签编码

* 优化算法选择

Adam, init_lr=3e-4，3e-4号称是Adam最好的初始学习率

学习率调整策略：

&emsp;&emsp;当我们使用梯度下降算法来优化目标函数的时候，当越来越接近Loss值的全局最小值时，学习率应该变得更小来使得模型尽可能接近这一点。


![](https://ai-studio-static-online.cdn.bcebos.com/15d80a33d34c4eafa5824032ecf3cd3bc4a5111b41e94703982b81b4f46bdf3b)


可以由上图看出，固定学习率时，当到达收敛状态时，会在最优值附近一个较大的区域内摆动；而当随着迭代轮次的增加而减小学习率，会使得在收敛时，在最优值附近一个更小的区域内摆动。（之所以曲线震荡朝向最优值收敛，是因为在每一个mini-batch中都存在噪音）。因此，选择一个合适的学习率，对于模型的训练将至关重要。下面来了解一些学习率调整的方法。

针对学习率的优化有很多种方法，而linearwarmup是其中重要的一种。

[飞桨2.0学习率调整相关API](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/optimizer/Overview_cn.html#about-lr)

当我们使用梯度下降算法来优化目标函数的时候，当越来越接近Loss值的全局最小值时，学习率应该变得更小来使得模型尽可能接近这一点，而余弦退火（Cosine annealing）可以通过余弦函数来降低学习率。余弦函数中随着x的增加余弦值首先缓慢下降，然后加速下降，再次缓慢下降。这种下降模式能和学习率配合，以一种十分有效的计算方式来产生很好的效果。

* 技巧应用

此部分通过MobileNetV2训练模型，并在模型中应用上述提到的技巧。
```python
from work.mobilenet import MobileNetV2

# 模型封装
model_res = MobileNetV2(class_dim=4)
model = paddle.Model(model_res)

# 定义优化器

scheduler = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=0.5, T_max=10, verbose=True)
sgd = paddle.optimizer.SGD(learning_rate=scheduler, parameters=linear.parameters())
# optim = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())
```

软标签&硬标签：

![](https://ai-studio-static-online.cdn.bcebos.com/ffcc361281034384af67d67e41df743bd03ff53aba6d403697a960571aea143f)

```python
# 配置模型
model.prepare(
    optim,
    paddle.nn.CrossEntropyLoss(soft_label=True),
    Accuracy()
    )

```
```python
# 模型训练与评估
model.fit(train_loader,
        val_loader,
        log_freq=1,
        epochs=5,
        # callbacks=Callbk(write=write, iters=iters),
        verbose=1,
        )
```
完整代码：

```python
# 数据读取部分
# 导入所需要的库
from sklearn.utils import shuffle
import os
import pandas as pd
import numpy as np
from PIL import Image

import paddle
import paddle.nn as nn
from paddle.io import Dataset
import paddle.vision.transforms as T
import paddle.nn.functional as F
from paddle.metric import Accuracy

import warnings
warnings.filterwarnings("ignore")

# 读取数据
train_images = pd.read_csv('data/data71799/lemon_lesson/train_images.csv', usecols=['id','class_num'])

# labelshuffling

def labelShuffling(dataFrame, groupByName = 'class_num'):

    groupDataFrame = dataFrame.groupby(by=[groupByName])
    labels = groupDataFrame.size()
    print("length of label is ", len(labels))
    maxNum = max(labels)
    lst = pd.DataFrame()
    for i in range(len(labels)):
        print("Processing label  :", i)
        tmpGroupBy = groupDataFrame.get_group(i)
        createdShuffleLabels = np.random.permutation(np.array(range(maxNum))) % labels[i]
        print("Num of the label is : ", labels[i])
        lst=lst.append(tmpGroupBy.iloc[createdShuffleLabels], ignore_index=True)
        print("Done")
    # lst.to_csv('test1.csv', index=False)
    return lst

# 划分训练集和校验集
all_size = len(train_images)
# print(all_size)
train_size = int(all_size * 0.8)
train_image_list = train_images[:train_size]
val_image_list = train_images[train_size:]

df = labelShuffling(train_image_list)
df = shuffle(df)

train_image_path_list = df['id'].values
label_list = df['class_num'].values
label_list = paddle.to_tensor(label_list, dtype='int64')
train_label_list = paddle.nn.functional.one_hot(label_list, num_classes=4)

val_image_path_list = val_image_list['id'].values
val_label_list = val_image_list['class_num'].values
val_label_list = paddle.to_tensor(val_label_list, dtype='int64')
val_label_list = paddle.nn.functional.one_hot(val_label_list, num_classes=4)

# 定义数据预处理
data_transforms = T.Compose([
    T.Resize(size=(224, 224)),
    T.RandomHorizontalFlip(224),
    T.RandomVerticalFlip(224),
    T.Transpose(),    # HWC -> CHW
    T.Normalize(
        mean=[0, 0, 0],        # 归一化
        std=[255, 255, 255],
        to_rgb=True)    
])
```

```python
# 构建Dataset
class MyDataset(paddle.io.Dataset):
    """
    步骤一：继承paddle.io.Dataset类
    """
    def __init__(self, train_img_list, val_img_list,train_label_list,val_label_list, mode='train'):
        """
        步骤二：实现构造函数，定义数据读取方式，划分训练和测试数据集
        """
        super(MyDataset, self).__init__()
        self.img = []
        self.label = []
        # 借助pandas读csv的库
        self.train_images = train_img_list
        self.test_images = val_img_list
        self.train_label = train_label_list
        self.test_label = val_label_list
        if mode == 'train':
            # 读train_images的数据
            for img,la in zip(self.train_images, self.train_label):
                self.img.append('data/data71799/lemon_lesson/train_images/'+img)
                self.label.append(la)
        else:
            # 读test_images的数据
            for img,la in zip(self.train_images, self.train_label):
                self.img.append('data/data71799/lemon_lesson/train_images/'+img)
                self.label.append(la)

    def load_img(self, image_path):
        # 实际使用时使用Pillow相关库进行图片读取即可，这里我们对数据先做个模拟
        image = Image.open(image_path).convert('RGB')
        return image

    def __getitem__(self, index):
        """
        步骤三：实现__getitem__方法，定义指定index时如何获取数据，并返回单条数据（训练数据，对应的标签）
        """
        image = self.load_img(self.img[index])
        label = self.label[index]
        # label = paddle.to_tensor(label)
        
        return data_transforms(image), paddle.nn.functional.label_smooth(label)

    def __len__(self):
        """
        步骤四：实现__len__方法，返回数据集总数目
        """
        return len(self.img)
```
```python
#train_loader
train_dataset = MyDataset(train_img_list=train_image_path_list, val_img_list=val_image_path_list, train_label_list=train_label_list, val_label_list=val_label_list, mode='train')
train_loader = paddle.io.DataLoader(train_dataset, places=paddle.CPUPlace(), batch_size=32, shuffle=True, num_workers=0)

#val_loader

val_dataset = MyDataset(train_img_list=train_image_path_list, val_img_list=val_image_path_list, train_label_list=train_label_list, val_label_list=val_label_list, mode='test')
val_loader = paddle.io.DataLoader(train_dataset, places=paddle.CPUPlace(), batch_size=32, shuffle=True, num_workers=0)
```
```python
# 模型训练部分
from work.mobilenet import MobileNetV2

# 模型封装
model_res = MobileNetV2(class_dim=4)
model = paddle.Model(model_res)

# 定义优化器

scheduler = paddle.optimizer.lr.LinearWarmup(
        learning_rate=0.5, warmup_steps=20, start_lr=0, end_lr=0.5, verbose=True)
optim = paddle.optimizer.SGD(learning_rate=scheduler, parameters=model.parameters())
# optim = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())

# 配置模型
model.prepare(
    optim,
    paddle.nn.CrossEntropyLoss(soft_label=True),
    Accuracy()
    )

# 模型训练与评估
model.fit(train_loader,
        val_loader,
        log_freq=1,
        epochs=10,
        # callbacks=Callbk(write=write, iters=iters),
        verbose=1,
        )
```
```python
# 保存模型参数
# model.save('Hapi_MyCNN')  # save for training
model.save('Hapi_MyCNN2', False)  # save for inference
```
```python
# 模型预测
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
    # plt.imshow(img)          #根据数组绘制图像
    # plt.show()               #显示图像
    
    #resize
    img = img.resize((32, 32), Image.BILINEAR) #Image.BILINEAR双线性插值
    img = np.array(img).astype('float32')

    # HWC to CHW 
    img = img.transpose((2, 0, 1))
    
    #Normalize
    img = img / 255         #像素值归一化
    # print(img)
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
    label_pre = []
    for i in range(len(infer_imgs)):
        data = infer_imgs[i]
        dy_x_data = np.array(data).astype('float32')
        dy_x_data = dy_x_data[np.newaxis,:, : ,:]
        img = paddle.to_tensor(dy_x_data)
        out = model(img)

        # print(out[0])
        # print(paddle.nn.functional.softmax(out)[0]) # 若模型中已经包含softmax则不用此行代码。

        lab = np.argmax(out.numpy())  #argmax():返回最大数的索引
        label_pre.append(lab)
        # print(lab)
        # print("样本: {},被预测为:{}".format(path, label_list[lab]))
    return label_pre
    # print("*********************************************")
```
```python
img_list = os.listdir('data/data71799/lemon_lesson/test_images/')
img_list.sort()
img_list
```
```python
image_path = []
submit = []
for root, dirs, files in os.walk('data/data71799/lemon_lesson/test_images/'):
    # 文件夹内图片
    for f in files:
        image_path.append(os.path.join(root, f))
        submit.append(f)
image_path.sort()       
submit.sort()

key_list = []
for i in range(len(image_path)):
    key_list.append(infer_img(path=image_path[i], use_gpu=True, model_file_path="Hapi_MyCNN1")[0])
    # time.sleep(0.5) #防止输出错乱
```
```python
import pandas as pd

img = pd.DataFrame(submit)
img = img.rename(columns = {0:"id"})
img['class_num'] = key_list


img.to_csv('submit123.csv', index=False)
```
## PaddleClas
&emsp;&emsp;PaddleClas是飞桨为工业界和学术界所准备的一个图像分类任务的工具集，助力使用者训练出更好的视觉模型和应用落地。PaddleClas提供了基于图像分类的模型训练、评估、预测、部署全流程的服务，方便大家更加高效地学习图像分类。

下面将从PaddleClas模型库概览、特色应用、快速上手、实践应用几个方面介绍PaddleClas实践方法：
1. PaddleClas模型库概览：概要介绍PaddleClas有哪些分类网络结构和预训练模型。
1. PaddleClas柠檬竞赛实战：重点介绍数据增广方法。

* PaddleClas模型库概览

&emsp;&emsp;图像分类模型有大有小，其应用场景各不相同，在云端或者服务器端应用时，一般情况下算力是足够的，更倾向于应用高精度的模型；在手机、嵌入式等端侧设备中应用时，受限于设备的算力和内存，则对模型的速度和大小有较高的要求。PaddleClas同时提供了服务器端模型与端侧轻量化模型来支撑不同的应用场景。

<center><img src="https://ai-studio-static-online.cdn.bcebos.com/5fe0a0a051374ab984b2c029ced6ff154af201c25f7441a3bbbd1b30af83bd7c" width="600" ></center>

    
&emsp;&emsp;这里我们使用MobileNetV2模型，因为它在预测速度和性能上都具有很大的优势，而且符合我们此次竞赛实战的要求，用户可以根据预测耗时的要求选择不同的网络。此外，PaddleClas也开源了预训练模型，我们可以基于此在自己的数据集上进行微调，提升效果。

更多模型详细介绍和模型训练技巧，可查看[PaddleClas模型库文档](https://paddleclas.readthedocs.io/zh_CN/latest/models/index.html)。

###  一、前置条件

1. 安装[Python3.5或更高版本](https://www.python.org/downloads/)版本。   
2. 安装PaddlePaddle 1.7或更高版本，具体安装方法请参见[快速安装](https://www.paddlepaddle.org.cn/install/quick)。由于图像分类模型计算开销大，推荐在GPU版本的PaddlePaddle下使用PaddleClas。
3. 下载PaddleClas的代码库。
```
cd path_to_clone_PaddleClas

以下二者任选其一
git clone https://github.com/PaddlePaddle/PaddleClas.git
git clone https://gitee.com/paddlepaddle/PaddleClas.git

```
4. 安装Python依赖库。Python依赖库在requirements.txt中给出。（本地）

```
pip install --upgrade -r requirements.txt
```

5. 设置PYTHONPATH环境变量（本地）
```
export PYTHONPATH=path_to_PaddleClas:$PYTHONPATH
```
开始：
```python
!git clone https://gitee.com/paddlepaddle/PaddleClas.git
```
```python
!pip install -r PaddleClas/requirements.txt
```
###  二、准备数据集

[PaddleClas数据准备文档](https://paddleclas.readthedocs.io/zh_CN/latest/tutorials/data.html)提供了ImageNet1k数据集以及flowers102数据集的准备过程。当然，如果大家希望使用自己的数据集，则需要至少准备以下两份文件。

* 训练集图像，以图像文件形式保存。
* 训练集标签文件，以文本形式保存，每一行的文件都包含文件名以及图像标签，以空格隔开。下面给出一个示例。
```
ILSVRC2012_val_00000001.JPEG 65
...
```

如果需要在训练的时候进行验证，则也同时需要提供验证集图像以及验证集标签文件。

以训练集配置为例，配置文件中对应如下
```
TRAIN: # 训练配置
    batch_size: 32 # 训练的batch size
    num_workers: 4 # 每个trainer(1块GPU上可以视为1个trainer)的进程数量
    file_list: "./dataset/flowers102/train_list.txt" # 训练集标签文件，每一行由"image_name label"组成
    data_dir: "./dataset/flowers102/" # 训练集的图像数据路径
    shuffle_seed: 0 # 数据打散的种子
    transforms: # 训练图像的数据预处理
        - DecodeImage: # 解码
            to_rgb: True
            to_np: False
            channel_first: False
        - RandCropImage: # 随机裁剪
            size: 224
        - RandFlipImage: # 随机水平翻转
            flip_code: 1
        - NormalizeImage: # 归一化
            scale: 1./255.
            mean: [0.485, 0.456, 0.406]
            std: [0.229, 0.224, 0.225]
            order: ''
        - ToCHWImage: # 通道转换
```

其中`file_list`即训练数据集的标签文件，`data_dir`是图像所在的文件夹。
```python
# 解压数据集
!unzip data/data71799/lemon_lesson.zip

!unzip lemon_lesson/train_images.zip -d lemon_lesson/

!unzip lemon_lesson/test_images.zip -d lemon_lesson/
```
```python
# 自己切分数据集
import pandas as pd
import codecs
import os
from PIL import Image

df = pd.read_csv('lemon_lesson/train_images.csv')

all_file_dir = 'lemon_lesson'

train_file = codecs.open(os.path.join(all_file_dir, "train_list.txt"), 'w')
eval_file = codecs.open(os.path.join(all_file_dir, "eval_list.txt"), 'w')

image_path_list = df['id'].values
label_list = df['class_num'].values

# 划分训练集和校验集
all_size = len(image_path_list)
train_size = int(all_size * 0.8)
train_image_path_list = image_path_list[:train_size]
train_label_list = label_list[:train_size]
val_image_path_list = image_path_list[train_size:]
val_label_list = label_list[train_size:]

image_path_pre = 'lemon_lesson/train_images'

for file,label_id in zip(train_image_path_list, train_label_list):
    # print(file)
    # print(label_id)
    try:
        img = Image.open(os.path.join(image_path_pre, file))
        
        # train_file.write("{0}\0{1}\n".format(os.path.join(image_path_pre, file), label_id))
        train_file.write("{0}{1}{2}\n".format(os.path.join(image_path_pre, file),' ', label_id))
        # eval_file.write("{0}\t{1}\n".format(os.path.join(image_path_pre, file), label_id))
    except Exception as e:
        pass
        # 存在一些文件打不开，此处需要稍作清洗
        # print('error!')

for file,label_id in zip(val_image_path_list, val_label_list):
    # print(file)
    # print(label_id)
    try:
        img = Image.open(os.path.join(image_path_pre, file))
        # train_file.write("{0}\t{1}\n".format(os.path.join(image_path_pre, file), label_id))
        eval_file.write("{0}{1}{2}\n".format(os.path.join(image_path_pre, file),' ', label_id))
    except Exception as e:
        # pass
        # 存在一些文件打不开，此处需要稍作清洗
        print('error!')

train_file.close()
eval_file.close()
```
#### 用PaddleX API一键切分数据集
[数据标注、转换、划分 » 图像分类](https://paddlex.readthedocs.io/zh_CN/develop/data/annotation/index.html)
- 先把每个分类的数据集都拎出来，归属到各自目录下
- 再用PaddleX自带的API，一键切分数据集
```python
!cp -r lemon_lesson  MyDataset
```
```python
!pip install paddlex 
```
```python
import pandas as pd
import codecs
import os
from PIL import Image

df = pd.read_csv('MyDataset/train_images.csv')
image_path_list = df['id'].values
label_list = df['class_num'].values
```
```python
df.class_num.value_counts()
```
```python
!mkdir MyDataset/0
!mkdir MyDataset/1
!mkdir MyDataset/2
!mkdir MyDataset/3

# class_0
class_0 = df[df.class_num==0]
class_0_list = class_0['id'].values

import shutil
for i in class_0_list:
    try:
        shutil.copy(os.path.join('MyDataset/train_images', i), os.path.join('MyDataset/0', i))
    except Exception as e:
        pass

# class_1
class_1 = df[df.class_num==1]
class_1_list = class_1['id'].values

import shutil
for i in class_1_list:
    try:
        shutil.copy(os.path.join('MyDataset/train_images', i), os.path.join('MyDataset/1', i))
    except Exception as e:
        pass

# class_2
class_2 = df[df.class_num==2]
class_2_list = class_2['id'].values
import shutil
for i in class_2_list:
    try:
        shutil.copy(os.path.join('MyDataset/train_images', i), os.path.join('MyDataset/2', i))
    except Exception as e:
        pass
# class_3
class_3 = df[df.class_num==3]
class_3_list = class_3['id'].values
import shutil
for i in class_3_list:
    try:
        shutil.copy(os.path.join('MyDataset/train_images', i), os.path.join('MyDataset/3', i))
    except Exception as e:
        pass
```

```python
!ls MyDataset/0 -l |grep "^-"|wc -l
!ls MyDataset/1 -l |grep "^-"|wc -l
!ls MyDataset/2 -l |grep "^-"|wc -l
!ls MyDataset/3 -l |grep "^-"|wc -l
```
***
查看某文件夹下文件的个数：ls -l |grep "^-"|wc -l 或 find ./company -type f | wc -l

ls -l

长列表输出该目录下文件信息(注意这里的文件，不同于一般的文件，可能是目录、链接、设备文件等)

grep "^-"

这里将长列表输出信息过滤一部分，只保留一般文件，如果只保留目录就是 ^d

wc -l

统计输出信息的行数，因为已经过滤得只剩一般文件了，所以统计结果就是一般文件信息的行数，又由于

一行信息对应一个文件，所以也就是文件的个数。

***
```python
# 使用PaddleX数据切分API要删除多余目录和list文件，否则会出现异常
!rm -r MyDataset/train_images
!rm -r MyDataset/test_images
!rm MyDataset/*.txt
```

```python
!paddlex --split_dataset --format ImageNet --dataset_dir MyDataset --val_value 0.2 --test_value 0.1
```
迁移学习(Transfer learning) 顾名思义就是就是把已学训练好的模型参数迁移到新的模型来帮助新模型训练。考虑到大部分数据或任务是存在相关性的，所以通过迁移学习我们可以将已经学到的模型参数（也可理解为模型学到的知识）通过某种方式来分享给新模型从而加快并优化模型的学习效率不用像大多数网络那样从零学习（starting from scratch，tabula rasa）。

### 三、模型训练与评估

在自己的数据集上训练分类模型时，更推荐加载预训练进行微调。

预训练模型使用以下方式进行下载。

```
python tools/download.py -a MobileNetV3_small_x1_0 -p ./pretrained -d True
```

更多的预训练模型可以参考这里：[https://paddleclas.readthedocs.io/zh_CN/latest/models/models_intro.html](https://paddleclas.readthedocs.io/zh_CN/latest/models/models_intro.html)

PaddleClas 提供模型训练与评估脚本：`tools/train.py`和`tools/eval.py`
#### 3.1 模型训练
准备好配置文件之后，可以使用下面的方式启动训练。


```
python tools/train.py \
    -c configs/quick_start/MobileNetV3_large_x1_0_finetune.yaml \
    -o pretrained_model="" \
    -o use_gpu=True
```


其中，`-c`用于指定配置文件的路径，`-o`用于指定需要修改或者添加的参数，其中`-o pretrained_model=""`表示不使用预训练模型，`-o use_gpu=True`表示使用GPU进行训练。如果希望使用CPU进行训练，则需要将`use_gpu`设置为`False`。

更详细的训练配置，也可以直接修改模型对应的配置文件。

运行上述命令，可以看到输出日志，示例如下：

- 如果在训练中使用了mixup或者cutmix的数据增广方式，那么日志中只会打印出loss(损失)、lr(学习率)以及该minibatch的训练时间。
    
`train step:890  loss:  6.8473 lr: 0.100000 elapse: 0.157s`
    
- 如果训练过程中没有使用mixup或者cutmix的数据增广，那么除了loss(损失)、lr(学习率)以及该minibatch的训练时间之外，日志中也会打印出top-1与top-k(默认为5)的信息。
    
`epoch:0    train    step:13    loss:7.9561    top1:0.0156    top5:0.1094    lr:0.100000    elapse:0.193s`
    

训练期间也可以通过VisualDL实时观察loss变化。

```python
%cd PaddleClas/

!python tools/download.py -a MobileNetV3_large_x1_0 -p ./pretrained -d True

!cp ../MobileNetV3_large_x1_0_finetune.yaml configs/quick_start/MobileNetV3_large_x1_0_finetune.yaml
```
在AI Studio上查看可视化效果：参考[VisualDL文档](https://github.com/PaddlePaddle/VisualDL)
> 设置日志文件并记录标量数据：
> ```python
> from visualdl import LogWriter
> # 在`./log/scalar_test/train`路径下建立日志文件
> with LogWriter(logdir="./log/scalar_test/train") as writer:
>  # 使用scalar组件记录一个标量数据
>  writer.add_scalar(tag="acc", step=1, value=0.5678)
>  writer.add_scalar(tag="acc", step=2, value=0.6878)
>  writer.add_scalar(tag="acc", step=3, value=0.9878)
> ```

因此，训练前可以改造一下`tools/train.py`的代码，加入VisualDL可视化，比如这里，把每轮验证集上的准确率结果记录下来：
```python
    # 在`./logdir`路径下建立日志文件
    with LogWriter(logdir="./logdir") as writer:
        for epoch_id in range(last_epoch_id + 1, config.epochs):
            net.train()
            # 1. train with train dataset
            program.run(train_dataloader, config, net, optimizer, lr_scheduler,
                        epoch_id, 'train')

            # 2. validate with validate dataset
            if config.validate and epoch_id % config.valid_interval == 0:
                net.eval()
                with paddle.no_grad():
                    top1_acc = program.run(valid_dataloader, config, net, None,
                                        None, epoch_id, 'valid')
                if top1_acc > best_top1_acc:
                    best_top1_acc = top1_acc
                    best_top1_epoch = epoch_id
                    if epoch_id % config.save_interval == 0:
                        model_path = os.path.join(config.model_save_dir,
                                                config.ARCHITECTURE["name"])
                        save_model(net, optimizer, model_path, "best_model")
                message = "The best top1 acc {:.5f}, in epoch: {:d}".format(
                    best_top1_acc, best_top1_epoch)
                logger.info("{:s}".format(logger.coloring(message, "RED")))
                # 使用scalar组件记录一个标量数据
                writer.add_scalar(tag="val_acc", step=epoch_id, value=top1_acc)
```
```python
!cp ../train.py tools/train.py

# 开始训练
!python tools/train.py -c ../MobileNetV3_large_x1_0_finetune.yaml
```
#### 3.2 模型微调
[30分钟玩转PaddleClas](https://paddleclas.readthedocs.io/zh_CN/latest/tutorials/quick_start.html)中包含大量模型微调的示例，可以参考该章节进行模型微调。
#### 3.3 模型评估
可以更改configs/eval.yaml中的ARCHITECTURE.name字段和pretrained_model字段来配置评估模型，也可以通过-o参数更新配置。
>注意： 加载预训练模型时，需要指定预训练模型的前缀，例如预训练模型参数所在的文件夹为output/ResNet50_vd/19，预训练模型参数的名称为output/ResNet50_vd/19/ppcls.pdparams，则pretrained_model参数需要指定为output/ResNet50_vd/19/ppcls，PaddleClas会自动补齐.pdparams的后缀。
```python
!python tools/eval.py \
    -c ../MobileNetV3_large_x1_0_finetune.yaml \
    -o pretrained_model="./output/MobileNetV3_large_x1_0/best_model/ppcls"\
    -o load_static_weights=False
```
### 四、图像增广

下面这个流程图是图片预处理并被送进网络训练的一个过程，需要经过解码、随机裁剪、水平翻转、归一化、通道转换以及组batch，最终训练的过程。
<center><img src="https://ai-studio-static-online.cdn.bcebos.com/07cac4aa7feb4e9f8f595dce4ad1abd61d63d0f953e440b5bf28e588f85c58c7" width="600" ></center>
<br>

1. 图像变换类：图像变换类是在随机裁剪与翻转之间进行的操作，也可以认为是在原图上做的操作。主要方式包括**AutoAugment**和**RandAugment**，基于一定的策略，包括锐化、亮度变化、直方图均衡化等，对图像进行处理。这样网络在训练时就已经见过这些情况了，之后在实际预测时，即使遇到了光照变换、旋转这些很棘手的情况，网络也可以从容应对了。

2. 图像裁剪类：图像裁剪类主要是在生成的在通道转换之后，在图像上设置掩码，随机遮挡，从而使得网络去学习一些非显著性的特征。否则网络一直学习很重要的显著性区域，之后在预测有遮挡的图片时，泛化能力会很差。主要方式包括：**CutOut**、**RandErasing**、**HideAndSeek**、**GridMask**。这里需要注意的是，在通道转换前后去做图像裁剪，其实是没有区别的。因为通道转换这个操作不会修改图像的像素值。

3. 图像混叠类：组完batch之后，图像与图像、标签与标签之间进行混合，形成新的batch数据，然后送进网络进行训练。这也就是图像混叠类数据增广方式，主要的有**Mixup**与**Cutmix**两种方式。

> 参考资料：
> - [深度学习中几种常用增强数据的库](https://blog.csdn.net/weixin_44111292/article/details/108930130)
> - [imgaug](https://github.com/aleju/imgaug)
> - [Albumentations](https://github.com/albumentations-team/albumentations)
```python
# 原项目random_erasing.py有错误，相当于运行时候没进行擦除处理，这里进行了替换，然后运行train.py，就会进行擦除了
!cp ../random_erasing.py ppcls/data/imaug/random_erasing.py

!python tools/train.py -c ../MobileNetV3_large_x1_0_finetune.yaml
```
#### 4.2 离线数据增广
这是一个离线数据增广的脚本，可以帮助我们利用PaddleClas的自动数据增强功能，快速进行离线数据扩充
```python
!cp ../img_aug.py ./

!mkdir ../img_aug

!python img_aug.py
```
### 五、模型推理
首先，对训练好的模型进行转换：
```
python tools/export_model.py \
    --model=模型名字 \
    --pretrained_model=预训练模型路径 \
    --output_path=预测模型保存路径
```
之后，通过推理引擎进行推理：
```
python tools/infer/predict.py \
    -m model文件路径 \
    -p params文件路径 \
    -i 图片路径 \
    --use_gpu=1 \
    --use_tensorrt=True
```

更多的参数说明可以参考[https://github.com/PaddlePaddle/PaddleClas/blob/master/tools/infer/predict.py](https://github.com/PaddlePaddle/PaddleClas/blob/master/tools/infer/predict.py)中的`parse_args`函数。

更多关于服务器端与端侧的预测部署方案请参考：[https://www.paddlepaddle.org.cn/documentation/docs/zh/advanced_guide/inference_deployment/index_cn.html](https://www.paddlepaddle.org.cn/documentation/docs/zh/advanced_guide/inference_deployment/index_cn.html)

```python
# 注意要写入类别数
!python tools/export_model.py \
    --model=MobileNetV3_large_x1_0 \
    --pretrained_model=output/MobileNetV3_large_x1_0/best_model/ppcls \
    --output_path=inference \
    --class_dim 4 
```
```python
# 可以预测整个目录
!python tools/infer/predict.py \
    --model_file inference/inference.pdmodel \
    --params_file inference/inference.pdiparams \
    --image_file ../lemon_lesson/test_images \
    --use_gpu=True
```
### 六、输出预测结果
做一些小幅改造，让预测结果以`sample_submit.csv`的格式保存，便于提交。
```python
!cp ../predict.py tools/infer/submit.py
```
```python
# 可以预测整个目录
!python tools/infer/submit.py \
    --model_file inference/inference.pdmodel \
    --params_file inference/inference.pdiparams \
    --image_file ../lemon_lesson/test_images \
    --use_gpu=True
```
PaddleClas github地址：[https://github.com/PaddlePaddle/PaddleClas/](https://github.com/PaddlePaddle/PaddleClas/)

PaddleClas教程文档地址：[https://paddleclas.readthedocs.io/zh_CN/latest/index.html](https://paddleclas.readthedocs.io/zh_CN/latest/index.html)

## 作业
本实践旨在通过一个美食分类的案列，让大家理解和掌握如何使用飞桨2.0搭建一个卷积神经网络。

特别提示：本实践所用数据集均来自互联网，请勿用于商务用途。

解压文件，使用train.csv训练，测试使用val.csv。最后以在val上的准确率作为最终分数。


```python
# 解压数据集
!unzip -oq data/data72793/lemon_homework.zip -d data/
!unzip -oq data/lemon_homework/lemon_lesson.zip -d data/
!unzip -oq data/lemon_lesson/train_images.zip -d data/
!unzip -oq data/lemon_lesson/test_images.zip -d data/
```


```python
# 导入所需要的库
from sklearn.utils import shuffle
import os
import pandas as pd
import numpy as np
from PIL import Image

import paddle
import paddle.nn as nn
from paddle.io import Dataset
import paddle.vision.transforms as T
import paddle.nn.functional as F
from paddle.metric import Accuracy

import warnings
warnings.filterwarnings("ignore")
```


```python
# 读取数据
train_images = pd.read_csv('data/lemon_lesson/train_images.csv', usecols=['id','class_num'])
#这里读出的是全部训练集的'id','class_num'信息

# labelshuffling 这里是打乱顺序，我们是对存放图片名称的文本进行内容的打乱。

def labelShuffling(dataFrame, groupByName = 'class_num'):

    groupDataFrame = dataFrame.groupby(by=[groupByName])
    labels = groupDataFrame.size()
    print("length of label is ", len(labels))
    maxNum = max(labels)
    lst = pd.DataFrame()
    for i in range(len(labels)):
        print("Processing label  :", i)
        tmpGroupBy = groupDataFrame.get_group(i)
        createdShuffleLabels = np.random.permutation(np.array(range(maxNum))) % labels[i]
        print("Num of the label is : ", labels[i])
        lst=lst.append(tmpGroupBy.iloc[createdShuffleLabels], ignore_index=True)
        print("Done")
    # lst.to_csv('test1.csv', index=False)
    return lst
```


```python
# 划分训练集和校验集
all_size = len(train_images)
# print(all_size)
train_size = int(all_size * 0.8)
train_image_list = train_images[:train_size]
val_image_list = train_images[train_size:]
```


```python
df = labelShuffling(train_image_list)
df = shuffle(df)

train_image_path_list = df['id'].values
label_list = df['class_num'].values
label_list = paddle.to_tensor(label_list, dtype='int64')
train_label_list = paddle.nn.functional.one_hot(label_list, num_classes=4)

val_image_path_list = val_image_list['id'].values
val_label_list = val_image_list['class_num'].values
val_label_list = paddle.to_tensor(val_label_list, dtype='int64')
val_label_list = paddle.nn.functional.one_hot(val_label_list, num_classes=4)

# 定义数据预处理
data_transforms = T.Compose([
    T.Resize(size=(224)),
    T.RandomHorizontalFlip(224),
    T.RandomVerticalFlip(224),
    T.Transpose(),    # HWC -> CHW
    T.Normalize(
        mean=[0, 0, 0],        # 归一化
        std=[255, 255, 255],
        to_rgb=True)    
    ])
```

    length of label is  4
    Processing label  : 0
    Num of the label is :  321
    Done
    Processing label  : 1
    Num of the label is :  207
    Done
    Processing label  : 2
    Num of the label is :  181
    Done
    Processing label  : 3
    Num of the label is :  172
    Done



```python
# 构建Dataset
class MyDataset(paddle.io.Dataset):
    """
    步骤一：继承paddle.io.Dataset类
    """
    def __init__(self, train_img_list, val_img_list,train_label_list,val_label_list, mode='train'):
        """
        步骤二：实现构造函数，定义数据读取方式，划分训练和测试数据集
        """
        super(MyDataset, self).__init__()
        self.img = []
        self.label = []
        # 借助pandas读csv的库，将前面得到的标签，名称list分别赋值
        self.train_images = train_img_list
        self.test_images = val_img_list
        self.train_label = train_label_list
        self.test_label = val_label_list
        if mode == 'train':
            # 读train_images的数据，这里是读训练的，所以括号里的是train的label和images
            for img,la in zip(self.train_images, self.train_label):
                self.img.append('data/train_images/'+img)
                self.label.append(la)
        else:
            # 读test_images的数据，这里是读验证的，所以括号里的应该是是val的label和images。
            for img,la in zip(self.test_images, self.test_label):
                #注意这里，因为我们从头到这里处理的都是list文件，本质上的test是从train划分出来的，所以路径应该为train_images
                self.img.append('data/train_images/'+img)
                self.label.append(la)

    def load_img(self, image_path):
        # 实际使用时使用Pillow相关库进行图片读取即可，这里我们对数据先做个模拟
        image = Image.open(image_path).convert('RGB')
        return image

    def __getitem__(self, index):
        """
        步骤三：实现__getitem__方法，定义指定index时如何获取数据，并返回单条数据（训练数据，对应的标签）
        """
        image = self.load_img(self.img[index])
        label = self.label[index]
        # label = paddle.to_tensor(label)
        
        return data_transforms(image), paddle.nn.functional.label_smooth(label)

    def __len__(self):
        """
        步骤四：实现__len__方法，返回数据集总数目
        """
        return len(self.img)

#train_loader
train_dataset = MyDataset(train_img_list=train_image_path_list, val_img_list=val_image_path_list, train_label_list=train_label_list, val_label_list=val_label_list, mode='train')
train_loader = paddle.io.DataLoader(train_dataset, places=paddle.CPUPlace(), batch_size=32, shuffle=True, num_workers=0)

#val_loader
#注意括号里的参数和train_loader的区别
val_dataset = MyDataset(train_img_list=train_image_path_list, val_img_list=val_image_path_list, train_label_list=train_label_list, val_label_list=val_label_list, mode='test')
val_loader = paddle.io.DataLoader(val_dataset, places=paddle.CPUPlace(), batch_size=32, shuffle=True, num_workers=0)
```


```python
from paddle.vision import MobileNetV2

# 模型封装
network = paddle.vision.models.resnet101(num_classes=4,pretrained=True)
model = paddle.Model(network)

# 定义优化器

scheduler = paddle.optimizer.lr.LinearWarmup(
        learning_rate=0.5, warmup_steps=20, start_lr=0, end_lr=0.5, verbose=True)
#optim = paddle.optimizer.SGD(learning_rate=scheduler, parameters=model.parameters())
optim = paddle.optimizer.Adam(learning_rate=3e-4, parameters=model.parameters())

# 配置模型
model.prepare(
    optim,
    paddle.nn.CrossEntropyLoss(soft_label=True),
    Accuracy()
    )

#vdl回调函数
visualdl = paddle.callbacks.VisualDL(log_dir='visualdl_log')

# 模型训练与评估
model.fit(train_loader,
        val_loader,
        log_freq=1,
        epochs=10,
        save_dir='./chk_points/',
        callbacks=[visualdl],
        shuffle=True,
        verbose=1,
        )
```

    2021-03-10 12:16:59,589 - INFO - unique_endpoints {''}
    [INFO 2021-03-10 12:16:59,589 download.py:154] unique_endpoints {''}
    2021-03-10 12:16:59,591 - INFO - File /home/aistudio/.cache/paddle/hapi/weights/resnet101.pdparams md5 checking...
    [INFO 2021-03-10 12:16:59,591 download.py:251] File /home/aistudio/.cache/paddle/hapi/weights/resnet101.pdparams md5 checking...
    2021-03-10 12:17:00,234 - INFO - Found /home/aistudio/.cache/paddle/hapi/weights/resnet101.pdparams
    [INFO 2021-03-10 12:17:00,234 download.py:184] Found /home/aistudio/.cache/paddle/hapi/weights/resnet101.pdparams


    Epoch 0: LinearWarmup set learning rate to 0.0.
    The loss value printed in the log is the current step, and the metric is the average value of previous step.
    Epoch 1/10
    step 41/41 [==============================] - loss: 0.5962 - acc: 0.9533 - 353ms/step         
    save checkpoint at /home/aistudio/chk_points/0
    Eval begin...
    The loss value printed in the log is the current batch, and the metric is the average value of previous step.
    step 7/7 [==============================] - loss: 0.4073 - acc: 0.9955 - 300ms/step        
    Eval samples: 221
    Epoch 2/10
    step 41/41 [==============================] - loss: 0.3910 - acc: 0.9992 - 342ms/step        
    save checkpoint at /home/aistudio/chk_points/1
    Eval begin...
    The loss value printed in the log is the current batch, and the metric is the average value of previous step.
    step 7/7 [==============================] - loss: 0.3584 - acc: 1.0000 - 296ms/step        
    Eval samples: 221
    Epoch 3/10
    step 41/41 [==============================] - loss: 0.4890 - acc: 1.0000 - 341ms/step         
    save checkpoint at /home/aistudio/chk_points/2
    Eval begin...
    The loss value printed in the log is the current batch, and the metric is the average value of previous step.
    step 7/7 [==============================] - loss: 0.3664 - acc: 1.0000 - 297ms/step        
    Eval samples: 221
    Epoch 4/10
    step 41/41 [==============================] - loss: 0.4458 - acc: 1.0000 - 341ms/step         
    save checkpoint at /home/aistudio/chk_points/3
    Eval begin...
    The loss value printed in the log is the current batch, and the metric is the average value of previous step.
    step 7/7 [==============================] - loss: 0.3891 - acc: 0.9955 - 297ms/step        
    Eval samples: 221
    Epoch 5/10
    step 41/41 [==============================] - loss: 0.3666 - acc: 0.9945 - 344ms/step         
    save checkpoint at /home/aistudio/chk_points/4
    Eval begin...
    The loss value printed in the log is the current batch, and the metric is the average value of previous step.
    step 7/7 [==============================] - loss: 0.3622 - acc: 1.0000 - 298ms/step        
    Eval samples: 221
    Epoch 6/10
    step 41/41 [==============================] - loss: 0.5063 - acc: 0.9860 - 351ms/step        
    save checkpoint at /home/aistudio/chk_points/5
    Eval begin...
    The loss value printed in the log is the current batch, and the metric is the average value of previous step.
    step 7/7 [==============================] - loss: 0.3615 - acc: 1.0000 - 297ms/step        
    Eval samples: 221
    Epoch 7/10
    step 41/41 [==============================] - loss: 0.3690 - acc: 0.9992 - 343ms/step        
    save checkpoint at /home/aistudio/chk_points/6
    Eval begin...
    The loss value printed in the log is the current batch, and the metric is the average value of previous step.
    step 7/7 [==============================] - loss: 0.3646 - acc: 1.0000 - 300ms/step        
    Eval samples: 221
    Epoch 8/10
    step 41/41 [==============================] - loss: 0.3787 - acc: 1.0000 - 341ms/step         
    save checkpoint at /home/aistudio/chk_points/7
    Eval begin...
    The loss value printed in the log is the current batch, and the metric is the average value of previous step.
    step 7/7 [==============================] - loss: 0.3562 - acc: 1.0000 - 300ms/step        
    Eval samples: 221
    Epoch 9/10
    step 41/41 [==============================] - loss: 0.4566 - acc: 1.0000 - 351ms/step         
    save checkpoint at /home/aistudio/chk_points/8
    Eval begin...
    The loss value printed in the log is the current batch, and the metric is the average value of previous step.
    step 7/7 [==============================] - loss: 0.3559 - acc: 1.0000 - 300ms/step        
    Eval samples: 221
    Epoch 10/10
    step 41/41 [==============================] - loss: 0.3803 - acc: 1.0000 - 351ms/step         
    save checkpoint at /home/aistudio/chk_points/9
    Eval begin...
    The loss value printed in the log is the current batch, and the metric is the average value of previous step.
    step 7/7 [==============================] - loss: 0.3550 - acc: 1.0000 - 297ms/step        
    Eval samples: 221
    save checkpoint at /home/aistudio/chk_points/final



```python
# 模型保存
model.save('MyCNN', False)
```

##  {'loss': [0.3568458], 'acc': 1.0}
### 验证集 精度满分，但是从后面 test_images的测试结果还是会有误判的。


```python
# 模型评价
model.evaluate(val_loader)
```

    Eval begin...
    The loss value printed in the log is the current batch, and the metric is the average value of previous step.
    step 7/7 - loss: 0.3568 - acc: 1.0000 - 310ms/step
    Eval samples: 221





    {'loss': [0.3568458], 'acc': 1.0}




```python
# 模型预测
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
    img = img.resize((224, 224), Image.BILINEAR) #Image.BILINEAR双线性插值
    img = np.array(img).astype('float32')

    # HWC to CHW 
    img = img.transpose((2, 0, 1))
    
    #Normalize  
    # img = img / 255         #像素值归一化
    mean = [0, 0, 0]   
    std = [255, 255, 255]
    img[0] = (img[0] - mean[0]) / std[0]
    img[1] = (img[1] - mean[1]) / std[1]
    img[2] = (img[2] - mean[2]) / std[2]
    
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
    label_pre = []
    for i in range(len(infer_imgs)):
        data = infer_imgs[i]
        dy_x_data = np.array(data).astype('float32')
        dy_x_data = dy_x_data[np.newaxis,:, : ,:]
        img = paddle.to_tensor(dy_x_data)
        out = model(img)

        # print(out[0])
        # print(paddle.nn.functional.softmax(out)[0]) # 若模型中已经包含softmax则不用此行代码。

        lab = np.argmax(out.numpy())  #argmax():返回最大数的索引
        label_pre.append(lab)
        # print(lab)
        print("样本: {}, 预测结果:{}\n".format(path, label_list[lab]))
    return label_pre
    
infer_img(path='data/test_images/test_1604.jpg',model_file_path='MyCNN',use_gpu=1)

infer_img(path='data/train_images/train_0000.jpg',model_file_path='MyCNN',use_gpu=1)
```


![png](output_10_0.png)


    样本: data/test_images/test_1604.jpg, 预测结果:3:規格外
    



![png](output_10_2.png)


    样本: data/train_images/train_0000.jpg, 预测结果:0:優良
    

```python
# 生成csv文件
import os
import csv
zemax = []
def create_csv_2(dirname):
    path = 'data/'+ dirname +'/' 
    name = os.listdir(path)
    # 对测试集文件名排序，test_0257.jpg，split出来数字字符，int转为整数
    name.sort(key=lambda x:  (int(x.split('.')[0].split('_')[1])))
    with open (dirname+'.csv','w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['id', 'label'])
        for n in name:
            if n[-4:] == '.jpg':
                print('测试样本：',n,)
                zemax = infer_img(path='data/test_images/'+ n,model_file_path='MyCNN',use_gpu=1)
                writer.writerow([str(n),zemax[0]])
            else:
                pass
            

create_csv_2('test_images')
```

    测试样本： test_0000.jpg



![png](output_12_1.png)


    样本: data/test_images/test_0000.jpg, 预测结果:0:優良
    
    测试样本： test_0001.jpg



![png](output_12_3.png)


    样本: data/test_images/test_0001.jpg, 预测结果:3:規格外
    
    测试样本： test_0002.jpg



![png](output_12_5.png)


    样本: data/test_images/test_0002.jpg, 预测结果:0:優良
    
    测试样本： test_0003.jpg



![png](output_12_7.png)


    样本: data/test_images/test_0003.jpg, 预测结果:3:規格外
    
    测试样本： test_0004.jpg



![png](output_12_9.png)


    样本: data/test_images/test_0004.jpg, 预测结果:0:優良
    
    测试样本： test_0005.jpg



![png](output_12_11.png)


    样本: data/test_images/test_0005.jpg, 预测结果:0:優良
    
    测试样本： test_0006.jpg



![png](output_12_13.png)


    样本: data/test_images/test_0006.jpg, 预测结果:0:優良
    
    测试样本： test_0007.jpg



![png](output_12_15.png)


    样本: data/test_images/test_0007.jpg, 预测结果:3:規格外
    
    测试样本： test_0008.jpg

总结：
* 代码是根据课件改的，使用了课件中的：label shuffling、独热编码、Adam学习率init_lr=3e-4、图像增强。另外把测试集的归一化改成和训练集一样了。
* 归一化 mean,std 数值的代码。上面没用到，看其他人文档记录下来的。

```python
import numpy as np
import cv2
import os
 
img_h, img_w = 640, 640   #图片大小
means, stdevs = [], []
img_list = []
 
imgs_path = 'data/train_images'  #训练集图片路径
imgs_path_list = os.listdir(imgs_path)  #返回指定的文件夹包含的文件或文件夹的名字的列表
 
for item in imgs_path_list:
    img = cv2.imread(os.path.join(imgs_path,item))
    img = cv2.resize(img,(img_w,img_h))
    img = img[:, :, :, np.newaxis] #增加第四维度
    img_list.append(img)

imgs_path = 'data/test_images' #测试集图片路径
imgs_path_list = os.listdir(imgs_path)

for item in imgs_path_list:
    img = cv2.imread(os.path.join(imgs_path,item))
    img = cv2.resize(img,(img_w,img_h))
    img = img[:, :, :, np.newaxis]
    img_list.append(img)

imgs = np.concatenate(img_list, axis=3)
imgs = imgs.astype(np.float32) / 255.
 
for i in range(3):
    pixels = imgs[:, :, i, :].ravel()  # 将多维数组转换为一维数组
    means.append(np.mean(pixels))
    stdevs.append(np.std(pixels))
 
# BGR --> RGB ， CV读取的需要转换，PIL读取的不用转换
means.reverse()
stdevs.reverse()
 
print("normMean = {}".format(means))
print("normStd = {}".format(stdevs))
```
** 代码中的一些点
```python
# np.newaxis 创建维度
a=np.array([1,2,3,4,5])
aa=a[:,np.newaxis]

print(a)
print(a.shape)
print(aa.shape)
print (aa)

# 输出：
# [1 2 3 4 5]
# (5,)
# (5, 1)
# [[1]
#  [2]
#  [3]
#  [4]
#  [5]]
```
```python
#np.concatenate拼接数组。axis就是对应的维度。https://www.cnblogs.com/rrttp/p/8028421.html
import numpy as np
 
a1 = np.asarray(range(0,6),dtype=int)
a2 = np.asarray(range(0,6),dtype=int)
aa1= a1.reshape([1,2,3])
aa2 =a2.reshape([1,2,3])
 
print("aa1,aa2:\n",aa1,"\n")
print("aa1,aa2_shape:\n",aa1.shape,"\n") #(1, 2, 3)

k1 = np.concatenate((aa1,aa2),axis=0)
k2 = np.concatenate((aa1,aa2),axis=1)
k3 = np.concatenate((aa1,aa2),axis=2)
print("k1:{}\n k2:{}\n k3:{}\n".format(k1,k2,k3))
 
# print(k1.shape) #(2, 2, 3)
# print(k2.shape) #(1, 4, 3)
# print(k3.shape) #(1, 2, 6)

# 输出：
# aa1,aa2:
#  [[[0 1 2]
#   [3 4 5]]] 

# aa1,aa2_shape:
#  (1, 2, 3) 

# k1:[[[0 1 2]
#   [3 4 5]]

#  [[0 1 2]
#   [3 4 5]]]
#  k2:[[[0 1 2]
#   [3 4 5]
#   [0 1 2]
#   [3 4 5]]]
#  k3:[[[0 1 2 0 1 2]
#   [3 4 5 3 4 5]]]
```

问题：
* 随机翻转：T.RandomHorizontalFlip(224)，T.RandomVerticalFlip(224)，其中参数224不知道原因，一般默认是概率p，但是224就不得而知了。

尝试对单张图片处理，没效果。先存着疑问。
```python
import paddle.vision.transforms as T
from PIL import Image
from matplotlib import pyplot as plt
%matplotlib inline

img = Image.open('lena.jpg')
data_transforms = T.Compose([
    T.Resize(size=(224)),
    T.RandomHorizontalFlip(224),
    T.RandomVerticalFlip(224),
    T.Transpose(),    # HWC -> CHW
    T.Normalize(
        mean=[0, 0, 0],        # 归一化
        std=[255, 255, 255],
        to_rgb=True)    
    ])
data_transforms(img)
plt.imshow(img)
plt.show()
```


