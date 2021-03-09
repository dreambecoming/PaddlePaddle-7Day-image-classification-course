# 百度飞桨领航团零基础图像分类速成营 课程总结5
![paddlepaddle](https://paddlepaddle-org-cn.cdn.bcebos.com/paddle-site-front/favicon-128.png  "百度logo")

[课程链接](https://aistudio.baidu.com/aistudio/course/introduce/11939)	`https://aistudio.baidu.com/aistudio/course/introduce/11939`  
[飞桨官网](https://www.paddlepaddle.org.cn/)	`https://www.paddlepaddle.org.cn/`   
[推荐学习网站](https://www.runoob.com/python3/python3-tutorial.html)	`https://www.runoob.com/python3/python3-tutorial.html`  

****
## 目录
* [竞赛全流程](#竞赛全流程)
* [调参](#调参)
* [建模实战](#建模实战)
* [作业](#作业)
## 参考资料
* [关于LeNet的前世今生](https://www.jiqizhixin.com/graph/technologies/6c9baf12-1a32-4c53-8217-8c9f69bd011b)



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
