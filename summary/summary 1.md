# 百度飞桨领航团零基础图像分类速成营 课程总结1（课前预习）
![paddlepaddle](https://paddlepaddle-org-cn.cdn.bcebos.com/paddle-site-front/favicon-128.png  "百度logo")

[课程链接](https://aistudio.baidu.com/aistudio/course/introduce/11939)	`https://aistudio.baidu.com/aistudio/course/introduce/11939`  
[飞桨官网](https://www.paddlepaddle.org.cn/)	`https://www.paddlepaddle.org.cn/`   

****
## 目录
* [图像处理的概念与基本操作](#图像处理的概念与基本操作)
* [OpenCV库进阶操作](#OpenCV库进阶操作)
* [图像分类任务概念导入](#图像分类任务概念导入)
* [PaddleClas数据增强代码解析](#PaddleClas数据增强代码解析)
* [预习作业](#预习作业)


# 课节1：课前资料

## python 的__call__()方法


对于定义的类，在Python中会创建一个MRO(Method Resolution Order)列表，它代表了类继承的顺序。查看MRO列表：

print C.mro()

调用父类(超类)的方法：

1. 明确指定：
```python
class  C(P):
     def __init__(self):
             P.__init__(self)
             print 'calling Cs construtor'
 ```
2. 使用super()方法 ：
```python
class  C(P):
    def __init__(self):
            super(C,self).__init__()
            print 'calling Cs construtor'
 
c=C()
```
* 能够在多继承中按照广度优先的继承树搜索关系链来有序地调用与父类相同名称的方法，且在每个子类拥有super的都会执行父类方法



## python 的__call__()方法


## python 的__call__()方法


## python 的__call__()方法
