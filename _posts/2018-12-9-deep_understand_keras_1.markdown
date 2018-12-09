---
layout:     post
title:      "深入理解keras系列（一）"
subtitle:   "keras线性堆叠模型"
date:       2018-12-08 12:00:00
author:     "Albert·Leiming·Du"
header-img: "img/post-bg-2015.jpg"
catalog: true
tags:
    - Keras
    - tf.keras
    - tensorflow
---

> “We are always the dust inside the large universe.”

# 1. 前言

[Keras](https://keras.io/)作为一个非常流行的python环境下的深度学习库，被广大科研工作者和机器学习爱好者广泛使用，但放眼网路上，除了官方的文档以外，系统总结Keras使用方法的文章寥寥可数。因此笔者推出《深入理解Keras系列》，将从源码剖析和使用注意事项两个方面系统总结keras使用过程中的一些技巧以及可能会遇到的一些问题。笔者功力尚浅，如有任何错误或者混淆不清的地方，欢迎各位在评论区或者邮件指正。

在官方Keras的包中，keras底层同时对[Tensorflow](https://www.tensorflow.org/), [Theano](https://pytorch.org/)和[CNTK](https://www.microsoft.com/en-us/cognitive-toolkit/)这几个非常流行的深度学习库进行了支持。由于笔者日常使用Tensorflow，因而此系列文章中将从**Tensorflow**的角度进行剖析。而由于主要剖析的Api几乎不涉及底层支持，因此得到的结论可以同时适用于于官方Keras及tf.keras。
# 2. tf.keras介绍

tf.keras与tf.estimator, tf.data并称为Tensorflow三大高阶Api，由于其高度的集成性和扩展性以及较低的入门成本，受到了广大工作者的追捧。tf.keras作为其中的佼佼者，支持非常广泛的数据输入(numpy array, tensor, tf.data.dataset or dataset iterator, generator), 非常简单的模型部署和分布式扩展以及非常高的callbacks的扩展。尤为重要的是，tf.keras对使用train, val, test 分类方式的数据有了原生的early stopping的支持。

tf.keras的模型主要分为两种，Sequential方式（**笔者称为线性模式**）的和Subclass方式的模型，第一种模型非常容易使用，命令也很简单，但对模型的自定义性支持不足，例如对**residual network等跨层连接和多输入**几乎不支持。第二种方式具有很高的扩展性，用户可以非常随意地定义自己的模型，但代码上比较繁琐，且需对tf.keras有深入理解后才能写出bug-free的模型。

# 3. tf.keras线性模型

## 3.1 tf.keras.Sequential() 使用

使用tf.keras.Sequential() 构建模型，模型必须是**线性累加**的，即各层只与相邻的层有联系，无法创建诸如resudual network等网络。

一个最简单的全连接网络建立如下：
```python
import tensorflow as tf
import numpy as np

#Create the numpy dataset
data = np.random.random((1000,30))
label = np.random.random((1000,10))
val_data = np.random.random((100,30))
val_label = np.random.random((100,10))

#Create the tf.data object
dataset = tf.data.Dataset.from_tensor_slices((data, label)).batch(32)
val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_label)).batch(32)

#Create the model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(64, activation='relu'))  # add the first layer
model.add(tf.keras.layers.Dense(10, activation='softmax'))   # add the second layer

#Compile the model
model.compile(optimizer=tf.train.AdamOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#Train and validate with numpy data and label
model.fit(data, label, epochs=10, batch_size=10, 
		validation_data=(val_data, val_label))

#Train and validate with tf.data object
model.fit(dataset, epochs=10, steps_per_epoch=30,
		validation_data=val_dataset, validation_steps=10)

#Evaluate with numpy data
model.evaluate(val_data, val_label, batch_size=10)

#Evaluate with tf.data data
model.evaluate(val_dataset, steps=10)

#Predict with numpy data
model.predict(val_data, batch_size=10)

#Predict with tf.data data
model.predict(val_dataset, steps=10)
 ```
上述代码即为tf.Sequential()的一般使用方式, 一般的使用步骤包括建立pipeline, add模型，compile模型，训练模型(model.fit)，评估模型(model.evaluate),预测模型（model.predict)。

代码中出现了两种数据pipeline的建立方式，不同的数据pipeline方式下需采取不同的策略和调用方法。
* * *
### numpy 输入
1. 使用numpy 建立的输入在fit的时候需要使用`model.fit(x=data, y=label)`方式进行数据输入。
2. 使用model.fit的时候必须设置**epochs**和**batch_size**参数。 
3. 使用model.fit的时候**validation_data**必须设置为如`(data, label)`tuple的样式以用来匹配参数。
4. 由于numpy的数据会全部驻留在内存中，所以numpy的数据输入方式值适合于小批量的数据注入。

#### tf.keras对numpy数据的实际处理方式
1. tf.keras.Sequential() 模块继承了tf.keras.Model()模块，tf.keras.Sequential()初始化之后，在使用**第一次**调用`model.add()`添加层的时候，如果有`tf.keras.layers.Input()`模块作为第一层，则会使用此层的输入来初始化整个model的输入，整个model的输入的shape由此input层决定。model的输出由最后一层的输出决定。对应到底层就是在图中建立输入和输出节点。
2. 如果没有`tf.keras.layers.Input()`模块作为第一层，则会在调用`model.fit(data, label)`的时候，会调用`self.self._standardize_user_data(data, label)`函数设置模型的输入节点，输入节点为相同shape的`tf.placeholder`。输出节点则会在调用compile函数的时候进行设置。

* * *
### tf.data 输入
1. 使用`tf.data`设置输入的时候，调用`model.fit(x=dataset, y=None)`进行数据输入。
2. 使用model.fit的时候需设置**steps_per_epoch**参数，从而确定一个batch的范围。
3. 使用tf.data可以训练大规模的数据，一般通过使用`tf.data.Dataset.from_tensor_slices()`或者`
tf.data.TFRecordDataset()`进行数据的读取。

#### tf.keras对tf.data数据的实际处理方式
1. 对layer的处理方式与上一节numpy的处理方式相同。
2. 在调用`model.fit(data, label)`的时候，会调用`self.self._standardize_user_data(data, label)`函数设置模型的输入节点, 会将此tf.data(tensor)作为network的input节点。此处不涉及重新建立`tf.placeholder`节点，用的是原生的tf.data的tensor节点。

* * *
## 3.2 tf.keras函数式模型
上一节介绍了采用tf.Sequential()进行添加层的模型，本节介绍使用tf.keras.Model()产生的函数式模型。
基本代码如下：
```python
import tensorflow as tf
import numpy as np

inputs = tf.keras.Input(shape=(32,))  # Returns a placeholder tensor

#A layer instance is callable on a tensor, and returns a tensor.
x = tf.keras.layers.Dense(64, activation='relu')(inputs)

x = x + x  #This fucntion would not work.

outputs = tf.keras.layers.Dense(10, activation='softmax')(x)

#Build the model
model = tf.keras.Model(inputs=inputs, outputs=outputs)

#Compile model
model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#Train with numpy
model.fit(data, label, batch_size=32, epochs=5)

#Train with tf.dada
model.fit(dataset, steps_per_epoch=10, validation_data=val_dataset)
```
* * *
### 函数式编程的特点：
1. 使用函数式编程更加灵活，但需要提前定义输入的shape，即使用`tf.keras.layers.Input()`定义输入模型。
2. 函数式编程中，每一层只能是继承`tf.layers.Layer()`的函数，否则会导致模型无法通过编译。诸如代码中`x=x+x`的函数将无法通过编译。

### 函数式编程的原理:
代码中的`tf.keras.Model()`实际上继承的是`tf.keras.Network()`模块，`inputs=inputs, outputs=outputs`会作为**变长参数**传递给Network模块的初始化函数，在Network模块的初始化过程中，inputs和outputs的shape会被解析，进而利用inputs和outputs的shape定义图的输入和输出节点。
具体设计的源码为：
```python
class Network(base_layer.Layer):
  """A `Network` is a composition of layers.

  It is the topological form of a "model". A `Model`
  is simply a `Network` with added training routines.
  """

  def __init__(self, *args, **kwargs):  # pylint: disable=super-init-not-called
    # Signature detection
    if (len(args) == 2 or
        len(args) == 1 and 'outputs' in kwargs or
        'inputs' in kwargs and 'outputs' in kwargs):
      # Graph network
      self._init_graph_network(*args, **kwargs)
    else:
      # Subclassed network
      self._init_subclassed_network(**kwargs)
```
* * *
# 4. 总结：
1. 使用线性模型，现阶段只支持单输入模型，即输入只能以一个numpy array或者Tensor的模式，无法传递tuple模式的输入。
2. tf.keras可以使用`tf.keras.layers.Input()`定义输入节点，也可以不定义，在传入数据使用`model.fit()`根据输入数据的类型和shape定义图中的输入和输出节点。归根结底是利用提前设置好的inputs和outputs的shape在图内同样shape的输入和输出节点的placeholder,最终在fit的时候实现数组的feed。
3. 关于使用Tensor作为输入和输出，笔者会在后序系列中专门剖析keras如何实现这些操作。

-----------Leiming 2018-12-09 于SG


