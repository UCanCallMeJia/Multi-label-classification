# Multi-label Classfication Problems  
## 多分类问题&多标签分类问题
### A --> Multi-class classification problem:
#### Problem Statement
> &nbsp; &nbsp; &nbsp;  多分类问题更加常见。相对于最简单的二分类问题（一个样本只具有一个属性，需要判断的只是0-1问题），多分类问题就是整个样本空间对应的所有类别总数大于2的分类问题。 eg: MNIST手写数字识别中有（0-9）10个标签，就是一个多分类问题。  
> &nbsp; &nbsp; &nbsp;  对于Multi-class classfication Problem, 我们的单个样本输入，对应着N个标签中的一个，往往网络的最后一层使用softmax来表示该标签对应的概率。进而计算交叉熵损失。

### B --> Multi-label classification problem:
#### Problem Statement
> &nbsp; &nbsp; &nbsp;  相比与Multi-class，简单概括Multi-label就是一个样本对应了多个属性，需要给一个样本在多个属性上做分类。  
> &nbsp; &nbsp; &nbsp;  例如：我们有许多穿着各种衣服的模特全身照片，需要准确识别出模特的衣着特征：什么颜色的衣服、什么颜色的裤子、什么类型的鞋子、带没带帽子等等...  
> &nbsp; &nbsp; &nbsp;  这种情况下，设计好几个NN来分别解决每个问题的思路是可行的，但是，我们在使用模型做测试的时候，还需要将各个模型的结果融合在一起，试想：如果分类种类很多很多 那这种方法会给开发人员带来极大的麻烦。Multi-label classification 中将这个复杂的问题做了一个简化：把多标签分类问题变成一个多个binary classification问题的组合。
> &nbsp; &nbsp; &nbsp; 例如:
> ```python
> # 假设共有八个属性，0代表不具备该属性，1代表具备该属性
> label_0 = [0, 1, 1, 1, 0, 1, 1, 0]
> ```
> &nbsp; &nbsp; &nbsp; 损失函数使用binary crossentropy，在每一个属性上计算一个2分类损失，加起来作为做后的损失，计算梯度反向传播。

## Example  
`structured_action_planner.py`
1. 设计简单的实验数据：
```python
# input 是一个长度为8的一维向量。
x = [[1,1,1,0,1,0,1,0],
    [1,1,1,0,1,0,0,1],
    [1,1,1,0,0,1,1,0],
    [1,1,1,0,0,1,0,1],
    [1,0,0,1,1,0,0,0],
    [1,0,0,1,0,1,0,0],
    [1,1,1,1,1,1,1,0],
    [1,1,1,1,1,1,0,1]]

# label 长度为8，即每个样本具有8个分类属性，0代表不具有该属性，1代表具有。
y = [[1,0,0,0,0,0,0,0],
    [0,1,0,0,0,0,0,0],
    [0,0,1,0,0,0,0,0],
    [0,0,0,1,0,0,0,0],
    [0,0,0,0,1,0,0,0],
    [0,0,0,0,0,1,0,0],
    [1,0,0,0,0,1,0,0],
    [0,1,0,0,0,1,0,0]]
```
2. 网络
```python
def get_model():
    # 输入长度8
    inputs = Input((8,))
    # 简单的全连接层
    x = Dense(20, activation='relu')(inputs)
    x = Dense(12, activation='relu')(x)
    # NOTICE！：这里使用sigmoid激活函数，将输出在每一维度上映射到0-1空间。
    out = Dense(8, activation='sigmoid')(x)
    fc_model = Model(inputs=inputs, outputs=out)
    return fc_model
```
3. 训练及测试
```python
def train_and_evaluate():
    x_train, y_train, x_test, y_test = get_data()
    model = get_model()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=1, epochs=100, verbose=1)
    results = model.predict(x_train)
    # print(results)
    for result,label in zip(results,y_train):
        tmp = np.array([0,0,0,0,0,0,0,0])
        # 这里我们将输出值 >0.5 认为标签为1，否则为0
        indexs = np.where(result>0.5)
        for index in indexs:
            tmp[index] = 1
        print('pred result: \n', tmp)
        print('True label: \n', label)
        input()
    # print('True:', y_test)
```
4. 训练&测试结果
#### 训练结果：
```python
Epoch 90/100
20/20 [==============================] - 0s 2ms/step - loss: 0.0521 - accuracy: 0.9750
Epoch 91/100
20/20 [==============================] - 0s 2ms/step - loss: 0.0512 - accuracy: 0.9750
Epoch 92/100
20/20 [==============================] - 0s 2ms/step - loss: 0.0511 - accuracy: 0.9750
Epoch 93/100
20/20 [==============================] - 0s 2ms/step - loss: 0.0497 - accuracy: 0.9812
Epoch 94/100
20/20 [==============================] - 0s 2ms/step - loss: 0.0491 - accuracy: 0.9875
Epoch 95/100
20/20 [==============================] - 0s 2ms/step - loss: 0.0479 - accuracy: 0.9875
Epoch 96/100
20/20 [==============================] - 0s 2ms/step - loss: 0.0473 - accuracy: 0.9875
Epoch 97/100
20/20 [==============================] - 0s 2ms/step - loss: 0.0464 - accuracy: 0.9875
Epoch 98/100
20/20 [==============================] - 0s 2ms/step - loss: 0.0457 - accuracy: 0.9937
Epoch 99/100
20/20 [==============================] - 0s 2ms/step - loss: 0.0448 - accuracy: 0.9937
Epoch 100/100
20/20 [==============================] - 0s 2ms/step - loss: 0.0442 - accuracy: 0.9937
```
#### 测试结果：
```python
pred result: 
 [0 0 0 1 0 0 0 0]
True label: 
 [0 0 0 1 0 0 0 0]

pred result: 
 [0 0 0 0 0 0 0 0]
True label: 
 [0 0 0 0 0 0 0 0]

pred result: 
 [0 0 0 0 1 0 0 0]
True label: 
 [0 0 0 0 1 0 0 0]

pred result: 
 [0 0 1 0 0 0 0 0]
True label: 
 [0 0 1 0 0 0 0 0]

pred result: 
 [1 0 0 0 0 1 0 0]
True label: 
 [1 0 0 0 0 1 0 0]

pred result: 
 [0 1 0 0 0 0 0 0]
True label: 
 [0 1 0 0 0 1 0 0]

pred result: 
 [0 1 0 0 0 0 0 0]
True label: 
 [0 1 0 0 0 0 0 0]

pred result: 
 [1 0 0 0 0 0 0 0]
True label: 
 [1 0 0 0 0 0 0 0]
```
`在这么简单的数据集上准确率还可以 ^~^`  
Contact:  jiazx@buaa.edu.cn
