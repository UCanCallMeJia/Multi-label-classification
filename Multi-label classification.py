from keras.layers import Dense,Dropout,BatchNormalization,Input
from keras.models import Model
import numpy as np 
import random

random.seed(1)
np.random.seed(1)

def get_data():
    x = [
            [1,1,1,0,1,0,1,0],
            [1,1,1,0,1,0,0,1],
            [1,1,1,0,0,1,1,0],
            [1,1,1,0,0,1,0,1],
            [1,0,0,1,1,0,0,0],
            [1,0,0,1,0,1,0,0],
            [1,1,1,1,1,1,1,0],
            # [1,1,1,1,1,1,1,0],
            [1,1,1,1,1,1,0,1]
            # [1,1,1,1,1,1,0,1]
        ]

    y = [
            [1,0,0,0,0,0,0,0],
            [0,1,0,0,0,0,0,0],
            [0,0,1,0,0,0,0,0],
            [0,0,0,1,0,0,0,0],
            [0,0,0,0,1,0,0,0],
            [0,0,0,0,0,1,0,0],
            [1,0,0,0,0,1,0,0],
            #[0,0,1,0,1,0,0,0],
            [0,1,0,0,0,1,0,0]
            #[0,0,0,1,1,0,0,0]
        ]


    a =[]
    for i in range(10):
        x_no_action_primitives = [0,0,0,0,random.randint(0,1),random.randint(0,1),random.randint(0,1),random.randint(0,1)]
        a.append(x_no_action_primitives)
    a = np.unique(a, axis=0)
    print(len(a))
    for i in range(len(a)):
        x.append(a[i])
        y.append([0,0,0,0,0,0,0,0])


    a =[]
    for i in range(20):
        x_only_move = [1,0,0,0,random.randint(0,1),random.randint(0,1),random.randint(0,1),random.randint(0,1)]
        a.append(x_only_move)
    a = np.unique(a, axis=0)
    print(len(a))
    for i in range(len(a)):
        x.append(a[i])
        y.append([0,0,0,0,0,0,0,0])


    # a =[]
    # for i in range(20):
    #     x_no_move = [0,random.randint(0,1),random.randint(0,1),random.randint(0,1),random.randint(0,1),random.randint(0,1),random.randint(0,1),random.randint(0,1)]
    #     a.append(x_no_move)
    # a = np.unique(a, axis=0)
    # print(len(a))
    # for i in range(len(a)):
    #     x.append(a[i])
    #     y.append([0,0,0,0,0,0,0,0])

    # a =[]
    # for i in range(20):
    #     x_no_objs = [random.randint(0,1),random.randint(0,1),random.randint(0,1),random.randint(0,1),0,0,0,0]
    #     a.append(x_no_objs)
    # a = np.unique(a, axis=0)
    # print(len(a))
    # for i in range(len(a)):
    #     x.append(a[i])
    #     y.append([0,0,0,0,0,0,0,0])

    print(len(x),len(y))
    # print(x,y)
    # input()

    # print(x, '\n', y)
    x = np.asarray(x, dtype=np.float)
    y = np.asarray(y, dtype=np.int64)
    # print(x.dtype, type(x))
    # print(y.dtype, type(y))
    # print(x.shape, '\n\n', y.shape)

    np.random.seed(1)
    state = np.random.get_state()
    np.random.shuffle(x)
    # print(x)

    np.random.set_state(state)
    np.random.shuffle(y)
    # print(y)

    TRAIN_NUM = 20
    x_train, y_train, x_test, y_test = x[0:TRAIN_NUM], y[0:TRAIN_NUM], x[TRAIN_NUM:], y[TRAIN_NUM:]
    print(x_train.shape, y_train.shape)
    return x_train, y_train, x_test, y_test

def get_model():

    inputs = Input((8,))
    x = Dense(20, activation='relu')(inputs)
    x = Dense(12, activation='relu')(x)
    out = Dense(8, activation='sigmoid')(x)
    fc_model = Model(inputs=inputs, outputs=out)
    return fc_model


def main():
    x_train, y_train, x_test, y_test = get_data()
    model = get_model()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=1, epochs=100, verbose=1)
    results = model.predict(x_train)
    # print(results)
    for result,label in zip(results,y_train):
        tmp = np.array([0,0,0,0,0,0,0,0])
        indexs = np.where(result>0.5)
        for index in indexs:
            tmp[index] = 1
        print('pred result: \n', tmp)
        print('True label: \n', label)
        print('\n')
        # input()
    # print('True:', y_test)

if __name__ == "__main__":
    main()