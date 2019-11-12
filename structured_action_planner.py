import numpy as np 
import random
np.random.seed(1)
random.seed(1)
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


    # a =[]
    # for i in range(20):
    #     x_only_move = [1,0,0,0,random.randint(0,1),random.randint(0,1),random.randint(0,1),random.randint(0,1)]
    #     a.append(x_only_move)
    # a = np.unique(a, axis=0)
    # print(len(a))
    # for i in range(len(a)):
    #     x.append(a[i])
    #     y.append([0,0,0,0,0,0,0,0])


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
    return x_train, y_train, x_test, y_test


def train(x_train, y_train, x_test, y_test):
    x_train = np.asarray(x_train, dtype=np.float)
    y_train = np.asarray(y_train, dtype=np.int64)
    # x_test = np.asarray(x_test, dtype=np.float)
    # y_test = np.asarray(y_test, dtype=np.int64) 
    x_test = x_train
    y_test = y_train

    from pystruct.learners import NSlackSSVM, OneSlackSSVM, SubgradientSSVM, LatentSSVM, SubgradientLatentSSVM, PrimalDSStructuredSVM
    from pystruct.models import MultiLabelClf, MultiClassClf

    clf = OneSlackSSVM(MultiLabelClf(), C=1,  show_loss_every=1, verbose=1, max_iter=1000)
    # print(x_train, y_train)
    # input()
    clf.fit(x_train, y_train)
    result = clf.predict(x_test)
    print('Result: \n', result)
    print('True label:\n', y_test)
    clf.score(x_test, y_test)
    print('\n')

    count = 0
    for i in range(len(result)):
        # print(np.sum(np.square(y_test[i]-result[i])))
        if np.sum(np.square(y_test[i]-result[i])) != 0:
            print('True label: ',y_test[i], 'Predict:  ', result[i])
            count += 1
    print(count)

    translate_vector(x_test,y_test)

def translate_vector(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    x_contexts = {0:'move', 1:'pick', 2:'place', 3:'push', 4:'apple', 5:'pear', 6:'plate_green', 7:'plate_white'}
    y_contexts = {
                    0: 'pick apple & place plate_green',
                    1: 'pick apple & place plate_white',
                    2: 'pick pear & place plate_green',
                    3: 'pick pear & place plate_white',
                    4: 'push apple',
                    5: 'push pear',
                    6: 'push plate_green',
                    7: 'push plate_white'
                }
    for x_test, y_predict in zip(x,y):
        index_x_test = np.where(x_test == 1)
        # print(index_x_test)
        print('CONTEXTS IN X: ')
        for index in index_x_test[0]:
            # print(index)
            print(x_contexts[int(index)])
        
        index_y_test = np.where(y_predict == 1)
        print('\nPREDICT ACTIONS: ')
        for index in index_y_test[0]:
            print(y_contexts[int(index)])
        print('\n\n\n')
if __name__ == "__main__":
    x_train, y_train, x_test, y_test = get_data()
    train(x_train, y_train, x_test, y_test)