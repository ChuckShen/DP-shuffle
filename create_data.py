import numpy as np
import random
import matplotlib.pyplot as plt

# 全局参数
pc = 0.6447
ncyc_train = 2500
ncyc_test = 100
PROP = 0.6447
# ptrain = [0.5 + x*0.01 for x in range(31)]
ptrain = [0.0 + x*0.025 for x in range(41)]
# ptrain = [0.0 + x*0.05 for x in range(21)]
# ptrain = [0.1, 0.6447, 0.90]
# ptrain = []
# ptrain = [0.1, 0.9]
ptest = ptrain


# DP生成函数
# ------------------------------------------------------
def doPercolationStep(vector, PROP, time):
    even = time % 2
    vector_copy = np.copy(vector)
    LENGTH = len(vector)
    for i in range(even, LENGTH, 2):
        if vector[i] == 1:
            pro1 = random.random()
            pro2 = random.random()
            if pro1 < PROP:
                vector_copy[(i + LENGTH - 1) % LENGTH] = 1
            if pro2 < PROP:
                vector_copy[(i + 1) % LENGTH] = 1
            vector_copy[i] = 0
    return vector_copy


# 局部函数
# ------------------------------------------------------

z = 1.

LENGTH_GET = [40]
STEP = [[1,41]] 

# z = 1.58
# LENGTH_GET = [28]
# STEP = [[1,29]] 
# LENGTH_GET = [40]
# STEP = [[1,41]] 
# LENGTH_GET = [8,16,32,48,64]
# STEP = [[1,9],[1,17],[1,33],[1,49],[1,65]]  # or Step = [10]

for size in range(len(LENGTH_GET)):
    # Size
    LENGTH = LENGTH_GET[size]
    # Time
    TIME = np.math.ceil(LENGTH**z)
    # TIME = 119
    # If ST = 0 Get T ,If ST = 1 Get L  0 横着取，1 竖着取
    ST = 0
    # 如果ST = 1则Step代表晶格序列, 如果ST = 0 则Step代表时间序列 The Step From [1,TIME+1]
    a_test = []
    #count = 0
    for p in ptest:
        for icyc in range(0, ncyc_test):
            vector = []
            vector_LT = []
            vector = np.ones(LENGTH)
            vector_LT.append(vector)    
            for i in range(TIME):
                vector = doPercolationStep(vector, p, i)
                vector_LT.append(vector)
            # if icyc == 99:
            #     plt.imshow(vector_LT)
            #     plt.show()
            # if icyc ==1800:
            #     print(vector_LT)
            a_test.append(vector_LT[STEP[size][0] - 1:STEP[size][1] - 1])

    np.save('./data/' + 'xtest_' + str(LENGTH) + '_' + str(TIME) + '.npy',
            a_test)

    fy = open('./data/' + 'ytest_' + str(LENGTH) + '_' + str(TIME) + '.dat',
              "w")
    for p in ptest:
        for icyc in range(0, ncyc_test):
            if (p < pc):
                fy.write("%5d \n" % 0)
            else:
                fy.write("%5d \n" % 1)
    fy.close()
    
    b_test = []
    for p in ptrain:
        for icyc in range(0, ncyc_train):
            vector = []
            vector_LT = []
            vector = np.ones(LENGTH)
            vector_LT.append(vector)
            for i in range(TIME):
                vector = doPercolationStep(vector, p, i)
                vector_LT.append(vector)

            b_test.append(vector_LT[STEP[size][0] - 1:STEP[size][1] - 1])

    np.save('./data/' + 'xtrain_' + str(LENGTH) + '_' + str(TIME) + '.npy',
            b_test)

    fy = open('./data/' + 'ytrain_' + str(LENGTH) + '_' + str(TIME) + '.dat',
              "w")
    for p in ptrain:
        for icyc in range(0, ncyc_train):
            if (p < pc):
                fy.write("%5d \n" % 0)
            else:
                fy.write("%5d \n" % 1)
    fy.close()
