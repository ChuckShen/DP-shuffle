import tensorflow as tf
import input_data
import sys
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import colors

# 训练参数
numberlabels = 2
hiddenunits = [2*512,64, 2, 64,512*2]
lamb = 0.001  # regularization parameter
batchsize_test = 1000
learning_rate = 0.001
batch_size = 48
trainstep = (2500 * 10 // batch_size)*20
kernel_size_1 = 2
kernel_size_2 = 3
# defining weighs and initlizatinon    权重初始化
def weight_variable(shape):
    initial = tf.truncated_normal(
        shape, stddev=0.01)  #generate tensor with shape=[x,y],  stddev 表示标准差
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(
        0.01, shape=shape)  #shape=[2]     [[0.01 0.01] [0.01 0.01]]
    return tf.Variable(initial)

# defining the layers
def layers(x, W, b):
    return tf.nn.sigmoid(tf.matmul(x, W)+b)

def layers_R(x, W, b):
    return tf.nn.relu(tf.matmul(x, W)+b)

def layers_N(x, W, b):
    return tf.matmul(x, W)+b

def conv2d(x,W,b):
    return tf.nn.relu(tf.nn.conv2d(x,W,[1,1,1,1],padding='SAME')+b)

def conv2d_S(x,W,b):
    return tf.nn.sigmoid(tf.nn.conv2d(x,W,[1,1,1,1],padding='SAME')+b)

def max_pool_2(x):
    return tf.nn.max_pool(x,[1,2,2,1],[1,2,2,1],padding='SAME')

sess = tf.Session()

# LENGTH_GET = [40]
# STEP = [[1,41]] 
LENGTH_GET = [16]
STEP = [[1,121]] 

for size in range(len(LENGTH_GET)):
    # Size
    lx = LENGTH_GET[size]
    # Time
    ly = STEP[size][1] - STEP[size][0]
    # defining the model
    #first layer
    #weights and bias
    # encoder weight lx*ly > 100
    W_1 = weight_variable([kernel_size_1,kernel_size_2,1,16])#16卷积层输出的深度==卷积核的个数。
    b_1 = bias_variable([16])
    # 100 > 50
    W_2 = weight_variable([kernel_size_1, kernel_size_2, 16, 8])
    b_2 = bias_variable([8])

    W_21 = weight_variable([kernel_size_1, kernel_size_2, 8, 8])
    b_21 = bias_variable([8])
    # 50 > 2
    W_3 = weight_variable([2*15*8, 2])
    b_3 = bias_variable([2])
    # decoder weight 2 > 50
    W_4 = weight_variable([2, 2*15*8])
    b_4 = bias_variable([2*15*8])
    # 50 > 100
    W_5 = weight_variable([kernel_size_1,kernel_size_2,8, 8])
    b_5 = bias_variable([8])
    
    W_51 = weight_variable([kernel_size_1,kernel_size_2,8, 16])
    b_51 = bias_variable([16])
    # 100 > lx*ly
    W_6 = weight_variable([kernel_size_1,kernel_size_2,16,16])
    b_6 = bias_variable([16])

    W_7 = weight_variable([kernel_size_1,kernel_size_2,16,1])
    b_7 = bias_variable([1])

    #Apply a sigmoid
    #x is input_data, y_ is the label
    x = tf.placeholder("float", shape=[None, lx * ly])   #默认是None,就是一维值，[None,3]表示列是3，行不定
    y_ = tf.placeholder("float", shape=[None, lx * ly])
    

    xinput_re = tf.reshape(x,[-1,lx,ly,1])
    # encoder
    O1 = conv2d_S(xinput_re, W_1, b_1)
    O2 = conv2d_S(max_pool_2(O1), W_2, b_2)
    O21 = conv2d_S(max_pool_2(O2), W_21, b_21)
    O2_re = tf.reshape(max_pool_2(O21),[-1,2*15*8])
    O3 = layers_N(O2_re, W_3, b_3)

    # decoder
    # O4 = layers(tf.nn.sigmoid(O3), W_4, b_4)
    O4 = layers(O3, W_4, b_4)
    O4_re = tf.reshape(O4,[-1,2,15,8])
    O5 = conv2d_S(tf.image.resize_images(O4_re,(4,30),method=1), W_5, b_5)
    O51 = conv2d_S(tf.image.resize_images(O4_re,(8,60),method=1), W_51, b_51)
    O6 = conv2d_S(tf.image.resize_images(O51,(16,120),method=1), W_6, b_6)
    O7 = conv2d_S(O6, W_7, b_7)

    y_conv = tf.reshape(O7,[-1,lx*ly])

    #Train and Evaluate the Model

    # cost function to minimize (with L2 regularization)
    cross_entropy = tf.reduce_mean( -y_*tf.log(tf.clip_by_value(y_conv,1e-10,1.0))-(1.0-y_)*tf.log((tf.clip_by_value(1-y_conv,1e-10,1.0))))  
    #defining the optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate)  #0.0001 is learn_rate
    train_step = optimizer.minimize(cross_entropy)

    # 判断预测正确与否

    #reading the data in the directory txt
    mnist = input_data.read_data_sets(numberlabels,
                                      lx,
                                      ly,
                                      './data/',
                                      one_hot=True)
 
    print(mnist)

    print('test.images.shape', mnist.test.images.shape)
    print('test.images.shape', mnist.test.labels.shape)
    print(
        "xxxxxxxxxxxxxxxxxxxxx Training START xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
        size)

    sess.run(tf.global_variables_initializer())  # 初始化参数

    # training
    for i in range(1,trainstep+1):

        batch = mnist.train.next_batch(batch_size)

        #每一步迭代，我们都会加载100个训练样本，然后执行一次train_step，并通过feed_dict将x 和 y_张量占位符用训练训练数据替代。

        if i % 2000 == 0:

            # batch_train = mnist.train.next_batch(batchsize_test)

            train_loss = sess.run(cross_entropy,
                                      feed_dict={
                                          x: batch[0],
                                          y_: batch[0]
                                      })
            print("step, train loss:", i, train_loss)

            
        #通过feed_dict对x&y_ 进行赋值，其中x为样例内容， y_为x对应的标签
        sess.run(train_step, feed_dict={x: batch[0], y_: batch[0]})   #0和1分别代表batch中的第一个和第二个元素

    print(
        "xxxxxxxxxxxxxxxxxxxxx Training Done xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    )

    print(
        "test loss",
        sess.run(cross_entropy,
                 feed_dict={
                     x: mnist.test.images,
                     y_: mnist.test.images
                 }))

    print("xxxxxxxxxxxxxxxxxxxxx Plot Data xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")

    #producing data to get the plots we like
    #output of neural net
   
    ncyc_test = 200
    plist = ptrain = [0.0 + x*0.025 for x in range(41)]
    # plist = ptrain = [0.1, 0.6447, 0.90]
    ptest = plist
    Ntemp = len(plist)  # number of different temperatures used in the simulation
    samples_per_T = int(mnist.test.num_examples / Ntemp)

    bb_test = []
    for p in ptest:
        for icyc in range(0, ncyc_test):
            array = []
            array = p
            bb_test.append(array)
    y = np.array(bb_test)
    ii = 0
    ress = []
    for i in range(Ntemp):
        for j in range(samples_per_T):
            res=sess.run(O3,feed_dict={x: [mnist.test.images[ii]]})
            ress.append(res)
            ii +=1
    A_reduction = np.array(ress)
    print(A_reduction)
    print(A_reduction.shape)
    X_reduction = np.reshape(A_reduction,(-1,2))
    print(X_reduction.shape)
    # golden_size = lambda width: (width, 2. * width / (1 + np.sqrt(5)))
    cm = plt.cm.get_cmap('rainbow')
    # plt.rc('font',**{'size':16})
    # fig, ax = plt.subplots(1,figsize=golden_size(8))
    # 设置输出的图片大小
    figsize = 10,8
    figure, ax = plt.subplots(figsize=figsize)
    sc = ax.scatter(X_reduction[:,0], X_reduction[:,1], c=y, s=4, cmap=cm)
    cb = plt.colorbar(sc)
    cb.ax.tick_params(labelsize=23)  #设置色标刻度字体大小。
    # font = {'family' : 'serif',
    #         'color'  : 'darkred',
    #         'weight' : 'normal',
    #         'size'   : 23,
    #         }
    # cb.set_label('colorbar',fontdict=font) #设置colorbar的标签字体及其大小
    # plt.legend()  #添加图例
    # 设置坐标刻度值的大小以及刻度值的字体
    plt.tick_params(labelsize=23)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    # print labels
    # [label.set_fontname('Times New Roman') for label in labels]
    plt.xlabel('${h_1}$',fontsize=23)
    plt.ylabel('${h_2}$',fontsize=23)
    # plt.figure(figsize=(8,6))
    # plt.title('Scatter Plot')
    # plt.imshow(figure, cmap='Greys_r')
    plt.savefig('autoencoder4.eps')
    plt.show()
    