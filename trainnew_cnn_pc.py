import tensorflow as tf
import input_data
import sys
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import colors
# seed=np.random.seed(42)
# data=np.random.randint(0,10,size=(10,10))

# font = 'TIMESBD.TTF'  #字体文件，可使用自己电脑自带的其他字体文件
# myfont = fm.FontProperties(fname=font)  #将所给字体文件转化为此处可以使用的格式

# 训练参数
numberlabels = 2
hiddenunits = [2*512,64, 1, 64,512*2]

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



LENGTH_GET = [16]
STEP = [[1,121]] 
# LENGTH_GET = [8,16,32,48,64]
# STEP = [[1,9],[1,17],[1,33],[1,49],[1,65]] 


# STEP = [[1, TIME + 1], [1, TIME + 1]]  # or Step = [10]

  # or Step = [10]

for size in range(len(LENGTH_GET)):
   
    # Size
    lx = LENGTH_GET[size]

    # Time
    ly = STEP[size][1] - STEP[size][0]

    # defining the model

    #first layer
    #weights and bias
    # encoder weight lx*ly > 100
    W_1 = weight_variable([kernel_size_1,kernel_size_2,1,16])
    b_1 = bias_variable([16])
    # 100 > 50
    W_2 = weight_variable([kernel_size_1, kernel_size_2, 16, 8])
    b_2 = bias_variable([8])

    W_21 = weight_variable([kernel_size_1, kernel_size_2, 8, 8])
    b_21 = bias_variable([8])
    # 50 > 2
    W_3 = weight_variable([2*15*8, 1])
    b_3 = bias_variable([1])
    # decoder weight 2 > 50
    W_4 = weight_variable([1, 2*15*8])
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

    # encoder
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
        if i % 2000 == 0:
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

    plist = ptrain = [0.0 + x*0.025 for x in range(41)]
    # plist = ptrain = [0.1, 0.6447, 0.9]
    ptest = plist
    Ntemp = len(
        plist)  # number of different temperatures used in the simulation
    print(Ntemp)

    samples_per_T = int(mnist.test.num_examples / Ntemp)

    f = open('./plot/' + 'nnoutlx104' + str(lx) + '_' + str(ly) + '.dat', 'w')
    ii = 0
    av_T =[]
    av_x_ALL= []
    av_y_ALL= []
    av_z_ALL= []
    for i in range(Ntemp):
        # av_z = []
        av=0.0
        for j in range(samples_per_T):
            #X[1, :]取第一行的所有列数据， X[:, 0]取所有行的第0列数据 
            res=sess.run(O3,feed_dict={x: [mnist.test.images[ii]]})  #0和1分别代表batch中的第一个和第二个元素
            av=av+res 
            ii +=1
        av=av/samples_per_T
        print(av)
        plt.scatter(plist[i],av[0][0],label="{}".format(plist[i]))
        f.write(str(plist[i])+' '+str(av[0,0])+' '+"\n")
    plt.xlabel('${p}$',fontsize=20)
    plt.ylabel('${h^*}$',fontsize=20)
    plt.savefig('ae_pc4.eps')
    plt.show()
