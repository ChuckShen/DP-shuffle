import numpy as np
import input_data
import sys
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib import rc,rcParams
import matplotlib
import random
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import time
start=time.time()

# pc = 0.6447
# ncyc_train = 100
ncyc_test = 1000
PROP = 0.6447
# ptrain = [0.447, 0.55]
# print(len(ptrain))
ptrain = [0.0 + x*0.025 for x in range(41)]
ptest = ptrain
plist = ptest

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

def shuffle_part(vector, p_choose):
    y2 = []
    for i in vector:
        y2.extend(i)
    point_num = int(len(y2)*p_choose)
    nums = np.random.random(vector.shape[1]*vector.shape[0])
    index = []
    point = []
    for i in np.argsort(nums)[0:point_num]:
        index0 = [i//vector.shape[1] , i % vector.shape[1]]
        index.append(index0)
        yy0 = vector[index0[0] , index0[1]]
        point.append(yy0)
    # print(index)
    # print(point)
    np.random.shuffle(point)
    # print(point)
    for i in range(0 , len(point)):
        vector[index[i][0], index[i][1]] = point[i]
    return vector

p_choose = 1.0
z = 1.75
LENGTH = 16
TIME = 120
# TIME = np.math.ceil(LENGTH**z)
print(TIME)
a_test = []
for p in ptest:
        for icyc in range(0, ncyc_test):
            vector = []
            vector_LT = []
            vector = np.ones(LENGTH)
            vector_LT.append(vector)    
            for i in range(TIME):
                vector = doPercolationStep(vector, p, i)
                vector_LT.append(vector)
            # print(vector_LT)
            AA = np.array(vector_LT)  
            vector_s = shuffle_part(AA, p_choose) 
            # print(vector_s) 
            AAA = vector_s.reshape(1,-1)
            # plt.imshow(AAA)
            # plt.show()
            # print(AAA)
            # print(AAA.shape)
            # Alist = AAA.tolist()
            AAAA = np.concatenate(AAA).ravel().tolist()
            # print(AAAA)
            a_test.append(AAAA)

X = np.array(a_test)
print(X.shape)
pca = PCA(n_components = 2, svd_solver='full')
pca.fit(X)
X_reduction = pca.transform(X)
print(pca.explained_variance_ratio_) 
print(X_reduction)
print(X_reduction.shape)
# XXXXXX= X_reduction/LENGTH
print (pca.n_components_)
XX = X_reduction[:,0]
print(XX)
print(XX.shape)
XXX = np.array(XX).reshape(len(ptrain),ncyc_test)
print(XXX)
print(XXX.shape)
XXXX = XXX.mean(axis=1)# print(c.mean(axis=1))#行# print(c.mean(axis=0))#列
print(XXXX)
print(XXXX.shape)
XXXXX = XXXX.tolist()
# colors = ['navy', 'turquoise', 'darkorange']
# To make plots pretty
# print(plist[0:])
print(XXXXX)

#     result.append(XXXXX)
# print(result)

# final = np.array(XXXXX)
# print(final.shape)
# print(final[0,:])
# print(final[3,:])

golden_size = lambda width: (width, 2. * width / (1 + np.sqrt(5)))
# cm = plt.cm.get_cmap('rainbow')
# plt.rc('font',**{'size':16})
# fig, ax = plt.subplots(1,figsize=golden_size(8))
# colors = ['red', 'orange', 'green',  'blue']
# GET = ['L = 10', 'L = 20', 'L = 30', 'L = 40']
plt.figure(figsize=golden_size(12))
# for i in range(len(LENGTH_GET)):
#     G  = GET[i]
plt.plot(plist[0:], XXXXX[0:], marker='o', linewidth=3)
f = open( 'non_nolmal_pca' + '_' + str(p_choose) + '_' + str(LENGTH) + '_' + str(TIME) + '.dat', 'w')  
for i in range(len(plist)): 
    f.write(str(plist[i])+' '+str(XXXXX[i])+' '+"\n")


plt.xlabel('${p}$',fontsize=20)
# plt.ylabel('${<p_1>/L}$')
plt.ylabel('${p_1}$',fontsize=20)
plt.tick_params(axis='both',which='both',direction='in')
# # plt.colorbar(sc, label='$0.25\\times$Temperature')
# # plt.colorbar(sc)
plt.legend()
plt.savefig('pca_p1_p.pdf')

end=time.time()
print('Running time: %s Seconds'%(end-start))

plt.show()