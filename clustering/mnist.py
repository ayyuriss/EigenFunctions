import sys
sys.path.append("../")
import core.utils as U
import core.picklize as pkl
import core.clustering as C
import networks.networks as N
import numpy as np

import spinets.spins as S
Spin=S.SpectralSpinet

print(Spin.name,Spin.info)
model = "MNISTF9Z"

print("Loading Data")
data, labels = pkl.gdepicklize("./../../datasets/mnist_all.pkl.gz")
size = 28
input_shape = (1,size,size)
data = data.reshape((-1,)+input_shape)
nearest = 5
k = 32
batch_size = 1024


""" Test data """
#############################
print("Preparing Test Data")
test_size = 10000
data_test = data[-test_size:]
labels_test = labels[-test_size:]

""" Train data """
#############################
print("Preparing Training data")
data_train = data[:-test_size]
labels_train = labels[:-test_size]
n = len(data_train)

""" Network """
#############################
print("Creating network")
spin = Spin(input_shape, k, network=N.FCSpectralNet, lr=1e-2, chol_alpha=1e-2,
                 ls_alpha = 0.5, ls_beta=0.25, ls_maxiter=30, log_freq=k,log_file=model)
spin.load("./checks/"+model)

""" Training """
#############################
from spinets.spintrainer import SpinTrainer
trainer = SpinTrainer(spin, reduce_ratio=0.8,best_interval=15)
d_learner = C.DistanceLearner(n,nearest,256)
i = 0
cluster_freq = 2*k
print("Training Starts")
while True:
    i+=1
    batch = np.random.choice(range(n), batch_size, replace = False)
    X = U.torchify(data_train[batch])
    W = d_learner.submit(X,batch).toarray()
    #Lap = U.torchify(C.build_laplacian(X,nearest).toarray())
    Lap = U.laplacian(U.torchify(W))
    err = trainer.train(X,Lap)
    
    if not i % 5:
        spin.save("./checks/"+model)
        V_pred = U.get(spin(U.torchify(data_test)))
        #for l in [4,5,6,7,8,9]:
        for l in range(10,k):
            ac,nmi = C.munkres_test(V_pred[:,1:l+1],labels_test)
            spin.logger.log("Acc %i"%l, ac)
            spin.logger.log("NMI %i"%l, nmi)
    if i > 30:
        D = []
        step = 10000
        for i in range(0,n,step):
            D.append(U.get(spin(U.torchify(data_train[i:min(i+step,n)]))))
        D = np.concatenate(D,axis=0)        
        for l in range(10,k):
            ac,nmi = C.munkres_test(D[:,1:l+1],labels_train)
            spin.logger.log("Acc Train %i"%l, ac)
            spin.logger.log("NMI Train %i"%l, nmi)
        i = 1
