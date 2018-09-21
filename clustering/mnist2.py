import sys
sys.path.append("../")
import core.utils as U
import core.picklize as pkl
import core.clustering as C
import networks.networks as N
import scipy.sparse as SP
import numpy as np

import spinets.spins as S
Spin=S.SpectralSpinet

print(Spin.name,Spin.info)
model = "MNISTFS"

print("Loading Data")
data, labels = pkl.gdepicklize("./../../datasets/mnist_all.pkl.gz")
size = 28
input_shape = (1,size,size)
data = data.reshape((-1,)+input_shape)
nearest = 5
k = 21
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
spin = Spin(input_shape, k, network=N.FCSpectralMNet, lr=1.0, chol_alpha=1e-2,
                 ls_alpha = 0.5, ls_beta=0.25, ls_maxiter=30, log_freq=k,log_file=model)
spin.load("./checks/"+model)

""" Training """
#############################
from spinets.spintrainer import SpinTrainer
trainer = SpinTrainer(spin, reduce_ratio=0.8,best_interval=15)
d_learner = C.DistanceLearner(n,nearest,2048)
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
    
    if not i%5:
        spin.save("./checks/"+model)
        V_pred = U.get(spin(U.torchify(data_test)))
        for l in [10,11,12,13,14,15,18,19,20]:
            spin.logger.log("Acc %i"%l, C.munkres_test(V_pred[:,1:l+1],labels_test))

    if i > 30:
        D = []
        step = 10000
        for i in range(0,n,step):
            D.append(U.get(spin(U.torchify(data_train[i:min(i+step,n)]))))
        D = np.concatenate(D,axis=0)        
        for l in range(4,k):
            spin.logger.log("Acc Train %i"%l, C.munkres_test(D[:,1:l+1],labels_train))
        i = 1
        




"""
D = []
step = 10000
for i in range(0,n,step):
    D.append(U.get(spin(U.torchify(data_train[i:min(i+step,n)]))))
D = np.concatenate(D,axis=0)
for l in [10,11,12,13,14,15,18,19,20]:
     print("Acc %i"%l, C.munkres_test(V_pred[:,1:l+1],labels_test))
"""
    #ac = C.munkres_test(V_pred[:,1:],labels_test)
#    if not i%cluster_freq:
#        V_pred = U.get(spin(U.torchify(data_test)))
#        pred_F = U.rayleigh_np(V_pred,L_test)
#        print("\n\t""%3.6f"%actual_F_test, "%3.6f"%pred_F)
#        spin.save("./checks/"+model)
        #C.munkres_acc(V_pred,labels_test,1,20)
#        ac = C.munkres_test(V_pred[:,1:],labels_test)
#        spin.logger.log("Accuracy",ac)
#        spin.logger.log("S Sparsity",len(d_learner.D.nonzero()[0]))
#        spin.logger.log("Acc 10", C.munkres_test(V_pred[:,1:11],labels_test))
#        spin.logger.log("Acc 15", C.munkres_test(V_pred[:,1:16],labels_test))
#        spin.logger.log("Acc 19", C.munkres_test(V_pred[:,1:],labels_test))
        #spin.logger.log()
#        i = 0