import sys
sys.path.append("../")
import core.utils as U
import core.picklize as pkl
import core.clustering as C
import networks.networks as N
import scipy.sparse as SP
import numpy as np

import spinets.spins as S
Spin = S.FinalNet
print(Spin.name,Spin.info)




print("Loading Data")
data, labels = pkl.gdepicklize("./../../datasets/mnist_all.pkl.gz")
size = 28
input_shape = (1,size,size)
data = data.reshape((-1,)+input_shape)
nearest = 5
k = 20
batch_size = 1024


""" Test data """
#############################
print("Preparing Test Data")
test_size = 10000
data_test = data[-test_size:]
labels_test = labels[-test_size:]
#L_test = C.build_laplacian(data_test, nearest)
#eigs_test,V_test = SP.linalg.eigsh(L_test,k=k,which="SM",tol=1e-12)
#actual_F_test = U.rayleigh_np(V_test[:,:k],L_test)
#print(actual_F_test,np.sum(eigs_test[:k]))
#C.munkres_acc(V_test,labels_test,1,30)

""" Train data """
#############################
print("Preparing Training data")
data_train = data[:-test_size]
labels_train = labels[:-test_size]
n = len(data_train)

""" Network """
#############################
print("Creating network")
spin = Spin(input_shape, k, network=N.ConvNet, lr=1.0, chol_alpha=1e-3,
                 ls_alpha = 0.5, ls_beta=0.25, ls_maxiter=30, log_freq=k,log_file="MNIST")
spin.load("./../checks/mnist")
from spinets.spintrainer import SpinTrainer
trainer = SpinTrainer(spin, reduce_ratio=0.5,best_interval=15)
i = 0
cluster_freq = 2*k
print("Training Starts")
while True:
    i+=1
    batch = np.random.choice(range(n), batch_size, replace = False)
    X = U.torchify(data_train[batch])
    Lap = U.torchify(C.build_laplacian(X,nearest).toarray())
    err = trainer.train(X,Lap)
    if not i%cluster_freq:
        V_pred = U.get(spin(U.torchify(data_test)))
#        pred_F = U.rayleigh_np(V_pred,L_test)
#        print("\n\t""%3.6f"%actual_F_test, "%3.6f"%pred_F)
        spin.save("./../checks/mnist")
        C.munkres_acc(V_pred,labels_test,1,20)
        i = 0
