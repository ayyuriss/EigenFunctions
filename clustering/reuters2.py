import sys
sys.path.append("../")

import networks.networks as N
import core.picklize as pkl
import core.utils as U
import core.clustering as C

import scipy.sparse as SP
import numpy as np

import spinets.spins as S

Spin = S.FinalNet
print(Spin.name,"\n",Spin.info)
print("Loading data")
model = "reuters2"
data, labels = pkl.gdepicklize("./../../datasets/reuters.pkl.gz")
input_shape = (data.shape[1],)
nearest = 10
k = 10
batch_size = 2048


""" Test data """
print("Preparing test data")
test_size = 10000
data_test = data[-test_size:]
labels_test = labels[-test_size:]
L_test = C.build_laplacian(data_test.toarray(),nearest)
eigs,V_test = SP.linalg.eigsh(L_test,k=k, which="SM",tol=1e-12)
ray_test = U.rayleigh_np(V_test,L_test)
print(ray_test,np.sum(eigs))
#U.munkres_acc(V_test,labels_test,1,15)
""" Train data """
print("Preparing train data")
data_train = data[:-test_size]
labels_train = labels[:-test_size]
n = data_train.shape[0]

"""
Training
"""
spin = Spin(input_shape, k, network=N.FCNet, lr=1.0, chol_alpha=2e-3,
                 ls_alpha = 0.5, ls_beta=0.25, ls_maxiter=30, log_freq=k,log_file=model)
spin.load("./../checks/"+model)
from spinets.spintrainer import SpinTrainer
trainer = SpinTrainer(spin, reduce_ratio=0.5,best_interval=15)
d_learner = C.DistanceLearner(n,nearest,2000)
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
    if not i%cluster_freq:
        V_pred = U.get(spin(U.torchify(data_test)))
#        pred_F = U.rayleigh_np(V_pred,L_test)
#        print("\n\t""%3.6f"%actual_F_test, "%3.6f"%pred_F)
        spin.save("./checks/"+model)
        #C.munkres_acc(V_pred,labels_test,1,20)
        ac = C.munkres_test(V_pred[:,1:],labels_test)
        spin.logger.log("Accuracy",ac)
        spin.logger.log("S Sparsity",len(d_learner.D.nonzero()[0]))
        i = 0