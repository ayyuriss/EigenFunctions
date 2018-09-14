import sys
sys.path.append("../")

import networks.networks as N
import core.picklize as pkl
import core.utils as U
import core.clustering as C

import numpy as np

import spinets.spins as S

Spin=S.SpectralSpinet
print(Spin.name,"\n",Spin.info)
print("Loading data")
model = "reutersF2"

data, labels = pkl.gdepicklize("./../../datasets/reuters.pkl.gz")
input_shape = (data.shape[1],)
nearest = 10
k = 11
batch_size = 2048


""" Test data """
####################################
print("Preparing test data")
test_size = 10000
data_test = data[-test_size:]
labels_test = labels[-test_size:]
#L_test = C.build_laplacian(data_test.toarray(),nearest)
#eigs,V_test = SP.linalg.eigsh(L_test,k=k, which="SM",tol=1e-12)
#ray_test = U.rayleigh_np(V_test,L_test)
#print(ray_test,np.sum(eigs))
#U.munkres_acc(V_test,labels_test,1,15)


""" Train data """
####################################
print("Preparing train data")
data_train = data[:-test_size]
labels_train = labels[:-test_size]
n = data_train.shape[0]

"""
Training
"""
####################################
spin = Spin(input_shape, k, network=N.FCSpectralNet, lr=1.0, chol_alpha=1e-2,
                 ls_alpha = 0.5, ls_beta=0.25, ls_maxiter=30, log_freq=k,log_file=model)
spin.load("./checks/"+model)

from spinets.spintrainer import SpinTrainer
trainer = SpinTrainer(spin, reduce_ratio=0.5,best_interval=15)
d_learner = C.DistanceLearner(n,nearest,5000)
i = 0
cluster_freq = 2*k
print("Starting Training")
while True:
    i+=1
    batch = np.random.choice(range(n), batch_size, replace = False)
    X = U.torchify(data_train[batch].toarray())
    W = d_learner.submit(data_train[batch],batch).toarray()
    #Lap = U.torchify(C.build_laplacian(X,nearest).toarray())
    Lap = U.laplacian(U.torchify(W))
    err = trainer.train(X,Lap)
    
    if i>5:
        spin.save("./checks/"+model)
        V_pred = U.get(spin(U.torchify(data_test.toarray())))
        #for l in [4,5,6,7,8,9]:
        for l in range(4,10):
            spin.logger.log("Acc %i"%l, C.munkres_test(V_pred[:,1:l+1],labels_test))
        i = 0



D = []
step = 10000
for i in range(0,n,step):
    D.append(U.get(spin(U.torchify(data_train[i:min(i+step,n)].toarray()))))
D = np.concatenate(D,axis=0)
l = 4
acc = C.munkres_test(D[:,1:l+1],labels_train)



import core.plots as P

data = spin.logger.load()

P.plt.plot(data["Acc 10"].dropna())
P.plt.plot(data["Spin lr"].dropna())


P.plt.figure()
P.plt.plot(data["Grsmn Dist Old"].dropna())
P.plt.figure()
P.plt.plot(data["Seq Rayleigh"].dropna())
P.plt.figure()
P.plt.plot(data["Rayleigh"].dropna())
P.plt.plot(data["Spin lr"].dropna())
"""    i+=1
7102
    batch = np.random.choice(range(n), batch_size, replace = False)
    X = U.torchify(data_train[batch].toarray())
    W = d_learner.submit(data_train[batch],batch).toarray()
    
    #Lap = U.torchify(C.build_laplacian(X,nearest).toarray())
    Lap = U.laplacian(U.torchify(W))
    err = trainer.train(X,Lap)
    if not i%cluster_freq :
        V_pred = U.get(spin(U.torchify(data_test.toarray())))
#        ray_pred = U.rayleigh_np(V_pred,L_test)
#        print(ray_test,ray_pred)
        spin.save("./checks/"+model)
        print("Testing clustering on test")
        spin.logger.log("Acc 4", C.munkres_test(V_pred[:,1:5],labels_test))
        #spin.logger.log("Acc 10", C.munkres_test(V_pred[:,1:11],labels_test))
        #spin.logger.log("Acc 15", C.munkres_test(V_pred[:,1:16],labels_test))
        #spin.logger.log("Acc 19", C.munkres_test(V_pred[:,1:],labels_test))
        i=0"""