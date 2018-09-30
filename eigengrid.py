from envs.grid import GRID
from base.wrappers import EnvWrapper
from spinets.spintrainer import SpinTrainer
import core.env as E
import networks.networks as N
import spinets.spins as S
import core.utils as U

import gc

gc.enable()
gc.collect()


Spin=S.SpectralSpinet
print(Spin.info)
grid_size = 36
size = grid_size*2
k = 32
input_shape=(3,size,size)
batch_size = 1024
game = "GridFinaRail"+str(size)

env = EnvWrapper(GRID(grid_size=grid_size,max_time= 5000,stochastic = True, square_size=2),
                 record_freq=10, size=size, mode="rgb", frame_count = 1)

action_set = [0,1,2,3]

spin = Spin(input_shape, k, network=N.FCConvNetBias, lr=1e-2, chol_alpha=1e-2,
                 ls_alpha = 0.5, ls_beta=0.25, ls_maxiter=30, log_freq=k,log_file=game)

builder = E.GraphBuilder(env,action_set,batch_size)
trainer = SpinTrainer(spin, reduce_ratio=0.5,best_interval=10)

spin.load("./checks/"+game)

i = 0
while True:
    i = i +1
    X, W = builder.get()
    X = U.torchify(X)
    Lap = U.laplacian(U.torchify(W))
    res = trainer.train(X,Lap)
    i+=1
    if i > 20:            
        spin.save("./checks/"+game)
        i=0