from base.wrappers import AtariWrapper,NetWrapper
import core.env as E
import networks.networks as N
from spinets.spintrainer import SpinTrainer
import spinets.spins as S
import core.utils as U
import gym
import gc

gc.enable()
gc.collect()
import gym.spaces

Spin=S.SpectralSpinet
print(Spin.info)
size = 84
k = 32
batch_size = 1024
game = "Breakout-v0"
action_set = [0,1,2,3]
env = gym.make(game)
env.name = game
env = AtariWrapper(env, size=size, frame_count = 3,frame_skip = 1,mode="rgb",record_freq = 20, crop=game)


input_shape = env.observation_space.shape
spin = Spin(input_shape, k, network=N.ConvNetBigAtari, lr=1.0, chol_alpha=1e-3,
                 ls_alpha = 0.5, ls_beta=0.25, ls_maxiter=30, log_freq=k,log_file=game)

spin.load("./checks/"+game)
builder = E.GraphBuilder(env,action_set,batch_size)
trainer = SpinTrainer(spin, reduce_ratio=0.5,best_interval=15)
i = 0
while True:

    X, W = builder.get()
    X = U.torchify(X)
    Lap = U.laplacian(U.torchify(W))
    res = trainer.train(X,Lap)

    if i > 32: 
        spin.save("./checks/"+game+str(size))
