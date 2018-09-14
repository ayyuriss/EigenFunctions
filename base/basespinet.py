import numpy as np
import torch
import core.utils as U
from base.logger import Logger
import networks.networks as N


class BaseSpinet(torch.nn.Module):
    info = "BaseSpinet Model"
    name = "BaseSpinet"
    
    def __init__(self, input_shape, spectral_dim, network=N.ConvNet, lr=1e-2,
                 ls_alpha = 0.5, ls_beta=0.5, ls_maxiter=50, log_freq=10,log_file=""):
        super(BaseSpinet,self).__init__()
        
        self.spectral_dim = spectral_dim
        self.network = network(input_shape, (spectral_dim,), owner_name=self.name)
        # Learning Parameters
        self.ls_maxiter = ls_maxiter
        self.ls_alpha = ls_alpha
        self.ls_beta = ls_beta
        self.lr = lr
        self.logger = Logger(log_file+self.name,history=log_freq)

        # Counters 
        self.step = 0
        # Logs
        self.log_freq = log_freq

    def load(self,fname):
        self.network.load(fname)
    def save(self,name):
        self.network.save(name)
    def forward(self,x):
        return self.network.forward(x)
    def forward_(self,x):
        return self.network.forward_(x)

    def ls_func(self,X,Lap):
        def func(theta):
            self.network.flaten.set(theta)
            return U.rayleigh(self.forward_(X).detach(),Lap)
        return func

    def train(self, X, Lap):
        self.logger.step()
        self.logger.log("Spin lr",self.lr)
        x = self.learn(X, Lap)
        self.step +=1
        if self.step == self.log_freq :
            self.logger.display()
            self.step = 0
        return x   

    def mean(self,X):
        return np.sum(X)/self.log_freq
    def display(self):
        X = [[k,self.mean(v[0]),self.mean(v[1])] for k,v in self.logs.items()]
        U.print_loss(*X)

    def log_stats(self,*stats):
        for k,i,v in stats:
            self.logs[k][i].append(v)

    def learn(self,X,Lap):
        raise NotImplementedError
    def gradient(self,Y):
        raise NotImplementedError
    def update_cholesky(self,X):
        raise NotImplementedError
    def summary(self):
        self.network.summary()