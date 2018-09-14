import numpy as np
from collections import deque
class SpinTrainer(object):
    def __init__(self,spin,reduce_ratio=0.5,best_interval = 15,verbose=0):
        self.best = np.inf
        self.spin = spin
        self.best_since = 0
        self.best_interval = best_interval
        self.rate = reduce_ratio
        self.past = deque([],best_interval)
        self.verbose = verbose
    def train(self,X,Y):
        res = 1.0
        new = self.spin.train(X,Y)
        self.past.append(new)
        if new < self.best:
            self.best = new
            self.best_since = 0
        else:
            self.best_since += 1
            if self.best_since>self.best_interval:
                self.best_since = 0
                self.best = np.mean(self.past)
                self.spin.lr = self.spin.lr * self.rate
                res = self.rate
                #for m in self.spin.cholesky.model:
                #    m.alpha = m.alpha*self.rate
                #print("New rates:", self.spin.lr,m.alpha)
                print("New rates:", self.spin.lr)
        if self.verbose :print(new, self.best_since)
        
        return res