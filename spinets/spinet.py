import networks.networks as N
import core.utils as U
def spincholesky(minispinet):

    class SpinCholesky(minispinet):
        name = "Cholesky"+minispinet.name
        info = "Cholesky and, " + minispinet.info
        def __init__(self,input_shape, spectral_dim, network=N.ConvNet, chol_alpha=5e-3,n_blocks=1,
                 lr=1.0,ls_alpha = 0.5, ls_beta=0.5, ls_maxiter=50, log_freq=10,log_file=""):
            super(SpinCholesky,self).__init__(input_shape, spectral_dim, network,lr,
                                              ls_alpha, ls_beta, ls_maxiter, log_freq,log_file)
            self.cholesky = N.CholeskyBlock((spectral_dim,),alpha=chol_alpha,n_blocks=n_blocks,owner_name=self.name)

        def load(self,fname):
            self.network.load(fname)
            self.cholesky.load(fname)

        def save(self,fname):
            self.network.save(fname)
            self.cholesky.save(fname)

        def forward(self,x):
            return self.cholesky(self.network(x))
        def forward_(self,x):
            return self.forward(x)/x.shape[0]**.5
        def train(self, X, Lap):

            x = super(SpinCholesky,self).train(X, Lap)
            grss1 = U.grassmann_distance(self.forward_(X).detach())
            self.update_cholesky(X)
            grss2 = U.grassmann_distance(self.forward_(X).detach())
            self.logger.log("Chol lr",self.cholesky.model[0].alpha)
            self.logger.log("Grsmn b4 Cholesky",grss1)
            self.logger.log("Grsmn after Cholesky",grss2)
            return x

        def update_cholesky(self,X):
            self.cholesky.update(self.network(X).detach())
        
        def summary(self):
            self.network.summary()
            self.cholesky.summary()

    return SpinCholesky