import core.utils as U
import networks.networks as N
from base.basespinet import BaseSpinet
import torch 

class SpectralSpinS(BaseSpinet):

    info = "sequential_rayleigh with double LS search"
    name = "SpectralSpinS"

    def __init__(self, input_shape, spectral_dim, network=N.ConvNet, lr=1e-2,
                 ls_alpha = 0.5, ls_beta=0.5, ls_maxiter=20, log_freq=10,log_file=""):
        super(SpectralSpinS,self).__init__(input_shape, spectral_dim, network, lr,
                 ls_alpha, ls_beta, ls_maxiter, log_freq,log_file)

        self.grmn_ratio = 0.9
        self.max_grsmn = 1e-3
        self.cg_iter = 10
        self.cg_damping=1e-5
        self.l1_coef = 1.0
        self.network.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.network.parameters()), lr=1e-3)
    def ls_func(self,X,Lap,L):
        def func(theta):
            self.network.flaten.set(theta)
            Y = self.network.forward_(X).detach()
            return U.sequential_rayleigh(Y,Lap),U.grassmann_distance(Y.mm(L.t()))
        return func
    def learn(self,X,Lap):

        Y = self.network.forward_(X)

        old_ray = U.rayleigh(Y.detach(),Lap)
        old_seq_ray = U.sequential_rayleigh(Y.detach(),Lap)
        grass_distance = U.grassmann_distance(Y)

        # Trust Region Part
        ray_seq_grad = self.network.flaten.flatgrad(old_seq_ray + self.l1_coef*self.network.l1_weight() ,retain=True)
        #ray_seq_grad = self.gradient(X,Lap)
        L = U.cholesky_inv(Y.t().mm(Y))
        #grsmn_grad = self.network.flaten.flatgrad((Y.t().mm(Y)-Y.t().mm(Y).detach()).norm()**2,retain=True,create=True)
        grsmn_grad = self.network.flaten.flatgrad(U.grassmann_distance(Y.mm(L.t())),retain=True,create=True)
        step_dir = U.conjugate_gradient(self.Fvp(grsmn_grad),ray_seq_grad,cg_iters=self.cg_iter)
        shs = .5*step_dir.dot(self.Fvp(grsmn_grad)(step_dir))
        lm = (shs/self.max_grsmn).sqrt()
        full_step = step_dir/lm


        fullstep_seq_ray = -self.lr*full_step
        expected_ray = -fullstep_seq_ray.dot(ray_seq_grad)*self.network.get_learning_rate()
        
        theta0 = self.network.flaten.get()
        func = self.ls_func(X,Lap,L)
        #self.network.step(fullstep_seq_ray)
        
        self.logger.log("Seq Rayleigh",old_seq_ray)
        self.logger.log("Grsmn Dist Old",grass_distance)
        self.logger.log("Rayleigh",old_ray)
        self.logger.log("Seq Grad norm",fullstep_seq_ray.norm())
        #self.logger.log("L1 wieghts before",theta0.abs().mean())
        self.logger.log("Expected Ray",expected_ray)
        constraint = lambda new : new<self.max_grsmn
        #coef =  U.linesearch(func, theta0, fullstep_seq_ray, expected_ray, self.ls_alpha, self.ls_beta, self.ls_maxiter)
        coef =  U.constrained_linesearch(func, theta0, fullstep_seq_ray, expected_ray, constraint, self.ls_alpha, self.ls_beta, self.ls_maxiter)
        fullstep_seq_ray.mul_(coef)
        expected_ray.mul_(coef)
        self.network.flaten.set(theta0+fullstep_seq_ray)


        """Y = self.forward_(X)
        ray_seq_grad = self.network.flaten.flatgrad(
                            U.sequential_rayleigh(Y,Lap)+ self.l1_coef*self.network.l1_weight(),retain=True)

        normed_seq_grad = ray_seq_grad/max(U.EPS,ray_seq_grad.norm())

        grass_distance_n = U.grassmann_distance(Y)
        grasmn_grad = self.network.flaten.flatgrad(grass_distance_n)

        normed_grsmn_grad = grasmn_grad/max(U.EPS,grasmn_grad.norm())
        proj_grsmn_grad = normed_grsmn_grad - normed_seq_grad.dot(normed_grsmn_grad)*normed_seq_grad
        #proj_grsmn_grad = proj_grsmn_grad
        fullstep_grass = -proj_grsmn_grad
        
        # Line search

        #expected_grass = -fullstep_grass.dot(grasmn_grad)
        #theta0 = self.network.flaten.get()
        func = self.cs_func(X,Lap)
        t0 = U.double_linesearch(func, theta0, fullstep_grass, fullstep_seq_ray, expected_ray, self.grmn_ratio*self.ls_alpha, 0.1 , 15)
        self.network.flaten.set(theta0+fullstep_grass*t0+fullstep_seq_ray)
        """
        Y = self.network.forward_(X).detach()
        # Metrics
        new_ray = U.rayleigh(Y,Lap)
        new_seq_ray = U.sequential_rayleigh(Y,Lap)
        new_grass = U.grassmann_distance(Y)

        # Log metrics Improve
        #self.logger.log("Grsmn grad norm",fullstep_grass.norm())
        self.logger.log("Seq Rayleigh Improve",old_seq_ray-new_seq_ray)
        self.logger.log("Grsmn Improve",grass_distance-new_grass)
        self.logger.log("Rayleigh Improve",old_ray-new_ray)
        #self.logger.log("CircleS",t0)
        self.logger.log("LS step",coef)
        self.logger.log("L1 wieghts after",self.network.l1_weight().detach())
        
        
        return U.get(new_seq_ray)
        
        #return U.get(new_grass)

    def Fvp(self,grad):
        def fisher_product(v):
            kl_v = (grad*v).sum()
            grad_grad_kl = self.network.flaten.flatgrad(kl_v, retain=True)
            return grad_grad_kl + v*self.cg_damping
        return fisher_product
    def gradient(self,X,Lap):
        
        Y = self.forward_(X)
        H = Y.detach()*0.0
        
        for i in range(Y.shape[1]):
            H[:,:i+1] += (torch.eye(X.shape[0])-Y[:,:i+1].mm(Y[:,:i+1].t())).mm(Lap.mm(Y[:,:i+1]))
        
        grad = self.network.flaten.flatgrad(Y.t().mm(H).trace(),retain=True)
        return grad
        
        