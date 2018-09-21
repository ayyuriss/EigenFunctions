import core.utils as U
import networks.networks as N
from base.basespinet import BaseSpinet

class SpectralSpin(BaseSpinet):

    info = "sequential_rayleigh with double LS search"
    name = "SpectralSpin"

    def __init__(self, input_shape, spectral_dim, network=N.ConvNet, lr=1e-2,
                 ls_alpha = 0.5, ls_beta=0.5, ls_maxiter=20, log_freq=10,log_file=""):
        super(SpectralSpin,self).__init__(input_shape, spectral_dim, network, lr,
                 ls_alpha, ls_beta, ls_maxiter, log_freq,log_file)

        self.max_grsmn = 1e-3
        self.cg_iter = 10
        self.cg_damping=1e-5
        self.l1_coef = 1.0
    def ls_func(self,X,Lap):
        def func(theta):
            self.network.flaten.set(theta)
            return U.sequential_rayleigh(self.forward_(X).detach(),Lap)
        return func

    def cs_func(self,X,Lap,theta0):
        self.network.flaten.set(theta0)
        Y0 = self.forward_(X).detach()
        def func(theta):
            self.network.flaten.set(theta)
            y = self.forward_(X).detach() 
            return U.sequential_rayleigh(y,Lap), (y.t().mm(y)-Y0.t().mm(Y0).detach()).norm()**2
        return func
    
    def learn(self,X,Lap):

        Y = self.forward_(X)

        old_ray = U.rayleigh(Y,Lap)
        old_seq_ray = U.sequential_rayleigh(Y,Lap)
        grass_distance = U.grassmann_distance(Y)

        # Trust Region Part
        ray_seq_grad = self.network.flaten.flatgrad(old_seq_ray + self.l1_coef*self.network.l1_weight() ,retain=True)
        grsmn_grad = self.network.flaten.flatgrad((Y.t().mm(Y)-Y.t().mm(Y).detach()).norm()**2,retain=True,create=True)
        step_dir = U.conjugate_gradient(self.Fvp(grsmn_grad),-ray_seq_grad,cg_iters=self.cg_iter)
        shs = .5*step_dir.dot(self.Fvp(grsmn_grad)(step_dir))
        lm = (shs/self.max_grsmn).sqrt()
        full_step = step_dir/lm


        fullstep_seq_ray = self.lr*full_step
        expected_ray = -fullstep_seq_ray.dot(ray_seq_grad)
        
        theta0 = self.network.flaten.get()
        func = self.cs_func(X,Lap,theta0)
        
        
        self.logger.log("Seq Rayleigh",old_seq_ray)
        self.logger.log("Grsmn Dist Old",grass_distance)
        self.logger.log("Rayleigh",old_ray)
        self.logger.log("Seq Grad norm",fullstep_seq_ray.norm())
        self.logger.log("L1 wieghts before",theta0.abs().mean())
        self.logger.log("Expected Ray",expected_ray)
        constraint = lambda x: x<1.5*self.max_grsmn
        
        coef =  U.constrained_linesearch(func, theta0, fullstep_seq_ray, expected_ray,constraint, self.ls_alpha, self.ls_beta, self.ls_maxiter)
        fullstep_seq_ray.mul_(coef)
        expected_ray.mul_(coef)
        self.network.flaten.set(theta0+fullstep_seq_ray)

        """
        Y = self.forward_(X)
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
        Y2 = self.forward_(X).detach()
        # Metrics
        new_ray = U.rayleigh(Y2,Lap)
        new_seq_ray = U.sequential_rayleigh(Y2,Lap)
        new_grass = U.grassmann_distance(Y2)
        div = (Y.t().mm(Y)-Y2.t().mm(Y2).detach()).norm()**2
        # Log metrics Improve
        self.logger.log("Grsmn Div",div)
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
