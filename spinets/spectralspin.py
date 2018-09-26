import core.utils as U
import networks.networks as N
from base.basespinet import BaseSpinet
# Changed
class SpectralSpin(BaseSpinet):
    info = "sequential_rayleigh with double LS search"
    name = "SpectralSpin"
    def __init__(self, input_shape, spectral_dim, network=N.ConvNet, lr=1.0,
                 ls_alpha = 0.5, ls_beta=0.5, ls_maxiter=20, log_freq=10,log_file=""):
        super(SpectralSpin,self).__init__(input_shape, spectral_dim, network, lr,
                 ls_alpha, ls_beta, ls_maxiter, log_freq,log_file)

        self.max_grsmn = 1e-3
        self.cg_iter = 10
        self.cg_damping= 1e-7
        self.l1_coef = 1.0
        self.normalize = 1.0

    def cs_func(self,X,Lap,theta0):
        self.network.flaten.set(theta0)
        Y0 = self.forward_(X).detach()
        L0 = U.cholesky_inv(Y0.t().mm(Y0))
        def func(theta):
            self.network.flaten.set(theta)
            y = self.forward_(X).detach()
            #return U.sequential_rayleigh(y,Lap), (y.t().mm(y)-Y0.t().mm(Y0).detach()).norm()**2
            return U.sequential_rayleigh(y,Lap), U.grassmann_distance(y.mm(L0.t()))
        return func
    def ls_func(self,X,Lap):
        def func(theta):
            self.network.flaten.set(theta)
            return U.sequential_rayleigh(self.forward_(X).detach(),Lap)
        return func
 
    def learn(self,X,Lap):

        Y = self.forward_(X)
        old_ray = U.rayleigh(Y,Lap)
        old_seq_ray = U.sequential_rayleigh(Y,Lap)
        grass_distance = U.grassmann_distance(Y)

        # Trust Region Part
        ray_seq_grad = self.network.flaten.flatgrad(old_seq_ray + self.l1_coef*self.network.l1_weight() ,retain=True)
        L = U.cholesky_inv(Y.detach().t().mm(Y.detach()))
#        grsmn_grad = self.network.flaten.flatgrad((Y.t().mm(Y)-Y.t().mm(Y).detach()).norm()**2,retain=True,create=True)
        grsmn_grad = self.network.flaten.flatgrad(U.grassmann_distance(Y.mm(L.t())),retain=True,create=True)
        step_dir, tol = U.conjugate_gradient(self.Fvp(grsmn_grad),-ray_seq_grad,cg_iters=self.cg_iter)
        shs = .5*step_dir.dot(self.Fvp(grsmn_grad)(step_dir))
        lm = (shs/self.max_grsmn).sqrt()
        full_step = step_dir/lm

        self.normalize = max(self.normalize, U.get(ray_seq_grad.norm()))
        
        fullstep_seq_ray = full_step*ray_seq_grad.norm()/self.normalize
        expected_ray = -fullstep_seq_ray.dot(ray_seq_grad)
        

        
        theta0 = self.network.flaten.get()
        
        self.logger.log("CG tol",tol)
        self.logger.log("Weight N",self.normalize)
        self.logger.log("Seq Rayleigh",old_seq_ray)
        self.logger.log("Grsmn Dist Old",grass_distance)
        self.logger.log("Rayleigh",old_ray)
        self.logger.log("Seq Grad norm",ray_seq_grad.norm())
        self.logger.log("Seq CG norm",fullstep_seq_ray.norm())
        self.logger.log("L1 wieghts before",theta0.abs().mean())
        self.logger.log("Expected Ray",expected_ray)


        func = self.cs_func(X,Lap,theta0)
        constraint = lambda x: x<=self.max_grsmn
        coef =  U.constrained_linesearch(func, theta0, fullstep_seq_ray, expected_ray,constraint, self.ls_alpha, self.ls_beta, self.ls_maxiter)
#        coef =  U.simple_constrained_linesearch(func, theta0, fullstep_seq_ray, constraint, self.ls_beta, self.ls_maxiter)
#        func = self.ls_func(X,Lap)
#        coef =  U.linesearch(func, theta0, fullstep_seq_ray, expected_ray, self.ls_alpha, self.ls_beta, self.ls_maxiter)        
#       
        fullstep_seq_ray.mul_(coef)
        expected_ray.mul_(coef)
        self.network.flaten.set(theta0+self.lr*fullstep_seq_ray)

        Y2 = self.forward_(X).detach()
        L2 = U.cholesky_inv(Y2.t().mm(Y2))
        # Metrics
        new_ray = U.rayleigh(Y2,Lap)
        new_seq_ray = U.sequential_rayleigh(Y2,Lap)
        new_grass = U.grassmann_distance(Y2)
        #div = (Y2.t().mm(Y2)-Y.t().mm(Y)).norm()**2
        div =  U.grassmann_distance(Y2.mm(L2.t()))
        # Log metrics Improve
        self.logger.log("Grsmn Div",div)
        self.logger.log("Seq Rayleigh Improve",old_seq_ray-new_seq_ray)
        self.logger.log("Grsmn Improve",grass_distance-new_grass)
        self.logger.log("Rayleigh Improve",old_ray-new_ray)
        #self.logger.log("CircleS",t0)
        self.logger.log("LS step",coef)
        self.logger.log("L1 wieghts after",self.network.l1_weight().detach())


        return U.get(new_seq_ray)


    def Fvp(self,grad):
        def fisher_product(v):
            kl_v = (grad*v).sum()
            grad_grad_kl = self.network.flaten.flatgrad(kl_v, retain=True)
            return grad_grad_kl + v*self.cg_damping
        return fisher_product
