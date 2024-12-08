import torch
import matplotlib.pyplot as plt
from timeit import default_timer
import copy

torch.manual_seed(0)

import numpy as np
import torch.nn as nn
from scipy.stats import norm
from scipy import integrate

class DiffusionModel:
    """ Base class for diffusion models with score-based sampling.
        Sampler evolves dx/dt = f(x,t) - g(t) * s(x,t) + sqrt{g(t)} * N(0,1) from T to 0,
        given the forward process dx/dt = f(x,t) +  sqrt{g(t)} * N(0,1) from 0 to T.
    """

    def __init__(self):
        self.T     = 1.
        self.eps   = 1e-3

    def SDEsampler(self, score_net, latents, num_steps=100):
        batch_size = latents.shape[0]

        # define initial samples
        init_T = self.T * torch.ones(batch_size, device=latents.device)
        init_x = latents * self.marginal_prob_std(init_T)[:, None]
        x = init_x

        # define steps
        time_steps = torch.linspace(self.T, self.eps, num_steps)
        dt = time_steps[0] - time_steps[1]

        with torch.no_grad():
            for (j,time) in enumerate(time_steps):
                batch_time = torch.ones(batch_size, device=latents.device) * time
                # evaluate score function
                sx = score_net(x, batch_time)
                # evaluate update to x
                f = self.drift(x, batch_time)
                g = self.diffusion_coeff(batch_time)
                drift = -1.*f + (g**2)[:,None]*sx
                x = x + dt * drift + torch.sqrt(dt)*g[:,None]*torch.randn_like(x)

        return x

    def ODEsampler(self, score_net, latents, err_tol=1e-5):
        batch_size = latents.shape[0]

        # extract device
        device=latents.device
        
        # define initial samples
        init_T = self.T * torch.ones(batch_size, device=latents.device)
        init_x = latents * self.marginal_prob_std(init_T)[:, None]

        def score_eval_wrapper(sample, time_steps):
            """A wrapper of the score-based model for use by the ODE solver."""
            sample = torch.tensor(sample, device=device, dtype=torch.float32).reshape(latents.shape)
            with torch.no_grad():    
                score = score_net(sample, time_steps)
            return score
  
        def ode_func(t, x):        
            """The ODE function for use by the ODE solver."""
            batch_time = torch.ones(batch_size, device=latents.device) * t
            g = self.diffusion_coeff(batch_time)
            f = self.drift(x.reshape(latents.shape), batch_time)
            rhs = f - 0.5*(g**2)[:,None] * score_eval_wrapper(x, batch_time)
            return rhs.detach().numpy().reshape((-1,)).astype(np.float64)
  
        # Run the RK solver
        res = integrate.solve_ivp(ode_func, (self.T, self.eps), init_x.reshape(-1).cpu().numpy(), \
                                  rtol=err_tol, atol=err_tol, method='RK45', dense_output=True)  
        
        x_shape = [latents.shape[0], latents.shape[1], len(res.t)]
        x = torch.tensor(res.y, device=latents.device, dtype=torch.float32).reshape(x_shape)
        return (res.t, x)
     
    def ODEsampler_fixedstep(self, score_net, latents, num_steps=100):
        batch_size = latents.shape[0]

        # define initial samples
        init_T = self.T * torch.ones(batch_size, device=latents.device)
        init_x = latents * self.marginal_prob_std(init_T)[:, None]
        x = init_x

        # define steps
        time_steps = torch.linspace(self.T, self.eps, num_steps)
        dt = time_steps[0] - time_steps[1]

        with torch.no_grad():
            for (j,time) in enumerate(time_steps):
                batch_time = torch.ones(batch_size, device=latents.device) * time
                # evaluate score function
                sx = score_net(x, batch_time)
                # evaluate update to x
                f = self.drift(x, batch_time)
                g = self.diffusion_coeff(batch_time)
                drift = (-1.*f + 0.5*(g**2)[:,None]*sx)
                x = x + dt * drift
        
        return (time_steps, x)

class VP(DiffusionModel):
    def __init__(self):
        super().__init__()
        self.beta_min = 0.001
        self.beta_max = 3

    def beta_t(self, t):
        """ Compute beta(t) factor in linear drift f(x,t) = -0.5*beta(t)*x
        """
        return self.beta_min + t*(self.beta_max - self.beta_min)

    def alpha_t(self, t):
        """ Compute alpha(t)=\int_0^t \beta(s)ds for beta defined in linear drift
        """
        return t*self.beta_min + 0.5 * t**2 * (self.beta_max - self.beta_min)

    def drift(self, x, t):
        """
        x: location of J particles in N dimensions, shape (J, N)
        t: time (number)
        returns the drift of a time-changed OU-process for each batch member, shape (J, N)
        """
        return -0.5*self.beta_t(t[:,None])*x

    def marginal_prob_mean(self, t):
        """ Compute the mean factor of $p_{0:t}(x(t) | x(0))$.
        """
        return torch.exp(-0.5 * self.alpha_t(t))

    def marginal_prob_std(self, t):
        """ Compute the standard deviation of $p_{0:t}(x(t) | x(0))$.
        """
        return torch.sqrt(1 - torch.exp(-self.alpha_t(t)))

    def diffusion_coeff(self, t):
        """Compute the diffusion coefficient of our SDE g(t).
        """
        return torch.sqrt(self.beta_t(t))

class VE(DiffusionModel):
    def __init__(self):
        super().__init__()
        self.sigma = 10.

    def drift(self, x, t):
        return torch.zeros(x.shape)

    def marginal_prob_mean(self, t):
        """ Compute the mean factor of $p_{0:t}(x(t) | x(0))$.
        """
        return torch.ones((1,))

    def marginal_prob_std(self, t):
        """Compute the standard deviation of $p_{0:t}(x(t) | x(0))$.
           The variance is given by \int_0^t g(s) ds.
        """    
        return torch.sqrt((self.sigma**(2 * t) - 1.) / 2. / np.log(self.sigma))

    def diffusion_coeff(self, t):
        """Compute the diffusion coefficient of our SDE g(t).
        """
        return self.sigma**t

class GMM_score(nn.Module):
    '''
        GMM score function which corresponds to the stationary point of the denoising score-matching loss function
    '''
    def __init__(self, train_data, marginal_prob_mean, marginal_prob_std):
        super().__init__()
        self.train_data = train_data
        self.marginal_prob_mean = marginal_prob_mean 
        self.marginal_prob_std  = marginal_prob_std

    def pdf_weights(self, x, t):
        # compute mean and sigma
        sigma = self.marginal_prob_std(t)
        meanf = self.marginal_prob_mean(t)
        # evaluate Gaussian densities
        logpdf_x_yi = torch.zeros((x.shape[0],self.train_data.shape[0]))
        for i in range(self.train_data.shape[0]):
            logpdf_x_yi[:,i] = self.log_normal_pdf(x, meanf[:,None] * self.train_data[i,:], sigma)
        # compute weighted average
        weights = torch.softmax(logpdf_x_yi, axis=1)
        return weights
        
    def forward(self, x, t):
        # compute weights
        weights = self.pdf_weights(x, t)     
        # compute sigma
        sigma = self.marginal_prob_std(t)
        # compute weighted average
        evals = torch.mm(weights, self.train_data) 
        evals[torch.isnan(evals)] = 0.0
        return (evals - x)/(sigma[:, None]**2)

    def log_normal_pdf(self, x, y, sigma):
        # ignoring normalization constant
        assert(x.shape[0] == len(sigma))
        return -0.5*torch.sum((x - y)**2,axis=1)/sigma**2

    def normal_pdf(self, x, y, sigma):
        return torch.exp(self.log_normal_pdf(x, y, sigma))

class GMM_score_TikhonovRegularized(GMM_score):
    '''
        GMM score function which corresponds to the stationary point of the regularized denoising
        score-matching loss function using Tikonov regularization with parameter const*g(t)^2/sigma(t)^2
    '''
    def __init__(self, train_data, marginal_prob_mean, marginal_prob_std, diffusion_coeff, constant=1.0):
        super().__init__(train_data, marginal_prob_mean, marginal_prob_std)
        self.diffusion_coeff = diffusion_coeff
        self.constant = constant
        
    def forward(self, x, t):
        # compute weights
        weights = self.pdf_weights(x, t)     
        # compute sigma and diffusion coefficient
        sigma = self.marginal_prob_std(t)
        g = self.diffusion_coeff(t)
        # compute weighted average
        evals = torch.mm(weights, self.train_data) 
        evals[torch.isnan(evals)] = 0.0
        return (evals - x)/(sigma[:, None]**2 + self.constant * g[:, None]**2)

class GMM_score_EmpiricalBayes(GMM_score):
    '''
        GMM score function which corresponds to the stationary point of the regularized denoising
        score-matching loss function using Empirical Bayes regularization with parameter tau_const
    '''
    def __init__(self, train_data, marginal_prob_mean, marginal_prob_std, tau_constant=1.0):
        super().__init__(train_data, marginal_prob_mean, marginal_prob_std)
        self.tau_constant = tau_constant
        
    def pdf_weights(self, x, t):
        # compute mean and sigma
        sigma = self.marginal_prob_std(t)
        meanf = self.marginal_prob_mean(t)
        # evaluate Gaussian densities
        logpdf_x_yi = torch.zeros((x.shape[0],self.train_data.shape[0]))
        for i in range(self.train_data.shape[0]):
            logpdf_x_yi[:,i] = self.log_normal_pdf(x, meanf[:,None] * self.train_data[i,:], sigma)
        # compute regularized weighted average
        log_denominator = torch.logsumexp(logpdf_x_yi, dim=1)
        logtau = torch.log(torch.tensor(self.tau_constant))
        reg_denominator = torch.max(log_denominator, logtau*torch.ones_like(log_denominator))
        weights = torch.exp(logpdf_x_yi - reg_denominator[:,None])
        return weights

    def forward(self, x, t):
        # compute weights
        weights = self.pdf_weights(x, t)     
        # compute sigma
        sigma = self.marginal_prob_std(t)
        # compute weighted average
        evals = torch.mm(weights, self.train_data) 
        evals -= torch.sum(weights,axis=1)[:,None] * x
        evals[torch.isnan(evals)] = 0.0
        return evals/(sigma[:, None]**2)
