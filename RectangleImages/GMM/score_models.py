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
    def __init__(self):
        self.T     = 1.
        self.eps   = 1e-3

    def SM_loss(self, net, x):
        """The loss function for training score-based generative models.
        Args:
            net: A PyTorch model of time-dependent score-based model
            x: A mini-batch of training data
        """
        random_t = torch.rand(x.shape[0], device=x.device) * (self.T - self.eps) + self.eps  
        z = torch.randn_like(x)
        mean = self.marginal_prob_mean(random_t)
        std  = self.marginal_prob_std(random_t)
        perturbed_x = x * mean[:, None] + z * std[:, None]
        score = net(perturbed_x, random_t)
        return torch.sum((score * std[:, None] + z)**2, dim=(1))

    def sampler(self, score_net, latents, num_steps=100):
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

        # extract devicev
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
  
        # Run the black-box ODE solver
        res = integrate.solve_ivp(ode_func, (self.T, self.eps), init_x.reshape(-1).cpu().numpy(), rtol=err_tol, atol=err_tol, method='RK45')  
        x = torch.tensor(res.y[:, -1], device=device).reshape(latents.shape)
        
        return x

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
        
        return x

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
        """    
        return torch.sqrt((self.sigma**(2 * t) - 1.) / 2. / np.log(self.sigma))

    def diffusion_coeff(self, t):
        """Compute the diffusion coefficient of our SDE g(t).
        """
        return self.sigma**t


class GMM_score_rescaled(nn.Module):
    def __init__(self, train_data, marginal_prob_std):
        super().__init__()
        self.train_data = train_data
        self.marginal_prob_std = marginal_prob_std

    def forward(self, x, t):
        # compute sigma
        sigma = self.marginal_prob_std(t)
        # evaluate Gaussian densities
        pdf_x_yi = torch.zeros((self.train_data.shape[0],x.shape[0]))
        for i in range(self.train_data.shape[0]):
            pdf_x_yi[i,:] = self.normal_pdf(self.train_data[i,:], x, sigma)
        # compute weighted sum
        evals = torch.sum(pdf_x_yi * self.train_data,axis=0) / torch.sum(pdf_x_yi,axis=0)
        # correct values to be zero instead of nan
        evals[torch.isnan(evals)] = 0.0
        evals = evals.reshape((x.shape[0],1))
        sigma2 = self.marginal_prob_std(t)**2
        return (evals - x)/sigma2[:, None]
        #return evals #(evals - x)/sigma.reshape(-1,1)))/sigma.reshape(-1,1)

    def normal_pdf(self, y, x, sigma):
        # ignoring normalization constant
        assert(x.shape[0] == len(sigma))
        return torch.exp(-0.5*torch.sum((x - y)**2,axis=1)/sigma**2)
