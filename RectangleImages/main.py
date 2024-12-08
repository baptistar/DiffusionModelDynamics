import torch
import dnnlib
from generate import edm_sampler, edm_sampler_fixednoise
import matplotlib.pyplot as plt
from timeit import default_timer
import copy

## Set parameters

device = torch.device('cuda:1')

Neval_samps  = 16
Ntrain_samps = 2

epochs = 50000
bsize = 2

cur_nimg = 1
lr_rampup_kimg = 10000
ema_halflife_kimg = 500
ema_rampup_ratio = 0.05

iters_plotting = 100
iters_checkpt  = 100

# Generate data 
data = torch.zeros((Ntrain_samps,1,64,64))
data[0,0,6:18,6:18] = 1.0
data[1,0,40:54,40:54] = 1.0

# fix the noise for sampling
Neval_samps = 16
fixed_noise_sequence = torch.randn((50,Neval_samps,1,64,64)).to(device)
fixed_latents = torch.randn(Neval_samps,1,64,64).to(device)

data_set = torch.utils.data.TensorDataset(data)
data_loader = torch.utils.data.DataLoader(data_set, batch_size=1, shuffle=True)

plt.figure()
for j in range(Ntrain_samps):
    plt.subplot(1,2,j+1)
    plt.imshow(data[j,0,...])
    plt.xticks([])
    plt.yticks([])

plt.savefig('truth.png')
plt.close()

# define network
network_kwargs = dnnlib.EasyDict()
network_kwargs.update(model_type='SongUNet', embedding_type='positional', encoder_type='standard', decoder_type='standard')
network_kwargs.update(channel_mult_noise=1, resample_filter=[1,1], model_channels=128, channel_mult=[2,2,2])
network_kwargs.class_name = 'training.networks.EDMPrecond'
network_kwargs.update(dropout=0.0, use_fp16=False)
interface_kwargs = dict(img_resolution=64, img_channels=1, label_dim=0)

net = dnnlib.util.construct_class_by_name(**network_kwargs, **interface_kwargs) # subclass of torch.nn.Module
net.train().requires_grad_(True).to(device)
ema = copy.deepcopy(net).eval().requires_grad_(False)

optimizer = torch.optim.Adam(net.parameters(), lr=10e-4, betas=[0.9,0.999], eps=1e-8)

loss_kwargs = dnnlib.EasyDict()
loss_kwargs.class_name = 'training.loss.EDMLoss'
loss_fn = dnnlib.util.construct_class_by_name(**loss_kwargs)

# define arrays to store loss
score_matching_loss = []
L2_manifold_loss  = []
Linf_manifold_loss  = []
L2_mean_manifold_loss  = []

for ep in range(epochs):
    err = 0.0
    count = 0
    net.train()
    
    t1 = default_timer()
    for x in data_loader:
        optimizer.zero_grad(set_to_none=True)

        x = x[0]
        x = x.to(device)
        loss = loss_fn(net, x).sum()/x.size(0)
        loss.backward()
        
        # Update weights
        for g in optimizer.param_groups:
            g['lr'] = 10e-4 * min(cur_nimg / max(lr_rampup_kimg * 1000, 1e-8), 1)
            
        for param in net.parameters():
            if param.grad is not None:
                torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
        optimizer.step()

        # Update EMA.
        ema_halflife_nimg = ema_halflife_kimg * 1000
        if ema_rampup_ratio is not None:
            ema_halflife_nimg = min(ema_halflife_nimg, cur_nimg * ema_rampup_ratio)
        ema_beta = 0.5 ** (x.size(0) / max(ema_halflife_nimg, 1e-8))
        for p_ema, p_net in zip(ema.parameters(), net.parameters()):
            p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))

        err += loss.item()

        cur_nimg += x.size(0)
        count += x.size(0)

    # # generate samples
    # ema.eval()
    # latents = torch.randn(Neval_samps,1,64,64).to(device)
    # with torch.no_grad(): 
    #     samples_rand = edm_sampler(ema, latents, num_steps=40).cpu()

    # # Compute Linf distance to data manifolds
    # Linf_dist = torch.zeros((Neval_samps))
    # for j in range(Neval_samps):
    #     L2_error = torch.zeros((Ntrain_samps))
    #     for k in range(Ntrain_samps):
    #         L2_error[k] = torch.sum((samples_rand[j,:,:,:] - data[k,:,:,:])**2)
    #     Linf_dist[j] = torch.min(L2_error)
    # Linf_manifold_loss.append(torch.max(Linf_dist))

    # store loss
    score_matching_loss.append(err/count)

    print(ep+1, err, default_timer() - t1)
    
    # generate new samples
    if (ep+1) % iters_plotting == 0:
        ema.eval()
        latents = torch.randn(Neval_samps,1,64,64).to(device)
        with torch.no_grad(): 
            samples_rand = edm_sampler(ema, latents, num_steps=40).cpu()

        plt.figure()
        for j in range(Neval_samps):
            plt.subplot(4,4,j+1)
            plt.imshow(samples_rand[j,0,...])
            plt.xticks([])
            plt.yticks([])

        plt.savefig('figures/new_samples_' + str(ep+1) + '.png')
        plt.close()

        # Compute Linf distance to data manifolds
        Linf_dist = torch.zeros((Neval_samps))
        L2_dist = torch.zeros((Neval_samps))
        for j in range(Neval_samps):
            L2_error = torch.zeros((Ntrain_samps))
            Linf_error = torch.zeros((Ntrain_samps))
            for k in range(Ntrain_samps):
                L2_error[k] = torch.sum((samples_rand[j,:,:,:] - data[k,:,:,:])**2)
                Linf_error[k] = torch.max((samples_rand[j,:,:,:] - data[k,:,:,:]))
            Linf_dist[j] = torch.min(Linf_error)
            L2_dist[j] = torch.min(L2_error)
        # append to data vector
        L2_manifold_loss.append(torch.max(L2_dist))
        L2_mean_manifold_loss.append(torch.mean(L2_dist))
        Linf_manifold_loss.append(torch.max(Linf_dist))

        # plot samples with fixed noise
        with torch.no_grad(): 
            samples_fixed = edm_sampler_fixednoise(ema, fixed_latents, fixed_noise_sequence, num_steps=40).cpu()

        plt.figure()
        for j in range(Neval_samps):
            plt.subplot(4,4,j+1)
            plt.imshow(samples_fixed[j,0,...])
            plt.xticks([])
            plt.yticks([])

        plt.savefig('figures/new_samples_fixed_' + str(ep+1) + '.png')
        plt.close()


    if (ep+1) % iters_checkpt == 0:
        torch.save(net.state_dict(), 'net.pt')
        # store losses
        torch.save({"Linf_loss":Linf_manifold_loss, "L2_loss":L2_manifold_loss, "L2_mean_loss":L2_mean_manifold_loss, "SM_loss":score_matching_loss}, 'losses_twosamps.pt')
