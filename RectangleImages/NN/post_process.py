import torch
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    'font.size': 18,
})

folder = './'
data = torch.load(folder+'losses_twosamps.pt')
Linf_loss  = data['Linf_loss']
SM_loss    = data['SM_loss']

Linf_steps = torch.arange(1,len(Linf_loss)*100,100)
SM_steps   = torch.arange(0,len(SM_loss))

#plt.figure(figsize=(6,5))

#plt.subplot(2,1,1)
#plt.plot(SM_steps, SM_loss)
#plt.xlabel('Iteration')
#plt.ylabel('Score matching loss')

#plt.subplot(2,1,2)
plt.figure()
plt.plot(Linf_steps, torch.tensor(Linf_loss))
plt.xlabel('Epoch')
plt.ylabel('$L^2$ distance to data manifold')

plt.savefig(folder+'losses.pdf')
