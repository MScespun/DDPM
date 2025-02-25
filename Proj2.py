import torch
import numpy as np
from diffusion import q_sample, posterior_q, Denoising, denoise_with_mu
from utils import pack_data, unpack_1d_data, scatter_pixels
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from operator import mul
from functools import reduce
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

#se si dispone di un'opprtuna scehda grafica i dati verrano mandati a questa per risultati più veloci
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# converto l'immagine in puntini con scatter pixels e la plotto
x, y = scatter_pixels('pika.png')
x = [x/25 - 3 for x in x]
y = [y/25 - 2 for y in y]
df = pd.DataFrame({'x': x,
                    'y':y
                    })
ax = sns.scatterplot(data=df,x='x',y='y')
plt.show()
## Salvo gli assi per plottare dopo
y_ax = ax.get_ylim()
x_ax = ax.get_xlim()
axes = (x_ax,y_ax)

# mando i dati al device
one_d_data = pack_data(x,y)
x_init = torch.tensor(one_d_data).to(torch.float32).to(device)

DATA_SIZE = len(x_init)


#Parametri di diffusione

beta_start = .0004
beta_end = .02
num_diffusion_timesteps = 30
betas = np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps) ** 2
print(betas)
alphas = 1 - betas
# mando parametri al device
betas = torch.tensor(betas).to(torch.float32).to(device)
alphas = torch.tensor(alphas).to(torch.float32).to(device)
list_bar_alphas = [alphas[0]]
for t in range(1, num_diffusion_timesteps):
    list_bar_alphas.append(reduce(mul, alphas[:t]))

list_bar_alphas = torch.cumprod(alphas, axis=0).to(torch.float32).to(device)
training_steps_per_epoch = 40

#definisco loss function e metodo di discesa del gradiente e la rete neurale
criterion = nn.MSELoss()
denoising_model = Denoising(DATA_SIZE, num_diffusion_timesteps).to(device)
denoising_model.emb = denoising_model.emb.to(device)
optimizer = optim.AdamW(denoising_model.parameters())

#training della rete neurale
pbar = tqdm(range(20))
for epoch in pbar:

    running_loss = 0.0
    Ts = np.random.randint(1, num_diffusion_timesteps, size=training_steps_per_epoch)
    for _, t in enumerate(Ts):
        q_t = q_sample(x_init, t, list_bar_alphas, device)
        mu_t, cov_t = posterior_q(x_init, q_t, t, alphas, list_bar_alphas, device)
        optimizer.zero_grad()
        mu_theta = denoising_model(q_t, t)
        loss = criterion(mu_theta, mu_t)
        loss.backward()
        optimizer.step()
        running_loss += loss.detach()
    pbar.set_description('Epoch: {} Loss: {}'.format(epoch, running_loss / training_steps_per_epoch))
print('Finished Training')

#genera dato casuale
data = torch.distributions.MultivariateNormal(loc=torch.zeros(DATA_SIZE),covariance_matrix=torch.eye(DATA_SIZE)).sample().to(device)

#quest'ultima parte serve a creare la gif che mostra il procedimento
from celluloid import Camera
import pandas as pd
import matplotlib.pyplot as plt

fig = plt.figure()
camera = Camera(fig)

data_plot = data.detach().cpu().numpy()
x_new, y_new = unpack_1d_data(data_plot)
df_new = pd.DataFrame({'x': x_new,
                           'y': y_new
                           })

graph = sns.scatterplot(data=df_new, x='x', y='y')
plt.pause(0.1)
graph.set_xlim(axes[0])
graph.set_ylim(axes[1])

camera.snap()

for d in range(1, num_diffusion_timesteps):
    data = denoise_with_mu(denoising_model,data,num_diffusion_timesteps-d, alphas, DATA_SIZE, device)
    data_plot = data.detach().cpu().numpy()
    x_new, y_new = unpack_1d_data(data_plot)
    df_new = pd.DataFrame({'x': x_new,
                           'y': y_new
                           })

    graph = sns.scatterplot(data=df_new, x='x', y='y')
    plt.pause(0.1)
    graph.set_xlim(axes[0])
    graph.set_ylim(axes[1])

    camera.snap()

anim = camera.animate(blit=False)
anim.save('output2.gif',fps=24, dpi=120)