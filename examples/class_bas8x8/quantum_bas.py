# %% markdown
# # Quantum-Assisted RBM training on the BAS Dataset for Reconstruction
# This is an example on quantum-assisted training of an RBM on the BAS(4,4)
# dataset.
# Developed by: Jose Pinilla
# %%
# Required packages
import qaml
import torch
SEED = 0
torch.manual_seed(SEED) # For deterministic weights

import matplotlib.pyplot as plt
import torchvision.transforms as torch_transforms

# %%
################################# Hyperparameters ##############################
M,N = SHAPE = (8,8)
DATA_SIZE = N*M
HIDDEN_SIZE = 64
EPOCHS = 50
SAMPLES = None
BATCH_SIZE = 400
TRAIN,TEST = SPLIT = 400,110
# Stochastic Gradient Descent
learning_rate = 0.1
weight_decay = 1e-4
momentum = 0.5

# %%
#################################### Input Data ################################
bas_dataset = qaml.datasets.BAS(*SHAPE,embed_label=True,transform=torch.Tensor)
train_dataset,test_dataset = torch.utils.data.random_split(bas_dataset,[*SPLIT])
train_sampler = torch.utils.data.RandomSampler(train_dataset,replacement=False,
                                               num_samples=SAMPLES)
train_loader = torch.utils.data.DataLoader(train_dataset,sampler=train_sampler,
                                           batch_size=BATCH_SIZE)

# PLot all data
fig,axs = plt.subplots(6,5)
for ax,(img,label) in zip(axs.flat,train_dataset):
    ax.matshow(img.view(*SHAPE),vmin=0,vmax=1); ax.axis('off')
plt.tight_layout()

# %%
################################# Model Definition #############################
# Specify model with dimensions
rbm = qaml.nn.RBM(DATA_SIZE,HIDDEN_SIZE)

# Initialize biases
torch.nn.init.constant_(rbm.b,0.5)
torch.nn.init.zeros_(rbm.c)
torch.nn.init.uniform_(rbm.W,-0.5,0.5)

# Set up optimizers
optimizer = torch.optim.SGD(rbm.parameters(), lr=learning_rate,
                            weight_decay=weight_decay,momentum=momentum)

# Set up training mechanisms
# Trainable inverse temperature with separate optimizer
beta = torch.nn.Parameter(torch.tensor(2.5), requires_grad=True)
beta_optimizer = torch.optim.SGD([beta],lr=0.01)
solver_name = "Advantage_system1.1"
qa_sampler = qaml.sampler.QASampler(rbm,solver=solver_name,beta=beta)

# Loss and autograd
CD = qaml.autograd.SampleBasedConstrastiveDivergence()
betaGrad = qaml.autograd.AdaptiveBeta()

# # %%
# ################################## Model Training ##############################
# Set the model to training mode
rbm.train()
err_log = []
accuracy_log = []
b_log = [rbm.b.detach().clone().numpy()]
c_log = [rbm.c.detach().clone().numpy()]
W_log = [rbm.W.detach().clone().numpy().flatten()]
for t in range(EPOCHS):
    epoch_error = 0
    epoch_error_beta = 0

    for img_batch, labels_batch in train_loader:
        input_data = img_batch.flatten(1)

        # Negative Phase
        vk, prob_hk = qa_sampler(BATCH_SIZE,auto_scale=False)
        # Positive Phase
        v0, prob_h0 = input_data, rbm(input_data)

        # Reconstruction error from Contrastive Divergence
        err = CD.apply((v0,prob_h0), (vk,prob_hk), *rbm.parameters())

        # Do not accumulate gradients
        optimizer.zero_grad()

        # Compute gradients
        err.backward()

        # Update parameters
        optimizer.step()

        #Accumulate error for this epoch
        epoch_error  += err.item()

    # Error Log
    b_log.append(rbm.b.detach().clone().numpy())
    c_log.append(rbm.c.detach().clone().numpy())
    W_log.append(rbm.W.detach().clone().numpy().flatten())
    err_log.append(epoch_error)
    print(f"Epoch {t} Reconstruction Error = {epoch_error}")
    ############################## CLASSIFICATION ##################################
    count = 0
    mask = torch.ones(1,M,N)
    for test_data, test_label in test_dataset:
        test_data[-2:,-1] = 0.5
        prob_hk = rbm(test_data.flatten())
        label_pred = rbm.generate(prob_hk).view(*SHAPE)[-2:,-1]
        if label_pred.argmax() == test_label.argmax():
            count+=1
    accuracy_log.append(count/TEST)
    print(f"Testing accuracy: {count}/{TEST} ({count/TEST:.2f})")

# Set the model to evaluation mode
# rbm.eval()

# # %%
# ############################## Logging Directory ###############################
import os
directory = 'BAS88_beta25_noscale_50_Adv11'
if not os.path.exists(directory):
        os.makedirs(directory)
seed = torch.initial_seed()
if not os.path.exists(f'{directory}/{seed}'):
        os.makedirs(f'{directory}/{seed}')

# # %%
# ############################ Store Model and Logs ##############################
torch.save(b_log,f"./{directory}/{seed}/b.pt")
torch.save(c_log,f"./{directory}/{seed}/c.pt")
torch.save(W_log,f"./{directory}/{seed}/W.pt")
torch.save(err_log,f"./{directory}/{seed}/err.pt")
torch.save(accuracy_log,f"./{directory}/{seed}/accuracy.pt")
torch.save(dict(qa_sampler.embedding),f"./{directory}/{seed}/embedding.pt")

# %% md
############################ Load Model and Logs ###############################
b_log = torch.load(f"./{directory}/{SEED}/b.pt")
c_log = torch.load(f"./{directory}/{SEED}/c.pt")
W_log = torch.load(f"./{directory}/{SEED}/W.pt")
err_log = torch.load(f"./{directory}/{SEED}/err.pt")
accuracy_log = torch.load(f"./{directory}/{SEED}/accuracy.pt")
embedding = torch.load(f"./{directory}/{SEED}/embedding.pt")

rbm.b.data = torch.tensor(b_log[-1])
rbm.c.data = torch.tensor(c_log[-1])
rbm.W.data = torch.tensor(W_log[-1]).view(rbm.H,rbm.V)
qa_sampler = qaml.sampler.QASampler(rbm,solver=solver_name,
                                    beta=beta,embedding=embedding)

# %%
################################# qBAS Score ###################################
scale = qa_sampler.scalar*qa_sampler.beta
num_samples = 1000
gibbs_sampler = qaml.sampler.GibbsNetworkSampler(rbm,beta=scale)
prob_v,_ = gibbs_sampler(torch.rand(num_samples,DATA_SIZE),k=200)
img_samples = prob_v.view(num_samples,*SHAPE).bernoulli()
# PLot some samples
fig,axs = plt.subplots(4,5)
for ax,img in zip(axs.flat,img_samples):
    ax.matshow(img.view(*SHAPE),vmin=0,vmax=1); ax.axis('off')
plt.tight_layout()
# Get and print score
p,r,score = bas_dataset.score(img_samples)
print(f"qBAS : Precision = {p:.02} Recall = {r:.02} Score = {score:.02}")

# %%
############################## RECONSTRUCTION ##################################
k = 5
hist = {}
count = 0
scale = qa_sampler.scalar*qa_sampler.beta
mask = torch_transforms.functional.erase(torch.ones(1,M,N),2,2,4,4,0).flatten()
for img, label in bas_dataset:
    clamped = mask*(img.flatten().detach().clone())
    prob_hk = rbm.forward(clamped + (1-mask)*0.5,scale=scale)
    prob_vk = rbm.generate(prob_hk,scale=scale).detach()
    for _ in range(k):
        masked = clamped + (1-mask)*prob_vk.data
        prob_hk.data = rbm.forward(masked,scale=scale).data
        prob_vk.data = rbm.generate(prob_hk,scale=scale).data
    recon = (clamped + (1-mask)*prob_vk).bernoulli().view(img.shape)
    if recon.equal(img):
        count+=1
    num = torch.count_nonzero(recon.to(bool).bitwise_xor(img.to(bool))).item()
    hist[num]=hist.get(num,0)+1
print(f"Dataset Reconstruction: {count/(TEST+TRAIN):.02}")
plt.bar(hist.keys(),hist.values())
plt.ylabel('Frequency')
plt.xlabel('Incorrect Bits')
# %%
############################ MODEL VISUALIZATION ###############################

# Testing accuracy graph
fig, ax = plt.subplots()
plt.plot(accuracy_log)
plt.ylabel("Testing Accuracy")
plt.xlabel("Epoch")
plt.savefig("quantum_accuracy.pdf")

# L1 error graph
fig, ax = plt.subplots()
plt.plot(err_log)
plt.ylabel("Reconstruction Error")
plt.xlabel("Epoch")
plt.savefig("quantum_err_log.pdf")

# Visible bias graph
fig, ax = plt.subplots()
ax.set_prop_cycle('color', list(plt.get_cmap('turbo',DATA_SIZE).colors))
lc_v = ax.plot(b_log)
plt.legend(iter(lc_v),[f'b{i}' for i in range(DATA_SIZE)],ncol=2,bbox_to_anchor=(1,1))
plt.ylabel("Visible Biases")
plt.xlabel("Epoch")
plt.savefig("quantum_b_log.pdf")

# Hidden bias graph
fig, ax = plt.subplots()
ax.set_prop_cycle('color', list(plt.get_cmap('turbo',HIDDEN_SIZE).colors))
lc_h = plt.plot(c_log)
plt.legend(lc_h,[f'c{i}' for i in range(HIDDEN_SIZE)],ncol=2,bbox_to_anchor=(1,1))
plt.ylabel("Hidden Biases")
plt.xlabel("Epoch")
plt.savefig("quantum_c_log.pdf")

# Weights graph
fig, ax = plt.subplots()
ax.set_prop_cycle('color', list(plt.get_cmap('turbo',rbm.V*rbm.H).colors))
lc_w = plt.plot(W_log)
plt.ylabel("Weights")
plt.xlabel("Epoch")
plt.savefig("quantum_W_log.pdf")

# %%
################################## ENERGY ######################################
scale=qa_sampler.scalar*qa_sampler.beta
data_energies = []
for img,label in bas_dataset:
    data = img.flatten()
    data_energies.append(rbm.free_energy(data).item())

rand_energies = []
rand_data = torch.rand(len(bas_dataset)*100,rbm.V)
for img in rand_data:
    rand_energies.append(rbm.free_energy(img.bernoulli()).item())

gibbs_energies = []
gibbs_sampler = qaml.sampler.GibbsNetworkSampler(rbm,beta=scale)
for img,label in bas_dataset:
    data = img.flatten()
    prob_v,prob_h = gibbs_sampler(data,k=50)
    gibbs_energies.append(rbm.free_energy(prob_v.bernoulli()).item())

qa_energies = []
qa_sampleset = qa_sampler(num_reads=BATCH_SIZE,auto_scale=True)
for s_v,s_h in zip(*qa_sampleset):
    qa_energies.append(rbm.free_energy(s_v.detach()).item())

plot_data = [(data_energies,  'Data',    'blue'),
             (rand_energies,  'Random',  'red'),
             (gibbs_energies, 'Gibbs-5', 'green'),
             (qa_energies,    'Quantum', 'orange')]

hist_kwargs = {'ec':'k','lw':2.0,'alpha':0.5,'histtype':'stepfilled','bins':100}
weights = lambda data: [1./len(data) for _ in data]

fig, ax = plt.subplots(figsize=(15,10))
for data,name,color in plot_data:
    ax.hist(data,weights=weights(data),label=name,color=color,**hist_kwargs)

plt.xlabel("Energy")
plt.ylabel("Count/Total")
plt.legend(loc='upper right')
plt.savefig("quantum_energies.pdf")

# %%
################################## VISUALIZE ###################################
plt.matshow(rbm.b.detach().view(*SHAPE))
plt.colorbar()
plt.savefig("quantum_b.pdf")
plt.matshow(rbm.c.detach().view(1,HIDDEN_SIZE))
plt.yticks([])
plt.colorbar()
plt.savefig("quantum_c.pdf")

fig,axs = plt.subplots(HIDDEN_SIZE//4,4)
for i,ax in enumerate(axs.flat):
    weight_matrix = rbm.W[i].detach().view(*SHAPE)
    ms = ax.matshow(weight_matrix)
    ax.axis('off')
fig.subplots_adjust(wspace=0.1, hspace=0.1)
cbar = fig.colorbar(ms, ax=axs.ravel().tolist(), shrink=0.95)
plt.savefig("quantum_weights.pdf")

# %%
########################### Check parameter range ##############################
h_range = qa_sampler.properties['h_range']
J_range = qa_sampler.properties['extended_j_range']
target_ising = qa_sampler.embed_bqm()
linear = target_ising.linear.values()
quad = target_ising.quadratic.values()
print(f"Linear range [{min(linear):.2} <> {max(linear):.2}] @ device={h_range}")
print(f"Quadratic range [{min(quad):.2} <> {max(quad):.2}] @ device={J_range}")
