# %% markdown
# RBM unsupervised training example of MNIST using Persistent Contrastive
# Divergence (PCD-1).

# %%
import qaml
import torch

import matplotlib.pyplot as plt

import torchvision.datasets as torch_datasets
import torchvision.transforms as torch_transforms

# %%
################################# Hyperparameters ##############################
EPOCHS = 5
BATCH_SIZE = 64
# Stochastic Gradient Descent
learning_rate = 1e-3
weight_decay = 1e-4
momentum = 0.5
# %%
#################################### Input Data ################################
train_dataset = torch_datasets.MNIST(root='./data/', train=True,
                                     transform=torch_transforms.ToTensor(),
                                     download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                           shuffle=True)

test_dataset = torch_datasets.MNIST(root='./data/', train=False,
                                    transform=torch_transforms.ToTensor(),
                                    download=True)
test_loader = torch.utils.data.DataLoader(test_dataset)
# %%
################################# Model Definition #############################
DATA_SIZE = len(train_dataset.data[0].flatten())
HIDDEN_SIZE = 128

# Specify model with dimensions
rbm = qaml.nn.RBM(DATA_SIZE, HIDDEN_SIZE)

# Set up optimizer
optimizer = torch.optim.SGD(rbm.parameters(), lr=learning_rate,
                                              weight_decay=weight_decay,
                                              momentum=momentum)
# Set up training mechanisms
sampler = qaml.sampler.PersistentGibbsNetworkSampler(rbm, BATCH_SIZE)
CD = qaml.autograd.ConstrastiveDivergence()
# %%
################################## Model Training ##############################
# Set the model to training mode
rbm.train()
err_log = []
for t in range(EPOCHS):
    epoch_error = torch.Tensor([0.])
    for img_batch, labels_batch in train_loader:

        input_data = img_batch.flatten(1)

        # Positive Phase
        v0, prob_h0 = input_data, rbm(input_data)
        # Negative Phase
        vk, prob_hk = sampler(len(v0), k=1)

        # Reconstruction error from Contrastive Divergence
        err = CD.apply((v0,prob_h0), (vk,prob_hk), *rbm.parameters())

        # Do not accumulated gradients
        optimizer.zero_grad()
        # Compute gradients. Save compute graph at last epoch
        err.backward(retain_graph=(t == EPOCHS-1))

        # Update parameters
        optimizer.step()
        epoch_error  += err
    err_log.append(epoch_error.item())
    print(f"Epoch {t} Reconstruction Error = {epoch_error.item()}")

plt.plot(err_log)
plt.ylabel("Reconstruction Error")
plt.xlabel("Epoch")
rbm.eval()

# %%
################################# VISUALIZE ####################################
# Computation Graph
from torchviz import make_dot
make_dot(err)

# %% raw
# Option to save for future use
torch.save(rbm,"mnist_unsupervised.pt")

# %% raw
# Option to load existing model
rbm = torch.load("mnist_unsupervised.pt")

# %% markdown
# Plot the distribution of energies for (a) the training data, (b) the test data
# (c) a set of random samples of visible configurations. The expected result is
# to have both (a) and (b) as clusters of lower energy, and (c) as a normal
# distribution to the right, i.e. high energy.

# %%
################################## ENERGY ######################################
data_energies = []
for img,_ in train_dataset:
    data_energies.append(rbm.free_energy(img.float().view(rbm.V)).item())

test_energies = []
for img,_ in test_dataset:
    test_energies.append(rbm.free_energy(img.float().view(rbm.V)).item())

rand_energies = []
for _ in range(len(train_dataset)):
    rand_energies.append(rbm.free_energy(torch.rand(rbm.V)).item())

plt.hist(data_energies,label="Data",bins=100)
plt.hist(test_energies,label="Test",bins=100)
plt.hist(rand_energies,label="Random",bins=100)
plt.ylabel("Count")
plt.xlabel("Energy")
plt.legend()

# %% markdown
# It's sometimes useful to visualize the distribution of linear biases and
# weights between visible and hidden layers

# %%
################################# VISIBLE ######################################
plt.matshow(rbm.b.detach().view(28, 28))
plt.colorbar()

# %%
################################# WEIGHTS ######################################
fig,axs = plt.subplots(HIDDEN_SIZE//8,8)
for i,ax in enumerate(axs.flat):
    weight_matrix = rbm.W[i].detach().view(28, 28)
    ms = ax.matshow(weight_matrix, cmap='viridis', vmin=-1, vmax=1)
    ax.axis('off')
fig.subplots_adjust(wspace=0.0, hspace=0.0)
cbar = fig.colorbar(ms, ax=axs.ravel().tolist(), shrink=0.95)

# %% markdown
# It's possible to sample from the joint probability of the model and plot the
# visible units of those samples. This doesn't necessarily sample an image of a
# number.

# %%
################################## SAMPLE ######################################
SAMPLES = 4
prob_vk,prob_hk = sampler(SAMPLES,k=3,init=torch.rand(BATCH_SIZE,rbm.V)*0.1)
fig,axs = plt.subplots(1,SAMPLES)
for ax,vk in zip(axs.flat,prob_vk):
    ax.matshow(vk.detach().view(28, 28))
    ax.axis('off')
fig.subplots_adjust(wspace=0.0, hspace=0.0)

# %%
############################ NOISE RECONSTRUCTION ##############################
input_data, label = train_loader.dataset[85] # Random input
corrupt_data = (input_data + torch.randn_like(input_data)*0.5).view(1,784)
prob_vk,prob_hk = sampler(1,k=1,init=corrupt_data.clone())
recon_data = prob_vk.detach()

fig,axs = plt.subplots(1,3)
axs[0].matshow(input_data.view(28,28))
axs[1].matshow(corrupt_data.view(28,28))
axs[2].matshow(recon_data.detach().view(28,28))

# %%
############################## RECONSTRUCTION ##################################
input_data, label = train_loader.dataset[4]
mask = torch.ones_like(input_data)
for i in range(0,15): # Is there a nicer way to create random masks?
    for j in range(0,15):
        mask[0][j][i] = 0

corrupt_data = (input_data*mask).view(1,784)

prob_vk,prob_hk = sampler(1,k=1,init=corrupt_data.clone())
fig,axs = plt.subplots(1,3)
axs[0].matshow(input_data.view(28, 28))
axs[1].matshow(corrupt_data.view(28, 28))
axs[2].matshow(prob_vk.detach().view(28, 28))

# %% markdown
# Even though this model was trained unsupervised, it is possible to now use it
# as a pre-trained model and "fit" its hidden layers and output and perform
# training on that.

# %%
############################### CLASSIFIER ####################################
LABEL_SIZE = len(train_dataset.classes)

model = torch.nn.Sequential(rbm,
                            torch.nn.Linear(HIDDEN_SIZE,LABEL_SIZE),)
loss_fn = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for t in range(10):
    for v_batch, labels_batch in train_loader:
        # Forward pass: compute predicted y by passing x to the model.
        y_pred = model(v_batch.view(len(v_batch),DATA_SIZE))

        # Compute and print loss.
        loss = loss_fn(y_pred, torch.nn.functional.one_hot(labels_batch,10)*1.0)

        # Before the backward pass, use the optimizer object to zero all of the
        # gradients for the Tensors it will update (which are the learnable weights
        # of the model)
        optimizer.zero_grad()

        # Backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()

        # Calling the step function on an Optimizer makes an update to its parameters
        optimizer.step()
    print(f"Epoch {t} Loss = {loss.item()}")

count = 0
for test_data, test_label in test_loader:
    label_pred = model(test_data.view(1,DATA_SIZE)).argmax()
    if label_pred == test_label:
        count+=1
print(f"Testing accuracy: {count}/{len(test_dataset)}")
