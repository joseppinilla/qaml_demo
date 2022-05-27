# %% markdown
# # Classical RBM training on the Bars-And-Stripes Dataset for Reconstruction
# This is an example on classical Gibbs training of an RBM on the BAS(4,4)
# dataset.
# Developed by: Jose Pinilla
# %%
# Required packages
import qaml
import torch
torch.manual_seed(0) # For deterministic weights

import matplotlib.pyplot as plt
import torchvision.transforms as torch_transforms

# %%
################################# Hyperparameters ##############################
M,N = SHAPE = (8,8)
DATA_SIZE = N*M
HIDDEN_SIZE = 64
EPOCHS = 200
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
rbm = qaml.nn.RBM(DATA_SIZE, HIDDEN_SIZE)

# Initialize biases
torch.nn.init.constant_(rbm.b,0.5)
torch.nn.init.zeros_(rbm.c)
torch.nn.init.uniform_(rbm.W,-0.5,0.5)

# Set up optimizer
optimizer = torch.optim.SGD(rbm.parameters(), lr=learning_rate,
                            weight_decay=weight_decay,momentum=momentum)

# Set up training mechanisms
gibbs_sampler = qaml.sampler.GibbsNetworkSampler(rbm)
CD = qaml.autograd.SampleBasedConstrastiveDivergence()


# %%
################################## Model Training ##############################
# Set the model to training mode
rbm.train()
err_log = []
accuracy_log = []
b_log = [rbm.b.detach().clone().numpy()]
c_log = [rbm.c.detach().clone().numpy()]
W_log = [rbm.W.detach().clone().numpy().flatten()]
for t in range(EPOCHS):
    epoch_error = torch.Tensor([0.])
    for img_batch, labels_batch in train_loader:
        input_data = img_batch.flatten(1)

        # Positive Phase
        v0, prob_h0 = input_data, rbm(input_data)
        # Negative Phase
        vk, prob_hk = gibbs_sampler(v0.detach(), k=5)

        # Reconstruction error from Contrastive Divergence
        err = CD.apply((v0,prob_h0), (vk,prob_hk), *rbm.parameters())

        # Do not accumulate gradients
        optimizer.zero_grad()

        # Compute gradients
        err.backward()

        # Update parameters
        optimizer.step()

        #Accumulate error for this epoch
        epoch_error  += err

    # Error Log
    b_log.append(rbm.b.detach().clone().numpy())
    c_log.append(rbm.c.detach().clone().numpy())
    W_log.append(rbm.W.detach().clone().numpy().flatten())
    err_log.append(epoch_error.item())
    print(f"Epoch {t} Reconstruction Error = {epoch_error.item()}")
    ############################## CLASSIFICATION ##################################
    count = 0
    for test_data, test_label in test_dataset:
        test_data[-2:,-1] = 0.5
        prob_hk = rbm(test_data.flatten())
        label_pred = rbm.generate(prob_hk).view(*SHAPE)[-2:,-1]
        if label_pred.argmax() == test_label.argmax():
            count+=1
    accuracy_log.append(count/TEST)
    print(f"Testing accuracy: {count}/{TEST} ({count/TEST:.2f})")

# %%
############################ MODEL VISUALIZATION ###############################
# Testing accuracy graph
fig, ax = plt.subplots()
plt.plot(accuracy_log)
plt.ylabel("Testing Accuracy")
plt.xlabel("Epoch")
plt.savefig("classical_accuracy.pdf")

# L1 error graph
fig, ax = plt.subplots()
plt.plot(err_log)
plt.ylabel("Reconstruction Error (L1)")
plt.xlabel("Epoch")
plt.savefig("classical_err_log.pdf")
