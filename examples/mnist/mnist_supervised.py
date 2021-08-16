import qaml
import torch

import matplotlib.pyplot as plt

import torchvision.datasets as torch_datasets
import torchvision.transforms as torch_transforms

################################# Hyperparameters ##############################
EPOCHS = 5
BATCH_SIZE = 64
# Stochastic Gradient Descent
learning_rate = 1e-3
weight_decay = 1e-4
momentum = 0.5

#################################### Input Data ################################
train_dataset = torch_datasets.MNIST(root='./data/', train=True,
                                     transform=torch_transforms.ToTensor(),
                                     target_transform=torch_transforms.Compose([
                                     lambda x:torch.LongTensor([x]),
                                     lambda x:torch.nn.functional.one_hot(x,10)]),
                                     download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                           shuffle=True)

test_dataset = torch_datasets.MNIST(root='./data/', train=False,
                                    transform=torch_transforms.ToTensor(),
                                    target_transform=torch_transforms.Compose([
                                    lambda x:torch.LongTensor([x]),
                                    lambda x:torch.nn.functional.one_hot(x,10)]),
                                    download=True)
test_loader = torch.utils.data.DataLoader(test_dataset)

################################# Model Definition #############################
DATA_SIZE = len(train_dataset.data[0].flatten())
LABEL_SIZE = len(train_dataset.classes)

VISIBLE_SIZE = DATA_SIZE + LABEL_SIZE
HIDDEN_SIZE = 128

# Specify model with dimensions
rbm = qaml.nn.RBM(VISIBLE_SIZE, HIDDEN_SIZE)

# Set up optimizer
optimizer = torch.optim.SGD(rbm.parameters(), lr=learning_rate,
                                              weight_decay=weight_decay,
                                              momentum=momentum)
# Set up training mechanisms
sampler = qaml.sampler.GibbsNetworkSampler(rbm)
CD = qaml.autograd.ConstrastiveDivergence()

# %%
################################## Model Training ##############################
# Set the model to training mode
rbm.train()
err_log = []
for t in range(EPOCHS):
    epoch_error = torch.Tensor([0.])
    for img_batch, labels_batch in train_loader:

        input_data = torch.cat((img_batch.flatten(1),labels_batch.flatten(1)),1)

        # Positive Phase
        v0, prob_h0 = input_data, rbm(input_data)
        # Negative Phase
        vk, prob_hk = sampler(v0, k=2)

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

# %%
################################# ANIMATION ####################################
from matplotlib.animation import FuncAnimation
img = torch.zeros(1,DATA_SIZE)
clamp = torch.nn.functional.one_hot(torch.LongTensor([3]),10)
img_data = []
for _ in range(1000):
    prob_hk = rbm.forward(torch.cat((img,clamp),dim=1).bernoulli())
    img,label = rbm.generate(prob_hk).split((DATA_SIZE,LABEL_SIZE),dim=1)
    img_data.append(img.detach().clone().view(28,28).numpy())

fig = plt.figure()
plot = plt.matshow(img_data[0],fignum=0)
def init():
    plot.set_data(img_data[0])
    return [plot]

def update(j):
    plot.set_data(img_data[j])
    return [plot]

anim = FuncAnimation(fig,update,init_func=init,frames=1000,interval=20,blit=True)
plt.show()
anim.save("./animation.gif","pillow")

# %%
############################## CLASSIFICATION ##################################
count = 0
for test_data, test_label in test_loader:
    prob_hk = rbm.forward(torch.cat((test_data.flatten(1),torch.zeros(1,LABEL_SIZE)),dim=1))
    _,label_pred = rbm.generate(prob_hk).split((DATA_SIZE,LABEL_SIZE),dim=1)
    if label_pred.argmax() == test_label.argmax():
        count+=1

print(f"Testing accuracy: {count}/{len(test_dataset)}")
