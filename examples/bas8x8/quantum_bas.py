# %% markdown
# # Quantum-Assisted RBM training on the BAS Dataset for Reconstruction
# This is an example on quantum-assisted training of an RBM on the BAS(8,8)
# dataset.
# Developed by: Jose Pinilla
# %%
# Required packages
import os
import qaml
import torch

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

# %%
# Experiment iterators
scaling = [True,False]
betas = [1.0,2.0,3.0,4.0]
seeds = [1,2,3]
weight_inits = [0.1,0.5,1.0,2.0,4.0]

# QAP settings
chain_strength = 1.6
num_reads = BATCH_SIZE
solver_name = "Advantage_system4.1"

for auto_scale in scaling:
    for beta in betas:
        for weight_init in weight_inits:

            directory = f"BAS88_lrWbeta{beta}_bcW{weight_init}"
            directory += f"{'' if auto_scale else 'no'}scale_"
            directory += f"{EPOCHS}_{'batch' if num_reads==BATCH_SIZE else num_reads}"
            directory += f"Adv_wd{weight_decay}"
            directory += f"_cs{chain_strength}"
            directory = directory.replace('.','')
            print(directory)

            for SEED in seeds:
                ######################################## RNG ###################################
                torch.manual_seed(SEED)
                ############################## Logging Directory ###############################
                if not os.path.exists(directory):
                        os.makedirs(directory)
                if not os.path.exists(f'{directory}/{SEED}'):
                        os.makedirs(f'{directory}/{SEED}')

                print(f"Running with SEED={SEED}")
                ################################# Model Definition #############################
                # Specify model with dimensions
                rbm = qaml.nn.RBM(DATA_SIZE,HIDDEN_SIZE)
                # Set up optimizers
                optimizer = torch.optim.SGD(rbm.parameters(), lr=learning_rate*weight_init,
                                            weight_decay=weight_decay,momentum=momentum)

                # Set up training mechanisms
                qa_sampler = qaml.sampler.QASampler(rbm,solver=solver_name,beta=beta)

                # Loss and autograd
                CD = qaml.autograd.SampleBasedConstrastiveDivergence()

                # Initialize biases
                _ = torch.nn.init.uniform_(rbm.b,-weight_init,weight_init)
                _ = torch.nn.init.uniform_(rbm.c,-weight_init,weight_init)
                _ = torch.nn.init.uniform_(rbm.W,-weight_init,weight_init)

                ################################## Model Training ##############################
                try:
                    # Set the model to training mode
                    rbm.train()
                    err_log = []
                    scalar_log = []
                    accuracy_log = []
                    b_log = [rbm.b.detach().clone().numpy()]
                    c_log = [rbm.c.detach().clone().numpy()]
                    W_log = [rbm.W.detach().clone().numpy().flatten()]
                    for t in range(EPOCHS):
                        epoch_error = 0
                        for img_batch, labels_batch in train_loader:
                            input_data = img_batch.flatten(1)

                            # Negative Phase
                            vk, prob_hk = qa_sampler(num_reads=num_reads,auto_scale=auto_scale,
                                                     num_spin_reversal_transforms=4,embed_kwargs={'chain_strength':chain_strength})
                            # Positive Phase
                            scale = qa_sampler.scalar*qa_sampler.beta if auto_scale else 1.0
                            v0, prob_h0 = input_data, rbm(input_data,scale=scale)

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
                        if auto_scale is True:
                            scalar_log.append(qa_sampler.scalar)
                        print(f"Epoch {t} Reconstruction Error = {epoch_error}")
                        ############################## CLASSIFICATION ##################################
                        count = 0
                        for test_data, test_label in test_dataset:
                            test_data[-2:,-1] = 0.5
                            recon_hk = rbm(test_data.flatten(),scale=scale)
                            label_pred = rbm.generate(recon_hk,scale=scale).view(*SHAPE)[-2:,-1]
                            if label_pred.argmax() == test_label.argmax():
                                count+=1
                        accuracy_log.append(count/TEST)
                        print(f"Testing accuracy: {count}/{TEST} ({count/TEST:.2f})")

                except Exception as e:
                    print(e)
                ############################ Store Model and Logs ##############################
                torch.save(b_log,f"./{directory}/{SEED}/b.pt")
                torch.save(c_log,f"./{directory}/{SEED}/c.pt")
                torch.save(W_log,f"./{directory}/{SEED}/W.pt")
                torch.save(err_log,f"./{directory}/{SEED}/err.pt")
                torch.save(scalar_log,f"./{directory}/{SEED}/scalar.pt")
                torch.save(accuracy_log,f"./{directory}/{SEED}/accuracy.pt")

                # Testing accuracy graph
                fig, ax = plt.subplots()
                plt.plot(accuracy_log)
                plt.ylabel("Testing Accuracy")
                plt.xlabel("Epoch")
                plt.savefig(f"./{directory}/{SEED}/accuracy.pdf")
