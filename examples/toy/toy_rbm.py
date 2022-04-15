# # Characterization of Samples from D-Wave QAPs as Boltzmann Machines

# Required packages
import qaml
import torch
import numpy as np

################################## Configuration ###############################
solver_name = "Advantage_system4.1"

weight_inits = [0.1,0.5,1.0,2.0,4.0]
num_samples = 100000
auto_scales = [True]
seeds = [1,2,3,4,5,6]:
shape = (8,8)
embed_kwargs = {'chain_strength':1.6}


beta_estimate = 4.0

################################### Iterations ################################
header = f"Seed W0.1 W0.5 W1.0 W2.0 W4.0\n"
for weight_init in weight_inits:
    with open("log_dist.txt", "a") as myfile:
        myfile.write(f"Shape: {shape}\n")
    with open("log_beta.txt", "a") as myfile:
        myfile.write(f"Shape: {shape}\n")
    for auto_scale in auto_scales:
        with open("log_dist.txt", "a") as myfile:
            myfile.write(f"auto_scale: {auto_scale}\n")
        with open("log_beta.txt", "a") as myfile:
            myfile.write(f"auto_scale: {auto_scale}\n")
        with open("log_dist.txt", "a") as myfile:
            myfile.write(header)
        with open("log_beta.txt", "a") as myfile:
            myfile.write(header)
        for seed in seeds:
            with open("log_dist.txt", "a") as myfile:
                myfile.write(f"{seed} ")
            with open("log_beta.txt", "a") as myfile:
                myfile.write(f"{seed} ")
            torch.manual_seed(seed)
            for weight_init in weight_inits:

                # Create model
                rbm = qaml.nn.RestrictedBoltzmannMachine(*shape)
                _ = torch.nn.init.uniform_(rbm.b,-weight_init,weight_init)
                _ = torch.nn.init.uniform_(rbm.c,-weight_init,weight_init)
                _ = torch.nn.init.uniform_(rbm.W,-weight_init,weight_init)
                # Create sampler
                qa_sampler = qaml.sampler.QASampler(rbm,solver=solver_name)
                _ = qa_sampler.embed_bqm(auto_scale=auto_scale)
                # Scale beta estimates if needed
                if auto_scale:
                    scalar = 4/weight_init
                    beta_range = np.linspace(beta_estimate*scalar-5,
                                             beta_estimate*scalar+5,11)
                    print(scalar,beta_range)
                else:
                    beta_range = np.linspace(1,10,10)
                # Sample and compute distance
                try:
                    vq,hq = qa_sampler(num_reads=10000,auto_scale=auto_scale,
                                       num_spin_reversal_transforms=4,
                                       embed_kwargs=embed_kwargs)

                    beta,dist = qaml.perf.distance_from_gibbs(rbm,(vq,hq),
                                          num_samples=num_samples,
                                          beta_range=beta_range)
                    if auto_scale: beta = beta/(4/weight_init)
                except:
                    beta = 'n/a'
                    dist = 'n/a'

                # Log results
                with open("log_dist.txt", "a") as myfile:
                    myfile.write(f"{dist} ")
                with open("log_beta.txt", "a") as myfile:
                    myfile.write(f"{beta} ")
                print(beta,dist)
                try:
                    del rbm
                    del vq
                    del hq
                except:
                    pass
            with open("log_dist.txt", "a") as myfile:
                myfile.write(f"\n")
            with open("log_beta.txt", "a") as myfile:
                myfile.write(f"\n")
