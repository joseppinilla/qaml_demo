# # Characterization of Samples from D-Wave QAPs as Boltzmann Machines

# Required packages
import qaml
import torch
import traceback

import numpy as np

################################## Configuration ###############################
solver_name = "Advantage_system4.1"

weight_inits = [0.1,0.5,1.0,2.0,4.0]
num_samples = 100000
auto_scales = [True,False]
seeds = [1,2,3,4,5,6]
betas = np.linspace(1,9,65) #[1-9] +=0.125

shape = (8,8)
embed_kwargs = {'chain_strength':1.6}

################################### Iterations ################################

for auto_scale in auto_scales:
    dist_filename = 'dist_scale.csv' if auto_scale else 'dist_noscale.csv'
    beta_filename = 'beta_scale.csv' if auto_scale else 'beta_noscale.csv'

    with open(dist_filename, "a") as myfile:
        myfile.write(f"0.1 0.5 1.0 2.0 4.0\n")
    with open(beta_filename, "a") as myfile:
        myfile.write(f"0.1 0.5 1.0 2.0 4.0\n")

    for seed in seeds:

        torch.manual_seed(seed)
        for weight_init in weight_inits:
            # Create model
            rbm = qaml.nn.RestrictedBoltzmannMachine(*shape)
            _ = torch.nn.init.uniform_(rbm.b,-weight_init,weight_init)
            _ = torch.nn.init.uniform_(rbm.c,-weight_init,weight_init)
            _ = torch.nn.init.uniform_(rbm.W,-weight_init,weight_init)
            # Create sampler
            qa_sampler = qaml.sampler.QASampler(rbm,solver=solver_name)

            # Sample and compute distance
            try:
                vq,hq = qa_sampler(num_reads=10000,auto_scale=auto_scale,num_spin_reversal_transforms=4,embed_kwargs=embed_kwargs)
                # Scale beta estimates if needed
                if auto_scale:
                    beta_range = betas*4/weight_init
                else:
                    beta_range = betas

                beta,dist = qaml.perf.distance_from_gibbs(rbm,(vq,hq),num_samples=num_samples,beta_range=beta_range)
            except:
                traceback.print_exc()
                beta = 'n/a'
                dist = 'n/a'

            # Log results
            with open(dist_filename, "a") as myfile:
                myfile.write(f"{dist} ")
            with open(beta_filename, "a") as myfile:
                myfile.write(f"{beta/qa_sampler.scalar} ")

        with open(dist_filename, "a") as myfile:
            myfile.write(f"\n")
        with open(beta_filename, "a") as myfile:
            myfile.write(f"\n")
