# Quantum Assisted Machine Learning
QAML is a library for generative learning models built as an extension of PyTorch.
This library provides a set of custom Pytorch classes based on Pytorch `Module`, `Function`, and `Dataset` to allow easy
integration with Pytorch models.

This compatibility and modularity is what makes possible the use of Quantum Processing Units to accelerate
the sampling process required in generative models.

# References:

I want to acknowledge the projects I used as inspiration for QAML.

[Gabriel Bianconi's pytorch-rbm](https://github.com/GabrielBianconi/pytorch-rbm): Perfect example of a baseline RBM implementation along with MNIST example.

[Jan Melchior's PyDeep](https://github.com/MelJan/PyDeep.git):  Most thorough Boltzmann Machine learning library out there with impressive modularity.

[PIQUIL's QuCumber](https://github.com/PIQuIL/QuCumber): Practical application with a PyTorch RBM backend.
