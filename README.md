S-AFEM and SA-PINVIT for Elliptic Eigenvalue Problems
=====================================================

![GitHub CI](https://github.com/luca-heltai/sa-pinvit/workflows/GitHub%20CI/badge.svg)

![Indent](https://github.com/luca-heltai/sa-pinvit/workflows/Indent/badge.svg)

![Doxygen](https://github.com/luca-heltai/sa-pinvit/workflows/Doxygen/badge.svg)

Smoothed AFEM and Smoothed Adaptive Perturbed Inverse Iteration is a matrix-free geometric multigrid application that solves the poisson problem or computes eigenvalues of the Laplace operator on locally refined grids using the Smoothed Adaptive FEM or the Smoothed Adaptive Perturbed Inverse Iteration method.

To build and install, make sure you have deal.II 9.3 or higher installed:

	mkdir build
	cd build
	cmake -DDEAL_II_DIR=/path/to/your/installation/of/deal.II ..

The documentation is built and deployed at each merge to master. You can 
find the latest documentation here:
https://luca-heltai.github.io/sa-pinvit/

You can compile the code directly from within Visual Studio Code, if you have the remote docker extension installed. When you open the directory that contains the repository, visual studio code should detect that a `.devcontainer` directory is present in the current workspace, and ask if you want to reopen the project in the container. If you say yes, then the project is reopened in an isolated docker environment, based on the latest dealii master build.

Licence
=======

See the file ./LICENSE for details
