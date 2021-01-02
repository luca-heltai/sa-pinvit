SA-PVINT for Elliptic Eigenvalue Problems
=========================================

![GitHub CI](https://github.com/luca-heltai/sa-pvint/workflows/GitHub%20CI/badge.svg)

![Indent](https://github.com/luca-heltai/sa-pvint/workflows/Indent/badge.svg)

![Doxygen](https://github.com/luca-heltai/sa-pvint/workflows/Doxygen/badge.svg)

Smoothed Adaptive Perturbed Inverse Iteration is a matrix-free geometric multigrid application that computes eigenvalues of the Laplace operator on locally refined grids using the Smoothed Adaptive Perturbed Inverse Iteration method.

To build and install, make sure you have deal.II 9.3 or higher installed:

	mkdir build
	cd build
	cmake -DDEAL_II_DIR=/path/to/your/installation/of/deal.II ..

The documentation is built and deployed at each merge to master. You can 
find the latest documentation here:
https://luca-heltai.github.io/sa-pvint/

Licence
=======

See the file ./LICENSE for details
