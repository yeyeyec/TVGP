### BRISC library for nearest neighbor Gaussian process (NNGP) approximation

This directory contains C/C++ implementation of NNGP approximation and its R interface (Saha and Datta, 2018). Code is adopted from <https://github.com/ArkajyotiSaha/BRISC>.

The SPGP appendix scripts compile this shared library automatically if it is missing.
To compile manually, run the following command from this `BRISC/` directory:
```
R CMD SHLIB BRISC.cpp BRISC_Predict.cpp util.cpp init.c lbfgs.c
```

Reference:

Saha, A. and Datta, A. (2018). BRISC: bootstrap for rapid inference on spatial covariances. _Stat_, 7(1):e184.
