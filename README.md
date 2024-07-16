# Out-of-Distribution Detection Through Soft Clustering with NNK-Means

Official repository for the paper "Out-of-Distribution Detection Through Soft Clustering with NNK-Means"

## Table of Contents

- [Abstract](#abstract)
- [Contents](#contents)

## Abstract

As language models become more general purpose, increased attention needs to be paid to detecting out-of-distribution (OOD) instances, i.e., those not belonging to any of the distributions seen during training. Existing methods for detecting OOD data are computationally complex and storage-intensive. We propose a novel soft clustering approach for OOD detection based on non-negative kernel regression. Our approach greatly reduces computational and space complexities (up to 11x improvement in inference time and 87% reduction in storage requirements) and outperforms existing approaches by up to 4 AUROC points on four different benchmarks. We also introduce an entropy-constrained version of our algorithm, which leads to further reductions in storage requirements (up to 97% lower than comparable approaches) while retaining competitive performance.Our soft clustering approach for OOD detection highlights its potential for detecting tail-end phenomena in extreme-scale data settings.

## Contents

1. `src/ec_nnk_means.py`: Core logic and implementation of the entropy-constrained NNK-Means algorithm. Also defines `NNKMU` a wrapper class for training and inference using the algorithm.

2. `src/data.py`: Utility functions for loading and preprocessing data.

3. `src/ood_detection.py`: Utility functions for fititng, inference and evaluation of OOD detection methods.

4. `src/main.py`: OOD Detection driver script that benchmarks our approach against various others.
