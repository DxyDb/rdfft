# Memory-Efficient Training with In-Place FFT Implementation

The official implementation of "Memory-Efficient Training with In-Place FFT Implementation" (rdFFT).

## Introduction

Fast Fourier Transforms (FFTs) are widely used to reduce memory and computational costs in deep learning. However, existing implementations, including standard FFT and real FFT (rFFT), cannot achieve true in-place computation. In particular, rFFT maps an input of size $n$ to a complex output of size $n/2+1$, causing dimensional mismatch and requiring additional memory allocation. 

In this work, we propose the first **real-domain, fully in-place FFT framework (rdFFT)** that strictly preserves input-output memory space consistency. By leveraging butterfly operation symmetry and conjugate properties in the frequency domain, we design an implicit complex encoding scheme that **eliminates intermediate cache usage entirely**. Experimental results across multiple natural language understanding tasks demonstrate that our method significantly reduces training memory costs, offering a promising new direction for frequency-domain lightweight adaptation.

## Citing our work

If you find this work or code helpful for your research, please consider citing our paper:

```bibtex
@article{ding2026memory,
  title={Memory-Efficient Training with In-Place FFT Implementation},
  author={Ding, Xinyu and Liu, Bangtian and Liao, Siyu and Wang, Zhongfeng},
  journal={Advances in Neural Information Processing Systems},
  volume={38},
  pages={122038--122056},
  year={2026}
}
