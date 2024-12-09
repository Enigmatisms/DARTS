<div align="center">

### ACM Transactions on Graphics 2024
### (SIGGRAPH Asia 2024 Journal Track)

## <a href="https://dl.acm.org/doi/10.1145/3687930">🎯 DARTS: Diffusion Approximated Residual Time Sampling for Time-of-flight Rendering in Homogeneous Scattering Media</a>

[Qianyue He](https://github.com/Enigmatisms), [Dongyu Du](https://dongyu-du.github.io/), [Haitian Jiang](https://github.com/Enigmatisms/DARTS), [Xin Jin*](https://scholar.google.com.hk/citations?user=FTikW50AAAAJ&hl=zh-CN&oi=sra)

[![Project Page](https://img.shields.io/badge/Project-Page-green.svg)](https://darts-paper.github.io/DARTS-proj-page/)
[![Paper PDF](https://img.shields.io/badge/Paper-PDF-red.svg)](https://dl.acm.org/doi/10.1145/3687930)
[![Code License](https://img.shields.io/badge/license-MIT-blue)](https://github.com/Enigmatisms/DARTS?tab=MIT-1-ov-file#readme)
</div>

The code is composed of two renderers:

- A modified version of [pbrt-v3](https://github.com/mmp/pbrt-v3)
- A modified version of [Tungsten](https://github.com/tunabrain/tungsten) (actually, [transient-Tungsten](https://github.com/GhostatSpirit))

A kind reminder: don't get misled. **Diffusion** here does not mean DDPM related generative AI. It is an optics concept which describes the "diffusion" of photons within the participating medium. The only part Pytorch gets used in this work, is the precomputation of the EDA direction sampling table, which takes merely 5 seconds to complete with `torch.compile` and no back-prop is used. Yeah, I know, this is old-fashioned, sorry about that if you accidentally click this repo and try to see how diffusion models are employed.

Compilation and using this code base can refer to the README in each folder. The main modification:

- Support ToF rendering (transient / time-gated) for both renderers.
- Added `GuidedHomogeneousMedium` class, `IsotropicDAPhase` class and etc. which support our sampling methods.
- Modified some of the BSDF implementation, so that Tungsten can have the same outputs as PBRT-v3 (Tungsten can have matched output as PBRT-v4 but not v3).

You can refer to [Enigmatisms/AnalyticalGuiding](https://github.com/Enigmatisms/AnalyticalGuiding) repo to test the 2D and 1D numerical tests and interactive visualization (in DearPyGui and Taichi).

---
### Scene availability
Check [this repo](https://github.com/Enigmatisms/DARTS-scenes) for GLOSSY DRAGON and STAIRCASE scene. For more test scenes mentioned in the supplementary document, open an issue and leave your e-mail for me so that I can send them to you.

---

### Rendering Results

![image](https://github.com/user-attachments/assets/75041d18-c9c5-4935-8747-760d1fe665aa)

![image](https://github.com/user-attachments/assets/e056af7d-610d-4d4c-9bf3-75119cc16bd6)

![image](https://github.com/user-attachments/assets/50cdd855-aa27-4430-9458-0bef26e8d50a)

---

### Acknowledgements & Guidances

Apart from other authors, here I'd like to extend my personal gratitude to [Yang Liu](https://github.com/GhostatSpirit), who is the first author of paper ["Temporally sliced photon primitives for time-of-flight rendering"](https://onlinelibrary.wiley.com/doi/abs/10.1111/cgf.14584). The implementation (photon point methods) of our method is based on the code base of his (listed above) and some of the discussions with him really pushed forward the modification of the Tungsten renderer part. His work for camera-unwarped transient rendering is very solid and inspiring, which definitely deserves more attention.

Other ToF rendering related paper:

- [Jarabo A, Marco J, Munoz A, et al. A framework for transient rendering[J]. ACM Transactions on Graphics (ToG), 2014, 33(6): 1-10.](https://cs.dartmouth.edu/~wjarosz/publications/jarabo14framework.html)
- [Marco J, Guillén I, Jarosz W, et al. Progressive transient photon beams[C]//Computer graphics forum. 2019, 38(6): 19-30.](https://onlinelibrary.wiley.com/doi/am-pdf/10.1111/cgf.13600)
- [Kim J, Jarosz W, Gkioulekas I, et al. Doppler Time-of-Flight Rendering[J]. ACM Transactions on Graphics (TOG), 2023, 42(6): 1-18.](https://dl.acm.org/doi/abs/10.1145/3618335)
- [Yi S, Kim D, Choi K, et al. Differentiable transient rendering[J]. ACM Transactions on Graphics (TOG), 2021, 40(6): 1-11.](https://dl.acm.org/doi/abs/10.1145/3478513.3480498)

---

### Citation

For the paper:

```tex
@article{he2024darts,
  title={DARTS: Diffusion Approximated Residual Time Sampling for Time-of-flight Rendering in Homogeneous Scattering Media},
  author={He, Qianyue and Du, Dongyu and Jiang, Haitian and Jin, Xin},
  journal={ACM Transactions on Graphics (TOG)},
  volume={43},
  number={6},
  pages={1--14},
  year={2024},
  publisher={ACM New York, NY, USA}
}
```

For the code base, it is recommended that PBRT-v3 and Tungsten are cited altogether:

```tex
@misc{He:2024:DARTS,
  title = {DARTS-ToF-Renderer},
  author = {He, Qianyue},
  year = {2024},
  url = {https://github.com/Enigmatisms/DARTS/}
}
```
