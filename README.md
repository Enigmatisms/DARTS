# DARTS

> ACM TOG (SIGGRAPH Asia 24 Journal Track)
> 
> DARTS: Diffusion Approximated Residual Time Sampling for Time-of-flight Rendering in Homogeneous Scattering Media
> 
> Qianyue He, Dongyu Du, Haitian Jiang, Xin Jin*

Coming soon. Jesus, never thought that I would say this one day, but yeah, for real, coming soon (patent-related problems). The code will eventually be open-sourced as the supp material (which is included in the submission), nevertheless, so even if I should forget to update this repo, which would be highly unlikely, I am sure anyone in need can find the code.

Well, this is the repository of our paper: DARTS: Diffusion Approximated Residual Time Sampling for Time-of-flight Rendering in Homogeneous Scattering Media.

Yet, the code release will be postponed, most likely before the date of the paper getting officially released. The code is composed of two renderers:
- A modified version of [pbrt-v3](https://github.com/mmp/pbrt-v3)
- A modified version of [Tungsten](https://github.com/tunabrain/tungsten) (actually, [transient-Tungsten](https://github.com/GhostatSpirit))

The code base will be huge (including several modified external packages used by pbrt-v3), therefore it takes time to release the code (that is gauranteed to be compile-able and runable). Since the arxiv version of this paper (I use IEEE template to avoid being recognized as SIGGRAPH submission) is already available, and judging by the reviews, the method in this work can be easily reproduced, so if you have any question about the implementation before the code upload, feel free to open an issue to have a discussion with me. 

Apart from other authors, here I'd like to extend my personal gratitude to [Yang Liu](https://github.com/GhostatSpirit), who is the first author of paper ["Temporally sliced photon primitives for time-of-flight rendering"](https://onlinelibrary.wiley.com/doi/abs/10.1111/cgf.14584). The implementation (photon point methods) of our method is based on the code base of his (listed above) and some of the discussions with him really pushed forward the modification of the Tungsten renderer part. His work for camera-unwarped transient rendering is very solid and inspiring, which definitely deserves more attention.
