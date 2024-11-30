# ToF-Tungsten
### I. Intros

This repository is based on [tunabrain/tungsten](https://github.com/tunabrain/tungsten) and [GhostatSpirit/tungsten-transient-public](https://github.com/GhostatSpirit/tungsten-transient-public) (the official implementation of 2022 CGF paper: temporally sliced photon primitives, also based on Tungsten). The same LICENSE should apply in this repository. The modified version: tof-tungsten is created by the authors of the DARTS paper for SIGGRAPH 24 submission. Related copyrights are reserved by the anonymous authors.

This repository is mainly used to:

- Test vanilla (transient) photon based methods, like Photon Points (2D) and Photon Beams (1D), and our DARTS based Photon Points (DARTS PP). We also implement DARTS PT in this repo, but compared with tof-pbrt-v3 (the other code repo we presented as supplementary code), the DARTS PT in this repo has inferior performance. Therefore, the PT related experiments are rendered by tof-pbrt-v3.

### II. Compile & Run

Please refer to the [original Tungsten repo](https://github.com/tunabrain/tungsten), our code is compiled in the same way.

### III. Other Code

In `pbrt-utils/`, we implement some format conversion code for pfm format conversion. However, we do recommend to use the code provided in `tof-pbrt-v3`, so the python script in this repo will not be introduced in detail.

#### 3.1 Scene file

Scene files are given in `exp_scenes`, the typical settings in the scene file are listed as follows (we introduced some new hyper-parameters):

```json
"volume_alpha": 0.6,
"enable_guiding": true,
"strict_time_width": true,
"enable_elliptical": true,
"frustum_culling": false
```

- `volume_alpha`: progressive alpha for volumetric photon based methods. Since in [GhostatSpirit/tungsten-transient-public](https://github.com/GhostatSpirit/tungsten-transient-public), surface rendering is not supported and we think that the progressive properties of surface photons and volume photon (primitives) will not be the same. 
- `enable_guiding`: setting this to be `true` enables DA-based distance sampling (for photon pass, in photon based methods, and for the entire volumetric path tracing)
- `strict_time_width`: described in our Supplmentary Material (PDF)
- `enable_elliptical`: setting this to be `true` enables elliptical sampling (for photon pass, in photon based methods, and for the entire volumetric path tracing)
- `frustum_culling`: should be `false`. `true` actually makes the rendering output worse.

To run our code, you can either use:

```shell
cd ./build/release/
./tungsten ../../exp_scenes/staircase/scene-points-darts-short.json -t 104
```

Or try running the scripts given in `./scripts/`.

---

#### copyright

Original repo: more info in their original repo. The authors reserve all copyright of the made modifications. **<u>This repo will be made public upon acceptance.</u>**
