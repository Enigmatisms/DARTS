# ToF-pbrt-v3

### I. Intros

This repository is based on [mmp/pbrt-v3](https://github.com/mmp/pbrt-v3) (source code of the third edition of *Physically Based Rendering: From Theory to Implementation*, by [Matt Pharr](http://pharr.org/matt), [Wenzel Jakob](http://www.mitsuba-renderer.org/~wenzel/), and Greg Humphreys). The same [BSD-2-Clause license](https://github.com/mmp/pbrt-v3#BSD-2-Clause-1-ov-file) should apply in this repository. The modified version: ToF-pbrt-v3 is created by the authors of the DARTS paper for SIGGRAPH 24 submission. Related copyrights are reserved by the anonymous authors.

This repository is mainly used to:

- Test vanilla path tracing (Vanilla PT) and DARTS path tracing (DARTS PT). pbrt-v3 does support `SPPM`, yet we didn't implement the photon based modification here. For photon based methods, please refer to the second code packge (ToF-tungsten)
- Evaluate the results: `temporal/`, `utils/` and `analysis/` folders contain the python code used in evaluation and presentable result generation.

### II. Compile & Run

Unlike [mmp/pbrt-v3](https://github.com/mmp/pbrt-v3), we incorporated git submodules inside this package, therefore there is no need to run the `git submodule` command. Note that [mmp/pbrt-v3](https://github.com/mmp/pbrt-v3) originally depends on a modified version of [wjakob/openexr](https://github.com/wjakob/openexr/tree/84793a726d77ad6cb9a510011c3907df809c32a4). However, this version is not C++17-ready. Therefore we further modified the underlying openexr lib.

The code is tested on Ubuntu 20.04 (mostly) and Ubuntu 22.04 system. Corresponding compiler should support C++17 features since we further incorporated [vectorclass/version2](https://github.com/vectorclass/version2) in our repo (see `src/sse`). To compile, run the following command:

```shell
cd <root-of-tof-pbrt-v3>
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=release ..; make -j16
```

Pretty similar to the compilation of original pbrt-v3. This will create a `pbrt` executable file under `build` folder. To run the code, you can use either:

```shell
cd build
# an example scene
./pbrt ../experiments/darts/staircase/staircase-gated-short.pbrt --nthreads=104
```

Or try running the scripts in `./scripts/` under the root folder.

**<u>Before you use the scene files in</u>** `./experiments/`, make sure to run:

```shell
cd experiments/
sudo chmod +x ./cp_scene_file.sh
./cp_scene_file.sh
```

This shell script will copy the scene files to the subfolders in need.

### III. Other Code

Here we briefly introduce the code for evaluation in this repo:

#### 3.1 Environment requirements

All the python code is tested with `py3.8`, `py3.9` and`py3.10`，you can use conda and pip to install the required packages in `requirements.txt`. Among them, `openexr` can only be installed through conda:

```shell
conda install openexr-python -c conda-forge
```

- `taichi`: required to perform shot noise removal
- `configargparse`: almost all the python scripts take corresponding `.conf` files in order to input the arguments.
- `rich`: console logging
- `dearpygui`: for visualization purposes
- `opencv-python`: image io, image and video processing

For almost all the scripts, you can use:

```shell
python3 <script-name>.py --config ./config/<corresponding configuration>
```

To change the settings, please modify the content of the corresponding configuration. If you are not sure what that parameter means, you can use:

```shell
python3 <script-name>.py --help
```

To find out.

#### 3.2 Functionality

- `temporal/` folder mainly provides the functionality to plot transient curves, load and convert HDR images (`exr`, `pfm`) to `png ` or `jpg` file:
  - `python3 ./transient_read.py --config ./config/guide.conf`. Set target folder (containing HDR images) to convert.
  - `python3 ./fluct_plot.py --config ./config/multiple.conf`. Set target folder (containing HDR images) to plot the transient curves.
- `utils/` Folder provides multiple useful functionalities:
  - `python3 ./post_process.py --config ./config/merge_frame.conf` will output several different images. Given the specified input folder, this script will merge all the HDR files in the folder, etc.
  - `python3 ./post_process.py --config ./config/transient_merge.conf`: Given the specified input folders, this script will merge all the HDR files across different folders, according to the image file name (index), this is used for merging transient outputs.
  - `python3 ./run_zooming.py --config ./config/cropping.conf`: this will initialize a dearpygui GUI, for you to draw selection box for different images. The selected area(s) will be saved, together with a json file (used in MSE analysis)
  - `python3 ./curve_analysis --config ./config/exp/xxx`: this script will output a graph figure to `./utils/output.png` and a corresponding GUI will show up. This is the script to make the curve graphs for section 6.3 of our paper.
- `analysis/`: in this folder, `cal_var.py` is used to calculate MSE across multiple folders and frames:
  - `python3 ./cal_var.py --config ./configs/input-xxx.conf`

#### 3.3 Scene file

Scene files are given in `experiments`, the typical settings in the scene file are listed as follows:

```
Integrator "volpath" "integer maxdepth" [180] "float rrthreshold" [0.0] "integer rseed_time" [1] "integer logTime" [1]
    "integer guiding" [1] "bool da_guide" "true" "integer time_sample" [1] "integer use_truncate" [0] "integer extra_nee_nums" [0] "bool fuse_origin" ["false"]
Sampler "random"
    "integer pixelsamples" [ 6000 ]

Film "image"
    "string filename" [ "dragon.exr" ]
    "integer yresolution" [ 960 ]
    "integer xresolution" [ 960 ]
    "float tmin_time" [7.1]
    "float t_interval" [0.05]
    "integer t_samplecnt" [1]

Accelerator "bvh"

MakeNamedMedium "mymedium" "string type" "guided_homo" "rgb sigma_s" [1.0 1.0 1.0] "rgb sigma_a" [0.005 0.005 0.005] "float g" [-0.0] "bool use_elliptical_phase" "false"
MediumInterface "" "mymedium"
```

Note that: we set the medium to be our `guided_homo` medium. To render steady state images, `t_samplecnt` should be set to be `0`. Other parameters:

- `"integer guiding" [1]`: should be 1 all the time. 0 is not recommended.
- `"bool da_guide" "true"`: `true` enables DA-based distance sampling.
- `"integer time_sample" [1]`: 1 enables elliptical sampling.
- `"integer use_truncate" [0] `: should be 0. This might improve volumetric rendering (in scenes without any triangles), but will introduce excessive runtime overhead.
- `"integer extra_nee_nums" [0]`, number of generalized shadow connection: 0 means we use reused ray direction to sample the elliptical control vertex. Value bigger than 0 will generate new direction samples. This might be helpful in multiple emitter scenes and non-delta emitter scenes. In our experiments, the improvement brought about by this setting is subtle.
- `"bool fuse_origin" ["false"]` Whether to incorporate direct shadow connection for medium events. Should be 0 for time-gated rendering. For transient rendering, this value can be set to 1 (but the improvement is negligible)
- `"bool use_elliptical_phase" "false"`: should usually be `true`. We did provide a new directional sampling method, but it leads to negligible improvements as discussed in our Discussion.
- `"integer rseed_time"`: when set to 1, the current system time will be used as random seed (otherwise, the random seed is fixed, so different realizations of the same stochastic sampling will be the same).
- `"integer logTime"`: when set to 1, a file named `time.log` will be created to keep track of the time used to render the current image (can be read by `cal_var.py` to analyze the time consumption).

---

#### copyright

Original repo: more info in their original repo. The authors reserve all copyright of the made modifications. **<u>This repo will be made public upon acceptance.</u>**
