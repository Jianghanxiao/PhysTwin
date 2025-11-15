# PhysTwin: Physics-Informed Reconstruction and Simulation of Deformable Objects from Videos

<span class="author-block">
<a target="_blank" href="https://jianghanxiao.github.io/">Hanxiao Jiang</a><sup>1,2</sup>,
</span>
<span class="author-block">
<a target="_blank" href="https://haoyuhsu.github.io/">Hao-Yu Hsu</a><sup>2</sup>,
</span>
<span class="author-block">
<a target="_blank" href="https://kywind.github.io/">Kaifeng Zhang</a><sup>1</sup>,
</span>
<span class="author-block">
<a target="_blank" href="https://www.linkedin.com/in/hnyu/">Hsin-Ni Yu</a><sup>2</sup>,
</span>
<span class="author-block">
<a target="_blank" href="https://shenlong.web.illinois.edu/">Shenlong Wang</a><sup>2</sup>,
</span>
<span class="author-block">
<a target="_blank" href="https://yunzhuli.github.io/">Yunzhu Li</a><sup>1</sup>
</span>

<span class="author-block"><sup>1</sup>Columbia University,</span>
<span class="author-block"><sup>2</sup>University of Illinois Urbana-Champaign</span>

### [Website](https://jianghanxiao.github.io/phystwin-web/) | [Paper](https://jianghanxiao.github.io/phystwin-web/phystwin.pdf) | [Arxiv](https://arxiv.org/abs/2503.17973)

### Overview
This repository contains the official implementation of the **PhysTwin** framework.

![TEASER](./assets/teaser.png)


### Update
**This repository will be actively maintained by the authors, with continuous updates introducing new features to inspire further research.**

- **Actively Developing:** In the long term, we aim to develop a comprehensive physics simulator focused on real-to-sim, serving as an easy-to-use platform for XR, VR, and robotics applications. **Feel free to reach out via email if you’re also interested in this direction and would like to collaborate on related research projects.**

- **[25.11.6] Extend PhysTwin wiht robot physics support:** Explore our extended system [Real2Sim-Eval](https://real2sim-eval.github.io/), which supports both keyboard and Gello-based robot control, enabling physics-based interactions with constructed PhysTwins. We are actively developing a full robotics simulator that will serve as an easy-to-use platform for diverse research applications. A demo version will also be released in this repository soon.

- **[25.10.26] Speed Acceleration for Self-Collision Cases:** For scenarios involving self-collision, instead of checking all particle pairs within a distance threshold, we introduce a mechanism to ignore topologically adjacent particle pairs. This significantly accelerates both optimization and inference in cloth-like cases where self-collision is activated. The main modification is implemented in [code](https://github.com/Jianghanxiao/PhysTwin/blob/release_collision_accelerate/qqtt/engine/trainer_warp.py#L179),and the feature is available in the branch `release_collision_accelerate`. This is a pre-released feature developed as part of an ongoing project. The fully accelerated system will be released once the complete system is done.

![accelerated_example](./assets/cloth_collision_accelerate.gif)

- **[25.7.22] Remote Control Feature & Bug Fix:** Fixed a deprojection error in the data processing pipeline. Added support for remote control—previously, the interactive playground only responded to physical keyboard input; it now accepts virtual keyboard signals from remote devices as well.

- **[25.4.15] GPU Memory Optimization:** Thanks to user feedback and testing, we've further optimized the code to reduce GPU memory usage in the interactive playground—now requiring only about 2GB in total. Previously, LBS initialization consumed a significant amount of GPU memory; it's now offloaded to the CPU and only needs to run once at startup. Everything runs smoothly as a result.

- **[25.4.8] Optmization Speed:** Regarding the questions on optimization speed, thanks to Nvidia Warp, our differentiable Spring-Mass simulator enables first-order optimization in approximately 5 minutes—and even faster with visualizations disabled—significantly outperforming prior work that typically requires hours. The zero-order, sampling-based optimization (CMA-ES) takes around 12 minutes, depending on the number of epochs. These statistics are based on the stuffed animal experiments without self-collision enabled.
  
- **[25.4.4] Material Visualization:** Show the experimental features to visualize the materials approximated from the underlying spring-mass model. (See below for detailed instructions)
<p align="center">
  <img src="./assets/material_rope.gif" width="30%">
  <img src="./assets/material_cloth.gif" width="30%">
  <img src="./assets/material_sloth.gif" width="30%">
</p>


- **[25.4.3] Multiple Objects Demos:** Show the experimental features for handling collisions among multiple PhysTwins we construct. (See below for detailed instructions)
<p align="center">
  <img src="./assets/rope_multiple.gif" width="45%">
  <img src="./assets/sloth_multiple.gif" width="45%">
</p>

- **[25.4.3] LBS GPU Memory Fix:** Clear intermediate variables to significantly reduce GPU memory usage in the interactive playground. The sloth case now requires only about 4GB in total. (Pull the latest code to apply the fix.)

- **[25.4.1] Force Visualization:** Visualize the forces applied to objects after optimization, aiding in force analysis from videos. (See below for detailed instructions)
<p align="center">
  <img src="./assets/force_rope.gif" width="30%">
  <img src="./assets/force_cloth.gif" width="30%">
  <img src="./assets/force_sloth.gif" width="30%">
</p>

#### Long-Term Plans
In the long term, we aim to develop a comprehensive physics simulator focused on real-to-sim, serving as an easy-to-use platform for XR, VR, and robotics applications. **Feel free to reach out via email if you’re also interested in this direction and would like to collaborate on related research projects.**


### Setup
#### 🐧Linux Setup
```
# Here we use cuda-12.1
export PATH={YOUR_DIR}/cuda/cuda-12.1/bin:$PATH
export LD_LIBRARY_PATH={YOUR_DIR}/cuda/cuda-12.1/lib64:$LD_LIBRARY_PATH
# Create conda environment
conda create -y -n phystwin python=3.10
conda activate phystwin

# Install the packages
# If you only want to explore the interactive playground, you can skip installing Trellis, Grounding-SAM-2, RealSense, and SDXL.
bash ./env_install/env_install.sh

# Download the necessary pretrained models for data processing
bash ./env_install/download_pretrained_models.sh
```

#### 🪟Windows Setup
Thanks to @GuangyanCai contributions, now we also have a windows setup codebase in `windows_setup` branch.

#### 🐳Docker Setup
Thanks to @epiception contributions, we now have Docker support as well.
```
export DOCKER_USERNAME="your_alias" # default is ${whoami} (optional)
chmod +x ./docker_scripts/build.sh
./docker_scripts/build.sh

# The script accepts architecture version from https://developer.nvidia.com/cuda-gpus as an additional argument
./docker_scripts/build.sh 8.9+PTX # For NVIDIA RTX 40 series GPUs
```

#### 🐧Linux Setup (RTX 5090 + CUDA 12.8 + Python 3.10 Specific)
```
# Here we use CUDA 12.8
export PATH={YOUR_DIR}/cuda/bin:$PATH
export LD_LIBRARY_PATH={YOUR_DIR}/cuda/lib64:$LD_LIBRARY_PATH
export CUDA_HOME={YOUR_DIR}/cuda

# Create conda environment
conda create -y -n phystwin python=3.10
conda activate phystwin

# Open gaussian_splatting/submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.h and add an include directive for cstdint
# Forcefully create a symbolic soft link between system libstdc++.so.6 and conda environment libstdc++.so.6 e.g. `ln -sf /usr/lib/x86_64-linux-gnu/libstdc++.so.6 {CONDA_PATH}/envs/phystwin/bin/../lib/libstdc++.so.6`

# Install the packages (if you only want to explore the interactive playground, you can skip installing TRELLIS, Grounded-SAM-2, Grounding-DINO, RealSense, and SDXL)
bash ./env_install/5090_env_install.sh

# Download the necessary pretrained models for data processing
bash ./env_install/download_pretrained_models.sh
```

### Download the PhysTwin Data
Download the original data, processed data, and results into the project's root folder. (The following sections will explain how to process the raw observations and obtain the training results.)
- [data](https://drive.google.com/file/d/1A6X7X6yZFYJ8oo6Bd5LLn-RldeCKJw5Z/view?usp=sharing): this includes the original data for different cases and the processed data for quick run. The different case_name can be found under `different_types` folder.
- [experiments_optimization](https://drive.google.com/file/d/1xKlk3WumFp1Qz31NB4DQxos8jMD_pBAt/view?usp=sharing): results of our first-stage zero-order optimization.
- [experiments](https://drive.google.com/file/d/1hCGzdGlzL4qvZV3GzOCGiaVBshDgFKjq/view?usp=sharing): results of our second-order optimization.
- [gaussian_output](https://drive.google.com/file/d/12EoxhEhE90NMAqLlQoj_zM_C63BOftNW/view?usp=sharing): results of our static gaussian appearance.
- [(optional) additional_data](https://drive.google.com/file/d/1Q9AFDr_yQD-n5YNAe157hViTBC9mo876/view?usp=sharing): data for extra clothing demos not included in the original paper.

### Play with the Interactive Playground
Use the previously constructed PhysTwin to explore the interactive playground. Users can interact with the pre-built PhysTwin using keyboard. The next section will provide a detailed guide on how to construct the PhysTwin from the original data.

![example](./assets/sloth.gif)

Run the interactive playground with our different cases (Need to wait some time for the first usage of interactive playground; Can achieve about 37 FPS using RTX 4090 on sloth case)

```
python interactive_playground.py \
(--inv_ctrl) \
--n_ctrl_parts [1 or 2] \
--case_name [case_name]

# Examples of usage:
python interactive_playground.py --n_ctrl_parts 2 --case_name double_stretch_sloth
python interactive_playground.py --inv_ctrl --n_ctrl_parts 2 --case_name double_lift_cloth_3
```
or in Docker
```
./docker_scripts/run.sh /path/to/data \
                        /path/to/experiments \
                        /path/to/experiments_optimization \
                        /path/to/gaussian_output \
# inside container
conda activate phystwin_env
python interactive_playground.py --inv_ctrl --n_ctrl_parts 2 --case_name double_lift_cloth_3
```

Options: 
-   --inv_ctrl: inverse the control direction
-   --n_ctrol_parts: number of control panel (single: 1, double: 2) 
-   --case_name: case name of the PhysTwin case

### Train the PhysTwin with the data
Use the processed data to train the PhysTwin. Instructions on how to get above `experiments_optimization`, `experiments` and `gaussian_output` (Can adjust the code below to only train on several cases). After this step, you get the PhysTwin that can be used in the interactive playground.
```
# Zero-order Optimization
python script_optimize.py

# First-order Optimization
python script_train.py

# Inference with the constructed models
python script_inference.py

# Train the Gaussian with the first-frame data
bash gs_run.sh
```

### Evaluate the performance of the contructed PhysTwin
To evaluate the performance of the constructed PhysTwin, need to render the images in the original viewpoint (similar logic to interactive playground)
```
# Use LBS to render the dynamic videos (The final videos in ./gaussian_output_dynamic folder)
bash gs_run_simulate.sh
python export_render_eval_data.py
# Get the quantative results
bash evaluate.sh

# Get the qualitative results
bash gs_run_simulate_white.sh
python visualize_render_results.py
```

### Data Processing from Raw Videos
The original data in each case only includes `color`, `depth`, `calibrate.pkl`, `metadata.json`. All other data are processed as below to get, including the projection, tracking and shape priors.
(Note: Be aware of the conflict in the diff-gaussian-rasterization library between Gaussian Splatting and Trellis. For data processing, you don't need to install the gaussian splatting; ignore the last section in env_install.sh)
```
# Process the data
python script_process_data.py

# Further get the data for first-frame Gaussian
python export_gaussian_data.py

# Get human mask data for visualization and rendering evaluation
python export_video_human_mask.py
```

### Control Force Visualization
Visualize the force applied by the hand to the object as inferred from our PhysTwin model, based solely on video data.
```
python visualize_force.py \
--n_ctrl_parts [1 or 2] \
--case_name [case_name]

# Examples of usage:
python visualize_force.py --case_name single_push_rope_1 --n_ctrl_parts 1 
python visualize_force.py --case_name single_clift_cloth_1 --n_ctrl_parts 1    
python visualize_force.py --case_name double_stretch_sloth 
```
The visualziation video is saved under `experiments` folder.

### Material Visualization
Experimental feature to visualize the approximated material from the constructed PhysTwin.
```
python visualize_material.py \
--case_name [case_name]

# Examples of usage:
python visualize_material.py --case_name double_lift_cloth_1
python visualize_material.py --case_name single_push_rope
python visualize_material.py --case_name double_stretch_sloth
```


### Multiple Objects Demos
Try the experimental features for handling collisions among the multiple PhysTwins we construct.

```
# The stuff is deployed in the 'claw_matchine' branch
git pull
git checkout claw_machine

# Play with the examples
python interactive_playground.py --n_ctrl_parts 1 --case_name single_push_rope_1 --n_dup 4
python interactive_playground.py --n_ctrl_parts 2 --case_name double_stretch_sloth --n_dup 2
```

### Follow-up and Potential Collaborations  
If you are interested in collaborating or extending this work for your research, feel free to contact us at `hanxiao.jiang@columbia.edu`.  

## Projects Developed or Benchmarked with PhysTwin
* **[NovaFlow: Zero-Shot Manipulation via Actionable Flow from Generated Videos](https://novaflow.lhy.xyz/).** Hongyu Li*, Lingfeng Sun*, Yafei Hu, Duy Ta, Jennifer Barry, George Konidaris, and Jiahui Fu
* **[Real-to-Sim Robot Policy Evaluation with Gaussian Splatting Simulation of Soft-Body Interactions](https://real2sim-eval.github.io/).** Kaifeng Zhang*, Shuo Sha*, Hanxiao Jiang, Matthew Loper, Hyunjong Song, Guangyan Cai, Zhuo Xu, Xiaochen Hu, Changxi Zheng, Yunzhu Li
* **[PhysWorld: From Real Videos to World Models of Deformable Objects via Physics-Aware Demonstration Synthesis](https://arxiv.org/abs/2510.21447)** Yu Yang, Zhilu Zhang, Xiang Zhang, Yihan Zeng, Hui Li, Wangmeng Zuo
* **[NeuSpring: Neural Spring Fields for Reconstruction and Simulation of Deformable Objects from Videos](https://arxiv.org/abs/2511.08310)** Qingshan Xu, Jiao Liu, Shangshu Yu, Yuxuan Wang, Yuan Zhou, Junbao Zhou, Jiequan Cui, Yew-Soon Ong, Hanwang Zhang
* ...

### Citation
If you find this repo useful for your research, please consider citing the paper
```
@article{jiang2025phystwin,
    title={PhysTwin: Physics-Informed Reconstruction and Simulation of Deformable Objects from Videos},
    author={Jiang, Hanxiao and Hsu, Hao-Yu and Zhang, Kaifeng and Yu, Hsin-Ni and Wang, Shenlong and Li, Yunzhu},
    journal={ICCV},
    year={2025}
}
```
