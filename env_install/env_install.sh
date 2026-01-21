conda install -y numpy==1.26.4
pip install warp-lang
pip install usd-core matplotlib
pip install "pyglet<2"
pip install open3d
pip install trimesh
pip install rtree 
pip install pyrender

conda install -y pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.4 -c pytorch -c nvidia
pip install stannum
pip install termcolor
pip install fvcore
pip install wandb
pip install moviepy imageio
conda install -y opencv
pip install cma
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt240/download.html

# # Install the env for realsense camera
pip install Cython
pip install pyrealsense2
pip install atomics
pip install pynput

# Install the env for grounded-sam-2
pip install git+https://github.com/IDEA-Research/Grounded-SAM-2.git
pip install --no-build-isolation git+https://github.com/IDEA-Research/GroundingDINO.git

# Install the env for image upscaler using SDXL
pip install diffusers
pip install accelerate

# # Install the env for trellis
cd data_process
git clone --recurse-submodules https://github.com/microsoft/TRELLIS.git
cd TRELLIS
. ./setup.sh --basic --xformers --flash-attn --diffoctreerast --spconv --mipgaussian --kaolin --nvdiffrast

cd ../..

pip install gsplat==1.4.0
pip install kornia
cd gaussian_splatting/
pip install --no-build-isolation submodules/diff-gaussian-rasterization/
pip install --no-build-isolation submodules/simple-knn/
cd ..
