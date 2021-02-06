### Environment Setup ###

conda create -n torch python=3.6
\
conda activate torch
\
conda install pip
\
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch
\
pip install numpy pandas tqdm matplotlib
