conda create -n vlm python=3.12 -y
conda activate vlm
pip install -r requirements.txt
conda install -c conda-forge librsvg -y
conda install -c conda-forge libiconv -y
sudo apt install libvips -y
