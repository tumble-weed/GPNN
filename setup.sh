sudo apt-get install libomp-dev -y
pip install -r requirements.txt
pip install faiss-gpu --no-cache

# conda install -c conda-forge faiss-gpu -y
# conda install -c pytorch faiss-gpu cudatoolkit=11.6
conda install -c pytorch faiss-gpu=1.7.3 cudatoolkit=11.3 -y