export http_proxy="http://127.0.0.1:7890"
export https_proxy="http://127.0.0.1:7890"

git clone https://github.com/bytedance/SandboxFusion.git
cd SandboxFusion

conda create -n sandbox -y python=3.12
conda activate sandbox
pip install poetry
# poetry source add --priority=primary sankuai https://pypi.sankuai.com/simple/
poetry lock
poetry install

mkdir -p docs/build

sed -i 's/python=3.10/python=3.12/' runtime/python/install-python-runtime.sh

cd runtime/python

sed -i 's/tensorflow==2.14.1/tensorflow==2.16.1/' requirements.txt
sed -i '/^torch==2.1.2/d' requirements.txt
sed -i 's/numba==0.58.1/numba==0.59.1/' requirements.txt
sed -i 's/PyQt5==5.15.10/PyQt5==5.15.11/' requirements.txt
sed -i 's/tree-sitter==0.20.4/tree-sitter==0.21.3/' requirements.txt
sed -i 's/tree-sitter-languages==1.9.1/tree-sitter-languages==1.10.2/' requirements.txt
sed -i 's/faiss-cpu==1.7.4/faiss-cpu==1.8.0/' requirements.txt
sed -i 's/gmpy2==2.1.5/gmpy2==2.2.1/' requirements.txt

rewrite ~/SandboxFusion/runtime/python/install-python-runtime.sh

bash install-python-runtime.sh

conda activate sandbox
cd SandboxFusion
make run-online

# conda create -p /home/hadoop-mtsearch-assistant/dolphinfs_hdd_hadoop-mtsearch-assistant/liujiarun02/envs/sandbox --clone sandbox
# conda activate /home/hadoop-mtsearch-assistant/dolphinfs_hdd_hadoop-mtsearch-assistant/liujiarun02/envs/sandbox