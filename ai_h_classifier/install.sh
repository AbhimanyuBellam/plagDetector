sudo apt update
sudo apt install python3-pip -y

pip3 install --upgrade pip

pip3 install openpyxl xlsxwriter setuptools gdown torchtext matplotlib torch transformers datasets accelerate nvidia-ml-py3 optimum

git config --global user.name bellamabhimanyu@gmail.com

git config --global user.email bellamabhimanyu@gmail.com

git clone https://github.com/AbhimanyuBellam/plagDetector

cd plagDetector/ai_h_classifier
python3 data_downloader.py
