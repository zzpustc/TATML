# TATML
A Target-oriented Meta Learning Method for WiFi-based Human Activity Recognition (HAR). 

This code is mainly about our new meta learning method TATML applying on the WiFi-based HAR.

# Platform
- python 3.x
- paddlepaddle 2.x

# Create Environment
We recommend you to use anaconda to create a new environment as follow (be careful about your cuda version):
```shell
conda create -n tatml python=3.7 paddlepaddle-gpu=2.1 cudatoolkit=10.2
```

Then install all required packages:
```shell
pip install -r requirements.txt
```

# Data Preparing
Our dataset is about WiFi-based human activity recognition. Here the original dataset will be provided soon. Before feeding data into the model, we have several preprocessing steps. You are recommended to realize these operations by yourself by referring to this [[paper](https://www2.cs.sfu.ca/~jcliu/Papers/WiCARWiFibased.pdf)]. 

# Training & Testing
Since our method mainly bases on MAML, we have several steps to train TATML. 
1. Training MAML
```python
python train_maml.py --src_id 2 --src_id 1 --root_path [your_data_path]
```
2. Training TATML
```python
python train_tatml.py --src_id 2 --src_id 1 --root_path [your_data_path]
```
Divide the target domain samples into validation and testing sets and then test the performance.

