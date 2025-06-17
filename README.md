# Federated Learning Pipeline (Thesis Project)

This repository contains the code and resources for a master's thesis project by Joran Verheijen (Joran.Verheijen@vub.be) and Robbe Vlaeminck (Robbe.Vlaeminck@vub.be). The project investigates the practical deployment of Federated Learning (FL) on the Jetson Nano. The primary objective is to evaluate how FL can deliver privacy-preserving, energy-efficient, and accurate object detection models on edge devices.

## Project Overview

Traditional centralized machine learning systems require users to send raw data to remote servers for training, introducing privacy risks and network bottlenecks. These concerns are especially important in assistive applications involving wearable cameras and personal sensing devices. FL addresses these issues by enabling decentralized training: user data remains on-device, and only model parameters (weights or gradients) are exchanged with a central aggregator. This approach enhances privacy, reduces bandwidth and energy consumption, and is well-suited for battery-powered edge devices.

This project develops a complete FL pipeline, including model design, client-server coordination, and systematic evaluation under varying configurations. The focus is on experimental evaluation and practical performance trade-offs in federated embedded learning.

## Research Objectives

- **Build a Functional FL Framework:** Develop a modular FL framework with a central aggregation server and multiple clients (Jetson Nano hardware and simulated nodes). Validate the framework using both the MNIST classification task and the Outdoor Obstacle Detection (OOD) dataset.
- **Decentralized Training for Privacy:** Ensure all training data remains on local devices, reducing privacy concerns and network traffic by exchanging only model parameters.
- **Compare FL with Centralized and Stand-Alone Learning:** Conduct controlled experiments to compare federated, centralized, and stand-alone learning. Evaluation metrics include classification accuracy, energy consumption, communication overhead, and convergence speed. Due to computational constraints, a custom lightweight CNN4 was developed for the OOD dataset instead of heavier architectures like YOLO or MobileNet.

## Key Contributions

- A modular and reproducible FL framework for embedded AI applications.
- Comparative analysis of federated, centralized, and stand-alone learning setups.
- A validated lightweight CNN model optimized for Jetson Nano.
- Design and deployment insights for FL in energy-constrained, privacy-sensitive environments.

## Project Structure

- `MNIST/`: Contains scripts and data for MNIST experiments. The dataset is fully loaded and partitioned on the Jetson Nano, with partitioning handled in `client.py`. `client2.py` provides additional testing utilities. The MNIST dataset is used exclusively within this folder, making it a good starting point for understanding the project workflow. 
- `Object_detect+utils/`: Contains scripts and utilities for both centralized and federated learning experiments, as well as configuration files.
    - **Centralised learning (`Centralised learning/`):** Scripts for centralized training, including `centralized_OOD.py`, `centralized_VIS.py`, and `TF_OOD.py`. These are designed for PC use (can run on Jetson Nano, but are slow with the full dataset). The dataset is reclassified into "movable" (e.g., car) and "non-movable" (e.g., trash bin, standstill) object classes, plus a background class. TFRecord files (not included in the repo) are required as input.
        - **OOD:** Base code, used for the learning of the Outdoor Obstacle Detection dataset. Other files in this folder, are modified from this.
        - **VIS:** Similar structure to centralized learning, but uses a different dataset, preprocessing, and class definitions.
        - **TF_ODD:** Implements transfer learning with fine-tuning. The main difference from centralized OOD scripts is in lines 99–184.
    - **Configuration Files (`Configs/`):** `config1.py` is adapted for federated learning; `TF_config.py` may not be used in the final project.
    - **Federated learning (`Federated learning/`):** Contains scripts for federated learning experiments. Note: `psutil` may need to be installed separately to monitor energy usage during training (as referenced in the thesis, pg 28), alongside `tegrastats`.
    - **Utils (`Utils/`):** Utility scripts for data handling and evaluation:
        - `eval_OOD.py`: Evaluates models on the OOD dataset.
        - `tf_splitter.py`: Splits large datasets; set the `DATASET` variable on line 26 for each run. The split is 80/10/10, already reflected in the TFRecord files.
        - `train_val_split.py`: Creates an 80/20 train/validation split.
        - `tf_splitter.py`: Splits a single, pre-labeled dataset.
- `Object_detection/`: Object detection scripts and configurations.
- `dataomvormen_yolo.py`: Data transformation scripts for YOLO.
- `Comparision/`: Contains scripts and models for comparing different YOLO and OOD approaches.
- `documents/`: Project documentation and Gantt chart scripts.
- `Improvements/object_detection1/`: Improved object detection scripts and federated learning models.
- `OOD.v1i.yolov5pytorch/`: OOD detection experiments with YOLOv5 in PyTorch.
    - Files inlcuded in the file from Roboflow, not used
- `standalone/`: Standalone scripts and models for YOLO and CNN experiments. Config files are often reused. 
    - folder is not used in thesis, mainly ysed for experimentations. 


## Datasets
- **MNIST Dataset:** [https://git-disl.github.io/GTDLBench/datasets/mnist_datasets/](https://git-disl.github.io/GTDLBench/datasets/mnist_datasets/) - might be some pre-processing involved?
- **Outdoor Obstacle Detection (OOD) Dataset:** [https://universe.roboflow.com/fpn/ood-pbnro](https://universe.roboflow.com/fpn/ood-pbnro)
- **Visually Impaired (VI) Dataset:** [https://universe.roboflow.com/all-mix/visually-impaired-dataset](https://universe.roboflow.com/all-mix/visually-impaired-dataset)
- [] TODO check datasets tomorrow.
- File names should be the same in the config files, with what we see in roboflow. 

## Getting Started

1. Clone the repository:
   ```powershell
   git clone https://github.com/Vlaeminksken/thesis_github
   ```
2. Install required Python packages (see individual script requirements).
3. Run scripts as needed for your experiments.

## Requirements

- Python 3.7+
- TensorFlow, PyTorch, and other dependencies (see script headers or requirements files)
- Jetson Nano DNN image ([Qengineering/Jetson-Nano-image](https://github.com/Qengineering/Jetson-Nano-image) recommended)


## Practical Setup Notes

- **Jetson Nano DNN Image:** Instead of manually installing JetPack SDK, TensorFlow, and CUDA, use the Jetson Nano DNN image. This approach saves significant setup time and ensures compatibility, especially with CUDA since conflicts can/will happen.
- **microSD Card Size:** The default DNN image is for 32GB microSD cards. For this project, modify the image to work with a 64GB card to provide sufficient swap space for model training with GParted.
- **Flashing the Image:** Use the Raspberry Pi Foundation's Imager tool with the "no filtering" and "custom" options to flash the image to the microSD card. The process takes about 30 minutes.
- **WiFi Module Compatibility:** The TP-Link TL-WN722N USB WiFi module is recommended for reliable out-of-the-box support. Avoid the TP-Link Archer T2U. Note that the TL-WN722N may block other USB ports, so plan USB device connections accordingly.
- **Jetson Nano Hardware Limitations:** The Jetson Nano P3450 (2019) may only support inference (not training) for some algorithms (e.g., YOLO). This limitation motivated the use of the lightweight CNN4 model for training.
- **Python Package Versions:** The repository's Python scripts do not specify exact package versions. On Jetson Nano DNN, scripts generally run without additional updates (except possibly NumPy) as of June 2025. On Windows 11, use the latest package versions via pip.
- **Data Format:** The tf.record format is used for importing images from RoboFlow.
- **Configuration Files:** Configuration files (e.g., `configX.py`) are imported as modules (e.g., `import configX as cfg`) in the main scripts. This allows flexible configuration management for different experiments.
- Config files in each folder, are for the Python files in each folder. Make sure that they are located next to each other when importing it into it.

## Set up virtual environment
It is recommended to use a virtual environment to ensure packages are installed correctly. This ensures the portability of the code across different machines. The `uv` package manager works well for this purpose, and is also suggested by the Marimo project. 

See the [uv installation guide](https://github.com/astral-sh/uv?tab=readme-ov-file#installation).

```
uv venv                              # setup virtual environment (venv) with uv
.venv\Scripts\activate               # activate venv (Windows)
#source .venv/bin/activate                  # activate venv (Linux/OSX)
uv pip install -r requirements.txt   # install marimo environment & dependencies
```

## Usage

- Refer to the scripts in each folder for specific experiments and usage instructions.
- Configuration files are provided in the `Configs/` directories.

## Results

- Model weights, training logs, and evaluation plots are stored in the respective `models/` directories.

## Documentation

- See `Master-Project-Description.pdf` and other documents in the `documents/` folder for more details.


## Notes More
- The OKDO box does not work. Licensed, but not the same. boot up, but got stuck in the booting screen and shut down.
- Buy newest, easy to install OS. Also more ram and computational strength. P3450. carolinn
- Ethernet does not work, but Eduroam "should" work. Hotspot gives more control over load IP and so on. 
- use ip config to find IP of laptop
- first step: for federated learning with object detection. only training. server running on laptop first (serverobj1.py) in federated learning/ base folder. Should be on the same wifi network as Jetson NAno e.g. connected to phone wifi. ipconfig ipv4 address note that down
- On PC it is 3.11.5, on JN 3.8.10 for python version.
-  sever listening should show, but it waits for a socket connection. 
- Location of dataset is hardcoded inside the clientobj1.py file. We need to set BASE_DIR to where the dataset is. 
- tf_splitter.py and train_val_splitter both kinda do the same thing. the tf_splitter allows for fancy parameters to be given to it in the cmd line, whereas train_val_spli has the params hard coded
- the test and the validation subsects are combined together, in train_val_split .8 .2 training and validation. VIS has small dataset items, so this is done to make things better. In robotform it should be in the correct form. 
- MNIST is easy version of object detection. kaggle hojjak dataset includes the files used.
- object detection is already splitted in three parts. but, we want to fine tune. so, we had to find data base to fine tune it. big datasets don't hvae tfrecord, so use conversions of xml. BUT, issues with mismatching image. Convinced, to one find that is avlaible and possible to down load in tfrecord. SO the coversion required the codes. Pro tip, use tfrecord as priority. easily used by jetson since it used low memory. also, easy to handle in tensorflow. This is more important than a big and better dataset.
- tfrecord = easy + low computational way
- pascal vlc dataset was got work, but it was too much to handle for jetson nano
- MNIST is not tfrecord, but it is OK since it is small. 
- splitter_tf is the one eventually used. 
- The split_XX folder is given to the jetson nano or virtual client
- clientobj1.py is different on the jn and pc. will be moved to the repo. The difference is in how the directory is handled.
- clientobj1.py in federated learning / base is to be loaded on the JN. line 155 till 162 is different. It is blind to what the dataset is being used, just specify the split_XX folder from splitter_tf.
- specifically, we use python3 clientobj.py --client_ID 0 where 0 is any integer you want. -> module not found? may require pip3 install psutil -> 7.0.0 installed
-  base conda environment does not have the packages, but the non virtual has no problem with the packages installed. 
- The client runner is used to run multiple clients quickly on the server. line32 in the clientrunner is used to specify what the client is to be run on the server. python clientrunner.py 9 where 9 is the number of virtual clients. When the enter is pressed and training is tarted, the clients are added. 
- in the config.py file, make sure the PRETRAINED_PATH is set to the correct location. For transfer tuning and fine tuning. in some code, it is hard coded so use due diligence. The comments allow selection of which dataset is used. SPLITS_DIR is the folder where the splits are defined.
- The training starts, when the clients are connected on the jetson nano. CLients on PC done in about 10 seconds. Takes 450 seconds ish to finish all epochs though. The server is waiting for the JN to send its weight. Training is done on the CPU, not the GPU since it is more stable. See comment on line 164 lol.
- MNIST can be forced on the GPU, but the other two are done on the CPU. lol cpu energy is incorrect. recieving weights is less intensive
- 1/10 dataset training, then go and average on the fed server, then the fine tuning on the local set again. 
- the higher it training ahppens, the longer time it takes for each round. Gets warm and toasty. power consumption increases.
- finger pointing to section which says it is using GPU
- what they would do different-> using newer JN or laptop / PC to run it
- jtop is used to get wattage -> lost when trying to run on GPU, also memory errors
- MNIST is possible on GPU, but the other too is not possible. reverts to using the CPU when the GPU is not able to handle much. 
- In the JetsonReady folder the clientobj1 is for the virtual clients, called by the client runner. in the base folder, the clientobj1 is for local use on the jetson, this is NOT running on the GPU be default. And try not to run on the GPU either.
- serverobj1 is for transfer learning. we do the learning on the smaller (?) VIS dataset on the server PC thingy. we trasnfer weights to the clients, and fine tuning is done there with the OOD dataset. clients finish and send wieght back to server and the federated avergae test is done on the server.
- using something that is not related, can still have impact with trasnfer learning, goals of thesis.
- server is ran the exact same as the other code in the trasnferlearning_finetuning. it is for fine tuning. config files are in jetson ready. just change the directories for the clients, and the pretrained path and mode. the secomd argument is the number of layers in line 108