## Final Project
## Taming Stable Diffusion for Lip Sync!

### ITCS 6166: Communication And Computer Networks

#### Team 11:

- **Ananya Reddy**
- **Nithya**
- **Vaishnav Reddy**
- **Tharun Sai Reddy**

## Project Description:

This project implements a Diffusion-based Latent Editing Framework for Realistic Talking Head Generation via Audio-Latent Space Synchronization. The project is based on the paper “LatentSync: High-Fidelity Talking Head Generation via Audio-Latent Space Sync”, published in 2023 by researchers at ByteDance. The implementation is done in PyTorch.

We enhanced the original pipeline by integrating it with a Streamlit interface for simplified interaction and deployed the entire system on Google Colab for accessible, GPU-powered execution. The model takes an input audio and a still face image to generate a high-quality, temporally aligned lip-synced video output using diffusion techniques applied in latent space.

## Project Structure:

- **LatentSync**
  - Contains the implementation of the LatentSync model. From [link](https://github.com/bytedance/LatentSync)
- **Tests**
  - Contains the test data for the project
- **models**
  - Contains the trained models for the project (LatentSync_unet and stabel_unet) (Exported)
- **requirements.txt**
  - Contains the required libraries for the project
- **Streamlit_run.py**
  - Contains the code to run the project on the Streamlit user interface
  - **Pyngorg**
  - 

## How to run the project (Colab without exported models):

- Open the colab notebook [Colab Link](https://colab.research.google.com/drive/1xNPglltHZws663KWjJxbF1jpD1dy6sA-?usp=sharing)
- Run the cells in the notebook

## How to run the project (Local with exported models):

- Clone the repository
- Create a virtual environment using the command `conda create -n <env_name> python=3.10`
- Install the required libraries using the command `pip install -r requirements.txt`
- Run the project using the command `python streamlit_run.py`

## Training

## 🏋️‍♂️ Training U-Net

Before training, you should process the data as described above. We released a pretrained SyncNet with 94% accuracy on both VoxCeleb2 and HDTF datasets for the supervision of U-Net training. You can execute the following command to download this SyncNet checkpoint:

```bash
huggingface-cli download ByteDance/LatentSync-1.5 stable_syncnet.pt --local-dir checkpoints
```

If all the preparations are complete, you can train the U-Net with the following script:

```bash
./train_unet.sh
```

We prepared three UNet configuration files in the ``configs/unet`` directory, each corresponding to a different training setup:

- `stage1.yaml`: Stage1 training, requires **23 GB** VRAM.
- `stage2.yaml`: Stage2 training with optimal performance, requires **30 GB** VRAM.
- `stage2_efficient.yaml`: Efficient Stage 2 training, requires **20 GB** VRAM. It may lead to slight degradation in visual quality and temporal consistency compared with `stage2.yaml`, suitable for users with consumer-grade GPUs, such as the RTX 3090.

Also remember to change the parameters in U-Net config file to specify the data directory, checkpoint save path, and other training hyperparameters. For convenience, we prepared a script for writing a data files list. Run the following command:

```bash
python -m tools.write_fileslist
```

## 🏋️‍♂️ Training SyncNet

In case you want to train SyncNet on your own datasets, you can run the following script. The data processing pipeline for SyncNet is the same as U-Net. 

```bash
./train_syncnet.sh
```

After `validations_steps` training, the loss charts will be saved in `train_output_dir`. They contain both the training and validation loss. If you want to customize the architecture of SyncNet for different image resolutions and input frame lengths, please follow the [guide](docs/syncnet_arch.md).

