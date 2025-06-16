# Semantic Segmentation of Amazon Rainforest Cover in Sentinel-2 Satellite Imagery
The Amazon rainforest, a vital global carbon sink, is under increasing threat from rising temperatures, prolonged droughts, deforestation, and wildfires. Timely, accurate monitoring of forest cover is critical for detecting changes and guiding ecosystem conservation efforts. In this project, I built a U-net model to segment Amazon rainforest cover in 10 m resolution Sentinel-2 satellite imagery (4 color channels – red, green, blue, near-infrared). My model achieves a 97% true positive rate in detecting “forest” pixels, thereby offering reliable rainforest surveying for users. This solution can be integrated into real-time monitoring platforms, providing conservation organizations and policymakers with actionable insights to support rapid response and long-term planning for Amazon biome preservation. 

---

## Table of Contents
- [Project workflow](#project_workflow)
- [How can one use the optimized model from this project in an external application?](#model_deployment)
- [How can one work with their own version of this repository?](#repo_installation)
  
## Project workflow <a name="project_workflow"></a>

![Project workflow diagram](https://github.com/akshaysuresh1/amazonforest_segmentation/blob/main/media/project_workflow.png?raw=True)

## How can one use the optimized model from this project in an external application? <a name="model_deployment"></a>

1. Sign up for a Weights & Biases API key. Store the received API key in the `WANDB_API_KEY` environment variable.
2. Have `numpy`, `torch`, `wandb`, and `segmentation_models_pytorch` installed in your local Python environment. Preferably, use [uv](https://docs.astral.sh/uv/concepts/projects/dependencies/) to handle package dependencies and resolve version conflicts. Here is the configuration I used during project development.
```bash
pip install numpy==1.26.4 torch==2.2.2 segmentation_models_pytorch==0.5.0 wandb==0.19.11
```
3. In your Python environment, download the model artifact files from W&B.
```python
import wandb
run = wandb.init()
artifact = run.use_artifact(
    "akshaysuresh1/amazonforest_segmentation/unet_with_se_resnet50:best_model",
    type="model",
)
artifact_dir = artifact.download()
```  
4. Load the downloaded weights into a U-Net model.
```python
import os
import torch
from segmentation_models_pytorch import Unet

# Set torch device.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get relevant metadata from artifact.
encoder = artifact.metadata.get("encoder")
batch_size = artifact.metadata.get("batch_size")
initial_learning_rate = artifact.metadata.get("lr_initial")
weights_file = f"{encoder}_batch{batch_size}_lr{initial_learning_rate:.1e}_weights.pt"

# Intialize a U-Net model with random weights.
model = Unet(
    encoder_name=encoder,
    encoder_weights=None,
    in_channels=4,  # No. of input channels in data
    activation="sigmoid",
)

# Load trained weights from downloaded W&B artifact into U-Net model.
state_dict = torch.load(os.path.join(artifact_dir, weights_file), map_location=device)
model.load_state_dict(state_dict)
```
Your model is now ready to be deployed.

## How can one work with their own version of this repository? <a name="repo_installation"></a>

1. Create an AWS account and set up an S3 bucket with GeoTiff data from the "AMAZON.rar" file shared by [Bragagnolo, L., da Silva, R. V., & Grzybowski, J. M. V. (2021)](https://zenodo.org/records/4498086). Ensure that the S3 bucket has the below folder structure.
```
amazon_segmentation_dataset/
├── train/
│   ├── images/
│   └── masks/
├── validation/
│   ├── images/
│   └── masks/
├── test/
│   ├── images/
│   └── masks/
``` 
2. Sign up for a Weights & Biases API key.
3. Clone this repository to your local device and `cd` into the root directory.
```bash
git clone git@github.com:akshaysuresh1/amazonforest_segmentation.git
cd amazonforest_segmentation
```
4. In the project root directory, set up the requisite environment variables in your .env file.
```bash
# Creates a copy of the .env.example file.
cp .env.example .env
# TO DO: Supply your AWS and W&B credentials in the .env file.
```
5. Create a virtual environment using `venv` or `conda`. The example below mimics the Python config of the CircleCI docker image called during project development.
```bash
conda create -n amazon_project_env python==3.12.7 ipython
conda activate amazon_project_env
```  
6. Install the package `amazon_seg_project`.
```bash
pip install --upgrade pip
pip install .
```
7. Verify that all tests pass successfully on your local device.
```bash
# Run unit tests.
pytest -v amazon_seg_project_tests/test_unit

# Run integration tests.
pytest -v amazon_seg_project_tests/test_integration
```
Your installation is now ready for use.
