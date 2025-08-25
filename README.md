# ay2425-s1-pg16
ST5188 AY2024/25 S1 Project Group 16 Repository Resources

# Enhancing Generative AI Models with Customized Evaluation Metrics: A Case Study in Realistic Flower Bouquet Generation 

Welcome to the Project! This guide will help you get started with data processing, training, and testing.

## Directory Structure

- **/data/**: Contains data processing scripts.
- **/diffusers/examples/text-to-image/**: Contains stable diffusion fine-tuning scripts.
- **/yolo/**: Contains yolov8 training script.
- **/test/**: Contains scripts and resources for testing the model.

## Setup Instructions

1. **Data Processing**
   
   Run data processing scripts in data folder to prepare training data for YOLO training and stable diffusion fine-tuning. Our source data is taken from Roboflow website.
   Final test captions generation script can be found here as well.


2. **Install Requirements**

   Before you begin training, make sure to install all necessary dependencies. 
   Navigate to **diffusers/examples/text-to-image/** and run the following command:

   ```bash
   pip install -r requirements.txt

3. **Model training**

   Train a YOLO model.

   Fine-tune a stable diffusion model using LoRa.
   cd diffusers/examples/text-to-image
   bash instruction.sh
   
4. **Model testing**
   
   Navigate to data folder first and use generate_prompt.py to generate 100 captions.
   Navigate to test/stable_diffusion_with_custom_loss_model.py for test image generation using the fine-tuned model.
