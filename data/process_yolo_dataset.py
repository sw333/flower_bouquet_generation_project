import argparse
import logging
import math
import os
import random
import shutil
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import datasets
from datasets import load_dataset
import os
import json
import yaml
from PIL import Image
import logging
import sys
from collections import Counter

logger = logging.getLogger(__name__)
log_formatter = '[%(asctime)s] %(levelname)s [%(filename)s:%(lineno)s:%(funcName)s] %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_formatter)

root = os.path.dirname(os.path.realpath(__file__))
output_folder = f'{root}/yolo_combined_dataset'
os.makedirs(output_folder, exist_ok=True)
train_dir = os.path.join(output_folder, 'train')
valid_dir = os.path.join(output_folder, 'valid')
test_dir = os.path.join(output_folder, 'test')
os.makedirs(f'{train_dir}/images', exist_ok=True)
os.makedirs(f'{train_dir}/labels', exist_ok=True)
os.makedirs(f'{valid_dir}/images', exist_ok=True)
os.makedirs(f'{valid_dir}/labels', exist_ok=True)
os.makedirs(f'{test_dir}/images', exist_ok=True)
os.makedirs(f'{test_dir}/labels', exist_ok=True)
probabilities = [0.85, 0.10, 0.05]


logger.info(f'root folder: {root}')

final_catefories = {
    'daisy':0,'dandelion':1,'grtz':2,'irises':3,'lilies':4,
    'lisianthuses':5,'chrysanthemums':6,
    'orchids':7,'peonies':8,'roses':9,'tulips':10,
    # 'mixed':11
}

subfolders = ['Dataset_1_1','Dataset_1_2','Dataset_1_3','Dataset_2_1']
for subfolder in subfolders:
    subroot = f'{root}/{subfolder}'
    image_dir = f'{subroot}/raw/images'
    label_dir = f'{subroot}/raw/labels'
    logger.info(f'processing pictures from: {image_dir}')

    yml_dir = f'{subroot}/data.yaml'
    with open(yml_dir, "r") as stream:
        try:
            cat_info = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise ImportError('Failed to load config.yml, please check on the file') from exc
    cat_names = cat_info['names']

    for image_filename in os.listdir(image_dir):
        logger.info(f'filename: {image_filename}')
        if image_filename.endswith('.jpg'):
            # Corresponding label file
            label_filename = os.path.splitext(image_filename)[0] + '.txt'
            image_path = os.path.join(image_dir, image_filename)
            label_path = os.path.join(label_dir, label_filename)

            image = Image.open(image_path)
            with open(label_path, 'r') as label_file:
                lines = label_file.readlines()
            
            if len(lines)<1:
                logger.info('no flower object in txt file, image skipped.')
                continue
            
            choice = random.choices(['train', 'valid', 'test'], probabilities)[0]
            new_image_path = os.path.join(f'{output_folder}/{choice}/images', image_filename)
            image.save(new_image_path)
            logger.info(f'saving image to: {new_image_path}')

            # Parse the bounding box and category information
            new_txt = []
            for line in lines:
                # print(line)
                tmp_txt = []
                parts = line.strip().split()
                category = parts[0]
                bbox = list(map(float, parts[1:5]))
                newcat = final_catefories[cat_names[int(category)]]
                tmp_txt.append(newcat)
                tmp_txt.extend(bbox)
                new_txt.append(tmp_txt)
            # print('checking new text file: ')
            # print(new_txt)
            new_txt_path = os.path.join(f'{output_folder}/{choice}/labels', label_filename)
            with open(new_txt_path, 'w') as file:
                for line in new_txt:
                    # Convert each number to string and join with spaces
                    file.write(' '.join(map(str, line)) + '\n')
            logger.info(f'saving label to: {new_txt_path}')
