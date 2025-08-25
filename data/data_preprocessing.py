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
output_dir = f'{root}/processed'
metadata_file = os.path.join(output_dir, 'metadata.json')
os.makedirs(output_dir, exist_ok=True)

logger.info(f'root folder: {root}')


def run(subfolder):
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
    # print('cat_names: ',cat_names)

    metadata_list = []

    # Iterate through all jpg files in the image directory
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

            # Parse the bounding box and category information
            bboxes = []
            categories = []
            for line in lines:
                # print(line)
                parts = line.strip().split()
                category = parts[0]
                bbox = list(map(float, parts[1:5]))
                categories.append(cat_names[int(category)])
                bboxes.append(bbox)
            
            value_counts = Counter(categories)
            value_counts_dict = dict(value_counts)
            
            # Create a category-specific directory if it doesn't exist
            if len(list(set(categories)))>1:
                logger.info(f'Multiple categories detected in the picture {image_filename}.jpg: {value_counts_dict}')
                category_name = 'mixed'
            else:
                category_name = categories[0]

            category_dir = os.path.join(output_dir, category_name)
            os.makedirs(category_dir, exist_ok=True)

            # Save the image as a PNG in the category-specific directory
            new_image_filename = os.path.splitext(image_filename)[0] + '.png'
            new_image_path = os.path.join(category_dir, new_image_filename)
            image.save(new_image_path)

            caption_str = 'A flower bouquet consisting of'
            for k,v in value_counts_dict.items():
                tmp_str = f' {str(v)} {k},'
                caption_str+=tmp_str
            caption_str = caption_str[:-1]
            print('caption_str: ', caption_str)

            # Prepare the metadata for this image
            metadata = {
                "file_name": f'{category_name}/{new_image_filename}',
                "additional_feature": caption_str,
                "objects": {
                    "bbox": bboxes,
                    "categories": categories
                }
            }
            metadata_list.append(metadata)
    return metadata_list
        

if __name__ == "__main__":
    subfolders = ['Dataset_1_1','Dataset_1_2','Dataset_1_3','Dataset_2_1']
    final_metadata = []
    for f in subfolders:
        metadata = run(f)
        final_metadata.extend(metadata)
    # Save all metadata to metadata.json
    with open(metadata_file, 'w') as f:
        json.dump(final_metadata, f, indent=4)

    logger.info(f"Processing complete.")