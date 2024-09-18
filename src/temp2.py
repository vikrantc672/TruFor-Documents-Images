import os
import glob
import argparse
import time  
from temp import process_single_image  
import numpy as np
import torch
from torch.nn import functional as F
from glob import glob
from config import update_config
from config import _C as config
from data_core import myDataset

def process_images_in_directory(input_dir, output_dir, gpu=0, save_np=False):
    start_time = time.time()  
    print("Gathering image files...")
    image_files = glob(os.path.join(input_dir, '**/*'), recursive=True)
    print(f"Time after gathering images: {time.time() - start_time:.6f} seconds")
    for image_path in image_files:
        print(f"Processing image: {image_path}")
        single_image_start = time.time() 
        if not os.path.isdir(image_path): 
            print(f"Processing file: {image_path}")
            process_start_time = time.time()
            success = process_single_image(image_path, output_dir)
            print(f"Time after processing image {image_path}: {time.time() - process_start_time:.6f} seconds")
            if not success:
                print(f"Failed to process image: {image_path}")

        print(f"Time after completing {image_path}: {time.time() - single_image_start:.6f} seconds")

    print(f"Total time for processing all images: {time.time() - start_time:.6f} seconds")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process images with TruFor model.')
    parser.add_argument('-gpu', '--gpu', type=int, default=-1, help='GPU device ID, use -1 for CPU')
    parser.add_argument('-in', '--input', type=str, required=True, help='Input directory')
    parser.add_argument('-out', '--output', type=str, required=True, help='Output directory')
    parser.add_argument('-save_np', '--save_np', action='store_true', help='Whether to save the Noiseprint++ or not')

    args = parser.parse_args()
    process_images_in_directory(args.input, args.output, args.gpu, args.save_np)
