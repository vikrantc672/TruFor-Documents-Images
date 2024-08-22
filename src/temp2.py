import os
import glob
import argparse
from temp import process_single_image  # Import the function from image_processor.py
import numpy as np
import torch
from torch.nn import functional as F
from glob import glob
from config import update_config
from config import _C as config
from data_core import myDataset
def process_images_in_directory(input_dir, output_dir,  gpu=0, save_np=False):
    # Get a list of all image files in the directory
    image_files = glob(os.path.join(input_dir, '**/*'), recursive=True)  # Correct usage of glob

    # device = 'cuda:%d' % gpu if gpu >= 0 else 'cpu'

    # if config.TEST.MODEL_FILE:
    #     model_state_file = config.TEST.MODEL_FILE
    # else:
    #     raise ValueError("Model file is not specified in the configuration.")

    # print('=> loading model from {}'.format(model_state_file))
    # checkpoint = torch.load(model_state_file, map_location=torch.device(device))

    # if config.MODEL.NAME == 'detconfcmx':
    #     from models.cmx.builder_np_conf import myEncoderDecoder as confcmx
    #     model = confcmx(cfg=config)
    # else:
    #     raise NotImplementedError('Model not implemented')

    # model.load_state_dict(checkpoint['state_dict'])
    # model = model.to(device)

    for image_path in image_files:
        if not os.path.isdir(image_path):  # Process only files, not directories
            success = process_single_image(image_path, output_dir)
            if not success:
                print(f"Failed to process image: {image_path}")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process images with TruFor model.')
    parser.add_argument('-gpu', '--gpu', type=int, default=-1, help='GPU device ID, use -1 for CPU')
    parser.add_argument('-in', '--input', type=str, required=True, help='Input directory')
    parser.add_argument('-out', '--output', type=str, required=True, help='Output directory')
    # parser.add_argument('-model', '--model', type=str, required=True, help='Path to the model state file')
    parser.add_argument('-save_np', '--save_np', action='store_true', help='Whether to save the Noiseprint++ or not')

    args = parser.parse_args()

    # Call the function to process images
    process_images_in_directory(args.input, args.output,  args.gpu, args.save_np)
