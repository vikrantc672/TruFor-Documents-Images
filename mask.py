#python3 mask.py --image_dir $(realpath ${INPUT_DIR}) --mask_npz_dir $(realpath ${OUTPUT_DIR}) --result_mask_dir $(realpath ${MASK_RESULT_DIR})



import argparse
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt



# Parse command-line arguments
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--image_dir', type=str, help='original image directory')
parser.add_argument('--mask_npz_dir', type=str, help='mask npz directory')
parser.add_argument('--result_mask_dir', type=str, help='directory to store masks')
args = parser.parse_args()

image_dir = args.image_dir
mask_npz_dir = args.mask_npz_dir
result_mask_dir = args.result_mask_dir

# Ensure the result directory exists
os.makedirs(result_mask_dir, exist_ok=True)

# List all images in the directory
image_files = [f for f in os.listdir(image_dir) if f.endswith(('jpg', 'jpeg', 'png', 'bmp'))]

for image_file in image_files:
    image_path = os.path.join(image_dir, image_file)
    output_path = os.path.join(mask_npz_dir, f"{image_file}.npz")
    if not os.path.exists(output_path):
        print(f"Output file {output_path} not found, skipping.")
        continue

    result = np.load(output_path)
    
    # Save localization map
    localization_map = result['map']
    plt.imsave(os.path.join(result_mask_dir, f"{image_file}_localization_map.png"), localization_map, cmap='RdBu_r', vmin=0, vmax=1)

print("Localization maps have been saved.")
