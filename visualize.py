import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# Parse command-line arguments
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--image_dir', type=str, help='input image directory')
parser.add_argument('--output_dir', type=str, help='output directory')
parser.add_argument('--mask_dir', type=str, default='', help='ground truth mask directory (optional)')
args = parser.parse_args()

image_dir = args.image_dir
output_dir = args.output_dir
mask_dir = args.mask_dir

# List all images in the directory
image_files = [f for f in os.listdir(image_dir) if f.endswith(('jpg', 'jpeg', 'png', 'bmp'))]
num_images = len(image_files)

# Create a single figure with multiple subplots
fig, axs = plt.subplots(num_images, 3, figsize=(20, 5 * num_images))
fig.suptitle('Image Analysis Results', fontsize=16)

# Initialize list to store file names and scores
file_scores = []

for idx, image_file in enumerate(image_files):
    image_path = os.path.join(image_dir, image_file)
    output_path = os.path.join(output_dir, f"{image_file}.npz")

    # Load result
    if not os.path.exists(output_path):
        print(f"Output file {output_path} not found, skipping.")
        continue

    result = np.load(output_path)

    # Display image with score
    axs[idx, 0].imshow(Image.open(image_path))
    axs[idx, 0].set_title(f'Image\nScore: {result["score"]:.3f} ,{image_file}')
    axs[idx, 0].axis('off')

    axs[idx, 1].imshow(result['map'], cmap='RdBu_r', clim=[0, 1])
    axs[idx, 1].set_title('Localization map')
    axs[idx, 1].axis('off')

    axs[idx, 2].imshow(result['conf'], cmap='gray', clim=[0, 1])
    axs[idx, 2].set_title('Confidence map')
    axs[idx, 2].axis('off')

    # Store file name and score
    file_scores.append([image_file, result["score"]])

plt.tight_layout()
plt.subplots_adjust(top=0.95, bottom=0.05)  # Adjust top and bottom margins

output_figure = "morph_plot.png"
plt.savefig(output_figure)

plt.show()


