import sys, os
import argparse
import numpy as np
from tqdm import tqdm
from glob import glob
import cv2
import torch
from torch.nn import functional as F
import torchvision.transforms.functional as TF

path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
if path not in sys.path:
    sys.path.insert(0, path)

from config import update_config
from config import _C as config
from data_core import myDataset

# Argument parser setup
parser = argparse.ArgumentParser(description='Test TruFor')
parser.add_argument('-gpu', '--gpu', type=int, default=0, help='device, use -1 for cpu')
parser.add_argument('-in', '--input', type=str, default='../images',
                    help='can be a single file, a directory or a glob statement')
parser.add_argument('-out', '--output', type=str, default='../output', help='output folder')
parser.add_argument('-save_np', '--save_np', action='store_true', help='whether to save the Noiseprint++ or not')
parser.add_argument('opts', help="other options", default=None, nargs=argparse.REMAINDER)

args = parser.parse_args()
update_config(config, args)

input = args.input
output = args.output
gpu = args.gpu
save_np = args.save_np

device = 'cuda:%d' % gpu if gpu >= 0 else 'cpu'
np.set_printoptions(formatter={'float': '{: 7.3f}'.format})

if device != 'cpu':
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

# Input files processing
if '*' in input:
    list_img = glob(input, recursive=True)
    list_img = [img for img in list_img if not os.path.isdir(img)]
elif os.path.isfile(input):
    list_img = [input]
elif os.path.isdir(input):
    list_img = glob(os.path.join(input, '**/*'), recursive=True)
    list_img = [img for img in list_img if not os.path.isdir(img)]
else:
    raise ValueError("input is neither a file nor a folder")

# Resizing function using OpenCV
def resize_image_if_needed(img):
    """
    Resize the image if any dimension exceeds 1200 pixels while maintaining the aspect ratio.
    
    Parameters:
    img -- A NumPy array representing the image.

    Returns:
    img_resized -- Resized NumPy array representing the image.
    """
    max_dim = 2400
    height, width = img.shape[:2]

    # Check if resizing is necessary
    if width > max_dim or height > max_dim:
        if width > height:
            new_width = max_dim
            new_height = int((max_dim / width) * height)
        else:
            new_height = max_dim
            new_width = int((max_dim / height) * width)

        # Resize image
        img_resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        print("resized")
    else:
        img_resized = img

    return img_resized

# Dataset and DataLoader setup
test_dataset = myDataset(list_img=list_img)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=1)

# Model loading
if config.TEST.MODEL_FILE:
    model_state_file = config.TEST.MODEL_FILE
else:
    raise ValueError("Model file is not specified.")

print('=> loading model from {}'.format(model_state_file))
checkpoint = torch.load(model_state_file, map_location=torch.device(device))

if config.MODEL.NAME == 'detconfcmx':
    from models.cmx.builder_np_conf import myEncoderDecoder as confcmx
    model = confcmx(cfg=config)
else:
    raise NotImplementedError('Model not implemented')

model.load_state_dict(checkpoint['state_dict'])
model = model.to(device)

# Processing each image
with torch.no_grad():
    for index, (rgb, path) in enumerate(tqdm(testloader)):
        path = path[0]

        if os.path.splitext(os.path.basename(output))[1] == '':
            root = input.split('*')[0]
            sub_path = path.replace(os.path.dirname(root), '').strip() if os.path.isfile(input) else path.replace(root, '').strip()
            if sub_path.startswith('/'):
                sub_path = sub_path[1:]
            filename_out = os.path.join(output, sub_path) + '.npz'
        else:
            filename_out = output

        if not filename_out.endswith('.npz'):
            filename_out = filename_out + '.npz'

        if not os.path.isfile(filename_out):
            try:
                # Convert tensor to NumPy array
                rgb_numpy = rgb.squeeze(0).permute(1, 2, 0).cpu().numpy()

                # Resize the image using OpenCV
                rgb_resized = resize_image_if_needed(rgb_numpy)

                # Convert back to tensor
                rgb_resized = torch.tensor(rgb_resized).permute(2, 0, 1).unsqueeze(0).to(device)

                model.eval()
                det = conf = None

                pred, conf, det, npp = model(rgb_resized)

                if conf is not None:
                    conf = torch.squeeze(conf, 0)
                    conf = torch.sigmoid(conf)[0]
                    conf = conf.cpu().numpy()

                if npp is not None:
                    npp = torch.squeeze(npp, 0)[0]
                    npp = npp.cpu().numpy()

                if det is not None:
                    det_sig = torch.sigmoid(det).item()

                pred = torch.squeeze(pred, 0)
                pred = F.softmax(pred, dim=0)[1]
                pred = pred.cpu().numpy()

                out_dict = {
                    'map': pred,
                    'imgsize': tuple(rgb_resized.shape[2:])
                }
                if det is not None:
                    out_dict['score'] = det_sig
                if conf is not None:
                    out_dict['conf'] = conf
                if save_np:
                    out_dict['np++'] = npp

                os.makedirs(os.path.dirname(filename_out), exist_ok=True)
                np.savez(filename_out, **out_dict)

            except Exception as e:
                print(f"Error processing {path}: {e}")
                traceback.print_exc()
