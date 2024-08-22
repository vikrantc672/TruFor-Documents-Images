import sys
import os
import argparse
import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm
from glob import glob
from multiprocessing import Process, Manager, Queue, current_process
import logging
from PIL import Image

# Set up logging for better debugging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
if path not in sys.path:
    sys.path.insert(0, path)

from config import update_config
from config import _C as config
from data_core import myDataset

# Custom exceptions
class TimeoutException(Exception):
    pass

def process_image(rgb, model_state_dict, device, result_queue):
    try:
        logging.debug(f"Starting process {current_process().name}")
        if config.MODEL.NAME == 'detconfcmx':
            from models.cmx.builder_np_conf import myEncoderDecoder as confcmx
            model = confcmx(cfg=config)
        else:
            raise NotImplementedError('Model not implemented')

        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(device)
        

        # Open and preprocess image
        rgb = rgb.to(device)
        model.eval()
        det = None
        conf = None

        pred, conf, det, npp = model(rgb)

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

        out_dict = dict()
        out_dict['map'] = pred
        out_dict['imgsize'] = tuple(rgb.shape[2:])
        if det is not None:
            out_dict['score'] = det_sig
        if conf is not None:
            out_dict['conf'] = conf
        if save_np:
            out_dict['np++'] = npp

        from os import makedirs

        makedirs(os.path.dirname(filename_out), exist_ok=True)
        np.savez(filename_out, **out_dict)
        print(f'Finished processing file {path}')
    except Exception as e:
        print(f"Error processing file {path}: {e}")
        # logging.debug(f"Loading image {rgb_path}")
        # rgb = Image.open(rgb_path).convert("RGB")
        # rgb = np.array(rgb)
        # rgb = torch.tensor(rgb.transpose(2, 0, 1), dtype=torch.float) / 256.0
        # rgb = rgb.unsqueeze(0).to(device)

    #     logging.debug("Image preprocessing complete, running model")
    #     pred, conf, det, npp = model(rgb)

    #     if conf is not None:
    #         conf = torch.squeeze(conf, 0)
    #         conf = torch.sigmoid(conf)[0]
    #         conf = conf.cpu().numpy()

    #     if npp is not None:
    #         npp = torch.squeeze(npp, 0)[0]
    #         npp = npp.cpu().numpy()

    #     if det is not None:
    #         det_sig = torch.sigmoid(det).item()

    #     pred = torch.squeeze(pred, 0)
    #     pred = F.softmax(pred, dim=0)[1]
    #     pred = pred.cpu().numpy()

    #     out_dict = dict()
    #     out_dict['map'] = pred
    #     out_dict['imgsize'] = tuple(rgb.shape[2:])
    #     if det is not None:
    #         out_dict['score'] = det_sig
    #     if conf is not None:
    #         out_dict['conf'] = conf
    #     if save_np:
    #         out_dict['np++'] = npp

    #     result_queue.put(('result', out_dict))
    #     logging.debug(f"Process {current_process().name} finished")
    # except Exception as e:
    #     logging.error(f"Exception in process: {e}")
    #     result_queue.put(('exception', str(e)))

def process_with_timeout(rgb_path, model_state_dict, device, timeout=200):
    with Manager() as manager:
        result_queue = manager.Queue()
        process = Process(target=process_image, args=(rgb_path, model_state_dict, device, result_queue))
        process.start()
        process.join(timeout)
        
        if process.is_alive():
            process.terminate()  # Terminate the process if it takes too long
            process.join()       # Ensure the process has ended
            raise TimeoutException("Processing image timed out.")
        
        if not result_queue.empty():
            result_type, result_data = result_queue.get()
            if result_type == 'exception':
                raise Exception(result_data)
            return result_data
        else:
            raise RuntimeError("No result from processing image.")

# Argument parsing
parser = argparse.ArgumentParser(description='Test TruFor')
parser.add_argument('-gpu', '--gpu', type=int, default=0, help='device, use -1 for cpu')
parser.add_argument('-in', '--input', type=str, default='../images', help='can be a single file, a directory or a glob statement')
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

if '*' in input:
    list_img = glob(input, recursive=True)
    list_img = [img for img in list_img if not os.path.isdir(img)]
elif os.path.isfile(input):
    list_img = [input]
elif os.path.isdir(input):
    list_img = glob(os.path.join(input, '**/*'), recursive=True)
    list_img = [img for img in list_img if not os.path.isdir(img)]
else:
    raise ValueError("Input is neither a file nor a folder")

test_dataset = myDataset(list_img=list_img)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=1,num_workers=0)

if config.TEST.MODEL_FILE:
    model_state_file = config.TEST.MODEL_FILE
else:
    raise ValueError("Model file is not specified.")

logging.debug('=> loading model from {}'.format(model_state_file))
checkpoint = torch.load(model_state_file, map_location=torch.device(device))
# Load model state inside process
if config.MODEL.NAME == 'detconfcmx':
    from models.cmx.builder_np_conf import myEncoderDecoder as confcmx
    model = confcmx(cfg=config)
else:
    raise NotImplementedError('Model not implemented')

model.load_state_dict(checkpoint['state_dict'])
model = model.to(device)
# model.eval()
logging.debug('Model state loaded successfully')

print("Starting processing...")

with torch.no_grad():
    for index, (rgb, path) in enumerate(tqdm(testloader)):
        try:
            path = path[0]
            logging.debug(f'Processing image {path}...')
            if os.path.splitext(os.path.basename(output))[1] == '': 
                path = path[0]
                root = input.split('*')[0]
                if os.path.isfile(input):
                    sub_path = path.replace(os.path.dirname(root), '').strip()
                else:
                    sub_path = path.replace(root, '').strip()

                if sub_path.startswith('/'):
                    sub_path = sub_path[1:]

                filename_out = os.path.join(output, sub_path) + '.npz'
            else:  # output is a filename
                filename_out = output

            if not filename_out.endswith('.npz'):
                filename_out = filename_out + '.npz'

            if not os.path.isfile(filename_out):
                model_state_dict = checkpoint['state_dict']
                result = process_with_timeout(rgb, model_state_dict, device, timeout=200)  # Increased timeout to 300 seconds

                if result is None:
                    logging.warning(f"Failed to process image {path}.")
                    continue

                from os import makedirs
                makedirs(os.path.dirname(filename_out), exist_ok=True)
                np.savez(filename_out, **result)
        except (TimeoutException, FileNotFoundError, IOError) as e:
            logging.error(f"Error processing file {path}: {e}")
            continue
        except Exception as e:
            logging.error(f"Error processing file {path}: {e}")
            continue
