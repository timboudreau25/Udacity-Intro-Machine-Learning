####################
## Tim Boudreau
## September 2019
## Udacity Image Classifier Project
## predict
####################


##### import libraries #####

from torchvision import datasets, transforms, models
import torch
from torch import nn
from torch import optim
import json
from collections import OrderedDict
from PIL import Image
import numpy as np
import os
import argparse
import random
import os


##### define functions #####

def arg_parser():
    ## parser
    parser = argparse.ArgumentParser(description="Neural Network Settings")
    
    ## checkpoint selection
    parser.add_argument('--checkpoint', 
                        type=str, 
                        help='Directory of the checkpoint to use')
    
    ## specific image directory selection
    parser.add_argument('--img_dir', 
                        type=str, 
                        help='Directory of a specific image to classify')
    
    ## top k image classification options
    parser.add_argument('--topk', 
                        type=str, 
                        help='Top k classifications reported',
                        default = 5)
    
    ## custom class map json
    parser.add_argument('--class_map', 
                        action="store_true", 
                        help='directory to custom class map',
                        default = 'cat_to_name.json')
    
    ## gpu option
    parser.add_argument('--gpu', 
                        action="store_true", 
                        help='Use GPU + Cuda for calculations, appearance means true')
    
    ## parse arguments
    args = parser.parse_args()
    return args


def load_checkpoint(file_dir):
    '''
    Load my nn model checkpoint.
    '''
    ## get checkpoint file
    checkpoint = torch.load(file_dir)
    
    ## build model
    mod = checkpoint['model']
    if type(mod) == type(None): 
        mod = 'vgg16'
    model = eval("models.{}(pretrained=True)".format(mod))
    model.name = mod
    
    ## freeze pre-loaded params
    for param in model.parameters():
        param.requires_grad = False
        
    ## load checkpoint information
    model.classifier = checkpoint['classifier']
    model.optimizer = checkpoint['optimizer']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
        
    return model


def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    
    ## open image
    img = Image.open(image_path)
    
    
    ## modify dimensions but keep aspect ratio
    width, height = img.size
    resize = 256
        
    ## resize image so shorter size is 256 pixels, keep aspect ratio too
    if height > width:
        height = int(height * (224/width))
        width = resize
    else:
        width = int(width * (224/height))
        height = resize
    
    img_new_dims = img.resize((width, height))
    
    
    ## crop center of image
    crop = 224
    
    x_dims = [(width - crop / 2), crop + (width - crop / 2)]
    y_dims = [(height - crop / 2), crop + (height - crop / 2)]

    cropped_img = img.crop((x_dims[0], y_dims[0], x_dims[1], y_dims[1]))
    
    
    ## encode image, then divide by max color to put into 0-1 range
    np_image = np.array(cropped_img) / 255.0
    
    
    ## normalize images
    mean_array = np.array([.485, .456, .406])
    sd_array = np.array([.229, .224, .225])
    np_img_array = (np_image - mean_array) / sd_array
    
    ## reoder features so color is first
    np_img_array = np_img_array.transpose((2, 0, 1))
    
    
    return np_img_array


def checkpoint_selection(chk):
    '''
    Select checkpoint
    '''
    ## Build and train your network
    if type(chk) == type(None): 
        chk = 'nn_checkpoint.pth'
        
    return chk


def get_device(gpu):
    '''
    Get device to run on, if possible
    '''
    if (gpu) & (torch.cuda.is_available()):
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
#     device = torch.device("cpu")
    print('Device is ', device)
    
    return device


def get_flower_names(classes, class_map_url):
    '''
    Get flower name map from json
    '''
    with open(class_map_url, 'r') as f:
        cat_to_name = json.load(f)
    
    names = []
    for i in classes:
        names.append(cat_to_name[i])
        
    return names


def flower_selection(flower):
    '''
    Select flower
    '''
    ## Build and train your network
    if type(flower) == type(None): 
        flowers = os.listdir('flowers/test')
        random_flower = random.choice(flowers)
        flower_image = random.choice(os.listdir('flowers/test/' + random_flower))
        flower = 'flowers/test/' + random_flower + '/' + flower_image
        
    return flower


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax


def predict(image_path, model, topk, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''    
    ## start the model and decide the method of use
    model.eval()
        
    ## open image from path and ocnvert to tensor
    image_array = process_image(image_path)
    tensor = torch.from_numpy(image_array).type(torch.FloatTensor).unsqueeze_(0)
    
    ## model either cpu or gpu
    model.to(device)
    
    ## run image through model
    with torch.no_grad():
        output = model.forward(tensor.to(device))
        
    ## get probabilities from log output (from logmax)
    probs = torch.exp(output)
    probs_topk = np.array(probs.topk(topk)[0])[0]
    idx_topk = np.array(probs.topk(topk)[1])[0]
    
    ## map index to classes - invert class to idx
    class_to_idx = model.class_to_idx
    index_to_class = {val: key for key, val in class_to_idx.items()}
    
    ## map onto image
    classes = []
    
    for idx in idx_topk:
        classes += [index_to_class[idx]]
        
    return probs_topk, classes



##### load model checkpoint and arguments #####

## retrieve arguments
args = arg_parser()

## run function to get model
model = load_checkpoint(checkpoint_selection(args.checkpoint))


##### Predicting the flower class #####

image_path = flower_selection(args.img_dir)

probs,classes = predict(image_path, model, int(args.topk),  get_device(args.gpu))

names = get_flower_names(classes, args.class_map)


##### print flowers and probabilities #####

print('\n\nFlower predicted is:', image_path, '\n')
for flower, prob in zip(names, probs):
    print(flower, prob)