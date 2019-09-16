####################
## Tim Boudreau
## September 2019
## Udacity Image Classifier Project
## train
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
import argparse



##### define functions #####

def arg_parser():
    ## parser
    parser = argparse.ArgumentParser(description="Neural Network Settings")
   
    ## select model
    parser.add_argument('--model', 
                        type=str, 
                        help='Choose vgg16 or alexnet from torchvision.models')
    
    ## select directory to save checkpoint
    parser.add_argument('--checkpoint_dir',
                        type=str,
                        help='DIrectory to save checkpoint',
                        default = '')
    
    ## select directory to read flowers from
    parser.add_argument('--flower_dir',
                        type=str,
                        help='Flower directory',
                        default='flowers')
    
    ## hyperparameters
    parser.add_argument('--learning_rate', 
                        type=float, 
                        help='Learning rate as float',
                        default = .001)
    parser.add_argument('--hidden_layers', 
                        type=int, 
                        help='Hidden layers as int',
                        default = 512)
    parser.add_argument('--epochs', 
                        type=int, 
                        help='Epochs for training as int',
                        default = 3)

    ## gpu option
    parser.add_argument('--gpu', 
                        action="store_true", 
                        help='Use GPU + Cuda for calculations, appearance means true')
    
    ## parse arguments
    args = parser.parse_args()
    return args


def model_selection(mod):
    '''
    Select model
    '''
    ## Build and train your network
    if type(mod) == type(None): 
        mod = 'vgg16'
    elif mod not in ('vgg16', 'alexnet'):
        print('Model is not accepted. Please use vgg16 or alexnet.')
        exit()
    model = eval("models.{}(pretrained=True)".format(mod))
    model.name = mod

    print('Model is ', mod)        
        
    ## freeze pre-loaded params
    for param in model.parameters():
        param.requires_grad = False
        
    return model


def get_features(model):
    '''
    Get number of features for both model types
    '''
    if model.name == 'vgg16':
        return model.classifier[0].in_features
    else:
        return model.classifier[1].in_features


def get_device(gpu):
    '''
    Get device to run on, if possible
    '''
    if (gpu) & (torch.cuda.is_available()):
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print('Device is ', device)
    
    return device


def validate(model, validateloader, criterion, device):
    # set values to zero
    validate_loss = 0
    accuracy = 0
    
    for images, labels in validateloader:
        images = images.to(device)
        labels = labels.to(device)
        
        log_ps = model(images)
        validate_loss += criterion(log_ps, labels)

        ps = torch.exp(log_ps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor))
        
    return validate_loss, accuracy



##### import data and define transforms, etc. #####

## get args 
args = arg_parser()

## define directories
data_dir = arg.flower_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

## Define your transforms for the training, validation, and testing sets
data_transforms = {
    'train': transforms.Compose([transforms.RandomRotation(30),
                                 transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize([.485, .456, .406],
                                                      [.229, .224, .225])]),
    'validate&test': transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([.485, .456, .406],
                                                              [.229, .224, .225])])
}

## Load the datasets with ImageFolder
image_datasets = {
    'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
    'validate': datasets.ImageFolder(valid_dir, transform=data_transforms['validate&test']),
    'test': datasets.ImageFolder(test_dir, transform=data_transforms['validate&test'])
}

## Using the image datasets and the trainforms, define the dataloaders
dataloaders = {
    'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
    'validate': torch.utils.data.DataLoader(image_datasets['validate'], batch_size=32),
    'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=32)
}

## Load mapping dictionary
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

    
    
##### Build and train model #####

device = get_device(args.gpu)
model = model_selection(args.model)

## variables
classifier_features_num = get_features(model)
hidden_layers = args.hidden_layers
flower_outputs = 102
learn_rate = args.learning_rate
epochs = args.epochs
print('Hidden layers: ', hidden_layers, '\nLearning rate: ', learn_rate, '\nepochs: ', epochs)


## classifier
classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(classifier_features_num, hidden_layers)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(hidden_layers, flower_outputs)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
model.classifier = classifier
    
## criterion and optimizer, with frozen parameters
criterion = nn.NLLLoss ()
optimizer = optim.Adam (model.classifier.parameters (), lr = learn_rate)



##### train and validate model #####

## print every 50 images
image_print = 50
steps = 0

## fetch loaders from above
trainloader = dataloaders['train']
validateloader = dataloaders['validate']

## assign device type to model - cuda or cpu
model.to(device)

## for each epoch, train and validate
for e in range(epochs):
    running_loss = 0
    
    ## for each image and label in training set
    for images, labels in trainloader:

        ## count the step/image we're on
        steps += 1
        
        ## depending on cpu or gpu, define variables
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        
        log_ps = model.forward(images)
        loss = criterion(log_ps, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        ## validate
        if steps % image_print == 0:

            # Turn off gradients for validation, saves memory and computations
            with torch.no_grad():
                model.eval()
                validate_loss, accuracy = validate(model, validateloader, criterion, device)

            model.train()

            ## print stats on training/validation
            print("Epoch: {}/{}.. ".format(e+1, epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss/len(trainloader)),
                  "Validation Loss: {:.3f}.. ".format(validate_loss/len(validateloader)),
                  "Validation Accuracy: {:.3f}".format(accuracy/len(validateloader)))



##### save model #####

## Save the checkpoint 
checkpoint = {'state_dict': model.state_dict(),
              'class_to_idx': image_datasets['train'].class_to_idx,
              'classifier': model.classifier,
              'model': model.name,
              'optimizer': optimizer.state_dict(),
              'epochs': epochs,
              'hidden_layers': hidden_layers}

torch.save(checkpoint, args.checkpoint_dir + '/nn_checkpoint.pth')