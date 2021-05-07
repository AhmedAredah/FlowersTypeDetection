"""
@ Author: Ahmed Aredah
@ Title: Flow name detector -- Predict File

"""



import argparse
from torchvision import datasets, transforms, models
import torch
from torch import nn
from torch import optim
from PIL import Image
import json


def arg_parser():
    parser = argparse.ArgumentParser(description="Neural Network Settings")
    
    parser.add_argument('--image', 
                        type=str, 
                        help='Point to impage file for prediction.',
                        required=True)
    
    parser.add_argument('--checkpoint', 
                        type=str, 
                        help='Point to checkpoint file as str.',
                        required=True)
    
    parser.add_argument('--category_names', 
                        type=str, 
                        help='Mapping from categories to real names.',
                        required=True)
    
    parser.add_argument('--top_k', 
                        type = int,
                        help='Choose top K matches as int.')

   
    # Parse args
    args = parser.parse_args()
    return args

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns a tensor
    '''
   
    # Open an image
    pil_image = Image.open(image)
    
    # Define transformations
    image_transforms = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=(0.485, 0.456, 0.406), 
                                                          std=(0.229, 0.224, 0.225))])
    
    pil_image = image_transforms(pil_image)
    
    return pil_image


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

def predict(img , model, cat_to_name, topk = 5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    #= "flowers/test/1/image_06743.jpg"
    # Turn on evaluation Mode
    model.eval()
    
    # Work on the CPU rather than the GPU for better references
    model.cpu()
    
    
    # Load an image
    img.unsqueeze_(0)
        
    # Turn gradient off to speed up the process
    with torch.no_grad():
        # Get prediction of an image
        logps = model.forward(img)
        
        # Get the highest propabilities and classes
        top_ps, top_class = torch.topk(logps, topk) 
#         top_ps, top_class = logps.topk(topk)
        
        # get the linear propabilities of the image
        top_linear_ps = torch.exp(top_ps)
        
        # get the mapping classes
        class_to_idx_rev = {model.class_to_idx[k]: k for k in model.class_to_idx}
        
        classes = []
        # Get labels associated with these probabilities
        for Ic in top_class.numpy()[0]:
            classes.append(class_to_idx_rev[Ic])
            
        names = [cat_to_name[img_class] for img_class in classes]
            
        return top_linear_ps.numpy()[0], classes, names
    
    


def load_checkpoint(filepath= 'checkpoint.pth'):
    checkpoint = torch.load(filepath)
    
    # Define the model arch
    model = models.__dict__[checkpoint['architecture']](pretrained=True)
    
    # Freeze parameters so we don't backprop through them
    for param in model.parameters(): param.requires_grad = False
    
    # Load mode.classifier
    model.classifier = checkpoint['classifier']
    # Load State Dict
    model.load_state_dict(checkpoint['state_dict'])
    # Load mapping of classes to indicies
    model.class_to_idx = checkpoint['class_to_idx']
    
#     # Define Optimizer
#     optimizer = optim.Adam(model.classifier.parameters(), lr = 0.01)
#     #Load Optimizer Data
#     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    

    #Load other data
    epoch = checkpoint['epoch']
    
    return model



def print_class(probs, flowers):
    
    for i, j in enumerate(zip(flowers, probs)):
        print ("Rank {}:".format(i+1),
               "Class: {}, liklihood: {:.2f}%".format(j[1], j[0]*100))
        

def Main():
    args = arg_parser()
    
    # Load categories to names json file
    with open(args.category_names, 'r') as f:
        	cat_to_name = json.load(f)
            
#     cp = ("/home/workspace/saved_models/checkpoint.pth" if type(args.checkpoint) == type(None) else args.checkpoint)
    model = load_checkpoint(args.checkpoint)
    
    img = process_image(args.image)
    
    top_k = (5 if type(args.top_k) == type(None) else args.top_k)
    
    props,clss, nams = predict(img, model, cat_to_name, top_k)
    
    print_class(nams, props)
    
    
if __name__ == '__main__': Main()