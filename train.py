"""
@ Author: Ahmed Aredah
@ Title: Flow name detector -- Training File

"""



import argparse
from torchvision import datasets, transforms, models
import torch
from torch import nn
from torch import optim
from collections import OrderedDict



    
    
def arg_parser():
    parser = argparse.ArgumentParser(description="Neural Network Settings")
    
    parser.add_argument('--arch', 
                        type=str, 
                        default='vgg11',
                        help='Choose architecture from torchvision.models as str')
    
    parser.add_argument('--save_dir', 
                        type=str, 
                        help='Define save directory for checkpoints as str. If not specified then model will be lost.')
    
    parser.add_argument('--learning_rate', 
                        type=float, 
                        help='Define gradient descent learning rate as float')

    parser.add_argument('--epochs', 
                        type=int, 
                        help='Number of epochs for training as int')

    # Add GPU Option to parser
    parser.add_argument('--gpu', 
                        action="store_true", 
                        default = True,
                        help='Use GPU + Cuda for calculations... Make sure you have Nvidia Graphics Card and developers drivers are installed!')
    
    # Parse args
    args = parser.parse_args()
    return args

def Check_GPU(use_gpu = True):
    if use_gpu == True:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            print("GPU is now being used!")
            return device
        else:
            device = torch.device("cpu")
            print("There is no GPU device recognized! ... CPU is now being used!")
            return device
    else:
        device = torch.device("cpu")
        print("CPU is now being used!")
        return device
        
        
        
    
    
    
    
def transform(folder_path, train = True):
    """
    Perform transformations on a dataset
    Arguments:
        folder_path: Path to the dataset
        train:  True: the transformer is applied to the train dataset
                False: the transformer is applied to the test/validation dataset
    """
   # Define transformation
    if train:
        train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])
        train_data = datasets.ImageFolder(folder_path, transform=train_transforms)
        return train_data
    
    else:
        test_transforms = transforms.Compose([transforms.Resize(256),
                                              transforms.CenterCrop(224),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406], 
                                                                   [0.229, 0.224, 0.225])])
        test_data = datasets.ImageFolder(folder_path, transform=test_transforms)
        return test_data
    
    
def load_dataset(the_dataset):
    theloader = torch.utils.data.DataLoader(the_dataset, batch_size=32, shuffle=True)
    return theloader

def load_model(architecture="vgg11"):
    """
    Load a pretrained model.
    Params:
        architecture: (string) torch model from models module
    """
    # Load Defaults if none specified
    model = models.__dict__[architecture](pretrained=True)
    
    return model


def initiate_classifier(model):
    
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(25088, 4096)),
                              ('relu', nn.ReLU()),
                              ('fc2', nn.Linear(4096, 102)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))

    return classifier
    

def train(model, trainloader, validloader, device, epochs = 2, lr = 0.01):
    optimizer = optim.Adam(model.classifier.parameters(), lr = lr)
    criterion = nn.NLLLoss()

    train_loss = 0
    training_losses, validating_losses = [], []
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.to(device)

    print("\nTraining is in Progress ....\n")

    for e in range(epochs):
        for image, label in trainloader:

             # Move input and label tensors to the GPU
            image, label = image.to(device), label.to(device)

            optimizer.zero_grad()

            output = model(image)

            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        else:
            valid_loss = 0
            accuracy = 0

            with torch.no_grad():
                model.eval()

                for image, label in validloader:

                    # Move input and label tensors to the GPU
                    image, label = image.to(device), label.to(device)

                    voutput = model(image)
                    valid_loss += criterion(voutput, label).item()

                    ps = torch.exp(voutput)
                    top_k, top_class = ps.topk(1, dim = 1)

                    is_equals = top_class == label.view(*top_class.shape)
                    accuracy += torch.mean(is_equals.type(torch.float))

            model.train()

            training_losses.append(train_loss/len(trainloader))
            validating_losses.append(valid_loss/len(validloader))

            print(f'Epoch: {e+1}/{epochs} | ',
                  f'Training Loss: {round(training_losses[-1],4)} | ',
                  f'Validating Loss: {round(validating_losses[-1],4)} | ',
                  'Accuracy:  {:.4f}%'.format(accuracy/len(validloader) * 100))

    print("Training Complete!\n")
    return model, optimizer, loss
    
    
def test_model(model, testloader, device):
    accuracy = 0
    for param in model.parameters():
        param.requires_grad = False 
        
    with torch.no_grad():
        model.eval()

        for image, label in testloader:

            # Move input and label tensors to the GPU
            image, label = image.to(device), label.to(device)

            toutput = model(image)

            ps = torch.exp(toutput)
            top_k, top_class = ps.topk(1, dim = 1)

            is_equals = top_class == label.view(*top_class.shape)
            accuracy += torch.mean(is_equals.type(torch.float))

    print(f'Accuracy:  {accuracy/len(testloader) * 100}%')
    
    
def save_model(model, optimizer, train_datasets, epochs, model_name, file_path = '/home/workspace/saved_models/checkpoint.pth'):
    model.class_to_idx = train_datasets.class_to_idx

    checkpoint = {'architecture': model_name,
                  'classifier': model.classifier,
                  'class_to_idx': model.class_to_idx,
                  'optimizer_state_dict': optimizer.state_dict(),
                  'state_dict': model.state_dict(),
                  'epoch': epochs}

    torch.save(checkpoint, file_path)
    print("Model Saved successfully!")
    
def Main():
    args = arg_parser()
    
    # Set directory for training
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    train_data = transform(train_dir, True)
    valid_data = transform(valid_dir, False)
    test_data = transform(test_dir, False)
    
    trainloader = load_dataset(train_data)
    validloader = load_dataset(valid_data)
    testloader = load_dataset(test_data)
    
    model = load_model(architecture=args.arch)
    
    model.classifier = initiate_classifier(model)
    
    lr = (0.01 if type(args.learning_rate) == type(None) else args.learning_rate)
    epochs = (2 if type(args.epochs) == type(None) else args.epochs)
    device = Check_GPU(args.gpu)
    
    trained_model = train(model, trainloader, validloader, device, epochs, lr = lr)
    
    test_model(trained_model[0], testloader, device)
    
    cp = ("/home/workspace/saved_models/checkpoint.pth" if type(args.save_dir) == type(None) else args.save_dir)
    save_model(trained_model[0], trained_model[1], train_data, epochs, args.arch , cp)
    
    
if __name__ == '__main__': Main()