'''
File    :   production.py
Time    :   2022/01/15 14:01:11
Author  :   Michiel Jacobs 
Version :   1.0
Contact :   michiel.jacobs@vub.be
License :   (C)Copyright 2022, Michiel Jacobs
'''

import os
import sys
import logging
import torch
import numpy as np
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from dataclasses import dataclass
from datetime import datetime
from models import LeNet5
from matplotlib import pyplot as plt

@dataclass
class settings:
    
    name = 'LeNet5'

    # General
    epochs: int = 5
    batch_size: int = 32

    # Model
    model = LeNet5(47)

    # Loss
    loss_function = torch.nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.1 , momentum=0.5)
    #optimizer = optim.Adam(model.parameters(), lr=0.001)
    #optimizer = optim.Adamax(model.parameters(), lr=0.002)
    #optimizer = optim.RMSprop(model.parameters(), lr=0.01)

    # Construct a labelmap for the EMNIST balanced set,
    # so I have a reference of each class label to a number or letter based from the paper
    label_map = {
        # not PEP8 formatted, but readable
        0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
        10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J',
        20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T',
        30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z',
        36: 'a', 37: 'b', 38: 'd', 39: 'e',
        40: 'f', 41: 'g', 42: 'h', 43: 'n', 44: 'q', 45: 'r', 46: 't',
    }

def get_loader(transformer, train):

    loader = DataLoader(
        datasets.EMNIST('../data/',
                        split='balanced',
                        train=train,
                        download=True,
                        transform=transformer),
        batch_size=settings.batch_size,
        shuffle=True)
    
    return loader

def plotLoss(train, test):
    plt.figure(figsize=(20,5))

    plt.plot(train, label='Train loss')
    plt.plot(test, label='Val loss')

    plt.title('Loss variation i.f.o. the epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.savefig(os.path.join(folder, 'loss.jpg'))
    plt.close()

def plotAccuracy(accuracy):
    plt.figure(figsize=(20,5))

    plt.plot(accuracy, label='Accuracy')

    plt.title('Accuracy i.f.o. the epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.savefig(os.path.join(folder, 'accuracy.jpg'))
    plt.close()

def plotPerClassAccuracy(accuracies):
    labels = list(settings.label_map.values())

    plt.figure(figsize=(20,5))
    plt.bar(labels, accuracies)
    plt.title('Accuracies')
    plt.xlabel('Alphanumeric ')
    plt.ylabel('Accuracy (%)')

    plt.savefig(os.path.join(folder, 'per_class_accuracy.jpg'))
    plt.close()

if __name__ == '__main__':

    # get a directory for this model
    PATH = os.path.dirname(os.path.abspath(__file__))
    runs = os.path.join(PATH, 'runs')
    if not os.path.exists(runs):
        os.mkdir(runs)
    
    now = datetime.now()

    modelname = f'{settings.name} on {now.strftime("%m-%d-%Y %H.%M.%S")}'
    modelname = modelname.replace(' ', '_')

    folder = os.path.join(runs, modelname)
    if not os.path.exists(folder):
        os.mkdir(folder)

    LOGPATH = os.path.join(folder, f'{modelname}.log')
    MODELPATH = os.path.join(folder, f'{modelname}')
    REPORTPATH = os.path.join(folder, f'{modelname}.pdf')

    # Setup logging for this file
    log = logging.getLogger()
    log.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.DEBUG)
    sh.setFormatter(formatter)

    fh = logging.FileHandler(LOGPATH)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    # add the log to the handler
    log.addHandler(sh)
    log.addHandler(fh)

    # First check if the GPU is available
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda:0')
        logging.info('Training on GPU!')
    else:
        DEVICE = torch.device('cpu')
        logging.warning('No GPU available, working on CPU!')

    # Downloads by torchvision, first, prepare a transformer
    logging.info('Building transformer')
    transformer = transforms.Compose([

        # Fix the import rotation and flip all images to readable format
        lambda img: transforms.functional.rotate(img, -90),
        lambda img: transforms.functional.hflip(img),
        
        # Pad the images to a larger size to get the maximum from the highest level feature detectors conf. LeCun
        transforms.Pad(2),
        
        transforms.ToTensor(),])
    
    logging.info('Building dataloaders')
    train_loader = get_loader(transformer, True)
    test_loader = get_loader(transformer, False)

    logging.info('Sending model to device')
    settings.model.to(DEVICE)

    # a list for the metrics
    epoch_train_losses = []
    epoch_val_losses = []
    epoch_val_accuracies = []

    # training-loop
    logging.info('Starting training')
    for epoch in range(settings.epochs):
        train_losses = []
        for batch in train_loader:
            # first, retrieve our data for this batch and send it to the GPU or CPU
            images = batch[0].to(DEVICE)
            labels = batch[1].to(DEVICE)
            
            # set the parameter gradients to zero
            settings.optimizer.zero_grad()
            
            # Make predictions based on the training set
            outputs = settings.model(images)
            
            # compute the loss based on model output and real labels
            loss = settings.loss_function(outputs, labels)
            
            # backpropagation
            loss.backward()
            
            # Optimize parameters
            settings.optimizer.step()

            # add loss to the list
            train_losses.append(loss.item())
            
        # === loss stats every epoch ===
        # put network in eval mode
        # switch to eval mode
        settings.model.eval()

        val_losses = []
        n_correct = 0
        n_total = 0
        # prepare to count predictions for each class
        np_classes_correct = { settings.label_map[key]: 0 for key in settings.label_map }
        np_classes_total = { settings.label_map[key]: 0 for key in settings.label_map }
        # I do all the calculations each epoch as it is unclear which one will be the last one.

        with torch.no_grad():
            for val_data in test_loader:
                # first, retrieve our data for this batch and send it to the GPU or CPU
                images = val_data[0].to(DEVICE)
                labels = val_data[1].to(DEVICE)

                # Make predictions based on the training set
                output = settings.model(images)

                # compute the loss based on model output and real labels
                val_loss = settings.loss_function(output, labels)

                # add loss to the list
                val_losses.append(val_loss.item())

                # Get the prediction which is most likely our target
                values, predictions = torch.max(output.data, 1)

                # Calculate the total ammount of labels passed
                n_total += labels.size(0)

                # Calculate the ammount of predictions that were correct
                for label, prediction in zip(labels, predictions):
                    if label == prediction:
                        n_correct += 1
                        np_classes_correct[settings.label_map[int(label)]] += 1
                    np_classes_total[settings.label_map[int(label)]] += 1

        # now calculate the accuracy
        val_accuracy = (n_correct / n_total) * 100

        # Calculate the per class accuracy
        per_class_accuracy = [(float(value) / np_classes_total[key]) * 100 for key, value in np_classes_correct.items()]

        # switch back to training
        settings.model.train()

        # Calculate the epoch losses
        epoch_train_losses.append(np.mean(train_losses))
        epoch_val_losses.append(np.mean(val_losses))

        # Add the epoch accuracies to a list
        epoch_val_accuracies.append(val_accuracy)

        # Print information on the training each epoch, easier to understand for me comming from tensorflow.
        logging.info(
            f'Epoch: {epoch+1}/{settings.epochs} Loss: {epoch_train_losses[-1]:.3f} Val_loss: {epoch_val_losses[-1]:.3f} Val_acc: {epoch_val_accuracies[-1]:.2f} %')

        # Plotting every epoch allows and easy way to check how its going without the help of tensorboard or alike
        plotLoss(epoch_train_losses, epoch_val_losses)
        plotAccuracy(epoch_val_accuracies)
        plotPerClassAccuracy(per_class_accuracy)

    logging.info('Training complete')

    logging.info('Saving model for inference')
    torch.save(settings.model.state_dict(), os.path.join(MODELPATH, f'{modelname}_inference.pt'))

    logging.info('Saving model for further training')
    torch.save(settings.model, os.path.join(MODELPATH, f'{modelname}.pt'))

    logging.info('Normal termination')