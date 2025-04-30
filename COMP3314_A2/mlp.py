from __future__ import print_function, division

import torch
import torch.optim as optim

from torch.optim import lr_scheduler
from torchvision import datasets, transforms
import time
import os

import torch.nn as nn

class Net(nn.Module):
    """
    Input - 1x32x32 (Gray image)
    Output - 10 (Number for classification)
    """
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),  # flatten the image
            nn.Linear(32 * 32 * 1, 512),  # fully connected layers
            nn.ReLU(),
            nn.Dropout(0.5),          # dropout for overfitting
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 10)
        )
        
    def forward(self, xb):
        return self.network(xb)

# you can try different types of data augementation to increase the performance on test data.
data_transforms = {
    'train': transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  
        transforms.RandomAffine(degrees=10, translate=(0,0.1)),
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # normalization
    ]),
    'test': transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]),
}

def train_test(model, criterion, optimizer, scheduler, num_epochs=25):
    train_loss = []
    train_accuracy = []
    val_loss = [] 
    val_accuracy = []
    history = dict()
    model.train()
    for epoch in range(num_epochs):
        running_training_loss = 0.0
        running_training_accuracy = 0.0
        iteration_training_loss = 0.0
        total_training_predictions = 0
       
        start_time = time.time()
        for i, data in enumerate(train_dataloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_training_loss += loss.item()*inputs.size(0)
            
            _, predicted = torch.max(outputs.data, 1)
            total_training_predictions += labels.size(0)
            running_training_accuracy += (predicted == labels).sum().item()
            iteration_training_loss += loss.item() 
            if (i+1) % 100 == 0:
                print('Epoch:[%d]-Iteration:[%d], training loss: %.3f' %
                      (epoch + 1,i+1,iteration_training_loss/(i+1)))
        end_time = time.time()
        print('Time cost of one epoch: [%d]s' % (end_time-start_time))
        
        epoch_training_accuracy = running_training_accuracy / train_size*100
        epoch_training_loss = running_training_loss / train_size
        
        print('Epoch:[%d], training accuracy: %.1f, training loss: %.3f' %
              (epoch + 1,epoch_training_accuracy, epoch_training_loss))
        
        train_loss.append(epoch_training_loss)
        train_accuracy.append(epoch_training_accuracy)
        
        scheduler.step()
        
    print('Finished Training')

    history['train_loss'] = train_loss
    history['train_accuracy'] = train_accuracy

    correct = 0
    total = 0
    model.eval()
    # Since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in test_dataloader:
            images, labels = data[0].to(device), data[1].to(device)
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print('Accuracy of the network on test images: %d %%' % (
            accuracy))
    return history, accuracy

if __name__ == '__main__':

    # change the data-path, recommand for relative path
    data_dir = './data'  # change with the true parh
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                            data_transforms[x])
                    for x in ['train', 'test']}

    data_dir = './data' # Suppose the dataset is stored under this folder
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                            data_transforms[x])
                    for x in ['train', 'test']} # Read train and test sets, respectively.

    train_dataloader = torch.utils.data.DataLoader(image_datasets['train'], batch_size=128,
                                                shuffle=True, num_workers=4)

    test_dataloader = torch.utils.data.DataLoader(image_datasets['test'], batch_size=128,
                                                shuffle=False, num_workers=4)

    train_size =len(image_datasets['train'])


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # Set device to "cpu" if you have no gpu

    end = time.time()
    model_ft = Net().to(device)
    print(model_ft.network)
    criterion = nn.CrossEntropyLoss()

    # paramters for optimizer
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=1e-3)  

    # learning rate scheduler
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=15, gamma=0.9)
    
    # learning epoch
    history, accuracy = train_test(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
               num_epochs=50)
    
    print("time required %.2fs" %(time.time() - end))
    model_ft.eval()   # switch to evaluation mode

    num_classes = len(image_datasets['test'].classes)
    class_correct = [0] * num_classes
    class_total   = [0] * num_classes

    with torch.no_grad():
        for inputs, labels in test_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model_ft(inputs)
            _, preds = torch.max(outputs, 1)

        # accumulate stats
        for label, pred in zip(labels, preds):
            class_total[label.item()] += 1
            class_correct[label.item()] += int(pred == label)

# print per-class results
print("\nPer-class accuracy:")
for i, class_name in enumerate(image_datasets['test'].classes):
    accuracy_i = 100.0 * class_correct[i] / class_total[i]
    print(f"  Class {class_name:>2s}: {accuracy_i:5.2f}%  ({class_correct[i]}/{class_total[i]})")
