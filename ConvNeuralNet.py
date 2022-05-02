
from matplotlib.cbook import flatten
import numpy as np
from numpy.linalg import norm
from scipy import rand 
import matplotlib.pyplot as mpimg
from skimage import io, filters
from skimage import exposure
from skimage import color
from skimage.transform import resize 
from skimage.feature import hog  
import os
import pandas as pd
from sympy import Q 
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms 
from torch.utils.data import DataLoader, random_split, Dataset 
import csv
from PIL import Image
from natsort import natsorted 
""" ======================  Function definitions ========================== """
def charToSymbol(firstChar):
    switcher = {
        'a': 10, #addition
        's': 11, #subtraction
        'm': 12, #multiplication
        'd': 13, #division
        'r': 15  #random image
    }
    return switcher.get(firstChar, -1) #not a valid filename

def getImageType(file):
    firstChar = file[0]
    if not firstChar.isdigit():  # char isn't a digit (0-9)
        firstChar = charToSymbol(firstChar) #get value for corresponding math symbol
    return firstChar

class Data_set (Dataset):
    def __init__(self, csv_file, root_dir, transform):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir 
        self.transform = transform

    

    def __len__(self):
        return len(self.annotations)
   
    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = io.imread(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[index, 1]))
        if (self.transform):
            image = self.transform(image)
        return (image, y_label)

class CNN(nn.Module):
    def __init__(self, in_channels = 1, num_classes= 16):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels=8, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.pool = nn.MaxPool2d(kernel_size=(2,2),stride =(2,2))
        self.conv2 = nn.Conv2d(in_channels = 8,out_channels=16, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.fc1 = nn.Linear(90000, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0],-1)
        x = self.fc1(x)
        return x
def check_accuracy(loader, model):

    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x,y in loader:
            x = x.to(device=device)
            x = x.to(device=device)


            scores = model(x)
            _,predictions = scores.max(1)
            num_correct += (predictions ==y).sum()
            num_samples += predictions.size(0) 
        print (f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')

def model_loader():
    print("loading model")
    model = torch.load('./HWtest.pth')
    return model



#set up Vectors 

if __name__ == '__main__':
    
    x1_vector = []
    y_vector = []
    count = 0
    ext = ('.jpg')
    filenames =os.listdir("D:\code\project-team-leafs")
    filenames =natsorted(filenames)
    with open ("D:\code\project-team-leafs\Book1.csv",'w', newline='', encoding='UTF8') as f:
        for files in filenames:
            if files.endswith(ext):
            #load data
                im = mpimg.imread(files) 
                writer = csv.writer(f)
                data = [files, 0]
                writer.writerow(data)
                #grayscale, normalize, and feature_extractor (Histogram of oriented Gradients) 
                resized_im = color.rgb2gray(resize(im,(300,300)))
                bw_im = resized_im < (filters.threshold_yen(resized_im))
                bw_im = bw_im.astype(np.float32) 
                new_file = str(files)
                image = Image.fromarray((bw_im * 255).astype(np.uint8))
                image.save('D:/code/project-team-leafs/pictures/'+ new_file)
                bw_im = bw_im.flatten()
                #fd = hog(bw_im, orientations = 9, pixels_per_cell=(4,4),cells_per_block=(2,2),block_norm='L1')
                x1_vector = np.append(x1_vector,bw_im, axis=0)
                count = count+1
            else:
                continue
    f.close()
mean = np.mean(x1_vector)
std = np.std(x1_vector) 
#x_train, x_test, y_train, y_test = train_test_split(x, y_vector ,test_size=.20, random_state = 36)

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("running on the GPU")
else:
    device = torch.device("cpu")
    print("running on the CPU")

in_channels = 1
num_classes = 16
num_epochs = 10
batch_size = 4 
learning_rate = 0.001
load_model = True

model = CNN().to(device)

if (load_model):
    dataset = Data_set(csv_file = 'Book1.csv', root_dir = 'D:\code\project-team-leafs\pictures', transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean,std)]))
    test_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle = True)
    model = model_loader()
    check_accuracy(test_loader, model)

else :
    dataset = Data_set(csv_file = 'Book1.csv', root_dir = 'D:\code\project-team-leafs\pictures', transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean,std)]))
    train_set, test_set = random_split(dataset,[300,29])
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle = True)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle = True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range (num_epochs):
        for batch_idx, (data, targets) in enumerate(train_loader):
            data = data.to(device=device)
            targets = targets.to(device=device)

            scores = model(data)
            loss = criterion(scores, targets)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

    PATH = './HWtest.pth'
    torch.save(model,PATH)
    print('done')



    model.train()
    check_accuracy(train_loader,model)
    check_accuracy(test_loader, model)

print("done")