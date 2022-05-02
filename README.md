[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-f059dc9a6f8d3a56e377f745f24479a46679e63a5d9fe6f495e02850cd0d8118.svg)](https://classroom.github.com/online_ide?assignment_repo_id=6719062&assignment_repo_type=AssignmentRepo)

The Repostory has several python files only train and test are nesscary all other files where used for experimentation

Train.py : to train model simply place all files in Repostory or project-team-leafs file path, then run code. It will use a random 5% for testing 
it will then create a final_model.sav file that be saved to file folder (file may be over 700 MB)

Test.py : To test just run code with final_model.sav file in file path or repository, an array of y labels is printed

ConvNeuralNet.py: runs a convolutional neural network on pictures stored in same folder, produces filtered pictures in pictures folder
prints training accuarcy and actual accuracy  

SupportVectorClassifier.py: is train.py and test.py combined controlled by load variable, if true its testing, if false its training

The Repostory also constains:
Book1.csv :  a csv file that stores the label values and title of the pictures in columns, label values are stored by first character in title string 
('final_model.sav') : https://drive.google.com/drive/folders/1fTtrSbZUf8MkC52WjJSXfNsk36VQzcFq?usp=sharing (<= google drive like to saved model)
My final model saved to google drive for testing, just put it into final-project folder then run test.py
HWtest.pth: saved version of Convolutional neural net model
pictures : simple empty file that stores pictured created by ConvNeuralNet File if ran. 