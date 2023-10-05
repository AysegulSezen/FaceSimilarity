#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 17:48:26 2023

@author: aysegulsezen
It was converted from Dr. Andrew Ng (coursera) homework code (Handwritten digit recognition)
It guess face similarity. 
"""

from PIL import Image
import glob
import numpy as np
import math
import skimage.io as io 
import os
import scipy.optimize as op
import matplotlib.pyplot as plt

def  displayData(X, example_width):

    example_width = round( math.sqrt(X.shape[1]));

    m= X.shape[0]
    n =X.shape[1]
    example_height = int( (n / example_width));

    # Compute number of items to display
    display_rows = math.floor(math.sqrt(m));
    display_cols = math.ceil(m / display_rows);

    # Between images padding
    pad = 1;

    # Setup blank display
    w1=pad + display_rows * (example_height + pad)
    h1=int(pad + display_cols * (example_width + pad))
    display_array = - np.ones( shape=(w1 ,h1 ) );
    

    # Copy each example into a patch on the display array
    curr_ex = 0;
    for j in range( 1,display_rows+1):
    	for i in range(1,display_cols+1):
            max_val = max( abs( X[curr_ex, :] ) )
            row0=pad + (j - 1) * (example_height + pad) 
            row1=pad + (j - 1) * (example_height + pad) + example_height
            col0=pad + (i - 1) * (example_width + pad) 
            col1=pad + (i - 1) * (example_width + pad) + example_width
            display_array[row0:row1,col0:col1]=X[curr_ex, :].reshape( example_height, example_width,order='F') / max_val;
            
            curr_ex = curr_ex + 1;
		
    # Display Image
    h=io.imshow(display_array,cmap='gray')
    io.show()

    return h, display_array

def sigmoid(z):
    g = 1 / (1 + np.exp(-z))
    return g

def sigmoidGradient(z):
    g=np.zeros(z.shape[0])
    g= sigmoid(z) * (1- sigmoid(z))
    return g

def randInitializeWeights(L_in, L_out):
    W = np.zeros( shape=( L_out, 1 + L_in))
    epsilon_init=0.12;
    W= np.random.rand(L_out,1+L_in)*2*epsilon_init-epsilon_init
    return W

def nnCostFunction(nn_params,input_layer_size,hidden_layer_size,num_labels,X,y,l):
    Theta1=0
    Theta2=0
    
    Theta1=nn_params[0:(hidden_layer_size*  (input_layer_size+1))].reshape(hidden_layer_size,(input_layer_size+1))
    Theta2=nn_params[(hidden_layer_size*  (input_layer_size+1)): ].reshape(num_labels,(hidden_layer_size+1))

    m = X.shape[0]
    J = 0;
    Theta1_grad = np.zeros( shape=( Theta1.shape) );
    Theta2_grad = np.zeros( shape=( Theta2.shape) );
    
    ###### Part 1 Finding Cost     
    oneColumn = np.ones(shape=(m,1))
    X = np.hstack((oneColumn,X))  #[np.ones(m, 1) X]; # add 1 numbers column in X (X matrixinin başına 1 lerden oluşan bir kolon ekle)    
    z2= np.dot(X,Theta1.transpose())
    a2=sigmoid(z2)
    a2=np.hstack((oneColumn,a2))
    z3=np.dot(a2,Theta2.transpose())
    hx=sigmoid(z3)
        
    
    #lamdaSum= l/(2*m) *  (sum(sum( np.square( Theta1[1:]))) + sum(sum( np.square( Theta2[1:])))  )  # multiclass classification
    lamdaSum= l/(2*m) *  (sum(sum( np.square( Theta1[1:]))) + sum(np.square( Theta2[1:])) ) # binary (one output) classification
    
    #J=  (1/m) * sum(sum(( -yMatrix * np.log(hx)-(1-yMatrix ) * np.log(1-hx))) ) + lamdaSum; # multiclass classification cost    
    J=  (1/m) * sum(sum(( -y * np.log(hx)-(1-y ) * np.log(1-hx))) ) + lamdaSum; # binary classification cost
    print('cost J:',J)   # to watch how cost descrise after iterations

    ##### Part 2 Finding Grad (Backpropagation Algorithm formulas)
    y.shape = [y.shape[0],1]
    delta3= hx-y
    delta2= np.dot(delta3 , Theta2[:,1:]) * sigmoidGradient(z2)
    Delta2= np.dot(delta3.transpose() , a2 )
    Delta1= np.dot(delta2.transpose() , X )
    Theta1_grad= 1/m * Delta1
    Theta2_grad= 1/m * Delta2
    

    ##### Part 3 regulization

    lambdaSumG1 = l/m * np.hstack( ( (np.zeros(shape=(Theta1.shape[0],1))) , Theta1[:,1:] ));
    lambdaSumG2 = l/m * np.hstack( ( (np.zeros(shape=(Theta2.shape[0],1))) , Theta2[:,1:] ));

    Theta1_grad = Theta1_grad + lambdaSumG1;
    Theta2_grad = Theta2_grad + lambdaSumG2;
    
    # Unroll gradients
    grad = np.concatenate([np.concatenate(Theta1_grad),np.concatenate(Theta2_grad)])  #np.array( [initial_Theta1 , initial_Theta2]);
    #print('grad shape:',grad.shape)
    #print('grad:',grad)

    return J,grad


def predict(Theta1,Theta2,X):
    m = X.shape[0] 
    oneColumn = np.ones(shape=(m,1))
    X = np.hstack((oneColumn,X))  #[np.ones(m, 1) X]; # add 1 numbers column in beginning of X matrix 
    z2= np.dot(X,Theta1.transpose())
    a2=sigmoid(z2)
    a2=np.hstack((oneColumn,a2))
    z3=np.dot(a2,Theta2.transpose())
    hx=sigmoid(z3)    # our prediction 
                      
        
    return all( hx> 0.25 ),hx

############ckecking #######
def debugInitializeWeights(fan_out, fan_in):
    W = np.zeros( shape=(fan_out, 1 + fan_in));
    W = np.reshape( np.sin(range(1,((fan_out*(1+fan_in))+1)) ), W.shape , order='F') / 10; # numel(W) = fan_out * (1+fan_in) count of object
    return W

def computeNumericalGradient(J, theta , input_layer_size,hidden_layer_size,num_labels,X,y,lamda):
    Theta1= theta[0]  #auto shaped,in octave Theta1 = reshape( nn_params(1:hi ......
    Theta2= theta[1]  #auto shaped,in octave Theta2 = reshape(nn_params((1 + (hidden_layer_size * (inp ....
    countOfArrayMember= theta.shape[0] 

    numgrad = np.zeros( np.array([countOfArrayMember,1])); # theta 38 row 1, column 3 input 5 hidden 3 output.a0,h0 (4*5)+(6*3)
    perturb = np.zeros( np.array([countOfArrayMember,1]));
    e = math.e - 4;

    #print('theta:',theta)
    
    for p in range( 1 , countOfArrayMember):
    # Set perturbation vector
        perturb[p] = e
        thetaMinus =np.array([x - y for x, y in zip(theta, perturb)]) #np.subtract(theta , perturb)[0]
        thetaPlus = np.array([x + y for x, y in zip(theta, perturb)])  #np.dot( thetaC , perturb)
        loss1 = J(thetaMinus,input_layer_size,hidden_layer_size,num_labels,X,y,lamda)[0];
        loss2 = J(thetaPlus,input_layer_size,hidden_layer_size,num_labels,X,y,lamda)[0];
        numgrad[p] = (loss2 - loss1) / (2*e);
        perturb[p] = 0;
    return numgrad


def checkNNGradients(lamda):
    if not lamda:
        lamda=0
    
    input_layer_size = 3;
    hidden_layer_size = 5;
    num_labels = 1;
    m = 5;
    
    # We generate some 'random' test data
    Theta1 = debugInitializeWeights(hidden_layer_size, input_layer_size);
    Theta2 = debugInitializeWeights(num_labels, hidden_layer_size);
    # Reusing debugInitializeWeights to generate X
    X  = debugInitializeWeights(m, input_layer_size - 1);
    y = np.array([1,0,1,1,0])
    
    # Unroll parameters
    nn_params= np.concatenate([np.concatenate(Theta1),np.concatenate(Theta2)])    

    cost,grad= nnCostFunction(nn_params,input_layer_size,hidden_layer_size,num_labels,X,y,lamda)
    numgrad = computeNumericalGradient(nnCostFunction, nn_params,input_layer_size,hidden_layer_size,num_labels,X,y,lamda)
    
    
    print('numgrad:',numgrad[19], ' grad:', grad[19]) # to compare grad and numgrad, difference should be very small
    print('diff 0:',numgrad[1]-grad[1])
        

def faceSimilarity():  # main function
    
    input_layer_size  = 16384;  # 128x128 Input Images of Digits
    hidden_layer_size =  1024;  # 1024;   # 1024 hidden units
    num_labels = 1;          # 1 output unit, targetPerson:1 , other:0 binary classification 
    
    ################-1-
    print('Loading and Visualizing Data ...')
    
    # Get images from folder and label them. 0: other person ; 1: target person    
    imageArrayList = []
    yList= []
    
    filePath='data/'
    folderList=os.listdir(filePath)
    
    
    for fldr in folderList:
        if not fldr.startswith('.'):
            filePathF = filePath + fldr +'/*.*'
            print('files_path:',filePathF)
            for filename in glob.glob(filePathF):
                im=Image.open(filename).convert('L')
                im = im.resize((128,128))
                imageArrayList.append(np.concatenate(np.asarray(im)))
                if fldr=='person':
                    yList.append(1)
                else: 
                    yList.append(0)
    
    #imageArray = np.asarray(imageArrayList)
    
    X=np.asarray(imageArrayList)
    y=np.asarray(yList)         

    m = X.shape[0] 
    
    # Randomly select 100 data points to display
    rand_indices= np.random.permutation(m)
    sel = X[rand_indices[1:100], :];

    displayData(sel,5);
    
    #print('X 00:',X[0][0])
    # Visualize first image 
    #print(X[0].reshape(128,128))
    #Image.fromarray(X[0].reshape(128,128)).show()
        
    
    ################-2-
    print('Creating initial Neural Network Parameters with 0...\n')
    # Create initial weight(theta) of neural network. All of them are zero.
    
    Theta1=np.zeros(shape=((1024,16385)))
    Theta2=np.zeros(shape=((1,1025)))
    
    nn_params = np.array( [ Theta1, Theta2]) 
    nn_params = np.concatenate([np.concatenate(nn_params[0]),np.concatenate(nn_params[1])]) 
        
    lmbda = 3

        
    ################-6-
    print('Initializing Neural Network Parameters (random)...')
    # Create random initial weight(theta) of neural network.
    
    initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
    initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

    # Unroll parameters, make vector from arrays
    initial_nn_params =np.concatenate([np.concatenate(initial_Theta1),np.concatenate(initial_Theta2)])

 
    ################-9-
    #print('Training Neural Network... ')
    
    def decoratedCost(Thetas):  # thanks stackexchange.
        return nnCostFunction(Thetas, input_layer_size, hidden_layer_size, num_labels, X, y, lmbda)[0]
    
    def decoratedGrad(Thetas):
        return nnCostFunction(Thetas, input_layer_size, hidden_layer_size, num_labels, X, y, lmbda)[1]

    #result=op.fmin_cg(decoratedCost,initial_nn_params,fprime=decoratedGrad,maxiter=50)
    result=op.fmin_cg(decoratedCost,nn_params,fprime=decoratedGrad,maxiter=10)
    
    
    Theta1=result[0:(hidden_layer_size*  (input_layer_size+1))].reshape(hidden_layer_size,(input_layer_size+1))
    Theta2=result[(hidden_layer_size*  (input_layer_size+1)): ].reshape(num_labels,(hidden_layer_size+1))

        
    ############## Predict all X again and find accuracy
    pred,hx = predict(Theta1,Theta2, X);
    print('Training Set Accuracy:', np.mean(np.double(np.where(pred == y,1,0))) * 100);

    ############# Predicting with image      
    im5=Image.open('targetperson1.png').convert('L')
    im5 = im5.resize((128,128))
    arrim5=np.asarray([np.concatenate(np.asarray(im5))])   
    p5,hx5=predict(Theta1,Theta2, arrim5) 
    print('Predict target person image 1:',hx5)
    
    im5=Image.open('targetperson2.png').convert('L')
    im5 = im5.resize((128,128))
    arrim5=np.asarray([np.concatenate(np.asarray(im5))])   
    p5,hx5=predict(Theta1,Theta2, arrim5) 
    print('Predict target person image 2:',hx5)
    
    im5=Image.open('targetperson3.png').convert('L')
    im5 = im5.resize((128,128))
    arrim5=np.asarray([np.concatenate(np.asarray(im5))])   
    p5,hx5=predict(Theta1,Theta2, arrim5) 
    print('Predict target person image 3:',hx5)
    
    ############## 
    im6=Image.open('01290.png').convert('L')
    im6 = im6.resize((128,128))
    arrim6=np.asarray([np.concatenate(np.asarray(im6))])
    p6,hx6=predict(Theta1,Theta2, arrim6) 
    print('Predict other person image 1:',hx6)
    
    im7=Image.open('01715.png').convert('L')
    im7 = im7.resize((128,128))
    arrim7=np.asarray([np.concatenate(np.asarray(im7))])
    p7,hx7=predict(Theta1,Theta2, arrim7) 
    print('Predict other person image 2:',hx7)
    
    im8=Image.open('01716.png').convert('L')
    im8 = im8.resize((128,128))
    arrim8=np.asarray([np.concatenate(np.asarray(im8))])
    p8,hx8=predict(Theta1,Theta2, arrim8) 
    print('Predict other person image 3:',hx8)
          
    
faceSimilarity()    
#checkNNGradients(0)

