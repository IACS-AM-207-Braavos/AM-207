"""
Michael S. Emanuel
Mon Oct 22 22:01:33 2018
"""

import numpy as np
import matplotlib.pyplot as plt
## Standard boilerplate to import torch and torch related modules
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler

# *************************************************************************************************
# Regression Parent Class
class Regression(object):
    
    def __init__(self):
        self.params = dict()
    
    def get_params(self, k):
        return self.params.get(k, None)
    
    def set_params(self, **kwargs):
        for k,v in kwargs.items():
            self.params[k] = v
        
                    
    def fit(self, X, y):
        raise NotImplementedError()
        
    def predict(self, X):
        raise NotImplementedError()
        
    def score(self, X, y):
        raise NotImplementedError()


# *************************************************************************************************
## Our PyTorch implementation of Logistic Regression
class LRPyTorch(nn.Module):

    ## the constructor is where we'll define all our layers (input, hidden, and output)
    def __init__(self):

        ## this line creates an instance of our parent (or base) class which in this case
        ## is nn.Module.
        super().__init__()

        ## in the lines below we'll create instance variables and assign them torch.nn Models
        ## in order to create our layers.  You should ordinarily have one variable definition for each layer
        ## in your neural network except for the output layer.  The output layer is defined by the number of
        ## outputs in your last layer. Since we're dealing with simple Artificial Neural Networks, we should
        ## predominantly be using nn.Linear.  
        self.l1 = nn.Linear(784, 10)

 
    # forwards takes as a parameter x -- the batch of inputs that we want to feed into our neural network model
    # and returns the output of the model ... i.e. the results of the output layer of the model after forward
    # propagation through our model. practically this means you should call each layer you defined in the
    # constructor in sequence plus any activation functions on each layer.
    def forward(self, x):
     
        # call all our layers on our input (in this case we only need one)
        x = self.l1(x)

        # Since we're using Cross Entropy Loss
        # we can return our output directly
        return x
    

# *************************************************************************************************
class MLP_PyTorch(nn.Module):
    """Multilayer perceptron model (placeholder for now)"""

    def __init__(self):

        super().__init__()

        self.l1 = nn.Linear(784, 10)
        

 
    def forward(self, x):
     
        # call all our layers on our input (in this case we only need one)
        x = self.l1(x)

        return x
    

# *************************************************************************************************

class MNIST_Logistic_Regression(Regression):
    
    def __init__(self, learning_rate, weight_decay, batch_size, epochs, validation_size):
        
        super().__init__()
                
        ## Add inputs to parameters
        self.set_params(learning_rate=learning_rate,
                        weight_decay=weight_decay,
                        batch_size=batch_size,
                        validation_size=validation_size,
                        epochs=epochs)   

        ## Load MNIST Data
        train_dataset, test_dataset, train_loader, validation_loader, test_loader = self.load_data()
        
        ## Add Datasets and Data Loaders to our params
        self.set_params(train_dataset=train_dataset, 
                        train_loader=train_loader,
                        validation_loader=validation_loader,
                        test_dataset=test_dataset,
                        test_loader=test_loader,
                        is_fit = False)
        
        
        ## Here we instantiate the PyTorch model that we so nicely defined previously
        model = LRPyTorch()

        ## Here we define our loss function.  We're using CrossEntropyLoss but other options include
        ## NLLLoss (negative log likelihood loss for when the log_softmax activation is explicitly defined
        ## on the output layer), MSELoss for OLS Regression, KLLDivLoss for KL Divergence, BCELoss
        ## for binary cross entropy and many others
        criterion = nn.CrossEntropyLoss()

        ## Here we define our optimizer.  In class we've been using SGD although in practice one will often
        ## use other optimizers like Adam or RMSProp.  The primary parameter the optimizer takes is the
        ## set of parameters in your model.  Fortunately those are easily accessible via model.paramters()
        ## where model is the instance of the model you defined.  Other useful parameters include lr for the
        ## learning rate and weight_decay for the rate of l2 regularization.
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        
        ## Set the rest of our parameters -- batch_size, learning_rate, epochs, optimizer, model and criterion
        
        ## Add Datasets and Data Loaders to our params
        self.set_params(optimizer=optimizer, 
                        model=model,
                        criterion=criterion)   

    def load_data(self):
        """load MNIST data; split into train / validation / test. Changed from Lab."""
        # Get the batch size and validation size
        batch_size = self.get_params('batch_size')
        validation_size = self.get_params('validation_size')

        # Load training data
        train_dataset = datasets.MNIST(root='./hw3_data',
                                    train=True,
                                    transform=transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize((0.1307,), (0.3081,)),
                                     ]),
                                    download=True)

        ## Load test data
        test_dataset = datasets.MNIST(root='./hw3_data',
                                   train=False,
                                   transform=transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize((0.1307,), (0.3081,)),
                                     ]),
                                   download=True)
    
        # Random, non-contiguous split
        num_train: int = len(train_dataset)
        indices = list(range(num_train))
        # Indices for the validation set
        validation_idx = np.random.choice(indices, size=validation_size, replace=False)
        # The training data is the complement of the validation data in the full training set
        train_idx = list(set(indices) - set(validation_idx))
        # Save these indices
        self.set_params(train_idx=train_idx, validation_idx=validation_idx)
        
        # Define samples to be SubsetRandomSampler
        train_sampler = SubsetRandomSampler(train_idx)
        validation_sampler = SubsetRandomSampler(validation_idx)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)        
        validation_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=validation_sampler)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return (train_dataset, test_dataset, train_loader, validation_loader, test_loader)

    def sample_training_images(self):
        """Create a set of sample images from the MNIST training images"""
        
        training_set = self.get_params('train_dataset')
        sample_indices = np.random.choice(training_set.train_data.size(0), 10)
        
        sample_images = training_set.train_data[sample_indices,:,:].numpy()
        sample_labels = [training_set.train_labels[x] for x in sample_indices]
        
        self.set_params(sample_training_images=sample_images)
        self.set_params(sample_training_labels=sample_labels)
        
    def save_misclassified(self, predictions, images, labels):
        """Create and save a set of sample images misclassified images by the model"""
             
        mislabeled_indices = [index for index,value in enumerate(predictions == labels) if value==False]
        sample_indices = np.random.choice(mislabeled_indices, 10)
        
        sample_images = images[sample_indices,:,:].numpy()
        sample_labels = [predictions[x] for x in sample_indices]
        true_labels = [labels[x] for x in sample_indices]

        self.set_params(misclassified_images=sample_images)
        self.set_params(misclassified_labels=sample_labels)
        self.set_params(misclassified_true_labels=true_labels)
        
        
    def viz_training_images(self):
        """Visualize/Plot sample training images"""
        
        if not self.get_params('training_labels'):
            self.sample_training_images()
        
        # get the images and labels
        sample_images = self.get_params("sample_training_images")
        sample_labels = self.get_params("sample_training_labels")
        
        fig, (ax1, ax2) = plt.subplots(2, 5, figsize=(20, 10))
        plt.suptitle("Some Sample Images from MNIST", fontsize=20, weight='heavy')

        for i in range(5):
            ax1[i].imshow(sample_images[i])
            ax1[i].set_title("MNIST Label: {}".format(sample_labels[i]))
            ax2[i].imshow(sample_images[i+5])
            ax2[i].set_title("MNIST Label: {}".format(sample_labels[i+5]), weight='bold')
            
        plt.show()

    def viz_misclassified_images(self):
        """Visualize/Plot misclassified training images"""

        # get the images and labels
        sample_images = self.get_params("misclassified_images")
        sample_labels = self.get_params("misclassified_labels")
        true_labels = self.get_params("misclassified_true_labels")

        if not sample_labels:
            raise(Exception("Please run predict() or score() with save_misclassified=True"))

        fig, (ax1, ax2) = plt.subplots(2, 5, figsize=(20, 10))
        plt.suptitle("Some Sample Misclassified Images", fontsize=20, weight='heavy')

        for i in range(5):
            ax1[i].imshow(sample_images[i])
            ax1[i].set_title("MNIST Label: {} Classified: {}".format(true_labels[i], sample_labels[i]), weight='bold')
            ax2[i].imshow(sample_images[i+5])
            ax2[i].set_title("MNIST Label: {} Classified: {}".format(true_labels[i+5], sample_labels[i+5]), weight='bold')

        plt.show()

    ## Stolen from excellent visualization from submission from Madeleine Duran/Sarah Walker
    def viz_training_loss(self, epochs):
        """Visualize/Plot our training loss"""
        
        losses = self.get_params("training_losses")
        
        if type(losses) == type(None):
            raise("Please run fit() to train data")
        
        fig, axes = plt.subplots(nrows=1, ncols=epochs, figsize=(20,5), sharex=True, sharey=True)
        
        for i in range(epochs):
            axes[i].plot(range(len(losses[i])), losses[i])
            axes[i].set_title("epoch {}".format(i))
            if i % 2 == 1:
                axes[i].axvspan(-10, 950, facecolor='gray', alpha=0.2)
        plt.subplots_adjust(wspace=0)
        plt.show()       
    
    def viz_validation_loss(self, epoch):
        """Visualize the validation loss"""
        losses = self.get_params('validation_losses')[0:epoch+1]
        weight_decay = self.get_params('weight_decay')
        fig, ax = plt.subplots(figsize=(16,8))
        ax.set_title(f'Validation Accuracy: Epoch {epoch} for $\lambda$={weight_decay}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Validation Accuracy')
        if epoch > 0:
            ax.plot(np.arange(epoch+1), losses)
        else:
            ax.axhline(losses[0])
        ax.grid()
        plt.show()
    
    def predict(self, dataset='Test', save_misclassified=True):
        """Classify images based on the fitted logistic regression model"""

        # Get Loader
        if dataset == 'Train':
            loader = self.get_params('train_loader')
            train_idx = self.get_params('train_idx')
            dataset_labels = loader.dataset.train_labels[train_idx]
            dataset_images = loader.dataset.train_data[train_idx]
        elif dataset == 'Validation':
            loader = self.get_params('validation_loader')
            validation_idx = self.get_params('validation_idx')
            dataset_labels = loader.dataset.train_labels[validation_idx]
            dataset_images = loader.dataset.train_data[validation_idx]
        elif dataset == 'Test':
            loader = self.get_params('test_loader')
            dataset_labels = loader.dataset.test_labels
            dataset_images = loader.dataset.test_data
        # Convert to numpy
        dataset_labels = dataset_labels.numpy()
        
        predictions = []
        correct = 0
        model = self.get_params('model')

        for inputs, labels in loader:

            ## get the inputs from the dataloader and turn into a variable for 
            ## feeding to the model
            inputs = Variable(inputs)

            ## Reshape so that batches work properly
            inputs = inputs.view(-1, 28*28)

            # run our model on the inputs
            outputs = model(inputs)

            # get the class of the max log-probability
            pred = outputs.data.max(1)[1]

            correct += (pred == labels).sum()

            # append current batch of predictions to our list
            predictions += list(pred)


        if save_misclassified:
            self.save_misclassified(predictions, dataset_images, dataset_labels)
            
        self.set_params(predictions=predictions, 
                        correct_predictions=correct,
                        prediction_dataset_length=len(dataset_labels),
                        prediction_dataset_labels=dataset_labels
                       )
        return np.array(predictions)
    
    
    def score(self, dataset='Test', save_misclassified=True ):
        """Calculate accuracy score based upon model classification"""
        
        self.predict(dataset=dataset, save_misclassified=save_misclassified)
        correct = self.get_params('correct_predictions')
        total = self.get_params('prediction_dataset_length')
        accuracy: float = float(correct)/float(total)
        print(f'Dataset: {dataset} \nAccuracy: {correct}/{total} ({100*accuracy:.1f}%)\n')
        
        return accuracy
        
        
    def fit(self):
        """Fit our logistic regression model on MNIST training set"""
        
        ## We defined a number of variables in our constructor -- let's reclaim them here
        optimizer=self.get_params("optimizer")
        model=self.get_params("model")
        epochs=self.get_params("epochs")
        criterion=self.get_params("criterion")
        train_loader=self.get_params("train_loader")
        
        ## Get the Total size of training set
        self.get_params('train_dataset')
        training_size = self.get_params('train_dataset').train_data.size(0)
        
        iterations = int(np.ceil(training_size/self.get_params("batch_size")))
        
        ## We need something to keep track of our losses
        losses = np.zeros((epochs, iterations)) 
        # Also track the losses at the end of every epoch
        losses_val = np.zeros(epochs)
        ## Set Loss Matrix for visualizing
        self.set_params(training_losses=losses)
        self.set_params(validation_losses=losses_val)
                
        ## Our training loop.  We can loop over a fixed number of epochs or
        ## using a sensitivity parameter (i.e. until net change in loss is
        ## below a certain tolerance).  Here we iterate over a fixed number of
        ## epochs
        for epoch in range(epochs):

            ## We defined our train_loader DataLoader earlier.  The train_loader is a
            ## sequence of tuples with the first element of each tuple being
            ## the batched training inputs (the batch_size being defined in your DataLoader)
            ## and the second second element of each tuple being the corresponding labels
            ## more or less all the pytorch classes are built to handle batching transparently

            ## loop through the DataLoader.  Each loop is one iteration.  All the loops
            ## form one epoch
            for batch_index, (inputs, labels) in enumerate(train_loader):

                # Convert the inputs/labels passed from the DataLoader into
                # autograd Variables.  The dataloader provides them as PyTorch Tensors
                # per the transforms.ToTensor() operation.
                inputs, labels = Variable(inputs), Variable(labels)

                ## as mentioned above we receive the inputs as tensors of size (batch_size,1, 28, 28)
                ## which is effectively (batch_size, 28, 28) basically as a 3 dimensional tensor
                ## representing a stack of (28x28) matrices with each matrix element a floating point number
                ## representing the value of that pixel in the image.  Unfortunately our Neural Network model
                ## can't handle that representation and needs a pixel matrices to be flattened into a row vector
                ## of inputs.  The model takes a 2d tensor representing batch of such row vectors each row vector
                ## representing one set of inputs corresponding to one image.  In order to accomplish this
                ## flattening we use the .view method defined on autograd Variables.
                inputs = inputs.view(-1, 28*28)

                # we need to zero out our gradients after each pass
                optimizer.zero_grad()


                ## This is the optimize - forward step - backwards step part of our design pattern

                # this is the forward step --> we calculate the new outputs based upon the input data from
                # this batch and store the outputs in a variable
                outputs = model(inputs)

                # we compare the outputs to the ground truth labels in the batch to calculate the loss for this step
                loss = criterion(outputs, labels)
                
                ## count the loss
                losses[epoch,batch_index] = loss.data[0]

                # we run backpropagation on the loss variable which repopulates the gradients all the way
                # back through our model to the input layer
                loss.backward()

                # Use the gradients calculated in the backprop that took place in .backwards() to do a new
                # gradient descent step
                optimizer.step()
                
            # Validation loss at the end of the epochs
            loss_val = self.score(dataset='Validation')
            losses_val[epoch] = loss_val
            
            # Visualize the validation loss
            self.viz_validation_loss(epoch)
                
        
        # Set flag indicating that the model is now fit
        self.set_params(is_fit=True)
        
        return self

    def is_fit(self):
        return self.get_params('is_fit')
    