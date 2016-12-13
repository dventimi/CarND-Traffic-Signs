# Load pickled data
import pickle

# TODO: Fill this in based on where you saved the training and testing data

training_file = '../train.p'
testing_file = '../test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']

### Replace each question mark with the appropriate value.

# TODO: Number of training examples

n_train = train['labels'].shape[0]

# TODO: Number of testing examples.

n_test = test['labels'].shape[0]

# TODO: What's the shape of an traffic sign image?

image_shape = train['features'].shape[1:3]

# TODO: How many unique classes/labels there are in the dataset.

# The labels are numeric (integers), so we could take advantage of the
# natural ordering of numbers by subtracting the minimum label value
# from the maximum label value and adding 1 (because both the min
# value AND the max value are represented among the labels).

n_classes = max(train['labels'])-min(train['labels'])+1

# However, by doing this we implicitly assume that every possible
# label (i.e., integer) in that interval is represented in the training
# set.  That need not be the case.  Therefore, a more reliable way is
# to count the number of unique elements within set of labels.

import numpy as np
n_classes = len(np.unique(train['labels']))

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

### Data exploration visualization goes here.
### Feel free to use as many code cells as needed.
import matplotlib.pyplot as plt
# Visualizations will be shown in the notebook.
%matplotlib inline

### Preprocess the data here.
### Feel free to use as many code cells as needed.

### Generate data additional data (OPTIONAL!)
### and split the data into training/validation/testing sets here.
### Feel free to use as many code cells as needed.

### Define your architecture here.
### Feel free to use as many code cells as needed.

### Train your model here.
### Feel free to use as many code cells as needed.

### Load the images and plot them here.
### Feel free to use as many code cells as needed.

### Run the predictions here.
### Feel free to use as many code cells as needed.

### Visualize the softmax probabilities here.
### Feel free to use as many code cells as needed.
