# Load pickled data
import pickle
import tensorflow as tf
import cv2

# from sklearn.model_selection import train_test_split

# TODO: Fill this in based on where you saved the training and testing data

training_file = '../train.p'
testing_file = '../test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']

# train_features, valid_features, train_coords, valid_coords, train_sizes, valid_sizes, train_labels, valid_labels = train_test_split(train['features'], train['coords'], train['sizes'], train['labels'], test_size=0.05)

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
# %matplotlib inline

# Utility functions

import math
def batches(batch_size, features, labels):
    assert len(features) == len(labels)
    outout_batches = []
    sample_size = len(features)
    for start_i in range(0, sample_size, batch_size):
        end_i = start_i + batch_size
        batch = [features[start_i:end_i], labels[start_i:end_i]]
        outout_batches.append(batch)
    return outout_batches


def image_gallery(images, n=16, ncols=4, filter=lambda x: (x, None)):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    rows = n // ncols + int(n % ncols > 0)
    fig = plt.figure()
    plt.subplots_adjust(wspace=0.001, hspace=0.001)
    for t in zip(range(n), np.random.choice(np.array(range(images.shape[0])), n, False)):
        fig.add_subplot(rows,ncols,t[0]+1)
        img = filter(images[t[1],])
        plt.imshow(img[0], cmap=img[1], interpolation='nearest')


plt.ion()

image_gallery(train['features'])

# Grayscale of n sign images

image_gallery(train['features'], filter=lambda x: (cv2.cvtColor(x, cv2.COLOR_BGR2GRAY), 'gray'))

# training set of grayscale images

train['flat_features'] = np.array([cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in train['features']])

# Count of each sign (histogram)

fig = plt.figure()
h = plt.hist(train['labels'], n_classes)

################################################################################
# Implement basic neural net first
################################################################################

x = tf.placeholder(tf.int32, [None, image_shape[0], image_shape[1]], name='x')
y_ = tf.placeholder(tf.int32, [None], name='y_')
W = tf.Variable(tf.zeros([image_shape[0]*image_shape[1], n_classes]))
b = tf.Variable(tf.zeros([n_classes]))
y = tf.nn.softmax(tf.matmul(tf.to_float(tf.reshape(x, [-1, image_shape[0]*image_shape[1]])), W)+b)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(tf.one_hot(y_, n_classes) * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
sess.run(train_step, feed_dict={x: train['flat_features'], y_: train['labels']})

# for batch_xs, batch_ys in batches(1000, train['flat_features'], train['labels']):
#     sess.run(train_step, feed_dict={i: batch_xs, j: batch_ys})

# correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(tf.one_hot(y_, n_classes),1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# print(sess.run(accuracy, feed_dict={x: , y_: }))
