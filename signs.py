# Import modules

from collections import deque
from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle
from tensorflow.contrib.layers import flatten
import math
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pdb
import pickle
import tensorflow as tf

# Set parameters

ACCURACY_THRESHOLD = 0.01
BATCH_SIZE = 100
LEARNING_RATE = 0.001
MAX_EPOCHS = 100
MIN_EPOCHS = 10
MU = 0
SIGMA = 0.1
VALIDATION_FRACTION = 0.2

# Load traffic sign data

training_file = '../train.p'
testing_file = '../test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_tests, y_tests = test['features'], test['labels']
n_train = train['features'].shape[0]
n_test = test['features'].shape[0]
image_shape = X_train.shape[1:]
n_classes = len(np.unique(train['labels']))

### Replace each question mark with the appropriate value.

# TODO: Number of training examples

n_train = train['labels'].shape[0]

# TODO: Number of testing examples.

n_test = test['labels'].shape[0]

# TODO: What's the shape of an traffic sign image?

image_shape = train['features'].shape[1:]

# TODO: How many unique classes/labels there are in the dataset.

n_classes = len(np.unique(train['labels']))

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

### Data exploration visualization goes here.
### Feel free to use as many code cells as needed.
# Visualizations will be shown in the notebook.

# Explore the data

# Sample of n sign images
plt.ion()
n = 16
columns = 4
rows = n // columns + int(n % columns > 0)
fig = plt.figure()
plt.subplots_adjust(wspace=0.001, hspace=0.001)
for t in zip(range(n), np.random.choice(np.array(range(n_train)), n, False)):
    fig.add_subplot(rows,columns,t[0]+1)
    plt.imshow(train['features'][t[1],], interpolation='nearest')

### Preprocess the data here.
### Feel free to use as many code cells as needed.

# Shuffle the training data

train['features'], train['labels'] = shuffle(X_train, y_train)

# Scale images

train['features'] = (train['features']-128.)/128.
test['features'] = (test['features']-128.)/128.

### Generate data additional data (OPTIONAL!)
### and split the data into training/validation/testing sets here.
### Feel free to use as many code cells as needed.

# Reserve a portion of training data as validation data

X_train, X_valid, y_train, y_valid = train_test_split(train['features'], train['labels'], test_size=VALIDATION_FRACTION, random_state=42)
n_train = X_train.shape[0]
n_valid = X_valid.shape[0]
n_tests = X_tests.shape[0]

assert(len(X_train) == len(y_train))
assert(len(X_valid) == len(y_valid))
assert(len(X_tests) == len(y_tests))

### Define your architecture here.
### Feel free to use as many code cells as needed.

# Define architecture

def SignNet(x, keep_prob, n_classes):    
    # Layer 1: Convolutional. Input = 32x32xinput_channels. Output = 28x28x6.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, x.get_shape()[3].value, 6), mean = MU, stddev = SIGMA), name='conv1_W')
    conv1_b = tf.Variable(tf.zeros(6), name='conv1_b')
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b
    # Activation.
    conv1 = tf.nn.relu(conv1)
    # Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    # Layer 2: Convolutional. Output = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = MU, stddev = SIGMA), name='conv2_W')
    conv2_b = tf.Variable(tf.zeros(16), name='conv2_b')
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    # Activation.
    conv2 = tf.nn.relu(conv2)
    # Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    # Flatten. Input = 5x5x16. Output = 400.
    fc0   = flatten(conv2)
    # Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean = MU, stddev = SIGMA), name='fc1_W')
    fc1_b = tf.Variable(tf.zeros(120), name='fc1_b')
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b
    # Activation.
    fc1    = tf.nn.relu(fc1)
    # Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = MU, stddev = SIGMA), name='fc2_W')
    fc2_b  = tf.Variable(tf.zeros(84), name='fc2_b')
    fc2    = tf.matmul(fc1, fc2_W) + fc2_b
    # Activation.
    fc2    = tf.nn.relu(fc2)
    # Dropout
    fc2_drop = tf.nn.dropout(fc2, keep_prob)
    # Layer 5: Fully Connected. Input = 84. Output = n_classes.
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(84, n_classes), mean = MU, stddev = SIGMA), name='fc3_W')
    fc3_b  = tf.Variable(tf.zeros(n_classes), name='fc3_b')
    logits = tf.matmul(fc2_drop, fc3_W) + fc3_b
    return logits

### Train your model here.
### Feel free to use as many code cells as needed.

# Define the model

tf.reset_default_graph()
x = tf.placeholder(tf.float32, (None,) + X_train.shape[1:])
y = tf.placeholder(tf.int32, (None))
keep_prob = tf.placeholder(tf.float32)
one_hot_y = tf.one_hot(y, n_classes)
logits = SignNet(x, keep_prob, n_classes)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = LEARNING_RATE)
training_operation = optimizer.minimize(loss_operation)
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Define evaluation function

def evaluate(sess, X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob:1.0})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

# Train the model, validate, and test

def train_model(X_train, y_train, BATCH_SIZE):
    accuracy_window = deque(np.zeros(5, dtype='f'), 5)
    accuracy_means = deque(np.zeros(2, dtype='f'), 2)
    num_examples = len(X_train)
    for i in range(MAX_EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob:0.5})
        valid_accuracy = evaluate(sess, X_valid, y_valid)
        train_accuracy = evaluate(sess, X_train, y_train)
        accuracy_window.append(valid_accuracy)
        mean_accuracy = np.mean(accuracy_window)
        accuracy_means.append(mean_accuracy)
        accuracy_delta = accuracy_means[1]-accuracy_means[0]
        print("{},{:.3f},{:.3f},{:.3f},{:.3f}".format((i+1), valid_accuracy, mean_accuracy, accuracy_delta, train_accuracy))
        if (abs(accuracy_delta)<ACCURACY_THRESHOLD and i>MIN_EPOCHS):
            break
        
saver = tf.train.Saver()
sess = tf.Session()
try:
    saver.restore(sess, os.getcwd() + "/model.ckpt")
except:
    sess.run(tf.global_variables_initializer())
    train_model(X_train, y_train, BATCH_SIZE)
    save_path = saver.save(sess, "model.ckpt")
    print(save_path)

print("Test Accuracy = {:.3f}".format(evaluate(sess, X_tests, y_tests)))

### Run the predictions here.
### Feel free to use as many code cells as needed.

# Test a Model on New Images

image_loader = \
        tf.image.resize_images(
            tf.image.decode_jpeg(
                tf.WholeFileReader().read(
                    tf.train.string_input_producer(
                        tf.train.match_filenames_once("./images/*.jpg")))[1]),
            [32, 32], method=1)

with tf.Session() as loader_sess:
    tf.initialize_all_variables().run()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    images = np.array([
        loader_sess.run(image_loader),
        loader_sess.run(image_loader),
        loader_sess.run(image_loader),
        loader_sess.run(image_loader),
        loader_sess.run(image_loader),
        loader_sess.run(image_loader),
        loader_sess.run(image_loader),
        loader_sess.run(image_loader),
        loader_sess.run(image_loader),
        loader_sess.run(image_loader),
        loader_sess.run(image_loader),
        loader_sess.run(image_loader)])
    coord.request_stop()
    coord.join(threads)

classifications = sess.run(prediction_operation, feed_dict={x:images, keep_prob:1.0})

with open('signnames.csv') as f:
    lines = f.read().splitlines()
splitlines = [line.split(',') for line in lines[1:]]
signnames = {line[0]:line[1] for line in splitlines}

plt.ion()
n = 12
columns = 3
rows = n // columns + int(n % columns > 0)
fig = plt.figure(figsize=(8.5, 11))
plt.subplots_adjust(wspace=4, hspace=0.001)
for t in zip(range(n), images, classifications):
    ax = fig.add_subplot(rows,columns,t[0]+1)
    ax.set_title(signnames[str(t[2])])
    plt.imshow(t[1], interpolation='nearest')

### Visualize the softmax probabilities here.
### Feel free to use as many code cells as needed.

# Visualize uncertainty

X_check, y_check = train['features'][:1000,], train['labels'][:1000,]

def count_in_top_n(sess, X_check, y_check, k=5):
    probability_operation = tf.nn.softmax(logits)
    prediction_operation = tf.argmax(probability_operation, 1)
    correct = sess.run(correct_prediction, feed_dict={x: X_check, y: y_check, keep_prob:1.0})
    probs = sess.run(probability_operation, feed_dict={x: X_check, y: y_check, keep_prob:1.0})
    predictions = sess.run(tf.argmax(probs, 1), feed_dict={x: X_check, y: y_check, keep_prob:1.0})
    topn = sess.run(tf.nn.top_k(probs, k=k), feed_dict={x: X_check, y: y_check, keep_prob:1.0})
    return sum([len(np.where(p[1]==p[0])[0])>0 for p in zip(y_check[~correct], topn[1][~correct])])

print(count_in_top_n(sess, X_check, y_check, 1))
print(count_in_top_n(sess, X_check, y_check, 2))
print(count_in_top_n(sess, X_check, y_check, 3))
print(count_in_top_n(sess, X_check, y_check, 4))
print(count_in_top_n(sess, X_check, y_check, 5))
    


