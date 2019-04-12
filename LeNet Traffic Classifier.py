#### Goal for this project was to create/train a ML framework that would achieve above a 93% accuracy on recognizing traffic signs
#### Approach that was used was to use Lenet Architecture to train the model. 
#### Written by Collin Feight


import tensorflow as tf
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import time
import glob
import cv2


### Read in Data
training_file = 'train.p'
validation_file = 'valid.p'
testing_file = 'test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

# Assign Data to variables
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']


# Number of training examples
n_train = X_train.shape[0]

# Number of validation examples
n_validation = X_valid.shape[0]

# Number of testing examples.
n_test = X_test.shape[0]

# shape of an traffic sign image
image_shape = X_train.shape[1:]

# unique classes/labels there are in the dataset.
n_classes = len(set(y_train))

name_values = np.genfromtxt('signnames.csv', skip_header=1, dtype=[('myint','i8'), ('mysring','S55')], delimiter=',')


def plot(figures, rows, columns, labels=None):
    fig, axs = plt.subplots(ncols=columns, nrows=rows, figsize=(12, 14))
    axs = axs.ravel()
    for index, title in zip(range(len(figures)), figures):
        axs[index].imshow(figures[title], plt.gray())
        if (labels != None):
            axs[index].set_title(labels[index])
        else:
            axs[index].set_title(title)

        axs[index].set_axis_off()

    plt.tight_layout()

### Data exploration visualization code goes here.
# Visualizations will be shown in the notebook.
# fig, axs = plt.subplots(3,5, figsize=(15, 6))
# fig.subplots_adjust(hspace = .2, wspace=.001)
# axs = axs.ravel()
# for i in range(15):
#     index = random.randint(0, len(X_train))
#     image = X_train[index]
#     axs[i].axis('off')
#     axs[i].imshow(image)
#     axs[i].set_title(y_train[index])


# Pre-Processing the data --> convert to grayscale and normalize
def preprocess(Data):
    # axis == rgb
    X_Train_Gray = np.sum(Data / 3, axis=3, keepdims=True)
    features_train = (X_Train_Gray-128)/128
    return features_train


# Organize data so everything in terms of features and label
features_train = preprocess(X_train)
features_valid = preprocess(X_valid)
features_test = preprocess(X_test)
labels_train = y_train
labels_valid = y_valid
labels_test = y_test


### Convolution Using LeNet model
# 'Static' Parameters
mu = 0
sigma = .1
kernel_size = [1,2,2,1]
pool_strides = [1, 2, 2, 1]
padding_str = 'VALID'


#out_height/width = ceil(float(in_height - filter_height + 1) / float(strides[1])) for valid padding
#filter length in convolution is 'filter height' in above equation
def convolution(data, length_in, depth_in, length_out, depth_out):
    filter_length = length_in-length_out + 1
    stride = [1,1,1,1]
    W = tf.Variable(tf.truncated_normal(shape=(filter_length, filter_length, depth_in, depth_out), mean=mu, stddev=sigma))
    b = tf.Variable(tf.zeros(depth_out))
    #Using Valid Padding method to calculate filter dimensions
    model = tf.nn.conv2d(data, W, strides=stride, padding=padding_str) + b
    #using relu as activation function (non-linear)
    return tf.nn.relu(model)


# Linear function used for fully connected layers
def linear(data, length_in, length_out):
    W = tf.Variable(tf.truncated_normal(shape=(length_in, length_out), mean=mu, stddev=sigma))
    b = tf.Variable(tf.zeros(length_out))
    model = tf.matmul(data,W) + b
    return model


# Generates Logits and calls convolution and linear functions
def Pipeline(data, keep_prob):

    #Layer One
    #32x32x1 --> 28x28x12
    conv1 = convolution(data, 32, 1, 28, 12)

    #28x28x12 --> 14x14x12
    conv1 = tf.nn.max_pool(conv1, ksize=kernel_size, strides=pool_strides, padding=padding_str)

    #Layer Two
    #14x14x12 --> 10x10x25
    conv2 = convolution(conv1, 14, 12, 10, 25)

    #10x10x25 --> 5x5x25
    conv2 = tf.nn.max_pool(conv2, ksize=kernel_size, strides=pool_strides, padding=padding_str)

    #Layer Three
    flat = tf.contrib.layers.flatten(conv2)
    dropped = tf.nn.dropout(flat, keep_prob)
    final_3 = linear(dropped, 625, 300)
    final_3 = tf.nn.relu(final_3)

    #Layer Four
    final_4 = linear(final_3, 300, 100)
    final_4 = tf.nn.relu(final_4)

    #Layer Five
    final_layer = linear(final_4, 100, n_classes)

    return final_layer


# Create Placeholders that will be used in the training/eval functions. These placeholders are what get updated
feature_shape = features_train.shape[1:]
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.int32, None)
one_hot = tf.one_hot(y, n_classes)
keep_prob = tf.placeholder(tf.float32)

logits = Pipeline(x, keep_prob)

# Tuning parameters that effect the rate and amount the model trains
epoch = 15
batch_size = 64
rate = .001

### Evaluation Pipeline
prediction_inc = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot, 1))
accuracy_inc = tf.reduce_mean(tf.cast(prediction_inc, tf.float32))

# Make sure keep_prob stays 1 here
def eval(X, Y):
    num_list = len(X)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_list, batch_size):
        batch_x, batch_y = X[offset:offset+batch_size], Y[offset:offset+batch_size]
        accuracy = sess.run(accuracy_inc, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
        total_accuracy += (accuracy*len(batch_x))
    return total_accuracy / num_list


### Training pipeline
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot)
loss = tf.reduce_mean(cross_entropy)
train_step = tf.train.AdamOptimizer(learning_rate=rate).minimize(loss)

# Used to save information that has been updated such as the weight values
saver = tf.train.Saver()
model_file = './model'


def train(X, Y, prob):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print('Let the training begin')
        for i in range(epoch):

            # train test split faster but slightly worse results than shuffling training set

            # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=int(time.time()))
            X_train, Y_train = shuffle(X,Y)
            begin_time = time.time()
            for offset in range(0, len(X_train), batch_size):
                end = offset + batch_size
                features, labels = X_train[offset:end], Y_train[offset:end]
                sess.run(train_step, feed_dict={x: features, y: labels, keep_prob: prob})
            valid = eval(X_train, Y_train)
            #valid = eval(X_test, Y_test)
            print("[{3:.1f}s] epoch {0}/{1}: validation = {2:.3f}".format(i + 1, epoch, valid,
                                                                              time.time() - begin_time))
            # save weight scores after each epoch iteration
            saver.save(sess, model_file)
            print("model saved to {}".format(model_file))


#### TRAIN DATA FUNCTION CALLED HERE

# train(features_train, labels_train, .8)

####
# At this point, model has been successfully trained and can be used on any images

# Loading in New Images from website


def getnewsigns(new_signs_fnc, new_labels_fnc):
    figures = {}
    labels = {}
    new_signs = []
    count = 0
    for sign in new_signs_fnc:
        img = cv2.cvtColor(cv2.imread(sign), cv2.COLOR_BGR2RGB)
        new_signs.append(img)
        figures[count] = img
        labels[count] = name_values[new_labels_fnc[count]][1].decode('ascii')
        count += 1
        # plot(figures, 3, 2, labels)
    return new_signs


new_images = sorted(glob.glob('new_images/Image*.png'))
new_labels = np.array([1,22,35,15,37,18])

my_new_signs = np.array(getnewsigns(new_images, new_labels))
gray_new_signs = preprocess(my_new_signs)

k_size = 5
softmax_logits = tf.nn.softmax(logits)
top_k = tf.nn.top_k(softmax_logits, k=k_size)

# Runs evaluation pipeline on validation and test data to determine how well model was trained
with tf.Session() as sess:
    # restores saved weights so don't have to retrain model before using on validation/testing/new image data-sets
    saver.restore(sess, model_file)
    train_eval = eval(features_train, labels_train)
    valid_eval = eval(features_valid, labels_valid)
    test_eval = eval(features_test, labels_test)
    print("accuracy in train set: {:.3f}".format(train_eval))

    print("accuracy in validation set: {:.3f}".format(valid_eval))
    #print("accuracy in test set: {:.3f}".format(test_eval))
    my_accuracy_eval = eval(gray_new_signs, new_labels)
    print("My Data Set Accuracy = {:.3f}".format(my_accuracy_eval))
    my_softmax_logits = sess.run(softmax_logits, feed_dict={x: gray_new_signs, keep_prob: 1.0})
    my_top_k = sess.run(top_k, feed_dict={x: gray_new_signs, keep_prob: 1.0})
    for i in range(6):
        for j in range(k_size):
            print('Guess {} : ({:.0f}%)'.format(j+1, 100*my_top_k[0][i][j]))




