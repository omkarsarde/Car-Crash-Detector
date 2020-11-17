import numpy
import itertools
import glob
import cv2
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
import tensorflow.keras
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import LSTM, TimeDistributed
from tensorflow.keras.models import load_model

"""
Accident Prediction from Dash cam videos

File to Train the proposed Model
Please change the path to accident folder to a folder containing accident clips
Please change the path to normal folder to a folder containing non accident clips
"""

# file path to crash 1500 folder and normal folder, please put a trailing slash "/"
# as shown in examples

# folder containing accident clips
# example: 'O:/CV/Crash-1500-NEW/'
img_filepath_to_accidentfolder = 'O:/CV/p1/'

# folder containing normal clips
# example: 'O:/CV/Normal-20200430T122509Z-002/'
img_filepath_to_normalFolder = 'O:/CV/n1/'
neg_all = glob.glob(img_filepath_to_normalFolder + '*.mp4')
pos_2 = glob.glob(img_filepath_to_accidentfolder + '*.mp4')

pos_all = pos_2
all_files = numpy.concatenate((pos_all, neg_all))
print(len(neg_all), len(pos_all))


def load_set(videofile):
    """
    Load frames from video
    """
    vidcap = cv2.VideoCapture(videofile)
    success, image = vidcap.read()
    count = 0
    success = True
    img = []
    while success:
        success, image = vidcap.read()
        if image is not None:
            resized_image = cv2.resize(image, (244, 244))
            img.append(resized_image)
        count += 1
    return img


def label_matrix(values):
    """
    Create label matrix
    """
    n_values = numpy.max(values) + 1
    return numpy.eye(n_values)[values]


labels = numpy.concatenate(
    ([1] * len(pos_all), [0] * len(neg_all[0:len(pos_all)])))
labels = label_matrix(labels)


def make_dataset(rand):
    """
    Create dataset
    """
    seq1 = []
    for i, fi in enumerate(rand):
        if fi[-4:] == '.mp4':
            t = load_set(fi)
            seq1.append(t)
    return numpy.asarray(seq1)


# split data into training,test and validation

print("Creating Dataset of Frames")
x_train, x_t1, y_train, y_t1 = train_test_split(all_files, labels, test_size=0.40, random_state=0)
x_train = numpy.array(x_train)
y_train = numpy.array(y_train)
x_train = make_dataset(x_train)
len_X = len(x_t1)
len_Y = len(y_t1)

# test set for model
size_x = len_X / 2
size_y = len_Y / 2
x_test = numpy.array(x_t1[int(size_x):])
y_test = numpy.array(y_t1[int(size_y):])
x_test = make_dataset(x_test)

# validation set for model
x_valid = numpy.array(x_t1[:int(size_x)])
y_valid = numpy.array(y_t1[:int(size_y)])
x_valid = make_dataset(x_valid)

# print basic info
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print(x_valid.shape[0], 'validation samples')
print("Done creating train test and validation sets, creating model")

# set hyper-parameters
batch_size = 60
num_classes = 2
epochs = 30

# number of hidden layers for HRNN
row_hidden = 128
col_hidden = 128

# Model Creation
frame, row, col, channel = (49, 244, 244, 3)
video = Input(shape=(frame, row, col, channel))
cnn_base = VGG19(input_shape=(244, 244, 3), weights='imagenet', include_top=False)
cnn_base.trainable = False
cnn_out = GlobalAveragePooling2D()(cnn_base.output)
cnn_out.trainable = False
cnn = Model(inputs=[cnn_base.input], outputs=cnn_out)
cnn.trainable = False
encoded_frames = TimeDistributed(cnn)(video)
encoded_sequence = LSTM(col_hidden)(encoded_frames)
hidden_layer = Dense(1028, activation='relu')(encoded_sequence)
output = Dense(num_classes, activation='softmax')(hidden_layer)
model = Model([video], output)
optimizer = tensorflow.keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["categorical_accuracy"])

# Model Training
training = []
scores = []
fit = model.fit(x_train, y_train, batch_size=1, epochs=10, verbose=2, validation_data=(x_valid, y_valid))
training.append(fit)
score = model.evaluate(x_valid, y_valid, verbose=1, batch_size=1)
scores.append(score)
print('Validation loss:', score[0])
print('Validation accuracy:', score[1])

# SAVE MODEL
model.save('team4_model.h5')
# Clear Model from memory to ensure saved model is used
del model

# load saved model
model = load_model('team4_model.h5')
optimizer = tensorflow.keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["categorical_accuracy"])

# reinitialize test set if sample size causes earlier data to go out of ram
size_x = len_X / 2
size_y = len_Y / 2
x_test = numpy.array(x_t1[int(size_x):])
y_test = numpy.array(y_t1[int(size_y):])
x_test = make_dataset(x_test)

# Model testing on test set
y_hat = model.predict(x_test, batch_size=1)

# PLOTS AND RESULTS
# referred from :
# https://www.dlology.com/blog/simple-guide-on-how-to-generate-roc-plot-for-keras-classifier/

# Plot linewidth.
lw = 2

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(num_classes):
    fpr[i], tpr[i], _ = sklearn.metrics.roc_curve(y_test[:, i], y_hat[:, i])
    roc_auc[i] = sklearn.metrics.auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = sklearn.metrics.roc_curve(y_test.ravel(), y_hat.ravel())
roc_auc["micro"] = sklearn.metrics.auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area

# First aggregate all false positive rates
all_fpr = numpy.unique(numpy.concatenate([fpr[i] for i in range(num_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = numpy.zeros_like(all_fpr)
for i in range(num_classes):
    mean_tpr += numpy.interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= num_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = sklearn.metrics.auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure(1)
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = itertools.cycle(['aqua', 'darkorange'])
for i, color in zip(range(num_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
                   ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()

# Zoom in view of the upper left corner.
plt.figure(2)
plt.xlim(0, 0.2)
plt.ylim(0.8, 1)
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = itertools.cycle(['aqua', 'darkorange'])
for i, color in zip(range(num_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
                   ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()

# Element wise class probability

print("ELEMENT CLASS WISE PROBABILITY")
print("FORMAT: Y_TEST Y_HAT")
for i in range(len(y_hat)):
    print("y_test", y_test[i], "y_hat", y_hat[i])
