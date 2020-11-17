import glob
import numpy
import cv2
import sklearn
import matplotlib.pyplot as plt
import itertools
from sklearn.model_selection import train_test_split
import tensorflow
from tensorflow.keras.models import load_model

"""
Accident Prediction from Dash cam videos

File to Train the proposed Model
Please change the path to accident folder to a folder containing accident clips
Please change the path to normal folder to a folder containing non accident clips
"""

# file paths to crash 1500 folder and normal folder

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
    Create frames from video
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
    Create label matrix for train set to calculate accuracy
    """
    n_values = numpy.max(values) + 1
    return numpy.eye(n_values)[values]


labels = numpy.concatenate(
    ([1] * len(pos_all), [0] * len(neg_all[0:len(pos_all)])))
labels = label_matrix(labels)


def make_dataset(rand):
    """
    Create the dataset
    """
    seq1 = []
    for i, fi in enumerate(rand):
        if fi[-4:] == '.mp4':
            t = load_set(fi)
            seq1.append(t)
    return numpy.asarray(seq1)


# split data into training,test and validation
print("Creating Testset of Frames, using standard train test split to generate test set")
x_train, x_t1, y_train, y_t1 = train_test_split(all_files, labels, test_size=0.40, random_state=0)
x_train = numpy.array(x_train)
y_train = numpy.array(y_train)
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

# print basic info
print(x_test.shape[0], 'test samples')
print("Done creating test set, loading model")

# load saved model
model = load_model('team4_model.h5')
optimizer = tensorflow.keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["categorical_accuracy"])

# Model testing on test set
y_hat = model.predict(x_test, batch_size=1)

# PLOTS AND RESULTS
# referred from :
# https://www.dlology.com/blog/simple-guide-on-how-to-generate-roc-plot-for-keras-classifier/

# Plot linewidth.
lw = 2

num_classes = 2
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
