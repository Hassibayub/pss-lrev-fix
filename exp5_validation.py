#!/usr/bin/env python
# coding: utf-8

# THIS MODULE IS FOR ARCHIVE20K DATASET. FOR MOST PART OF THE CODE THIS WONT WORK.

import model
import model_img
import fastText
from keras.callbacks import ModelCheckpoint, Callback
from keras.models import load_model, Model
from keras.optimizers import *
from keras.layers import *
from keras import regularizers
from importlib import reload
from sklearn import metrics as sklm
import numpy as np
import keras.backend as K
from keras.backend.tensorflow_backend import set_session

import tensorflow as tf

# get_ipython().run_line_magic('env', 'CUDA_DEVICE_ORDER=PCI_BUS_ID')
# get_ipython().run_line_magic('env', 'CUDA_VISIBLE_DEVICES=6')
#
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# set_session(tf.Session(config=config))
reload(model_img)

reload(model)

# ## Text model features


# data_text_train = model.read_csv_data("data/archive20k/text/dataset.train")
# data_text_test = model.read_csv_data("data/archive20k/text/dataset.validation")
# # MAJORITY BASELINE
#
#
# print("archive26k")
# tmp = model.read_csv_data("data/archive20k/text/dataset.test")
# _, y_tmp, _, _, _ = zip(*tmp)
# print(sklm.accuracy_score([1 if y == 'NextPage' else 0 for y in y_tmp], [1] * len(y_tmp)))
# print(sklm.cohen_kappa_score([1 if y == 'NextPage' else 0 for y in y_tmp], [1] * len(y_tmp)))
# print(sklm.f1_score([1 if y == 'FirstPage' else 0 for y in y_tmp], [1] * len(y_tmp), average='binary', pos_label=1))
# print(sklm.f1_score([1 if y == 'FirstPage' else 0 for y in y_tmp], [1] * len(y_tmp), average='macro'))

print("tobacco")
tmp = model.read_csv_data("data/test.csv", csvformat='Tobacco800')
_, y_tmp, _, _, _ = zip(*tmp)
print(sklm.accuracy_score([1 if y == 'NextPage' else 0 for y in y_tmp], [1] * len(y_tmp)))
print(sklm.cohen_kappa_score([1 if y == 'NextPage' else 0 for y in y_tmp], [1] * len(y_tmp)))
if 'ft' not in locals():

    ft = fastText.load_model("embeddings/wiki.de.bin")
    model.ft = ft
# model_text = load_model("models/exp1_single-page_repeat-07.hdf5")

model_text = load_model("models/exp1_prev-page_repeat-01.hdf5")
_, y_true, _, _, _ = zip(*data_text_test)

y_true = [1 if y == 'FirstPage' else 0 for y in y_true]
y_predict = np.round(model_text.predict_generator(model.TextFeatureGenerator2(data_text_test)))
print("Accuracy: " + str(sklm.accuracy_score(y_true, y_predict)))
print("Kappa: " + str(sklm.cohen_kappa_score(y_true, y_predict)))
model_text.layers.pop()

model_text_features = Model(model_text.input, model_text.layers[-1].output)
text_features_train = model_text_features.predict_generator(model.TextFeatureGenerator2(data_text_train))

text_features_test = model_text_features.predict_generator(model.TextFeatureGenerator2(data_text_test))

# ## Image model features


img_dim = (224, 224)
data_image_train = model_img.read_csv_data("data/archive20k/text/dataset.train")

data_image_test = model_img.read_csv_data("data/archive20k/text/dataset.validation")
# model_image = load_model("models/exp2_img_repeat-07.hdf5")

model_image = load_model("models/exp2_prev-page_repeat-05.hdf5")
_, y_true, _, _, _ = zip(*data_text_test)

y_true = [1 if y == 'FirstPage' else 0 for y in y_true]
y_predict = np.round(
    model_image.predict_generator(model_img.ImageFeatureGenerator(data_image_test, img_dim, prevpage=True)))
print("Accuracy: " + str(sklm.accuracy_score(y_true, y_predict)))
print("Kappa: " + str(sklm.cohen_kappa_score(y_true, y_predict)))
model_image.layers.pop()

model_image_features = Model(model_image.input, model_image.layers[-1].output)
image_features_train = model_image_features.predict_generator(

    model_img.ImageFeatureGenerator(data_image_train, img_dim, prevpage=True))
image_features_test = model_image_features.predict_generator(
    model_img.ImageFeatureGenerator(data_image_test, img_dim, prevpage=True))

# # Training and test targets


# Training data
_, data_train_y, _, _, _ = zip(*data_text_train)
data_train_y = [1 if y == 'FirstPage' else 0 for y in data_train_y]
# Test data
_, data_test_y, _, _, _ = zip(*data_text_test)
data_test_y = [1 if y == 'FirstPage' else 0 for y in data_test_y]

class ValidationCheckpoint(Callback):

    def __init__(self, filepath, validation_x, validation_y, metric='kappa'):
        self.metric = metric
        self.max_metric = float('-inf')
        self.max_metrics = None
        self.filepath = filepath
        self.history = []
        self.validation_x = validation_x
        self.validation_y = validation_y

    def on_epoch_end(self, epoch, logs={}):
        predicted_labels = np.round(self.model.predict(self.validation_x))
        true_labels = self.validation_y

        eval_metrics = {
            'accuracy': sklm.accuracy_score(true_labels, predicted_labels),
            'f1_micro': sklm.f1_score(true_labels, predicted_labels, average='micro'),
            'f1_macro': sklm.f1_score(true_labels, predicted_labels, average='macro'),
            'f1_binary': sklm.f1_score(true_labels, predicted_labels, average='binary', pos_label=1),
            'kappa': sklm.cohen_kappa_score(true_labels, predicted_labels)
        }
        eval_metric = eval_metrics[self.metric]
        self.history.append(eval_metric)

        if epoch > -1 and eval_metric > self.max_metric:
            print("\n" + self.metric + " improvement: " + str(eval_metric) + " (before: " + str(
                self.max_metric) + "), saving to " + self.filepath)
            self.max_metric = eval_metric  # optimization target
            self.max_metrics = eval_metrics  # all metrics
            self.model.save(self.filepath)


# ## LDA Features


lda_train_x = []
with open("data/archive20k/lda_train.csv") as f:
    next(f)
    for l in f:
        lda_train_x.append([float(n) for n in l.split(",")])
lda_train_x = np.array(lda_train_x)
print(lda_train_x.shape)

lda_test_x = []
with open("data/archive20k/lda_validation.csv") as f:
    next(f)
    for l in f:
        lda_test_x.append([float(n) for n in l.split(",")])
lda_test_x = np.array(lda_test_x)
print(lda_test_x.shape)
features_x_train = lda_train_x

features_x_test = lda_test_x
sequence_x_train = np.empty((len(features_x_train), 2, len(features_x_train[0])))
sep_tp_x_train = np.empty((len(features_x_train), len(features_x_train[0])))
sep_pp_x_train = np.empty((len(features_x_train), len(features_x_train[0])))
for i, d in enumerate(features_x_train):
    if d[3] == "":
        prev_page = np.zeros((1, len(features_x_train[0])))
    else:
        prev_page = features_x_train[i - 1]
    sequence_x_train[i][0] = sep_pp_x_train[i] = prev_page
    sequence_x_train[i][1] = sep_tp_x_train[i] = features_x_train[i]

sequence_x_test = np.empty((len(features_x_test), 2, len(features_x_test[0])))
sep_tp_x_test = np.empty((len(features_x_test), len(features_x_test[0])))
sep_pp_x_test = np.empty((len(features_x_test), len(features_x_test[0])))
for i, d in enumerate(features_x_test):
    if d[3] == "":
        prev_page = np.zeros((1, len(features_x_test[0])))
    else:
        prev_page = features_x_test[i - 1]
    sequence_x_test[i][0] = sep_pp_x_test[i] = prev_page
    sequence_x_test[i][1] = sep_tp_x_test[i] = features_x_test[i]
print(sep_tp_x_train.shape)
print(sep_pp_x_train.shape)
print(sep_tp_x_test.shape)
print(sep_pp_x_test.shape)
sequence_x_2inputs_train = [sep_tp_x_train, sep_pp_x_train]

sequence_x_2inputs_test = [sep_tp_x_test, sep_pp_x_test]

# # Late Fusion


model_text = load_model("models/exp1_prev-page_repeat-01.hdf5")
model_image = load_model("models/exp2_prev-page_repeat-05.hdf5")
model_lda = load_model("models/exp3_img-text_lda_model.hdf5")
p_t = model_text.predict_generator(model.TextFeatureGenerator2(data_text_test))

p_v = model_image.predict_generator(model_img.ImageFeatureGenerator(data_image_test, img_dim, prevpage=True))
p_l = model_lda.predict(sequence_x_2inputs_test)
p_t = np.concatenate([1 - p_t, p_t], axis=1)  # probability from text model
p_v = np.concatenate([1 - p_v, p_v], axis=1)  # probability from visual model
p_l = np.concatenate([1 - p_l, p_l], axis=1)  # probability from lda model
i = 0.4

j = 0.1
k = 0.2

y_predict = np.argmax(np.power(p_t, i) * np.power(p_v, j) * np.power(p_l, k), axis=1)
acc = sklm.accuracy_score(y_true, y_predict)
kappa = sklm.cohen_kappa_score(y_true, y_predict)

print(str(i) + " " + str(j) + " " + str(k))
print("Accuracy: " + str(acc))
print("Kappa: " + str(kappa))

# Best results test set: 
# Accuracy: 0.9338567222767419
# Kappa: 0.7078080262749252
# Best results validation set (hold out): 
# Accuracy: 0.8899456521739131
# Kappa: 0.5914350798654169

sklm.confusion_matrix(y_true, y_predict)



# fp: 92
# fn: 398


# ## single vs multipage docs


def get_filter(text_features, single_page=True):
    bool_filter = []
    for i in range(len(text_features) - 1):
        if text_features[i][1] == 'FirstPage' and text_features[i + 1][1] != 'NextPage':
            bool_filter.append(True)
        else:
            bool_filter.append(False)
    if text_features[len(text_features) - 1][1] == 'FirstPage':
        bool_filter.append(True)
    else:
        bool_filter.append(False)

    if single_page:
        return bool_filter
    else:
        return [False if y else True for y in bool_filter]

np_y_true = np.array(y_true)

np_y_pred = np.array(y_predict)
sp_docs = get_filter(data_text_test, single_page=True)

acc = sklm.accuracy_score(np_y_true[sp_docs], np_y_pred[sp_docs])
kappa = sklm.cohen_kappa_score(np_y_true[sp_docs], np_y_pred[sp_docs])

print(np.sum(sp_docs))
print(acc)
print(kappa)

sklm.confusion_matrix(np_y_true[sp_docs], np_y_pred[sp_docs])
mp_docs = get_filter(data_text_test, single_page=False)


acc = sklm.accuracy_score(np_y_true[mp_docs], np_y_pred[mp_docs])
kappa = sklm.cohen_kappa_score(np_y_true[mp_docs], np_y_pred[mp_docs])

print(np.sum(mp_docs))
print(acc)
print(kappa)

sklm.confusion_matrix(np_y_true[mp_docs], np_y_pred[mp_docs])

# Divided into Single and Multi-Page Documents

def filter_dataset(text_features, img_features, lda_features, y, single_page=True):
    filtered_txt = []
    filtered_img = []
    filtered_lda_tp = []
    filtered_lda_pp = []
    filtered_y = []

    if single_page:
        for i in range(len(text_features) - 2):
            if text_features[i][1] == 'FirstPage' and text_features[i + 1][1] != 'NextPage':
                filtered_txt.append(text_features[i])
                filtered_img.append(img_features[i])
                filtered_lda_tp.append(lda_features[0][i])
                filtered_lda_pp.append(lda_features[1][i])
                filtered_y.append(y[i])
        i = len(text_features) - 1
        if text_features[i][1] == 'FirstPage':
            filtered_txt.append(text_features[i])
            filtered_img.append(img_features[i])
            filtered_lda_tp.append(lda_features[0][i])
            filtered_lda_pp.append(lda_features[1][i])
            filtered_y.append(y[i])
    else:
        for i in range(len(text_features) - 2):
            if (text_features[i][1] == 'FirstPage' and text_features[i + 1][1] != 'FirstPage') or (
                    text_features[i][1] == 'NextPage'):
                filtered_txt.append(text_features[i])
                filtered_img.append(img_features[i])
                filtered_lda_tp.append(lda_features[0][i])
                filtered_lda_pp.append(lda_features[1][i])
                filtered_y.append(y[i])
        i = len(text_features) - 1
        if text_features[i][1] != 'FirstPage':
            filtered_txt.append(text_features[i])
            filtered_img.append(img_features[i])
            filtered_lda_tp.append(lda_features[0][i])
            filtered_lda_pp.append(lda_features[1][i])
            filtered_y.append(y[i])
    return filtered_txt, filtered_img, [filtered_lda_tp, filtered_lda_pp], filtered_y

feat_txt, feat_img, feat_lda, y_true_filtered = filter_dataset(data_text_test, data_image_test, sequence_x_2inputs_test,

                                                               y_true,
                                                               single_page=False)
print(len(y_true_filtered))
p_t = model_text.predict_generator(model.TextFeatureGenerator2(feat_txt))

p_v = model_image.predict_generator(model_img.ImageFeatureGenerator(feat_img, img_dim, prevpage=True))
p_l = model_lda.predict(feat_lda)
p_t = np.concatenate([1 - p_t, p_t], axis=1)  # probability from text model
p_v = np.concatenate([1 - p_v, p_v], axis=1)  # probability from visual model
p_l = np.concatenate([1 - p_l, p_l], axis=1)  # probability from lda model
i = 0.4

j = 0.1
k = 0.2

y_predict = np.argmax(np.power(p_t, i) * np.power(p_v, j) * np.power(p_l, k), axis=1)
acc = sklm.accuracy_score(y_true_filtered, y_predict)
kappa = sklm.cohen_kappa_score(y_true_filtered, y_predict)

print(str(i) + " " + str(j) + " " + str(k))
print("Accuracy: " + str(acc))
print("Kappa: " + str(kappa))

# Single Page Docs
# Accuracy: 0.41922290388548056
# Kappa: 0.0

# Multi-Page Docs
# Accuracy: 0.9488028527763627
# Kappa: 0.6868376125359246

fn = 0

fp = 0
for i in range(len(y_predict)):
    if y_predict[i] == 1 and y_true_filtered[i] == 0:
        fp += 1
    if y_predict[i] == 0 and y_true_filtered[i] == 1:
        fn += 1
print(fp)
print(fn)
# Multi-page

# FP: 84
# FN: 117

# Single-page
# FP: not defined
# FN: 282
sklm.confusion_matrix(y_true_filtered, y_predict)
len(y_true_filtered)

3926 + 489


len(data_text_test)
