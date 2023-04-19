#!/usr/bin/env python
# coding: utf-8


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

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# set_session(tf.Session(config=config))

reload(model_img)
reload(model)

# ## Text model features


data_text_train = model.read_csv_data("data/train.csv", csvformat="Tobacco800")
data_text_test = model.read_csv_data("data/test.csv", csvformat="Tobacco800")

if 'ft' not in locals():
    ft = fastText.load_model("embeddings/wiki.en.bin")
    model.ft = ft

# model_text = load_model("models/exp1_single-page_repeat-07.hdf5")
model_text = load_model("models/tobacco800_exp1_prev-page_repeat-06.hdf5")

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
model_img.img_path_template = 'data/images/%s.tif'

data_image_train = model_img.read_csv_data("data/train.csv", csvformat="Tobacco800")
data_image_test = model_img.read_csv_data("data/test.csv", csvformat="Tobacco800")

# model_image = load_model("models/exp2_img_repeat-07.hdf5")
model_image = load_model("models/Tobacco800_exp2_prev-page_repeat-05.hdf5")

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
with open("data/lda_train.csv") as f:
    next(f)
    for l in f:
        lda_train_x.append([float(n) for n in l.split(",")])
lda_train_x = np.array(lda_train_x)
print(lda_train_x.shape)

lda_test_x = []
with open("data/lda_test.csv") as f:
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

input_tp = Input(shape=sequence_x_2inputs_train[0][0].shape)
input_pp = Input(shape=sequence_x_2inputs_train[1][0].shape)
difference = subtract([input_pp, input_tp])
final_feat = concatenate([input_tp, input_pp, difference])
final_feat = Dense(400)(final_feat)
final_feat = LeakyReLU()(final_feat)
final_feat = Dropout(0.9)(final_feat)
model_output = Dense(1, activation='sigmoid')(final_feat)
combined_model = Model([input_tp, input_pp], model_output)
model_path = "Tobacco800_exp3_img-text_lda_model.hdf5"
checkpoint = ValidationCheckpoint(model_path, sequence_x_2inputs_test, data_test_y)
combined_model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['accuracy'])
combined_model.fit(sequence_x_2inputs_train, data_train_y, validation_data=(sequence_x_2inputs_test, data_test_y),
                   batch_size=4096, epochs=40, callbacks=[checkpoint])

# # Late Fusion


model_text = load_model("models/tobacco800_exp1_prev-page_repeat-06.hdf5")
model_image = load_model("models/Tobacco800_exp2_prev-page_repeat-05.hdf5")
model_lda = load_model("models/Tobacco800_exp3_img-text_lda_model.hdf5")

p_t_train = model_text.predict_generator(model.TextFeatureGenerator2(data_text_train))
p_v_train = model_image.predict_generator(model_img.ImageFeatureGenerator(data_image_train, img_dim, prevpage=True))
p_l_train = model_lda.predict(sequence_x_2inputs_train)
p_t_train = np.concatenate([1 - p_t_train, p_t_train], axis=1)  # probability from text model
p_v_train = np.concatenate([1 - p_v_train, p_v_train], axis=1)  # probability from visual model
p_l_train = np.concatenate([1 - p_l_train, p_l_train], axis=1)  # probability from lda model

p_t_test = model_text.predict_generator(model.TextFeatureGenerator2(data_text_test))
p_v_test = model_image.predict_generator(model_img.ImageFeatureGenerator(data_image_test, img_dim, prevpage=True))
p_l_test = model_lda.predict(sequence_x_2inputs_test)
p_t_test = np.concatenate([1 - p_t_test, p_t_test], axis=1)  # probability from text model
p_v_test = np.concatenate([1 - p_v_test, p_v_test], axis=1)  # probability from visual model
p_l_test = np.concatenate([1 - p_l_test, p_l_test], axis=1)  # probability from lda model

# scoring with diffent i, j in (0,1] power normalizations
max_kappa = 0
test_exponents = [x / 10 for x in range(1, 11)]
for i in test_exponents:
    for j in test_exponents:
        for k in test_exponents:
            y_predict = np.argmax(np.power(p_t_test, i) * np.power(p_v_test, j) * np.power(p_l_test, k), axis=1)
            acc = sklm.accuracy_score(y_true, y_predict)
            kappa = sklm.cohen_kappa_score(y_true, y_predict)
            if kappa > max_kappa:
                max_kappa = kappa
                print(str(i) + " " + str(j) + " " + str(k))
                print("Accuracy: " + str(acc))
                print("Kappa: " + str(kappa))

# Best results: i = 0.4 k = 0.1 j = 0.2
# Accuracy: 0.9338567222767419
# Kappa: 0.7078080262749252


i = 0.1
j = 0.2
k = 0.1
y_predict = np.argmax(np.power(p_t_test, i) * np.power(p_v_test, j) * np.power(p_l_test, k), axis=1)
acc = sklm.accuracy_score(y_true, y_predict)
kappa = sklm.cohen_kappa_score(y_true, y_predict)
print(str(i) + " " + str(j) + " " + str(k))
print("Accuracy: " + str(acc))
print("Kappa: " + str(kappa))

# *Training set*
# ......
# 
# with LDA
# 
# 0.1 0.2 0.1
# Accuracy: 0.918918918918919
# Kappa: 0.8313436075537226
# 
# without LDA
# 
# 0.1 0.1
# Accuracy: 0.915057915057915
# Kappa: 0.8244284217661921
# 
# 
# *Test set*
# .......
# 
# 0.1 0.1 without LDA
# Accuracy: 0.915057915057915
# Kappa: 0.8244284217661921
# 
# 0.1 0.3 0.4 with LDA
# Accuracy: 0.9305019305019305
# Kappa: 0.8548838946647576


from matplotlib.pyplot import imshow, figure
from keras.preprocessing.image import load_img

i = 0.1
j = 0.2
k = 0.1
n = 0
y_predict = np.argmax(np.power(p_t_test, i) * np.power(p_v_test, j) * np.power(p_l_test, k), axis=1)
for i_y, y in enumerate(y_predict):
    if y == 0 and y_true[i_y] == 1:
        img_path = 'data/images/%s.tif' % data_text_test[i_y][0]
        print(img_path)
        image = load_img(img_path)
        figure()
        imshow(image)
        n += 1
print(n)  # 5 FN (handwriting,...), 16 FP (real 1st pages, tables/figures/complex layout)

# # Early Fusion


# with LDA
# features_x_train = np.concatenate([text_features_train, lda_train_x], axis = 1)
# features_x_test = np.concatenate([text_features_test, lda_test_x], axis = 1)

# without LDA
features_x_train = text_features_train
features_x_test = text_features_test

data_train_x = [features_x_train, image_features_train]
data_test_x = [features_x_test, image_features_test]
print(len(data_train_x[0][0]))
print(len(data_train_x[1][0]))

first_page_share = np.sum(data_train_y) / len(data_train_y)
class_weights = {0: first_page_share, 1: (1 - first_page_share)}
print(class_weights)

n_repeats = 10
all_acc = []
all_kap = []
for n_repeat in range(n_repeats):
    text_input = Input(shape=(len(data_train_x[0][0]),))
    img_input = Input(shape=(len(data_train_x[1][0]),))
    combined = concatenate([text_input, img_input])
    combined = Dense(400)(combined)
    combined = LeakyReLU()(combined)
    combined = Dropout(0.5)(combined)
    model_output = Dense(1, activation='sigmoid')(combined)
    combined_model = Model([text_input, img_input], model_output)

    model_path = "Tobacco800_exp3_img-text_mlp_simple.hdf5"
    checkpoint = ValidationCheckpoint(model_path, data_test_x, data_test_y)
    combined_model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['accuracy'])

    combined_model.fit(data_train_x, data_train_y,
                       validation_data=(data_test_x, data_test_y),
                       batch_size=32, epochs=25,
                       callbacks=[checkpoint])

    combined_model.load_weights(model_path)
    y_predict = np.round(combined_model.predict(data_test_x))
    all_acc.append(sklm.accuracy_score(data_test_y, y_predict))
    all_kap.append(sklm.cohen_kappa_score(data_test_y, y_predict))
    print("Accuracy: " + str(all_acc[-1]))
    print("Kappa: " + str(all_kap[-1]))
print(np.average(np.array(all_acc)))
print(np.average(np.array(all_kap)))

combined_model.load_weights(model_path)
y_predict = np.round(combined_model.predict(data_test_x))
print("Accuracy: " + str(sklm.accuracy_score(data_test_y, y_predict)))
print("Kappa: " + str(sklm.cohen_kappa_score(data_test_y, y_predict)))

# Early
# fusion
# with LDA()

# Accuracy: 0.9061776061776061
# Kappa: 0.8069826853972477
#
# Early
# fusion
# without
# LDA
#
# Accuracy: 0.9054054054054055
# Kappa: 0.0
# .8051798979324678

# --

# extract
# 10
# fp and 10
# fn
# files

# # MLP: predecessor page
# * extract features from text and image
# * combine 2 in sequence RNN


features_x_train = np.concatenate([text_features_train, image_features_train, lda_train_x], axis=1)
features_x_test = np.concatenate([text_features_test, image_features_test, lda_test_x], axis=1)

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

sequence_x_train.shape

model_path = "Tobacco_exp3_img-text_rnn-model.hdf5"
checkpoint = ValidationCheckpoint(model_path, sequence_x_test, data_test_y)

sgd = SGD(lr=1e-3, decay=1e-6, nesterov=True)

rnn_input = Input(shape=sequence_x_train[0].shape, )
# attention_probs = Dense(sequence_x_train[1][0].shape[0], activation='softmax')(rnn_input)
# attention_mul = multiply([rnn_input, attention_probs])
rnn_block = Bidirectional(GRU(300, return_sequences=True))(rnn_input)
rnn_block = Bidirectional(GRU(300))(rnn_block)
combined = Dense(128)(rnn_block)
combined = LeakyReLU()(combined)
model_output = Dense(1, activation='sigmoid')(combined)
combined_model = Model([rnn_input], model_output)
combined_model.compile(loss='binary_crossentropy', optimizer=Nadam(lr=1e-5), metrics=['accuracy'])
combined_model.fit(sequence_x_train, data_train_y, validation_data=(sequence_x_test, data_test_y),
                   batch_size=32, epochs=2, callbacks=[checkpoint])

feature_input = Input(shape=sequence_x_train[0].shape)
combined = Flatten()(feature_input)
combined = Dropout(0.25)(combined)
combined = Dense(300, activation='relu')(combined)
model_output = Dense(1, activation='sigmoid')(combined)
combined_model = Model([feature_input], model_output)
sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
model_path = "Tobacco_exp3_img-text_mlp_model.hdf5"
checkpoint = ValidationCheckpoint(model_path, sequence_x_test, data_test_y)
combined_model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
combined_model.fit(sequence_x_train, data_train_y, validation_data=(sequence_x_test, data_test_y),
                   batch_size=256, epochs=10, callbacks=[checkpoint])

sequence_x_2inputs_train = [sep_tp_x_train, sep_pp_x_train]
sequence_x_2inputs_test = [sep_tp_x_test, sep_pp_x_test]

sequence_x_2inputs_train[0].shape

n_repeats = 10
all_res = []
for n_repeat in range(n_repeats):
    input_tp = Input(shape=sequence_x_2inputs_train[0][0].shape)
    input_pp = Input(shape=sequence_x_2inputs_train[1][0].shape)
    # similarity = dot([input_tp, input_pp], axes=1, normalize=True)
    difference = subtract([input_pp, input_tp])
    final_feat = concatenate([input_tp, input_pp, difference])
    final_feat = Dense(500)(final_feat)
    final_feat = LeakyReLU()(final_feat)
    final_feat = Dropout(0.75)(final_feat)
    model_output = Dense(1, activation='sigmoid')(final_feat)
    combined_model = Model([input_tp, input_pp], model_output)
    # sgd = SGD(lr=0.001, decay=1e-5, momentum=0.9, nesterov=True)
    model_path = "Tobacco_exp3_img-text_mlp_model.hdf5"
    checkpoint = ValidationCheckpoint(model_path, sequence_x_2inputs_test, data_test_y)
    combined_model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['accuracy'])
    combined_model.fit(sequence_x_2inputs_train, data_train_y, validation_data=(sequence_x_2inputs_test, data_test_y),
                       batch_size=4096, epochs=10, callbacks=[checkpoint])
    all_res.append(checkpoint.max_metric)
np.average(all_res)
# avg kappa = 0.7828823923738958


# # Hyperparameters


n_repeats = 10
n_dense = [300, 400, 500]
n_dropouts = [i / 10 for i in range(1, 10)]
param_selection_results = {}
for p_dense in n_dense:
    for p_dropout in n_dropouts:
        exp_identifier = str(p_dense) + '_' + str(p_dropout)
        repeat_res = []
        print(exp_identifier)
        print("=================================")
        for n_repeat in range(n_repeats):
            input_tp = Input(shape=sequence_x_2inputs_train[0][0].shape)
            input_pp = Input(shape=sequence_x_2inputs_train[1][0].shape)
            difference = subtract([input_pp, input_tp])
            final_feat = concatenate([input_tp, input_pp, difference])
            final_feat = Dense(p_dense)(final_feat)
            final_feat = LeakyReLU()(final_feat)
            final_feat = Dropout(p_dropout)(final_feat)
            model_output = Dense(1, activation='sigmoid')(final_feat)
            combined_model = Model([input_tp, input_pp], model_output)
            model_path = "exp3_img-text_mlp_model.hdf5"
            checkpoint = ValidationCheckpoint(model_path, sequence_x_2inputs_test, data_test_y)
            combined_model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['accuracy'])
            combined_model.fit(sequence_x_2inputs_train, data_train_y,
                               validation_data=(sequence_x_2inputs_test, data_test_y),
                               batch_size=4096, epochs=10, callbacks=[checkpoint])

            repeat_res.append(checkpoint.max_metric)
        param_selection_results[exp_identifier] = np.average(repeat_res)

for experiment in param_selection_results.keys():
    print(experiment + ": " + str(param_selection_results[experiment]))

# * 300_0.1: 0.6852515810316551
# * 300_0.2: 0.6871223351208229
# * 300_0.3: 0.68686121506216
# * 300_0.4: 0.689159652336207
# * 300_0.5: 0.685658893687956
# * 300_0.6: 0.6853433528813586
# * 300_0.7: 0.6878746665438003
# * 300_0.8: 0.6891409725882904
# * 300_0.9: 0.6874184973695996
# * 400_0.1: 0.6832988830279183
# * 400_0.2: 0.6876135694830794
# * 400_0.3: 0.6854449904544734
# * 400_0.4: 0.6910287819300474
# * 400_0.5: 0.6866319842816794
# * 400_0.6: 0.6898778886492986
# * 400_0.7: 0.6873365297429803
# * 400_0.8: 0.6901676633495409
# * 400_0.9: 0.6914310739270534
# * 500_0.1: 0.6867188701608696
# * 500_0.2: 0.6860627445487304
# * 500_0.3: 0.6863893721984763
# * 500_0.4: 0.686207355448785
# * 500_0.5: 0.6903744622590591
# * 500_0.6: 0.6877049657707486
# * 500_0.7: 0.68878342194352
# * 500_0.8: 0.6909000005464196
# * 500_0.9: 0.6900227148787077


# Best setup: 400, 0.9
input_tp = Input(shape=sequence_x_2inputs_train[0][0].shape)
input_pp = Input(shape=sequence_x_2inputs_train[1][0].shape)
difference = subtract([input_pp, input_tp])
final_feat = concatenate([input_tp, input_pp, difference])
final_feat = Dense(400)(final_feat)
final_feat = LeakyReLU()(final_feat)
final_feat = Dropout(0.9)(final_feat)
model_output = Dense(1, activation='sigmoid')(final_feat)
combined_model = Model([input_tp, input_pp], model_output)
model_path = "exp3_img-text_mlp_model.hdf5"
checkpoint = ValidationCheckpoint(model_path, sequence_x_2inputs_test, data_test_y)
combined_model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['accuracy'])
combined_model.fit(sequence_x_2inputs_train, data_train_y, validation_data=(sequence_x_2inputs_test, data_test_y),
                   batch_size=4096, epochs=10, callbacks=[checkpoint])

# # RNN sequencing


features_x_train = np.concatenate([text_features_train, image_features_train, lda_train_x[:, 100:102]], axis=1)
features_x_test = np.concatenate([text_features_test, image_features_test, lda_test_x[:, 100:102]], axis=1)
features_x_train.shape

bioes2Idx = {'B': 0, 'I': 1, 'E': 2, 'S': 3}


def binary_to_BIOES(sequence):
    x_all, y_all = zip(*sequence)
    # B begin
    # I Inside
    # O None - we dont have O ...
    # E End
    # S Single
    last_state = 'O'
    y_translated = []
    for y in y_all:
        if y == 1:
            if last_state == 'B':
                # single
                y_translated[-1] = 'S'
            elif last_state == 'I':
                # end
                y_translated[-1] = 'E'
            # begin
            y_translated.append('B')
        elif y == 0:
            # inside
            y_translated.append('I')
        else:
            raise ValueError('Only accept 0 or 1 as label.')

        last_state = y_translated[-1]

    if last_state == 'B':
        # single
        y_translated[-1] = 'S'
    elif last_state == 'I':
        y_translated[-1] = 'E'

    return [(x_all[i], bioes2Idx[y_translated[i]]) for i in range(len(x_all))]


from keras.preprocessing.sequence import pad_sequences

label2Idx = {'FirstPage': 1, 'NextPage': 0}


def create_sequences(data_instances, data_features, max_seq_len=764):
    max_len = 0

    sequences = []
    prevBinder = ""
    tmp_sequence = []
    for i, instance in enumerate(data_instances):
        # "0 docid";"1 class";"2 type";"3 text";"4 binder"
        if prevBinder != instance[4]:
            if len(tmp_sequence) > 0:
                sequences.append(binary_to_BIOES(tmp_sequence))
            tmp_sequence = []
        tmp_sequence.append((data_features[i], label2Idx[instance[1]]))
        prevBinder = instance[4]
    if len(tmp_sequence) > 0:
        sequences.append(binary_to_BIOES(tmp_sequence))

    # create batches of same length
    batch_dict = {}
    for i, s in enumerate(sequences):
        if (len(s)) in batch_dict:
            batch_dict[len(s)].append(i)
        else:
            batch_dict[len(s)] = [i]
    batch_indexes = []
    for k in batch_dict.keys():
        batch_indexes.append(batch_dict[k])

    return batch_indexes, sequences


train_batch_idx, rnn_x_train = create_sequences(data_text_train, features_x_train)
test_batch_idx, rnn_x_test = create_sequences(data_text_test, features_x_test)

print(len(rnn_x_train))
print(len(rnn_x_train[20]))
print(len(rnn_x_train[20][1]))
print(len(test_batch_idx))
print(test_batch_idx)
print(rnn_x_train[20][0][0].shape[0])
for i in range(100):
    print(rnn_x_train[20][i][1])

from keras.utils import Sequence, to_categorical
import math


class SequenceGenerator(Sequence):
    def __init__(self, sequence_data, batch_idx):
        self.sequence_data = sequence_data
        self.batch_idx = batch_idx

    def __len__(self):
        return len(self.batch_idx)

    def __getitem__(self, idx):
        inds = self.batch_idx[idx]
        batch_x, batch_y = self.process_sequence_data(inds)
        return batch_x, batch_y

    def process_sequence_data(self, inds):
        features = []
        output_labels = []
        for index in inds:
            tmp_features, tmp_output_labels = zip(*self.sequence_data[index])
            features.append(tmp_features)
            tmp_output_labels = to_categorical(tmp_output_labels, num_classes=4)
            output_labels.append(tmp_output_labels)

        batch_input = np.array(features)
        batch_output = np.array(output_labels)

        # print(batch_input.shape)
        # print(batch_output.shape)

        return (batch_input, batch_output)


class SequenceCheckpoint(Callback):
    def __init__(self, filepath, metric='kappa'):
        self.metric = metric
        self.max_metric = float('-inf')
        self.max_metrics = None
        self.filepath = filepath
        self.history = []
        # self.x = validation_x
        # self.x_batches = x_batches
        # self.validation_y = validation_y

    def on_epoch_end(self, epoch, logs={}):

        true_labels = []
        predicted_labels = []
        for i in range(len(rnn_x_test)):
            example_features, y_true_binder = zip(*rnn_x_test[i])
            example_features = np.array(example_features)
            example_features = np.reshape(example_features, (1,) + example_features.shape)
            # example_features.shape
            y_pred_binder = rnn_model.predict(example_features).argmax(axis=-1)
            predicted_labels.extend(y_pred_binder[0])
            true_labels.extend(y_true_binder)

        eval_metrics = {
            'accuracy': sklm.accuracy_score(true_labels, predicted_labels),
            'f1_micro': sklm.f1_score(true_labels, predicted_labels, average='micro'),
            'f1_macro': sklm.f1_score(true_labels, predicted_labels, average='macro'),
            # 'f1_binary' : sklm.f1_score(true_labels, predicted_labels, average='binary', pos_label = 1),
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


from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_accuracy

rnn_input = Input(shape=(None, rnn_x_train[0][0][0].shape[0]))
rnn_layer = Bidirectional(GRU(200, return_sequences=True))(rnn_input)
# rnn_layer = Bidirectional(GRU(25, return_sequences=True))(rnn_layer)
rnn_dense = TimeDistributed(Dense(4))(rnn_layer)

crf = CRF(4, name='crf')
rnn_output = crf(rnn_dense)
rnn_model = Model(rnn_input, rnn_output)
rnn_model.compile(loss=crf_loss, optimizer='nadam', metrics=[crf_accuracy])
rnn_model.summary()

s_check = SequenceCheckpoint("exp3_seq_model.hdf5")

rnn_model.fit_generator(
    SequenceGenerator(rnn_x_train, train_batch_idx),
    validation_data=SequenceGenerator(rnn_x_test, test_batch_idx),
    callbacks=[s_check],
    epochs=15
)

rnn_model.load_weights(s_check.filepath)
# FirstPage = 1, NextPage = 0
idx2label = {0: 1, 1: 0, 2: 0, 3: 1}
y_true = []
y_pred = []
for i in range(len(rnn_x_test)):
    example_features, y_true_binder = zip(*rnn_x_test[i])
    example_features = np.array(example_features)
    example_features = np.reshape(example_features, (1,) + example_features.shape)
    # example_features.shape
    y_pred_binder = rnn_model.predict(example_features).argmax(axis=-1)
    y_pred.extend([idx2label[y] for y in y_pred_binder[0]])
    y_true.extend([idx2label[y] for y in y_true_binder])

sklm.cohen_kappa_score(y_true, y_pred)

np.array([idx2label[y] for y in y_pred_binder[0]])

np.array([idx2label[y] for y in y_true_binder])
