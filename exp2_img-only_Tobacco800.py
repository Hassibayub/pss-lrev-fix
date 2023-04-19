#!/usr/bin/env python
# coding: utf-8


import gc
import time
import model_img
import fasttext as fastText
from keras.callbacks import ModelCheckpoint
from importlib import reload
from keras import backend as K
import sklearn.metrics as sklm

# import tf.keras.backend.set_session as set_session
import tensorflow as tf

# get_ipython().run_line_magic('env', 'CUDA_DEVICE_ORDER=PCI_BUS_ID')
# get_ipython().run_line_magic('env', 'CUDA_VISIBLE_DEVICES=2')

# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# tf.keras.backend.set_session(tf.Session(config=config))


reload(model_img)

img_dim = (224, 224)
model_img.img_path_template = 'data/images/%s.tif'

data_train = model_img.read_csv_data("data/train.csv", csvformat="Tobacco800")
data_test = model_img.read_csv_data("data/test.csv", csvformat="Tobacco800")

model_singlepage = model_img.compile_model_singlepage(img_dim, print_summary=False)

# del model_singlepage
# model_singlepage.summary()


model_file = "Tobacco800_exp2_single-page_repeat-%02d.hdf5" % (0,)
checkpoint = model_img.ValidationCheckpoint(model_file, data_test[:200], img_dim, metric='kappa')
model_singlepage.fit_generator(generator=model_img.ImageFeatureGenerator(data_train[:1000], img_dim),
                               callbacks=[checkpoint],
                               epochs=5)

import numpy as np

np.round(model_singlepage.predict_generator(generator=model_img.ImageFeatureGenerator(data_train[:10], img_dim)))

# # Experiment: VGG CNN model (single page)
# * optimize for Kappa
# * 10 repeats


n_repeats = 10
n_epochs = 20
exp_history = []
optimize_for = 'kappa'
for i in range(n_repeats):
    print("Repeat " + str(i + 1) + " of " + str(n_repeats))
    print("-------------------------")

    if 'model_singlepage' in locals():
        del model_singlepage
        del checkpoint
        gc.collect()
        time.sleep(2)
        K.clear_session()
        time.sleep(2)

    model_singlepage = model_img.compile_model_singlepage(img_dim)
    model_file = "Tobacco800_exp2_img_repeat-%02d.hdf5" % (i,)
    checkpoint = model_img.ValidationCheckpoint(model_file, data_test, img_dim, metric=optimize_for)
    model_singlepage.fit_generator(
        generator=model_img.ImageFeatureGenerator(data_train, img_dim),
        callbacks=[checkpoint],
        epochs=n_epochs
    )

    #     model_slices = model_img.compile_model_singlepage_slices(img_dim, print_summary=False)
    #     model_file = "exp1_slices_repeat-%02d.hdf5" % (i,)
    #     checkpoint = model_img.ValidationCheckpoint(model_file, data_test, img_dim, prev_page_generator='slices', metric=optimize_for)
    #     model_slices.fit_generator(
    #         generator=model_img.ImageFeatureGenerator(data_train, img_dim, slices = True),
    #         callbacks = [checkpoint],
    #         epochs = n_epochs
    #     )

    with open("current_learning_progress.txt", "a") as pfile:
        pfile.write(str(i) + ": " + str(checkpoint.max_metrics) + '\n')

    exp_history.append(checkpoint.max_metrics)

avg_result = sum([m['kappa'] for m in exp_history]) / n_repeats
avg_acc = sum([m['accuracy'] for m in exp_history]) / n_repeats
print("-------------------------")
print(avg_result)
print(avg_acc)

for i, r in enumerate(exp_history):
    model_file = "Tobacco800_exp2_img_repeat-%02d.hdf5" % (i,)
    print(str(i) + ' ' + str(r['kappa']) + ' ' + str(r['accuracy']) + ' ' + model_file)
print("------------------------- VGG16 non-trainable")
print(avg_result)
print(avg_acc)

# # Experiment: Predecessor page

# *IMPORTANT RESULT*: 
# So far, it looks that predecessor image information does not contribute to PSS!


reload(model_img)
# model_img.img_path_template = 'data/Tobacco800/images/%s.tif.small.png'


n_repeats = 10
n_epochs = 15
exp2b_history = []
optimize_for = 'kappa'
for i in range(n_repeats):
    print("Repeat " + str(i + 1) + " of " + str(n_repeats))
    print("-------------------------")

    if 'model_prevpage' in locals():
        del model_prevpage
        del checkpoint
        gc.collect()
        time.sleep(2)
        K.clear_session()
        time.sleep(2)

    model_prevpage = model_img.compile_model_prevpage(img_dim)
    model_file = "Tobacco800_exp2_prev-page_repeat-%02d.hdf5" % (i,)
    checkpoint = model_img.ValidationCheckpoint(model_file, data_test, img_dim, prev_page_generator=True,
                                                metric=optimize_for)
    model_prevpage.fit_generator(generator=model_img.ImageFeatureGenerator(data_train, img_dim, prevpage=True),
                                 callbacks=[checkpoint],
                                 epochs=n_epochs)
    exp2b_history.append(checkpoint.max_metrics)

    with open("current_learning_progress.txt", "a") as pfile:
        pfile.write(str(i) + ": " + str(checkpoint.max_metrics) + '\n')

avg_kappa = sum([m['kappa'] for m in exp2b_history]) / n_repeats
avg_acc = sum([m['accuracy'] for m in exp2b_history]) / n_repeats
print("-------------------------")
print(avg_kappa)
print(avg_acc)

# -------------------------
# 0.7590367347473503
# 0.8849420849420848
for i, r in enumerate(exp2b_history):
    model_file = "Tobacco800_exp2_prev-page_repeat-%02d.hdf5" % (i,)
    print(str(i) + ' ' + str(r['kappa']) + ' ' + str(r['accuracy']) + ' ' + str(r['f1_macro']) + ' ' + model_file)

# load best model
model_prevpage.load_weights("Tobacco800_exp2_prev-page_repeat-05.hdf5")

y_predict = np.round(
    model_prevpage.predict_generator(model_img.ImageFeatureGenerator(data_test, img_dim, prevpage=True)))
y_true = [model_img.label2Idx[x[1]] for x in data_test]
print("Accuracy: " + str(sklm.accuracy_score(y_true, y_predict)))
print("Kappa: " + str(sklm.cohen_kappa_score(y_true, y_predict)))
