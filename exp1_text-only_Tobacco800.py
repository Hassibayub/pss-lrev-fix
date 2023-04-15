#!/usr/bin/env python
# coding: utf-8

# In[1]:


import model
import fasttext as fastText
from keras.callbacks import ModelCheckpoint
from importlib import reload
import sklearn.metrics as sklm
import numpy as np


# In[2]:


# from keras.backend.tensorflow_backend import set_session
import tensorflow as tf

#get_ipython().run_line_magic('env', 'CUDA_DEVICE_ORDER=PCI_BUS_ID')
#get_ipython().run_line_magic('env', 'CUDA_VISIBLE_DEVICES=0')


# In[4]:


reload(model)


# In[5]:


data_train = model.read_csv_data("data/train.csv", csvformat="Tobacco800")
data_test = model.read_csv_data("data/test.csv", csvformat="Tobacco800")


# In[7]:


if 'ft' not in locals():
    ft = fastText.load_model("../embeddings/wiki.en.bin")
    model.ft = ft


# # Experiment: Simple CNN model (single page)
# * optimize for Kappa
# * 10 repeats

# In[11]:


n_repeats = 10
n_epochs = 20
exp_history = []
optimize_for = 'kappa'
for i in range(n_repeats):
    print("Repeat " + str(i+1) + " of " + str(n_repeats))
    print("-------------------------")
    model_singlepage = model.compile_model_singlepage()
    model_file = "tobacco800_exp1_single-page_repeat-%02d.hdf5" % (i,)
    print(model_file)
    checkpoint = model.ValidationCheckpoint(model_file, data_test, metric=optimize_for)
    model_singlepage.fit_generator(generator=model.TextFeatureGenerator(data_train),
                    callbacks = [checkpoint],
                    epochs = n_epochs)
    exp_history.append(checkpoint.max_metrics)

avg_result = sum([m['kappa'] for m in exp_history]) / n_repeats
avg_acc = sum([m['accuracy'] for m in exp_history]) / n_repeats
print("-------------------------")
print(avg_result)
print(avg_acc)


# In[12]:


# average: 0.6867630405833453
for i, r in enumerate(exp_history):
    model_file = "tobacco800_exp1_single-page_repeat-%02d.hdf5" % (i,)
    print(str(i) + ' ' + str(r['kappa']) + ' ' + str(r['accuracy']) + ' ' + str(r['f1_macro']) + ' ' + model_file)


# # Experiment: Predecessor page

# In[13]:


n_repeats = 10
n_epochs = 20
exp1b_history = []
optimize_for = 'kappa'
for i in range(n_repeats):
    print("Repeat " + str(i+1) + " of " + str(n_repeats))
    print("-------------------------")
    model_prevpage = model.compile_model_prevpage()
    model_file = "tobacco800_exp1_prev-page_repeat-%02d.hdf5" % (i,)
    print(model_file)
    checkpoint = model.ValidationCheckpoint(model_file, data_test, prev_page_generator = True, metric=optimize_for)
    model_prevpage.fit_generator(generator=model.TextFeatureGenerator2(data_train),
                    callbacks = [checkpoint],
                    epochs = n_epochs)
    exp1b_history.append(checkpoint.max_metrics)

avg_result = sum([m['kappa'] for m in exp1b_history]) / n_repeats
print("-------------------------")
print(avg_result)


# In[14]:


avg_kappa = sum([m['kappa'] for m in exp1b_history]) / n_repeats
avg_acc = sum([m['accuracy'] for m in exp1b_history]) / n_repeats
print("-------------------------")
print(avg_kappa)
print(avg_acc)


# In[15]:


for i, r in enumerate(exp1b_history):
    model_file = "tobacco800_exp1_prev-page_repeat-%02d.hdf5" % (i,)
    print(str(i) + ' ' + str(r['kappa']) + ' ' + str(r['accuracy']) + ' ' + model_file)


# In[16]:


# load best model
model_prevpage.load_weights("tobacco800_exp1_prev-page_repeat-06.hdf5")
y_predict = np.round(model_prevpage.predict_generator(model.TextFeatureGenerator2(data_test, batch_size=256)))
y_true = [model.label2Idx[x[1]] for x in data_test]
print("Accuracy: " + str(sklm.accuracy_score(y_true, y_predict)))
print("Kappa: " + str(sklm.cohen_kappa_score(y_true, y_predict)))


# In[ ]:




