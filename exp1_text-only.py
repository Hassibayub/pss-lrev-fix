#!/usr/bin/env python
# coding: utf-8

# In[1]:


import model
import fastText
from keras.callbacks import ModelCheckpoint
from importlib import reload


# In[2]:


from keras.backend.tensorflow_backend import set_session
import tensorflow as tf

get_ipython().run_line_magic('env', 'CUDA_DEVICE_ORDER=PCI_BUS_ID')
get_ipython().run_line_magic('env', 'CUDA_VISIBLE_DEVICES=0')

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))


# In[8]:


reload(model)


# In[4]:


data_train = model.read_csv_data("data/archive20k/text/dataset.train")
data_test = model.read_csv_data("data/archive20k/text/dataset.test")


# In[5]:


# print(model.simple_tokenizer(data_train[12][2]))


# In[6]:


if 'ft' not in locals():
    ft = fastText.load_model("../embeddings/wiki.de.bin")
    model.ft = ft


# In[21]:


# model.vocab = {}
# model.vocab["PADDING_TOKEN"] = 0
# model.vocab["UNKNOWN_TOKEN"] = 1
# for d in data_train:
#     sentence = model.simple_tokenizer(d[2])
#     for w in sentence:
#         model.vocab[w] = len(model.vocab)
# # print(vocab)
# model.nb_words = len(model.vocab)
# print(model.nb_words)


# # Experiment: Simple CNN model (single page)
# * optimize for Kappa
# * 10 repeats

# In[9]:


n_repeats = 10
n_epochs = 10
exp_history = []
optimize_for = 'kappa'
for i in range(n_repeats):
    print("Repeat " + str(i+1) + " of " + str(n_repeats))
    print("-------------------------")
    model_singlepage = model.compile_model_singlepage()
    model_file = "exp1_single-page_repeat-%02d.hdf5" % (i,)
    print(model_file)
    checkpoint = model.ValidationCheckpoint(model_file, data_test, metric=optimize_for)
    model_singlepage.fit_generator(generator=model.TextFeatureGenerator(data_train),
                    callbacks = [checkpoint],
                    epochs = n_epochs)
    exp_history.append(checkpoint.max_metrics)

avg_result = sum([m['kappa'] for m in exp_history]) / n_repeats
print("-------------------------")
print(avg_result)


# In[14]:


# average: 0.5828771586417484, 0.6197080667414943
for i, r in enumerate(exp_history):
    model_file = "exp1_single-page_repeat-%02d.hdf5" % (i,)
    print(str(i) + ' ' + str(r['kappa']) + ' ' + str(r['accuracy']) + ' ' + str(r['f1_macro']) + ' ' + model_file)


# # Experiment: Predecessor page

# In[8]:


import os, warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings(action='once')


# In[ ]:


# [features_page, features_dot]
n_repeats = 10
n_epochs = 10
exp1b_history = []
optimize_for = 'kappa'
for i in range(n_repeats):
    print("Repeat " + str(i+1) + " of " + str(n_repeats))
    print("-------------------------")
    model_prevpage = model.compile_model_prevpage()
    model_file = "exp1_prev-page_repeat-%02d.hdf5" % (i,)
    print(model_file)
    checkpoint = model.ValidationCheckpoint(model_file, data_test, prev_page_generator = True, metric=optimize_for)
    model_prevpage.fit_generator(generator=model.TextFeatureGenerator2(data_train),
                    callbacks = [checkpoint],
                    epochs = n_epochs)
    exp1b_history.append(checkpoint.max_metrics)

avg_result = sum([m['kappa'] for m in exp1b_history]) / n_repeats
print("-------------------------")
print(avg_result)


# In[11]:


for i, r in enumerate(exp1b_history):
    model_file = "exp1_prev-page_repeat-%02d.hdf5" % (i,)
    print(str(i) + ' ' + str(r['kappa']) + ' ' + str(r['accuracy']) + ' ' + model_file)


# In[ ]:


# load best model
model.load_weights("exp1_prevapge_model.hdf5")
y_predict = model.predict_generator(TextFeatureGenerator2(data_test, batch_size=256)).argmax(axis=-1)
y_true = [label2Idx[x[1]] for x in data_test]
print("Accuracy: " + str(sklm.accuracy_score(y_true, y_predict)))
print("Kappa: " + str(sklm.cohen_kappa_score(y_true, y_predict)))

