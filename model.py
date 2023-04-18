import csv, re, math
import sklearn.metrics as sklm
import fasttext as fastText
import numpy as np
import cv2 as cv
import os 
import json

import pandas as pd
from sklearn.model_selection import train_test_split

from keras.utils import Sequence
from keras.models import Sequential, Model
from keras.layers import *
from keras.utils import *
from keras.callbacks import ModelCheckpoint, Callback

nb_embedding_dims = 300 # ft.get_dimension()
nb_sequence_length = 150
word_vectors_ft = {}
label2Idx = {'FirstPage' : 1, 'NextPage' : 0}


def preprocess_raw_data(data_path: str) -> None:
    # load tiff images in a list
    text_data_path = os.path.join(data_path, "ocr_text")
    images_data_path = os.path.join(data_path, "images")
    
    print("Loading images from: " + images_data_path)
    images = {}
    for filename in os.listdir(images_data_path):
        if filename.endswith(".tif"):
            img = cv.imread(os.path.join(images_data_path, filename), cv.IMREAD_GRAYSCALE)
            img = cv.resize(img, (224, 224))
            img = cv.GaussianBlur(img, (5, 5), 0)
            img = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]
            filename_split = filename.split('.')[0]
            images[filename_split] = img
    
    # load text data
    print("Loading text data from: " + text_data_path)
    text_data = {}
    for filename in os.listdir(text_data_path):
        if filename.endswith(".json"):
            with open(os.path.join(text_data_path, filename), 'r') as f:
                json_data = json.load(f)
                
                all_words = []
                for word in json_data['analyzeResult']['readResults'][0]['lines']:
                    all_words.extend(word['text'].split())
                word_text = ' '.join(all_words)
                
                filename_split = filename.split('.')[0]
                text_data[filename_split] = word_text
    
    print(len(images), len(text_data))
    # print(images[0], text_data[0])
    # merge two dictionaries into one based on the key
    data_combine = {k: (images[k], text_data[k]) for k in images.keys()}
    df = pd.DataFrame(data_combine).T.rename(columns={0: 'image', 1: 'text'})
    print(df.head())
    
def preprocess_raw_data2(data_path: str) -> None:
    """Preprocess raw data and save it as a csv file"""
    images_dir = os.path.join(data_path, "images")
    ocr_dir = os.path.join(data_path, "ocr_text")
    xml_dir = os.path.join(data_path, "xml")
    
    
    print("Loading images from: " + images_dir)
    data = []

    for filename in os.listdir(images_dir):

        img_path = os.path.join(images_dir, filename)
        ocr_path = os.path.join(ocr_dir, filename.split('.')[0] + ".json")
        
        img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
        with open(ocr_path, 'r') as f:
            json_data = json.load(f)
            # all_words = []
            # for word in json_data['analyzeResult']['readResults'][0]['lines']:
                # all_words.extend(word['text'].split())
            # word_text = ' '.join(all_words)
            
        data.append((filename, img, json_data))
    
    # extract text region from ocr json
    print("Extracting text regions from OCR data")
    text_regions = []
    
    for filename, img, ocr_data in data:
        for line in ocr_data['analyzeResult']['readResults'][0]['lines']:
            for word in line['words']:
                bounding_box = [int(x) for x in word['boundingBox']]
                left = bounding_box[0]
                top = bounding_box[1]
                width = bounding_box[2] - bounding_box[0]
                height = bounding_box[5] - bounding_box[1]
                text = word['text']
                
                text = re.sub(r',', '', text) # remove commas 
                text = re.sub(r';', '', text) # remove semicolons
                 
                text_regions.append((filename, left, top, width, height, text))
    
    # assign type 
    print("Assigning type to text regions")
    labeled_regions = []
    for filename, left, top, width, height, text in text_regions:
        docid = filename.split('.')[0]
        class_label = 'paper'
        type_label = assign_type(left, top, width, height)
        labeled_regions.append((docid, class_label, type_label, text))
    
    # Add binder column
    print("Adding binder column")
    binder = "mybinder"
    labeled_regions_with_binder = [(binder, docid, class_label, type_label, text) for docid, class_label, type_label, text in labeled_regions]

    
    # write CSV file
    print("Writing CSV file")
    # import csv 
    csv_path = "data/dataset.csv"
    df = pd.DataFrame(labeled_regions_with_binder, columns=['binder', 'docid', 'class', 'type', 'text'])
    df.to_csv(csv_path, sep=';', index=False)

    # with open(csv_path, 'w', newline='') as f:
        # writer = csv.writer(f, delimiter=';')
        # writer.writerow(['binder', 'docid', 'class', 'type', 'text'])
        # writer.writerows(labeled_regions_with_binder)

def preprocess_raw_data3(data_path: str) -> None:
    """Preprocess raw data and save it as a csv file"""

    images_dir = os.path.join(data_path, "images")
    text_data_path = os.path.join(data_path, "ocr_text")
    
    # load filename without extension
    files_no_ext = load_files_no_extension(images_dir)

    df = pd.DataFrame(columns=['binder', 'docid', 'class', 'type', 'text'])
    df['docid'] = files_no_ext
    df.set_index('docid', inplace=True)
    
    
    # load text data from ocrtxt files
    print("Loading text data from: " + text_data_path)
    for file in files_no_ext:
        ocr_file_path = os.path.join('data', 'ocr_text', file + '.json')
        
        with open(ocr_file_path, 'r') as f:
            json_data = json.load(f)

        json_data_lines = json_data['analyzeResult']['readResults'][0]['lines']
        
        page_text = []
        for line in json_data_lines:
            page_text.append(line['text'])
        
        page_text = ' '.join(page_text)
        
        df.loc[file, 'text'] = page_text
    
    # Adding const values to df
    print("Adding const values to df")
    df['class'] = df.apply(assign_class_label, axis=1)
    df['type'] = 'paper'
    df['binder'] = 'mybinder'
    
    # replace semicolons and commas from df['text']
    df['text'] = df['text'].str.replace(r',', '').str.replace(r';', '')
    
    df.reset_index(inplace=True)
    
    # export csv
    print("Exporting csv file")
    df.to_csv('data/dataset.csv', sep=';', index=False)
    

def assign_class_label(row: str) -> str:
    # get index value
    docid = row.name
    res_group = re.search(r'[-_]\d$', docid)
    if res_group:
        res = res_group.group(0)
        page_no = re.sub(r'[-_]', '', res)

        if int(page_no) > 1:
            return 'NextPage'
        else:
            return 'FirstPage'
    else:
        return 'FirstPage'
        
    
    
def load_files_no_extension(data_path: str) -> list:
    print("Loading filenames without extension")
    filenames = []
    for filename in os.listdir(data_path):
        filenames.append(filename.split('.')[0])
    
    return filenames
    
def assign_type(left, top, width, height):
    """Assign type to text region"""
    if height > 50:
        return "heading"
    elif width > 300:
        return "caption"
    else:
        return "body"


def split_csv_into_80_20(csv_path):
    """Split the csv file into train and test sets"""
    print("Splitting CSV file into train and test sets")
    df = pd.read_csv(csv_path, delimiter=';')
    train, test = train_test_split(df, test_size=0.2)
    train.to_csv("data/train.csv", index=False, sep=';')
    test.to_csv("data/test.csv", index=False, sep=';')

    
def simple_tokenizer(textline):
    textline = re.sub(r'http\S+', 'URL', textline)
    words = re.compile(r'[#\w-]+|[^#\w-]+', re.UNICODE).findall(textline.strip())
    words = [w.strip() for w in words if w.strip() != '']
    # print(words)
    return(words)


def read_csv_data(csvfile, csvformat = "archive20k", return_DC = False):
    
    data_instances = []
    instance_ids = []
    current_id = 0
    
    prevBinder = ""
    prevPageText = ""
    prevPageClass = ""
    
    data = pd.read_csv(csvfile, delimiter=';', skiprows=1, header=None, encoding='UTF-8')
    for instance in data.itertuples():
        # "binder";"docid";"class";"type";"text"
        if prevBinder == instance[1]:
            prevPage = prevPageText
        else:
            prevPage = ""
                
        if csvformat == "Tobacco800":
            data_instances.append([instance[2], instance[3], instance[4], prevPage, instance[1]])
            prevBinder = instance[1]
            prevPageText = instance[4]
            prevPageClass = instance[3]
        else:
            if return_DC:
                if instance[3] == "FirstPage" or prevPageClass == "FirstPage":
                    data_instances.append([instance[2], instance[3], instance[5], prevPage, instance[1], instance[4]])
                    instance_ids.append(current_id)
            else:
                data_instances.append([instance[2], instance[3], instance[5], prevPage, instance[1]])
            prevBinder = instance[1]
            prevPageText = instance[5]
            prevPageClass = instance[3]
                
        current_id += 1
        
    if len(instance_ids) > 0:
        return data_instances, instance_ids
    else:
        return data_instances

    
    
def _read_csv_data(csvfile, csvformat = "archive20k", return_DC = False):
    
    data_instances = []
    instance_ids = []
    current_id = 0
    
    prevBinder = ""
    prevPageText = ""
    prevPageClass = ""
    
    with open(csvfile, 'r', encoding='UTF-8') as f:
        datareader = csv.reader(f, delimiter=';')
        next(datareader)
        for instance in datareader:
            # "binder";"docid";"class";"type";"text"
            if prevBinder == instance[0]:
                prevPage = prevPageText
            else:
                prevPage = ""
                
            if csvformat == "Tobacco800":
                data_instances.append([instance[1], instance[2], instance[3], prevPage, instance[0]])
                prevBinder = instance[0]
                prevPageText = instance[3]
                prevPageClass = instance[2]
            else:
                if return_DC:
                    if instance[2] == "FirstPage" or prevPageClass == "FirstPage":
                        data_instances.append([instance[1], instance[2], instance[4], prevPage, instance[0], instance[3]])
                        instance_ids.append(current_id)
                else:
                    data_instances.append([instance[1], instance[2], instance[4], prevPage, instance[0]])
                prevBinder = instance[0]
                prevPageText = instance[4]
                prevPageClass = instance[2]
                
            current_id += 1
    if len(instance_ids) > 0:
        return data_instances, instance_ids
    else:
        return data_instances



class TextFeatureGenerator(Sequence):
    def __init__(self, text_data, batch_size = 32):
        self.text_data = text_data
        self.indices = np.arange(len(self.text_data))
        self.batch_size = batch_size
        self.kappa = []
        self.accuracy = []

    def __len__(self):
        return math.ceil(len(self.text_data) / self.batch_size)
    
    def on_epoch_end(self):
        # print("Shuffling ....")
        np.random.shuffle(self.indices)

    def __getitem__(self, idx):
        inds = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x, batch_y = self.process_text_data(inds)
        return batch_x, batch_y

    def process_text_data(self, inds):

        word_embeddings = []
        output_labels = []

        for index in inds:
            
            temp_word = []
            
            # tokenize
            sentence = simple_tokenizer(self.text_data[index][2])
            temp_output = label2Idx[self.text_data[index][1]]
            
            # trim to max sequence length
            if (len(sentence) > nb_sequence_length):
                half_idx = int(nb_sequence_length / 2)
                tmp_sentence = sentence[:half_idx]
                tmp_sentence.extend(sentence[(len(sentence) - half_idx):])
                sentence = tmp_sentence

            # padding
            words_to_pad = nb_sequence_length - len(sentence)

            for i in range(words_to_pad):
                sentence.append('PADDING_TOKEN')

            # create data input for words
            for w_i, word in enumerate(sentence):

                if word == 'PADDING_TOKEN':
                    word_vector = [0] * nb_embedding_dims
                else:
                    word_vector = ft.get_word_vector(word.lower())
                temp_word.append(word_vector)

            word_embeddings.append(temp_word)
            # temp_output = to_categorical(temp_output, len(label2Idx))
            output_labels.append(temp_output)

        return ([np.array(word_embeddings)], np.array(output_labels))

    
class TextFeatureGenerator2(Sequence):
    def __init__(self, text_data, batch_size = 32):
        self.text_data = text_data
        self.indices = np.arange(len(self.text_data))
        self.batch_size = batch_size
        self.kappa = []
        self.accuracy = []

    def __len__(self):
        return math.ceil(len(self.text_data) / self.batch_size)
    
    def on_epoch_end(self):
        # print("Shuffling ....")
        np.random.shuffle(self.indices)

    def __getitem__(self, idx):
        inds = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x, batch_y = self.process_text_data(inds)
        return batch_x, batch_y

    def process_text_data(self, inds):

        word_embeddings = []
        prev_embeddings = []
        output_labels = []

        for index in inds:
            
            word_embeddings.append(self.text_to_embedding(self.text_data[index][2]))
            prev_embeddings.append(self.text_to_embedding(self.text_data[index][3]))
            
            temp_output = label2Idx[self.text_data[index][1]]
            # temp_output = to_categorical(temp_output, len(label2Idx))
            output_labels.append(temp_output)

        return ([np.array(word_embeddings), np.array(prev_embeddings)], np.array(output_labels))
    
    def text_to_embedding(self, textsequence):
        temp_word = []
            
        # tokenize
        sentence = simple_tokenizer(textsequence)
        
        # trim to max sequence length
        if (len(sentence) > nb_sequence_length):
            half_idx = int(nb_sequence_length / 2)
            tmp_sentence = sentence[:half_idx]
            tmp_sentence.extend(sentence[(len(sentence) - half_idx):])
            sentence = tmp_sentence

        # padding
        words_to_pad = nb_sequence_length - len(sentence)

        for i in range(words_to_pad):
            sentence.append('PADDING_TOKEN')

        # create data input for words
        for w_i, word in enumerate(sentence):

            if word == 'PADDING_TOKEN':
                word_vector = [0] * nb_embedding_dims
            else:
                word_vector = ft.get_word_vector(word.lower())
            
            temp_word.append(word_vector)
            
        return temp_word    
    

def compile_model_singlepage(print_summary = False):
    model_input_ft = Input(shape = (nb_sequence_length, nb_embedding_dims))
    
    gru_block = Bidirectional(GRU(128, dropout = 0.5, return_sequences=True))(model_input_ft)
    
    filter_sizes = (3, 4, 5)
    conv_blocks = []
    for sz in filter_sizes:
        conv = Conv1D(
            filters = 200,
            kernel_size = sz,
            padding = "same",
            strides = 1
        )(gru_block)
        conv = LeakyReLU()(conv)
        conv = GlobalMaxPooling1D()(conv)
        conv = Dropout(0.5)(conv)
        conv_blocks.append(conv)
    model_concatenated = concatenate(conv_blocks)
    model_concatenated = Dense(128)(model_concatenated)
    model_concatenated = LeakyReLU()(model_concatenated)
    model_output = Dense(1, activation = "sigmoid")(model_concatenated)
    model = Model([model_input_ft], model_output)
    model.compile(loss='binary_crossentropy', optimizer='nadam', metrics = ['accuracy'])
    if print_summary:
        model.summary()
    return model

def compile_model_prevpage(include_top = True, print_summary = False):
    
    filter_sizes = (3, 4, 5)
    
    model_input_tp = Input(shape = (nb_sequence_length, nb_embedding_dims))    
    gru_block_tp = Bidirectional(GRU(128, dropout = 0.5, return_sequences=True))(model_input_tp)
    conv_blocks_tp = []
    for sz in filter_sizes:
        conv = Conv1D(
            filters = 200,
            kernel_size = sz,
            padding = "same",
            strides = 1
        )(gru_block_tp)
        conv = LeakyReLU()(conv)
        conv = GlobalMaxPooling1D()(conv)
        conv = Dropout(0.5)(conv)
        conv_blocks_tp.append(conv)
    model_concatenated_tp = concatenate(conv_blocks_tp)
    model_concatenated_tp = Dense(128)(model_concatenated_tp)
    model_concatenated_tp = LeakyReLU()(model_concatenated_tp)
    
    model_input_pp = Input(shape = (nb_sequence_length, nb_embedding_dims))    
    gru_block_pp = Bidirectional(GRU(128, dropout = 0.5, return_sequences=True))(model_input_pp)
    conv_blocks_pp = []
    for sz in filter_sizes:
        conv = Conv1D(
            filters = 200,
            kernel_size = sz,
            padding = "same",
            strides = 1
        )(gru_block_pp)
        conv = LeakyReLU()(conv)
        conv = GlobalMaxPooling1D()(conv)
        conv = Dropout(0.5)(conv)
        conv_blocks_pp.append(conv)
    model_concatenated_pp = concatenate(conv_blocks_pp)
    model_concatenated_pp = Dense(128)(model_concatenated_pp)
    model_concatenated_pp = LeakyReLU()(model_concatenated_pp)

    # concat both + another dense
    page_sequence = concatenate([model_concatenated_tp, model_concatenated_pp])
    page_sequence = Dense(256)(page_sequence)
    page_sequence = LeakyReLU()(page_sequence)
    
    if include_top:
        # prediction layer
        model_output = Dense(1, activation = "sigmoid")(page_sequence)

        # combine final model
        model = Model([model_input_tp, model_input_pp], model_output)
        model.compile(loss='binary_crossentropy', optimizer='nadam', metrics = ['accuracy'])
        
        if print_summary:
            model.summary()
        
        return model
    else:
        model = Model([model_input_tp, model_input_pp], page_sequence)
        
        if print_summary:
            model.summary()
        
        return model_input_tp, model_input_pp, model
        



def predict(model, data, prev_page_generator = False, batch_size=256):
    if prev_page_generator:
        y_predict = np.round(model.predict_generator(TextFeatureGenerator2(data, batch_size=batch_size)))
    else:
        y_predict = np.round(model.predict_generator(TextFeatureGenerator(data, batch_size=batch_size)))
    return y_predict


class ValidationCheckpoint(Callback):
    def __init__(self, filepath, test_data, prev_page_generator = False, metric = 'kappa'):
        self.test_data = test_data
        self.metric = metric
        self.max_metric = float('-inf')
        self.max_metrics = None
        self.filepath = filepath
        self.history = []
        self.prev_page_generator = prev_page_generator

    def on_epoch_end(self, epoch, logs={}):
        
        predicted_labels = predict(self.model, self.test_data, self.prev_page_generator)
        true_labels = [label2Idx[x[1]] for x in self.test_data]

        eval_metrics = {
            'accuracy' : sklm.accuracy_score(true_labels, predicted_labels),
            'f1_micro' : sklm.f1_score(true_labels, predicted_labels, average='micro'),
            'f1_macro' : sklm.f1_score(true_labels, predicted_labels, average='macro'),
            'f1_binary' : sklm.f1_score(true_labels, predicted_labels, average='binary', pos_label = 1),
            'kappa' : sklm.cohen_kappa_score(true_labels, predicted_labels)
        }
        eval_metric = eval_metrics[self.metric]
        self.history.append(eval_metric)
        
        if epoch > -1 and eval_metric > self.max_metric:
            print("\n" + self.metric + " improvement: " + str(eval_metric) + " (before: " + str(self.max_metric) + "), saving to " + self.filepath)
            self.max_metric = eval_metric     # optimization target
            self.max_metrics = eval_metrics   # all metrics
            self.model.save(self.filepath)



if __name__ == "__main__":
    data_path = 'data/'
    csv_path = "data/dataset.csv"
    preprocess_raw_data3(data_path)
    split_csv_into_80_20(csv_path)