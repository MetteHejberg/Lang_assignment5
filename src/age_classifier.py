# import libraries
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

# utils
import utils.classifier_utils as clf

# scikit learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import classification_report

# tensorflow keras
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, 
                                    Flatten,
                                    Conv1D, 
                                    MaxPooling1D, 
                                    Embedding)
from tensorflow.keras.optimizers import Adam

# plotting function for loss and accuracy
def plot_history(H, eps):
    plt.style.use("seaborn-colorblind")

    # loss function plot
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.plot(np.arange(0, eps), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, eps), H.history["val_loss"], label="val_loss", linestyle=":")
    plt.title("Loss curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.legend()

    # accuracy plot
    plt.subplot(1,2,2)
    plt.plot(np.arange(0, eps), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, eps), H.history["val_accuracy"], label="val_acc", linestyle=":")
    plt.title("Accuracy curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.legend()
    plt.show()

# let's load the data
def load_clean_data():
    # get the paths
    paths = []
    directory_path = os.path.join("self_assigned", "data")
    # get the filenames
    filenames = os.listdir(directory_path)
    for file in filenames:
        # if a file doesn't end with .cha
        if not file.endswith (".cha"):
            # pass 
            pass
        # else
        else:
            # append the dictory and the filename to a list
            input_path = os.path.join(directory_path, file)
            paths.append(input_path)
    datasets = []
    for path in paths:
        data = pd.read_csv(path, sep="\t").reset_index()
        datasets.append(data)
    cleaned = []
    # for every dataset
    for dataset in datasets:
        # get language data
        text = list(dataset[dataset["index"]=="*CHI:"]["@UTF8"])
        # get metadata
        metadata = dataset[dataset["index"]=="@ID:"].iloc[0]
        # get the age
        age = metadata["@UTF8"].split("|")[3]
        for example in text:
            # append 
            cleaned.append((age, example))
    # transform to a dataframe
    cleaned_data = pd.DataFrame(cleaned)
    # use regex to clean the speech column
    cleaned_data[1] = cleaned_data[1].str.replace(r"( \.|\([a-z]{1,9}\)|\(\.\)|\?|\[: [a-z]{1,9}\]|\[= .*\]|\[\]|!|:|xxx|\[.*\])", "", regex=True)
    # use regex to clean the age column
    cleaned_data[0] = cleaned_data[0].str.replace(r"(\.[0-9]{1,4}|\;[0-9]{1,4})", "", regex=True)
    # rename columns
    # label = age 
    cleaned_data.rename(columns = {0:"label", 1:"text"}, inplace=True)
    # convert age column from string to float
    cleaned_data["label"] = cleaned_data["label"].astype(float)
    # print how many there is of each class 
    print("number of tokens from 1-year-olds:")
    print(len(cleaned_data[cleaned_data["label"]==1]))
    print("number of tokens from 2-year-olds:")
    print(len(cleaned_data[cleaned_data["label"]==2]))
    print("number of tokens from 3-year-olds:")
    print(len(cleaned_data[cleaned_data["label"]==3]))
    print("number of tokens from 4-year-olds:")
    print(len(cleaned_data[cleaned_data["label"]==4]))
    print("number of tokens from 5-year-olds:")
    print(len(cleaned_data[cleaned_data["label"]==5]))
    
    # I remove the data from the 1 and 5-year-olds because there's less data 
    cleaned_data.drop(cleaned_data[cleaned_data["label"] == 5].index, inplace = True)
    cleaned_data.drop(cleaned_data[cleaned_data["label"] == 1].index, inplace = True)
    # sort the age column in ascending order
    cleaned_data.sort_values(by=["label"])
    
    return cleaned_data

# balance data for second run of model
def balance(cleaned_data):
    # take 23019 samples (the number of tokens in the 4-year-old category)
    balanced_data = clf.balance(cleaned_data, 23019)
    return balanced_data

# build the model
def mdl(data, EPOCHS, BATCH_SIZE, plot_name, report_name):
    # create train-test-split
    X_train, X_test, y_train, y_test = train_test_split(data["text"], data["label"], test_size=0.33, random_state=42)
    # define out-of-vocabulary token
    t = Tokenizer(oov_token = '<UNK>')
              # oov = Out Of Vocabulary
              # UNK = Unknown
        
    # fit the tokenizer on the documents 
    t.fit_on_texts(X_train)

    # set padding value
    t.word_index["<PAD>"] = 0
            # PAD = padding 
    
    X_train_seqs = t.texts_to_sequences(X_train)
        # seqs = sequence
    
    X_test_seqs = t.texts_to_sequences(X_test)
    X_train_pad = sequence.pad_sequences(X_train_seqs, padding="post", maxlen = 44)
    X_test_pad = sequence.pad_sequences(X_test_seqs, padding = "post", maxlen = 44)

    # create one-hot encodings
    lb = LabelBinarizer()
    y_train_lb = lb.fit_transform(y_train)
    y_test_lb = lb.transform(y_test)
    # label encoder 
    le = LabelEncoder()
    y_train_le = le.fit_transform(y_train)
    y_test_le = le.transform(y_test)
    y_train = lb.fit_transform(y_train)
    y_test = lb.transform(y_test)
    
    # define parameters for model
    # overall vocabulary size
    VOCAB_SIZE = len(t.word_index)

    # number of dimensions for embeddings
    EMBED_SIZE = 300

    # create the model
    model = Sequential()
    # embedding layer
    model.add(Embedding(VOCAB_SIZE, 
                    EMBED_SIZE, 
                    input_length=44))

    # first convolution layer and pooling
    model.add(Conv1D(filters=128, 
                        kernel_size=4, 
                        padding='same',
                        activation='relu'))
    model.add(MaxPooling1D(pool_size=2))

    # second convolution layer and pooling
    model.add(Conv1D(filters=64, 
                        kernel_size=4, 
                        padding='same', 
                        activation='relu'))
    model.add(MaxPooling1D(pool_size=2))

    # third convolution layer and pooling
    model.add(Conv1D(filters=32, 
                        kernel_size=4, 
                        padding='same', 
                        activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    
    # define optimizer
    adam = Adam(learning_rate=0.00001)

    # fully-connected classification layer
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(3, activation='sigmoid'))
    model.compile(loss='categorical_crossentropy', 
                        optimizer=adam, 
                        metrics=['accuracy'])
    history = model.fit(X_train_pad, y_train_lb,
                    epochs = EPOCHS, # I suggest between 5 and 15
                    batch_size = BATCH_SIZE, # I suggest 128 or 64
                    validation_data = (X_test_pad, y_test_lb),
                    verbose = True)
   
    # plot loss and accuracy
    plot_history(history, EPOCHS)
    # save figure
    plt.savefig(os.path.join("out", plot_name))
    # clear figure
    plt.clf()
    
    # classification report
    # get the labels
    labels = ["2", "3", "4"]
    # get predictions
    predictions =  model.predict(X_test_pad)
    y_pred = np.argmax(predictions, axis=1)
    # create classification report
    report = classification_report(y_test_le, y_pred, target_names = labels)
    # print it
    print(report)
    p = os.path.join("out", report_name)
    # save the report
    with open(p, "w") as outfile:
        outfile.write(report)

def parse_args():
    # initialize argparse
    ap = argparse.ArgumentParser()
    # add command line parameters
    ap.add_argument("-e", "--EPOCHS", type=int, required=True, help="the number of epochs of the model")
    ap.add_argument("-b", "--BATCH_SIZE", type=int, required=True, help="the batch size of the model")
    args = vars(ap.parse_args())
    return args

# let's run the code
def main():
    args = parse_args()        
    cleaned_data = load_clean_data()
    print("[INFO]: training the model with unbalanced data")
    mdl(cleaned_data, args["EPOCHS"], args["BATCH_SIZE"], "unbalanced_model_plot.jpg", "unbalanced_model_report.txt")
    balanced_data = balance(cleaned_data)
    print("[INFO]: training the model with balanced data")
    mdl(balanced_data, args["EPOCHS"], args["BATCH_SIZE"], "balanced_model_plot.jpg", "balanced_model_report.txt")

if __name__ == "__main__":
    main()

