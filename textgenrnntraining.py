!pip install -q textgenrnn
#from google.colab import files
from textgenrnn import textgenrnn
from datetime import datetime
import os

model_cfg = {
    'word_level': False,   # set to True if want to train a word-level model (requires more data and smaller max_length)
    'rnn_size': 128,   # number of LSTM cells of each layer (128/256 recommended)
    'rnn_layers': 4,   # number of LSTM layers (>=2 recommended)
    'rnn_bidirectional': True,   # consider text both forwards and backward, can give a training boost
    'max_length': 10,   # number of tokens to consider before predicting the next (20-40 for characters, 5-10 for words recommended)
    'max_words': 10000,   # maximum number of words to model; the rest will be ignored (word-level model only)
}

train_cfg = {
    'line_delimited': False,   # set to True if each text has its own line in the source file
    'num_epochs': 20,   # set higher to train the model for longer
    'gen_epochs': 10,   # generates sample text from model after given number of epochs
    'train_size': 0.8,   # proportion of input data to train on: setting < 1.0 limits model from learning perfectly
    'dropout': 0.0,   # ignore a random proportion of source tokens each epoch, allowing model to generalize better
    'validation': True,   # If train__size < 1.0, test on holdout dataset; will make overall training slower
    'is_csv': False   # set to True if file is a CSV exported from Excel/BigQuery/pandas
}

#the following three lines are if you are using google colab
#uploaded = files.upload() 
#all_files = [(name, os.path.getmtime(name)) for name in os.listdir()]
#latest_file = sorted(all_files, key=lambda x: -x[1])[0][0]

file_name = "/Users/Luna Lovegood/Documents/deep learning/AUTHORS/AUTHORS/SHAKESPEARE/total shakespeare.txt" #change to use different training files
#os.mkdir(file_name)
model_name = '4layerBidirectional'   # change to set file name of resulting trained models/texts

textgen = textgenrnn(name=model_name)

train_function = textgen.train_from_file if train_cfg['line_delimited'] else textgen.train_from_largetext_file

train_function(
    file_path=file_name,
    new_model=True,
    num_epochs=train_cfg['num_epochs'],
    gen_epochs=train_cfg['gen_epochs'],
    batch_size=1024,
    train_size=train_cfg['train_size'],
    dropout=train_cfg['dropout'],
    validation=train_cfg['validation'],
    is_csv=train_cfg['is_csv'],
    rnn_layers=model_cfg['rnn_layers'],
    rnn_size=model_cfg['rnn_size'],
    rnn_bidirectional=model_cfg['rnn_bidirectional'],
    max_length=model_cfg['max_length'],
    dim_embeddings=100,
    word_level=model_cfg['word_level'])

print(textgen.model.summary())#creating a model summary

# changing the temperature schedule can result in wildly different output!
temperature = [1.0, 0.5, 0.2, 0.2]   
prefix = None   # if you want each generated text to start with a given seed text this is where I want the user input from the gui to go

if train_cfg['line_delimited']:
  n = 1000
  max_gen_length = 60 if model_cfg['word_level'] else 300
else:
  n = 1
  max_gen_length = 2000 if model_cfg['word_level'] else 10000
  

textgen.generate_samples()