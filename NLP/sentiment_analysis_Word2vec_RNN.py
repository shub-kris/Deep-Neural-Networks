import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, GRU, Embedding, Input
from sklearn.model_selection import train_test_split
from gensim import models
from keras.preprocessing.text import Tokenizer
from keras.initializers import Constant

def clean_text(text):
    text = re.sub('[\W_]+', " ", text)
    text = text.lower()
    return text
    
# print(y[:5])

def remove_stopwords(data):
    tokenized_data = [sent.split() for sent in data]
    stop_words = stopwords.words('english')
    # print(stop_words)
    new_X = [[wd for wd in sent if wd not in stop_words] for sent in tokenized_data]
    X = [' '.join(sent) for sent in new_X]
    return X

'''
Word2vec Embeddings for the training_data
'''

def get_word_embeddings(word2vec, train_word_index, dim):
    embd_matrix = np.zeros((len(train_word_index)+ 1, dim))
    for wd, index in train_word_index.items():
        embd_matrix[index, :] = word2vec[wd] if wd in word2vec else np.random.rand(dim)
    return embd_matrix
 
def load_word2vec(path):
    embedding = models.KeyedVectors.load_word2vec_format(path, binary = True)
    return embedding
 
def build_sentiment_model(embd_matrix, seq_length, vocab_size, word_dim):
    input = Input(shape=(seq_length,), dtype='int32')
    embd_layer = Embedding(vocab_size, word_dim, weights = [embd_matrix], input_length = seq_length, trainable = False)
    embd_ip = embd_layer(input)
    gru = GRU(units = 32, dropout = 0.2, recurrent_dropout = 0.2)(embd_ip)
    op = Dense(1, activation = 'sigmoid')(gru)
    model = Model(input, op)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])
    print(model.summary())
    return model


def main():
    imdb = pd.read_csv('imdb_labelled.tsv.txt', header = None, delimiter = '\t', names = ['Reviews', 'Labels'])
    X = imdb['Reviews']
    y = imdb['Labels']
    
    X = X.apply(lambda x: clean_text(x))
    X = remove_stopwords(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15)
    Y_train, Y_test = y_train.values, y_test.values

    train_vocab = set([wd for sent in X_train for wd in sent])
    n_uniq_words = len(train_vocab)
    
    seq_length = 50
    word_dim = 300
    num_epochs = 100
    batch_size = 64
    '''Tokenizer and Padding'''
    tokenizer = Tokenizer(num_words = n_uniq_words)  # vectorize a text corpus, by turning each text into a sequence of integers
    tokenizer.fit_on_texts(X_train)
    train_seq = tokenizer.texts_to_sequences(X_train)
    train_word_index = tokenizer.word_index
    vocab_size = len(train_word_index) + 1
    train_data = pad_sequences(train_seq, maxlen = seq_length)
    
    test_seq = tokenizer.texts_to_sequences(X_test)
    test_data = pad_sequences(test_seq, maxlen = seq_length)
    
    wd_emb = load_word2vec('GoogleNews-vectors-negative300.bin.gz')
    embd_matrix = get_word_embeddings(wd_emb, train_word_index, word_dim)
    
    print(embd_matrix.shape)
    
    model = build_sentiment_model(embd_matrix, seq_length, vocab_size, word_dim)
    
    model.fit(train_data, Y_train, epochs = num_epochs, validation_split = 0.2, shuffle = True, batch_size = batch_size)
    


if __name__ == '__main__':
    main()

    