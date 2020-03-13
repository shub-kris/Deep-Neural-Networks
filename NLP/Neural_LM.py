import numpy as np
import pandas as pd
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import LSTM, Dense, GRU, Embedding
from sklearn.model_selection import train_test_split

def open_file(filename):
    with open(filename, 'r') as f:
        text = f.read()
        
    text = text.lower()
    
    return text
    
    
def enocde_text(text):
    uniq_chars = sorted(list(set(text)))
    char_int = {ch: i for i, ch in enumerate(uniq_chars)}
    int_char = {i: ch for i, ch in enumerate(uniq_chars)}
    
    encoded_data = [char_int[ch] for ch in text]
    
    return char_int, int_char, encoded_data
    


# char_int, int_char, encoded_data = enocde_text(text)
# print(encoded_data)
    
def prepare_data(seq_length, vocab_size, text):
    X = []
    y = []
    n = len(text)
    
    
    for i in range(0, n - seq_length):
        input = text[i: i + seq_length]
        target = text[i + seq_length]
        X.append(input)
        y.append(target)
        
    X = np.array(X)
    y =  to_categorical(y, num_classes = vocab_size)
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size = 0.1, random_state=42)
    return X_tr, X_val, y_tr, y_val
        
# X, y = prepare_data(30, char_int, encoded_data)
# print(X[: 2], len(X[0]))


def build_model(vocab_size, hid_dim, seq_length):
    model = Sequential()
    model.add(Embedding(vocab_size, hid_dim, input_length = seq_length))
    model.add(LSTM(100, return_sequences = True))
    model.add(LSTM(100))
    model.add(Dense(vocab_size, activation  = 'softmax'))
    return model

def generate_text(model, char_int, seq_length, in_txt, n_chars):

    #Generate fixed number of chars
    for i in range(n_chars):
        encoded = [char_int[ch] for ch in in_txt]
        encoded = pad_sequences([encoded], maxlen = seq_length, truncating = 'pre')  #For a fixed length input and adds if less
        pred_class = model.predict_classes(encoded, verbose = 0)
        # print(pred_char, type(pred_char))
        out_ch = ''
        for ch, int in char_int.items():
            if int == pred_class:
                out_char = ch
                break
            in_txt += out_ch
    return in_txt


def main():
    text = open_file('decl_independance.txt')
    char_int, int_char, encoded_data = enocde_text(text)
    vocab_size = len(char_int)
    seq_length = 30
    X_tr, X_val, y_tr, y_val = prepare_data(seq_length, vocab_size, encoded_data)
    hid_dim = 50
    LM = build_model(len(char_int), hid_dim, seq_length)
    # print(LM.summary())
    LM.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer='adam')
    LM.fit(X_tr, y_tr, epochs = 100, verbose = 2, validation_data = (X_val, y_val))
    inp = 'We hold these truths'
    print(generate_text(LM, char_int, seq_length, inp, 30))


if __name__ == '__main__':
    main()

    