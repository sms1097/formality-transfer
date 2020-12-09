import pickle
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.tokenize.casual import TweetTokenizer

class FormalityClassifier:
    def __init__(self):
        with open('metrics/formality-misc/formality-tokenizer.pickle', 'rb') as handle:
            self.tokenizer = pickle.load(handle)
        self.tweeter = TweetTokenizer()
        self.vocab_size = len(self.tokenizer.word_index) + 1
        
        self.model = tf.keras.Sequential([
            tf.keras.layers.Embedding(self.vocab_size, 200, mask_zero=True),
            tf.keras.layers.Dropout(0.8),  
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(1024, return_sequences=True)),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(1024)),
            tf.keras.layers.Dropout(0.8), 
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(2, activation='softmax')
        ])
        
        self.model.load_weights('metrics/formality-misc/formality_classifier')
        
        
    def _tokenize(self, corpus):
        """ Tokenize data and pad sequences """
        
        process_seq = lambda seq: '<start> ' + ' '.join(self.tweeter.tokenize(seq)) + ' <end>'
        x = [process_seq(seq) for seq in corpus]
        seqs = self.tokenizer.texts_to_sequences(x)
        padded_seqs = pad_sequences(seqs, padding='post')

        return padded_seqs,
        

    def classify(self, f_corpus):
        tokenized_corpus = self._tokenize(f_corpus)
        
        results = self.model.predict(tokenized_corpus)
        
        return np.mean(results[:, 1])
