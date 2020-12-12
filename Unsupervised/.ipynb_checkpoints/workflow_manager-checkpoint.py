import numpy as np
import tensorflow as tf

import re 
import os
from datetime import datetime

from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import TweetTokenizer

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

    
BATCH_SIZE = 64
EMBEDDING_DIM = 200

class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, encoder_units, weight_matrix):
        super(Encoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, 
                                                   embedding_dim,
                                                   weights=[weight_matrix])
        self.lstm_1 = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                encoder_units,
                return_sequences=True,
                return_state=True,
                kernel_regularizer=tf.keras.regularizers.l2(0.01),
                recurrent_regularizer=tf.keras.regularizers.l2(0.01),
                bias_regularizer=tf.keras.regularizers.l2(0.01)
            )
        )

    def call(self, x, hidden_state):
        """Shove into latent space"""
        # x shape: (batch_size x max_length x embedding_dim)
        x = self.embedding(x)

        # output shape: (batch_size x max_length x 2 * encoder_units)
        # h_f, h_b shapes: (batch_size x encoder_units)
        output, h_f, _, h_b, _ = self.lstm_1(x, initial_state=hidden_state)

        return output, h_f, h_b 
    
    
class GlobalAttention(tf.keras.layers.Layer):
    """
    This is called GlobalAttention since that's what
    Bahdanau attention is called in the Luong paper
    and most of this implementation follows that 
    scaffolding. 

    The difference between this and global attention
    is concatenating the forward and backward hidden
    states from the 
    """
    def __init__(self, units):
        """These parameters follow notation from Bahdanau paper"""
        super(GlobalAttention, self).__init__()
        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units)
        self.v = tf.keras.layers.Dense(1)

    def call(self, enc_opt, hidden_f, hidden_b):
        # concatenate hidden states as in eq 7 Bahdanau
        # hidden shape: (batch_size, 2 * lstm_units)
        hidden = tf.concat([hidden_f, hidden_b], axis=-1)

        # expand dims to meet shape of latent tensor
        # hidden_broad shape: (batch_size, 1, encoder_units * 2)
        hidden_broad = tf.expand_dims(hidden, 1)

        # Alignment model score from A.1.2 of Bahdanau et al 
        # score shape: (batch_size, max_length, v_units)
        score = self.v(tf.nn.tanh(self.W(hidden_broad) + self.U(enc_opt)))

        # softmax generalization of eq(7) Luong
        # attention_weights shape: (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)  

        # This takes weighted average with attention weights
        # context shape: (batch_size, 2 * encoder_units)
        context = attention_weights * enc_opt
        context = tf.reduce_sum(context, axis=1)

        return context, attention_weights

    
class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, attention_units, 
                 decoder_units):
        super(Decoder, self).__init__()

        self.attention = GlobalAttention(attention_units)
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.lstm_1 = tf.keras.layers.LSTM(decoder_units,
                                           return_sequences=True,
                                           return_state=True)
        self.flatten = tf.keras.layers.Flatten()
        self.opt = tf.keras.layers.Dense(vocab_size)

    def call(self, x, hidden_f, hidden_b, encoder_output):
        context_vector, attention_weights = self.attention(encoder_output, 
                                                           hidden_f, hidden_b)

        # x shape: (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # x shape: (batch_size, 1, embedding_dim + 2 * encoder_units)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # output shape: (batch_size, 1, decoder_units)
        # this shape is only important for expanding decoder depth
        output, h_f, c_f = self.lstm_1(x)

        # flatten to feed into opt
        # output shape: (batch_size, hidden_size)
        output = self.flatten(output)

        # get logits
        # x shape: (batch_size, vocab)
        x = self.opt(output)

        return x, h_f, attention_weights

    
def Discriminator(vocab_size, embedding_dim, weights):
    return tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, EMBEDDING_DIM, weights=[weights], mask_zero=True),
        tf.keras.layers.Dropout(0.8),  
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(1024, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(1024)),
        tf.keras.layers.Dropout(0.8), 
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(2, activation='softmax')
    ])

def tokenize(corpus, tokenizer=None, maxlen=None):
    """ Tokenize data and pad sequences """
    if not tokenizer: 
        tokenizer = Tokenizer(filters='', 
                              oov_token='<OOV>',
                              lower=False)
        tokenizer.fit_on_texts(corpus)
        
    seqs = tokenizer.texts_to_sequences(corpus)
    padded_seqs = pad_sequences(seqs, padding='post', maxlen=maxlen)

    return padded_seqs, tokenizer

def load_and_tokenize(BASE_PATH):
    ## Declare paths
    FORMAL_PATH_TRAIN = '{}/Supervised Data/Entertainment_Music/S_Formal_EM_Train.txt'.format(BASE_PATH)
    INFORMAL_PATH_TRAIN = '{}/Supervised Data/Entertainment_Music/S_Informal_EM_Train.txt'.format(BASE_PATH)

    FORMAL_PATH_HOLDOUT = '{}/Supervised Data/Entertainment_Music/S_Formal_EM_ValTest.txt'.format(BASE_PATH)
    INFORMAL_PATH_HOLDOUT = '{}/Supervised Data/Entertainment_Music/S_Informal_EM_ValTest.txt'.format(BASE_PATH)

    EMBEDDING_PATH = '{}/glove.6B.200d.txt'.format(BASE_PATH)
    
    ## Open Data
    formal = open(FORMAL_PATH_TRAIN).read()
    informal = open(INFORMAL_PATH_TRAIN).read()

    formal_holdout = open(FORMAL_PATH_HOLDOUT).read()
    informal_holdout = open(INFORMAL_PATH_HOLDOUT).read()
    
    ## initialize tokenzier
    tok = TweetTokenizer()
    
    ## process data
    process = lambda seq: '<start> ' + ' '.join(tok.tokenize(seq)) + ' <end>'
    f_corpus = [process(seq) for seq in formal.split('\n')]
    if_corpus = [process(seq) for seq in informal.split('\n')]

    f_holdout = [process(seq) for seq in formal_holdout.split('\n')]
    if_holdout = [process(seq) for seq in informal_holdout.split('\n')]

    ## Tokenize
    input_train, input_tokenizer = tokenize(if_corpus)
    target_train, target_tokenizer = tokenize(f_corpus)
    
    input_test, _ = tokenize(if_holdout, input_tokenizer)
    target_test, _ = tokenize(f_holdout, target_tokenizer)
    
    ## Create TF Dataset
    buffer_size = len(input_train)
    steps_per_epoch = len(input_train) // BATCH_SIZE
    input_vocab_size = len(input_tokenizer.word_index) + 1
    target_vocab_size = len(target_tokenizer.word_index) + 1

    train = tf.data.Dataset.from_tensor_slices((input_train, target_train)).shuffle(buffer_size)
    train = train.batch(BATCH_SIZE, drop_remainder=True)

    test = tf.data.Dataset.from_tensor_slices((input_test, target_test)).batch(1)
    context = {
        'input_tokenizer': input_tokenizer,
        'target_tokenizer': target_tokenizer,
        'input_vocab_size': input_vocab_size, 
        'embedding_dim:': EMBEDDING_DIM,
        
    }
    return train, test, context

def embedding_matrix(tokenizer, vocab_size, base_path):
    embeddings_index = {}
    embedding_dim = EMBEDDING_DIM
    embedding_path = '{}/glove.6B.200d.txt'.format(base_path)
    with open(embedding_path) as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    embeddings_matrix = np.zeros((vocab_size, embedding_dim))
    for word, i in tokenizer.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embeddings_matrix[i] = embedding_vector

    return embeddings_matrix
