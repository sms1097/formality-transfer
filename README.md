# Formality Transfer
This repo contains all of the work for my senior project and class project for CS 880. In order to run this code with the GYAFC data set authorization from Yahoo is needed to access the L6 corpus, and then authorization must be shown to Joel Tetreault. Full details can be found [here](https://github.com/raosudha89/GYAFC-corpus).

## CS 880
#### [Vanilla Encoder-Decoder](https://github.com/sms1097/formality-transfer/blob/master/supervised/Baselines/Vanilla%20Encoder%20Decoder.ipynb)
This is a basic Encoder-Decoder model with no attention.

#### [RNN with Global Attention](https://github.com/sms1097/formality-transfer/blob/master/supervised/Baselines/Global%20Attention%20Model.ipynb)
This is the global attention implementaiton.

#### [CRF POS with Global Attention](https://github.com/sms1097/formality-transfer/blob/master/supervised/Multi-Encoder%20RNN/CRF%20POS%20Concat.ipynb)
This is the parallel encoder model with global attention. The CRF POS model was trained in [this](https://github.com/sms1097/formality-transfer/blob/master/supervised/Multi-Encoder%20RNN/POS%20Generation.ipynb) notebook.

## Senior Project
### Transformer Based Models
All models trained using OpenNMT-tf are configured using a data.yaml file, a bash script for running the model, and a translate script. The multi-column encoder models involve a special transformer model that was written on top of the existing transformer model inside of the library. 

#### [Custom Transformer](https://github.com/sms1097/formality-transfer/blob/master/supervised/Baselines/Transformer%20Model.ipynb)
As mentioned in the paper this model was not used since the requirements for training were too great. 

#### [ONMT Transformer](https://github.com/sms1097/formality-transfer/tree/master/supervised/Baselines/onmt-transformer)
This is the folder that contains configuration for the  baseline transformer model. 

#### [Formality Discrimination](https://github.com/sms1097/formality-transfer/tree/master/semi-supervised/Formality%20Discrimination)
This folder contains all fo the work done on formaltiy transfer. 
- [Formality Discrimination](https://github.com/sms1097/formality-transfer/blob/master/semi-supervised/Formality%20Discrimination/Formality%20Discrimination.ipynb) is the notebook where all the data was split up before feeding into google translate. 
- [Formality Classifier](https://github.com/sms1097/formality-transfer/blob/master/semi-supervised/Formality%20Discrimination/Formality%20Classifier.ipynb) is where the classifier was trained to be able to detect formal sequences and the augmented training set was determined. 
- [formality-discrimination](https://github.com/sms1097/formality-transfer/tree/master/semi-supervised/Formality%20Discrimination/formality-discrimination) is the folder that contains the OpenNMT-tf model configuration.

#### [Backtranslation](https://github.com/sms1097/formality-transfer/tree/master/semi-supervised/backtranslation)
Two folders for this one:
- [backtranslation](https://github.com/sms1097/formality-transfer/tree/master/semi-supervised/backtranslation/backtranslation) contains the OpenNMT-tf model configuration for translating sequences from formal to informal.
- [translate](https://github.com/sms1097/formality-transfer/tree/master/semi-supervised/backtranslation/translate) contains the model configuration for translating with the extra data from back translation.

#### [POS Assisted](https://github.com/sms1097/formality-transfer/tree/master/supervised/Multi-Encoder%20Transformer/crf-pos/transformer-crf)
Model configuration for a multi-column that feeds in sequences that are tagged with part of speech. 

#### [Rule-Assisted](https://github.com/sms1097/formality-transfer/tree/master/supervised/Multi-Encoder%20Transformer/rule-assisted)
Configuration for model that takes the normal sequence and sequence cleaned by rules in parallel. 

### RNN Based Models
#### [RNN with Global Attention](https://github.com/sms1097/formality-transfer/blob/master/supervised/Baselines/Global%20Attention%20Model.ipynb)
This is an Encoder/Decoder RNN with a global attention mechanism. 

#### [RNN with Bahdanau Attention](https://github.com/sms1097/formality-transfer/blob/master/supervised/Baselines/Bahdanau%20Attention%20Model.ipynb)
This is an Encoder/Decoder RNN with Bahdanau attention. The big difference between this and global attention is that Bahdanau attention concatenates the forward and backward hidden states from the last bidirectional LSTM of the encoder, whereas the global attention mechanism doe not.

#### [RNN POS Assisted with Concat Global Attention](https://github.com/sms1097/formality-transfer/blob/master/supervised/Multi-Encoder%20RNN/CRF%20POS%20Concat.ipynb)
This is an Encoder/Decoder model that uses multiple encoders and conatenates hidden states and encoder output. Then uses global attention through on the new hidden state and encoder output.

#### [RNN POS Assisted with Averaged Global Attention](https://github.com/sms1097/formality-transfer/blob/master/supervised/Multi-Encoder%20RNN/CRF%20POS%20Avg.ipynb) 
Same as the other RNN with POS assist, except this one uses averaged encoder output and hidden state.

#### [RNN with Rule Assist](https://github.com/sms1097/formality-transfer/blob/master/supervised/Multi-Encoder%20RNN/Rule%20Concat.ipynb)
This is the same structure as the RNN with POS assist, except this one uses rule preprocessing for the second encoder. This rule preprocessing includes actions such as capitalizing the first letter, moving a sequence that is all caps to lower case, removing profanity, and others. 

### Other files
#### [GAN experimentation](https://github.com/sms1097/formality-transfer/tree/master/semi-supervised/GANs) 
This was basic experimentation with generating data from a trained encoder/decoder to learn new sequences. The goal was to augment data and match based on content retrieval methods to create new sequence pairs. This method never was trained successfully and the GAN suffers from mode collapse. Further work will need to be done to find good model structures for the GAN and most likely an autoencoder will need to be trained.

#### [Perplexity Metric](https://github.com/sms1097/formality-transfer/blob/master/metrics/Formality%20Benchmarking.ipynb)
This was an early stage idea about how to get a good formality metric. The idea was to train a language model and measure how close the perplexity of output sequences were to a test set. This metric was abandoned for the other formality classifier.

#### [Formality classifer wrapper](https://github.com/sms1097/formality-transfer/blob/master/metrics/formality_classifier.py)
This is what is imported into the results notebook to measure formality for test data.

### [Learned Ensemble Decoder](https://github.com/sms1097/formality-transfer/blob/master/supervised/Multi-Encoder%20RNN/Rule%20Ensemble.ipynb)
In the experiments with training two encoders in parallel, I experimented with training two full encoder-decoder networks in parallel and then trying to learn a weighted ensemble of both networks. This was going to be used with the POS or rule asssited features. This ended up requiring too much memory to train and was never fully implemented.

### [Results](https://github.com/sms1097/formality-transfer/blob/master/Results%20Analysis.ipynb)
Overview of results for all models tested. This is also where all of the plots are generated as well.


