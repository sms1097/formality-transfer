# Formality Transfer
This repo contains all of the work for my senior project and class project for CS 880. <br>
In order to run this code authorization from Yahoo to access the L6 corpus is needed 
and authorization must be shown to Joel Tetreault. Full details can be found [here](https://github.com/raosudha89/GYAFC-corpus)

## CS 880
### [Vanilla Encoder-Decoder](https://github.com/sms1097/formality-transfer/blob/master/supervised/Baselines/Vanilla%20Encoder%20Decoder.ipynb)
This is a basic Encoder-Decoder model with no attention.

### [RNN with Global Attention](https://github.com/sms1097/formality-transfer/blob/master/supervised/Baselines/Global%20Attention%20Model.ipynb)
This is the global attention implementaiton.

### [CRF POS with Global Attention](https://github.com/sms1097/formality-transfer/blob/master/supervised/Multi-Encoder%20RNN/CRF%20POS%20Concat.ipynb)
This is the parallel encoder model with global attention. The CRF POS model was trained in [this](https://github.com/sms1097/formality-transfer/blob/master/supervised/Multi-Encoder%20RNN/POS%20Generation.ipynb) notebook.

## Senior Project
### Transformer Based Models
#### [Custom Transformer](https://github.com/sms1097/formality-transfer/blob/master/supervised/Baselines/Transformer%20Model.ipynb)
As mentioned in the paper this model was not used since the requirements for training were too great. 

#### [ONMT Transformer](https://github.com/sms1097/formality-transfer/tree/master/supervised/Baselines/onmt-transformer)
This is the folder that contains configuration for the models being trained.

#### [Formality Discrimination](https://github.com/sms1097/formality-transfer/tree/master/semi-supervised/Formality%20Discrimination)


#### [Backtranslation](https://github.com/sms1097/formality-transfer/tree/master/semi-supervised/backtranslation)
Two folders are here: one for the back translation model and one for the transformer with the augmented data

#### [POS Assisted](https://github.com/sms1097/formality-transfer/tree/master/supervised/Multi-Encoder%20Transformer/crf-pos/transformer-crf)
Multi-column encoder with sequences and POS.

#### [Rule-Assisted](https://github.com/sms1097/formality-transfer/tree/master/supervised/Multi-Encoder%20Transformer/rule-assisted)



