# Formality Transfer
This repo contains all of the work for my senior project and class project for CS 880. In order to run this code with the GYAFC data set authorization from Yahoo is needed to access the L6 corpus, and then authorization must be shown to Joel Tetreault. Full details can be found [here](https://github.com/raosudha89/GYAFC-corpus).

# Models 
All models trained using OpenNMT-tf are configured using a data.yaml file, a bash script for running the model, and a translate script. The multi-column encoder models involve a special transformer model that was written on top of the existing transformer model inside of the library. 

## Baselines
### Transformers
- [Custom Transformer](https://github.com/sms1097/formality-transfer/blob/master/supervised/Baselines/Transformer%20Model.ipynb)
This is the from scratch implementation of the Transformer
- [ONMT Transformer](https://github.com/sms1097/formality-transfer/tree/master/supervised/Baselines/onmt-transformer)
This is the folder that contains configuration for the  baseline transformer model. 
### RNNs
- [Vanilla Encoder-Decoder](https://github.com/sms1097/formality-transfer/blob/master/supervised/Baselines/Vanilla%20Encoder%20Decoder.ipynb)
This is a basic Encoder-Decoder model with no attention.
- [RNN with Global Attention](https://github.com/sms1097/formality-transfer/blob/master/supervised/Baselines/Global%20Attention%20Model.ipynb)
This is an Encoder/Decoder RNN with a global attention mechanism. 
- [RNN with Bahdanau Attention](https://github.com/sms1097/formality-transfer/blob/master/supervised/Baselines/Bahdanau%20Attention%20Model.ipynb)
This is an Encoder/Decoder RNN with Bahdanau attention. The big difference between this and global attention is that Bahdanau attention concatenates the forward and backward hidden states from the last bidirectional LSTM of the encoder, whereas the global attention mechanism doe not.
- [RNN POS Assisted with Concat Global Attention](https://github.com/sms1097/formality-transfer/blob/master/supervised/Multi-Encoder%20RNN/CRF%20POS%20Concat.ipynb)
This is an Encoder/Decoder model that uses multiple encoders and conatenates hidden states and encoder output. Then uses global attention through on the new hidden state and encoder output.

## [Formality Discrimination](https://github.com/sms1097/formality-transfer/tree/master/semi-supervised/Formality%20Discrimination)
[Zhang et al (2020)](https://arxiv.org/pdf/2005.07522.pdf) proposed formality discrimination, which  
augments data through a round trip translation to a pivot language. In this implem-entation 
the training data is translated 
to a pivot language and then translated back. This round trip translation often results in more 
formal sequences which can be used as additional data. 

- [Formality Discrimination](https://github.com/sms1097/formality-transfer/blob/master/semi-supervised/Formality%20Discrimination/Formality%20Discrimination.ipynb) is the notebook where all the data was split up before feeding into google translate. 
- [Formality Classifier](https://github.com/sms1097/formality-transfer/blob/master/semi-supervised/Formality%20Discrimination/Formality%20Classifier.ipynb) is where the classifier was trained to be able to detect formal sequences and the augmented training set was determined. 
- [formality-discrimination](https://github.com/sms1097/formality-transfer/tree/master/semi-supervised/Formality%20Discrimination/formality-discrimination) is the folder that contains the OpenNMT-tf model configuration.

## [Backtranslation](https://github.com/sms1097/formality-transfer/tree/master/semi-supervised/backtranslation)
Backtranslation augments data through a reverse training process. This process starts by 
training a machine translation system to do the reverse task of converting formal sequences 
to informal sequences. Modifications to training were performed using techniques discussed in 
[Edunov et al. (2018)](https://arxiv.org/pdf/1808.09381.pdf). 

For this task, random sampling is used of the 
top 10 choices, and noise is added to the decoding. Edunov et al. showed that random sampling 
of the synthetic data is much more effective than using a standard beam search for generating data through
backtranslation. The approach used here is to sample the k most likely 
words form the target distribution, re-normalize the new distribution and sample once more. 

Edunov et al. also showed that introducing noise in the decoding process can 
increase the quality of sequences generated. This technique performs a regular beam-search decoding but
also introduces dropout, filler token replacement, and deleting words. This technique 
was what was adopted in the training of the back translation model used in this survey

- [backtranslation](https://github.com/sms1097/formality-transfer/tree/master/semi-supervised/backtranslation/backtranslation) contains the OpenNMT-tf model configuration for translating sequences from formal to informal.
- [translate](https://github.com/sms1097/formality-transfer/tree/master/semi-supervised/backtranslation/translate) contains the model configuration for translating with the extra data from back translation.

## Rule-Assisted

The approach here is similar to ass-isting a formality transfer model
as shown in [Wang et al. (2019)](https://www.aclweb.org/anthology/D19-1365.pdf).
Their approach included using rule pre-processing on input sequences and feeding 
a concatenated sequence into two encoders and using hier-archal attention on GPT blocks. \par 
This approach uses the ideas introduced in [Chen et al (2018)](https://arxiv.org/pdf/1804.09849.pdf)
by using parallel encoders. Two sequences are fed in parallel to two different
encoders and the hidden states learned by the encoders are concatenated.  \par
For the Transformer
based architecture used by Wang et al.,  hierarchal attention appears to be the best solution
specifically for formality transfer. Chen at al. found for RNN architectures the concatenation 
of the hidden states gave superior results in machine translation. The concatenation approach
is implemented, in addition to averaging the hidden states and encoder output.  

- [Transformer Based](https://github.com/sms1097/formality-transfer/tree/master/supervised/Multi-Encoder%20Transformer/rule-assisted)
- [RNN Based](https://github.com/sms1097/formality-transfer/blob/master/supervised/Multi-Encoder%20RNN/Rule%20Concat.ipynb)

![Rule Assisted Paradigm](https://github.com/sms1097/formality-transfer/blob/master/paper/Diagrams/Rule%20Concat.png)

#### POS Assisted
This model follows the same paradigm as the rule based encoder, except using part of 
speech labels for the sequence instead of rule pre-processed sequences. A CRF was trained 
to detect parts of speech on a separate corpus and used 
to create assisted data. This data was then fed through two encoders, and their hidden spaces
were concatenated. Two versions of this model were trained, one with concatenated hidden 
states and one with averaged hidden states. 

![POS Assist](https://github.com/sms1097/formality-transfer/blob/master/paper/Diagrams/CRF%20Encoder-Decoder.png)

- [Transformer Based](https://github.com/sms1097/formality-transfer/tree/master/supervised/Multi-Encoder%20Transformer/crf-pos/transformer-crf)
- [RNN Based](https://github.com/sms1097/formality-transfer/blob/master/supervised/Multi-Encoder%20RNN/CRF%20POS%20Avg.ipynb) 

## Other files
#### [GAN experimentation](https://github.com/sms1097/formality-transfer/tree/master/semi-supervised/GANs) 
This approach is based on ideas expressed in 
[Donahue et al (2018)](https://arxiv.org/pdf/1810.06640.pdf). A Seq2Seq model is trained 
using available parallel data until minimally acceptable results are achieved.
A generator is then fed random noise to create tensors that are equal in size to the 
intermediate tensor in between the encoder and decoder. The discriminator is fed both 
tensors and attempts to distinguish between which sequences are real and which are fake. 
The generator is rewarded for fooling the discriminator and the discriminator is rewarded
for detecting the generator outputs. Training is stopped once the discriminator is no 
longer able to distinguish if a tensor came from the generator of the Seq2Seq model.

It is important to distinguish that the GAN must be trained on the intermediate sequence
instead of output sequences or tokens. When training the GAN we are essentially playing a 
minimax game, such that the loss of the Discriminator is maximized while the loss of the
generator is minimized. In a normal RNN architecture, the network is trained to minimize 
cross entropy with the target token at the step of a sequence. With this minimax algorithm 
the new goal for the RNN is to minimize the loss of the discriminator. When iterating 
through the sequence and choosing the most likely token, we were performing an operation 
that was not differentiable, since the loss was calculated according to how well that token
matched. In order to be able to apply a loss function to the discriminator, we need a 
differentiable operation, and we can achieve this by learning to generate the intermediate
sequence between the encoder and decoder since it is continuous. 

Using the GAN we can generate useful data in two ways. First a pre-trained
back translation model could be used to generate informal sequences from the 
formal sequences. The upside of this approach is the potential for unlimited data, however
there exists potential problems with the balance of generated data and original data. 
The quality of the data would have to be assessed, which would require the training of a metric 
to ensure the backtranslated data fits the informal distribution. 

The second approach for using GAN data is using similarity metrics between
the informal corpus and generated sequences to pair rewrites. Jacc-ard similarity is
a retrieval metric which
computes the intersection over union for sequences, and could be used
to find potential matches between the data. Another similarity metric is cosine similarity, 
which computes distances between vectors representing
term counts. Similarly to Jaccard similarity this metric could be used to retrieve 
sequences from the generated sequences that are close to the informal sequences. A minimum
distance could be selected based upon results. 

Due to time and resource constraints this GAN approach could not be fully developed.
Further experimentation needs to be done to produce a GAN that can generate 
adequate data. All implementations that currently exist suffer from mode collapse. 
Different combinations of additional noise, learning rate tuning, and label smoothing 
were attempted, but with no success.
Likely the current approach will not prove to be adequate and a pure autoencoder
would need to be trained. In experimentation, it was difficult to find an autoencoder
that produced sequences of high enough quality for the generation to be worth 
pursuing. This approach might also require expanding the data set with the other half
of the supervised corpus. 

#### [Perplexity Metric](https://github.com/sms1097/formality-transfer/blob/master/metrics/Formality%20Benchmarking.ipynb)
This was an early stage idea about how to get a good formality metric. The idea was to train a language model and measure how close the perplexity of output sequences were to a test set. This metric was abandoned for the other formality classifier.

#### [Formality classifer wrapper](https://github.com/sms1097/formality-transfer/blob/master/metrics/formality_classifier.py)
This is what is imported into the results notebook to measure formality for test data.

#### [Learned Ensemble Decoder](https://github.com/sms1097/formality-transfer/blob/master/supervised/Multi-Encoder%20RNN/Rule%20Ensemble.ipynb)
In the experiments with training two encoders in parallel, I experimented with training two full encoder-decoder networks in parallel and then trying to learn a weighted ensemble of both networks. This was going to be used with the POS or rule asssited features. This ended up requiring too much memory to train and was never fully implemented.

#### [Results](https://github.com/sms1097/formality-transfer/blob/master/Results%20Analysis.ipynb)
All results for both projets are in this notebook.

