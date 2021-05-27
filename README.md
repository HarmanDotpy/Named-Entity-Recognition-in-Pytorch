# Named Entity Recognition

We do experimental work to report the gain in performance when going from a simple Bi-LSTM (without pretrained word embeddings) to modifying our architecture and word embeddings to finally using a Bi-LSTM with CRF layer which gave best results . We use the GMB dataset for all our experiments.

A detailed report can be found [here](https://github.com/HarmanDotpy/Named-Entity-Recognition-in-Pytorch/blob/main/NER_report.pdf). 


## For reproducing the experiments
first download glove embeddings data file "glove.6B.100d.txt" from https://nlp.stanford.edu/projects/glove/

### Training
```bash

# clone the repo
git clone <this repo url>

# unzip data
tar -xf ner-gmb.tar.gz #will create a folder called "ner-gmb" having 3 files, test.txt, train.txt and dev.txt

# go into the scripts/ folder
cd scripts/

# run the experiment
python3 train_ner.py --initialization glove --char_embeddings 1 --layer_normalization 0 --crf 0 --output_file output_model.pth --data_dir ../NER_Dataset/ner-gmb --glove_embeddings_file ../glove_data/glove.6B.100d.txt --vocabulary_output_file output_vocab.vocab

```
- ***char_embeddings*** - set to 1 if you want to use character level embeddings
- ***layer_normalization*** - set to 1 if you want to use layer_normalized lstm cell in the BiLSTM
- ***crf*** - set to 1 if you want to use crf layer on top of the BiLSTM
- ***output_file*** output path for saving the trained model
- ***data_dir*** cleaned ner-data path, for example, the one which we get after unzipping ner-gmb.tar.gz
- ***glove_embeddings_file*** path to glove embeddings file "glove.6B.100d.txt"
- ***vocabulary_output_file*** output vocabulary file, created inside the training script only, which will be used while testing


### Testing
from inside scripts/ directory run
```bash
python3 test_ner.py --model_file output_model.pth --char_embeddings 1 --layer_normalization 0 --crf 0 --test_data_file ../ner-gmb/test.txt --output_file output_predictions.txt --glove_embeddings_file ../glove_data/glove.6B.100d.txt --vocabulary_input_file output_vocab.vocab

```
- ***model_file*** is the same trained model which we generated using the train script above
- ***vocabulary_input_file*** is the same vocabulary file which we generated using the train script above


## Named Entity Recognition (NER) using Bi-LSTM

We use [[Lample et al., 2016](https://arxiv.org/abs/1603.01360)] for NER tagging on the publicly available GMB dataset. The objective here is to compare different variants of the model, and justify the performance gains as the complexity, or the prior we impose on the model, of the model increases. Following are the variants:

1. Bi-LSTM with randomly initialised word embeddings
2. Bi-LSTM with pretrained Glove word embeddings
3. Bi-LSTM with Glove word embeddings and character embeddings
4. Bi-LSTM with Glove word embeddings, character embeddings, and Layer Normalization (hard-coded LSTM for it)
5. Bi-LSTM with Glove word embeddings, character embeddings, Layer Normalization, and a CRF layer at the top.

## Some Results

<img width="800" alt="Screenshot 2021-05-27 at 11 34 55 PM" src="https://user-images.githubusercontent.com/50492433/119875924-f3914f80-bf44-11eb-8778-cd59fcf1fc0d.png">


![img](https://lh3.googleusercontent.com/fS3zoIEsb8Xtp4w7-Bha0yLJnIjns5HsmFp5h2kM1xTYMEmmUrYUCcRp-TTbGXak93L0WRBXaT4nXX_Uio5cG7b-2iBgXcYYGco1AoPjArjUQoRrGoViCTHumwBnTb8np4oUf77y)















