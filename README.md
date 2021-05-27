# Named Entity Recognition

We do experimental work to report the gain in performance when going from a simple Bi-LSTM (without pretrained word embeddings) to modifying our architecture and word embeddings to finally using a Bi-LSTM with CRF layer which gave best results . We use the GMB dataset for all our experiments.

A detailed report can be found [here](https://github.com/HarmanDotpy/Named-Entity-Recognition-in-Pytorch/blob/main/NER_report.pdf). 

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















