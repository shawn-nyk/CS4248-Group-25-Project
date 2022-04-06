# CS4248_proj
imdb sentimental analysis

This is CS4248 Group 25’s code repository. For our project, our team worked on the sentiment analysis of IMDb movie reviews using RNNs. This repo contains all the code we used and findings we referenced in our report, which consists of data preprocessing as well as recursive neural network (RNN) models. The dataset that was used can be found here: https://ai.stanford.edu/~amaas/data/sentiment/ 

The following are the files that can be found within the code folder:
- Basic_gru_rnn_model.ipynb: Initial basic GRU RNN model
- Basic_simple_rnn_model.ipynb: I will delete this lol
- cs4248-data-viz.ipynb:  python notebook for data exploration (wordcloud and ngram frequencies) 
- Imdb.py: Basic LSTM model
- imdb_w_TFpreproc.py: Tensorflow TFIDF/Glove embedding/Ngram/Stop word + punctuation removal utility functions with basic LSTM sample use case
- Simple_rnn_vs_gru_vs_lstm.ipynb: Implementation and comparison of Simple RNN, GRU, and LSTM models

- Project RNN.ipynb: Basic GRU RNN model
- Glove Project RNN.ipynb: GRU RNN model with GloVe Embeddings
- Preprocessing (Lemma) Project RNN.ipynb: GRU RNN model with GloVe Embeddings and lemmatization
- Preprocessing (Lemma + Stopword) Project RNN.ipynb: GRU RNN model with GloVe Embeddings and lemmatization and stopword
- Preprocessing_(Lemma_+_Stopword)_with_tfidf_Project_RNN_v0.ipynb: GRU RNN model with GloVe Embeddings x tfidf and lemmatization and stopword
- Preprocessing (Lemma + Stopword) with tfidf plus Project RNN.ipynb: GRU RNN model with GloVe Embeddings + tfidf and lemmatization and stopword
- Preprocessing (Lemma + Stopword) no glove Project RNN.ipynb: GRU RNN model with lemmatization and stopword
