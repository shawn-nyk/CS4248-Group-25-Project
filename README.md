# CS4248 Project - Sentiment Analysis of Movie Reviews

This is CS4248 Group 25’s code repository. For our project, our team worked on the sentiment analysis of IMDb movie reviews using RNNs. This repo contains all the code we used and findings we referenced in our report, which consists of data preprocessing as well as recursive neural network (RNN) models.

The dataset that was used can be found here: https://ai.stanford.edu/~amaas/data/sentiment/ 

The following are the files that can be found within the code folder:
- Glove Project RNN.ipynb: GRU RNN model with GloVe Embeddings
- Preprocessing (Lemma + Stopword) Project RNN.ipynb: GRU RNN model with GloVe Embeddings and lemmatization and stopword removal
- Preprocessing (Lemma + Stopword) no glove Project RNN.ipynb: GRU RNN model with lemmatization and stopword removal
- Preprocessing (Lemma + Stopword) with tfidf plus Project RNN.ipynb: GRU RNN model with GloVe Embeddings + tf-idf and lemmatization and stopword removal
- Preprocessing (Lemma) Project RNN.ipynb: GRU RNN model with GloVe Embeddings and lemmatization
- Preprocessing_(Lemma_+_Stopword)_with_tfidf_Project_RNN_v0.ipynb: GRU RNN model with GloVe Embeddings x tf-idf and lemmatization and stopword removal
- Project RNN.ipynb: Basic GRU RNN model
- basic_gru_rnn_model.ipynb: Initial basic GRU RNN model
- cs4248-data-viz.ipynb: Python notebook for data exploration (word cloud and n-gram frequencies) 
- imdb.py: Initial basic LSTM model
- imdb_w_TFpreproc.py: Tensorflow tf-idf/GloVe embedding/N-gram/Stopword + punctuation removal utility functions with basic LSTM sample use case
- simple_rnn_vs_gru_vs_lstm.ipynb: Implementation and comparison of Simple RNN, GRU, and LSTM models
