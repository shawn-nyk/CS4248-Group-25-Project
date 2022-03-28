import tensorflow as tf
import pickle
from tensorflow import keras

training_dir = "aclImdb_v1/aclImdb/train"
testing_dir = "aclImdb_v1/aclImdb/test"


IS_TRAIN = True

batch_size = 64
seed = 0
max_features = 10000
sequence_length = 250


def load_data():

    training_data = tf.keras.preprocessing.text_dataset_from_directory(
        training_dir,
        batch_size=batch_size,
        validation_split=0.2,
        subset='training',
        seed=seed
    )

    validation_data = tf.keras.preprocessing.text_dataset_from_directory(
        training_dir,
        batch_size=batch_size,
        validation_split=0.2,
        subset='validation',
        seed=seed
    )

    testing_data = tf.keras.preprocessing.text_dataset_from_directory(
        testing_dir,
        batch_size=batch_size)

    return training_data, validation_data, testing_data


def generate_vectorizer(input_text):

    vectorizer = tf.keras.layers.experimental.preprocessing.TextVectorization(
        standardize='lower_and_strip_punctuation',
        max_tokens=max_features,
        output_mode='int',
        output_sequence_length=sequence_length)

    train_text = input_text.map(lambda x, y: x)
    vectorizer.adapt(train_text)

    return vectorizer

def generate_tfidf_vectorizer(input_text,s):

    vectorizer = tf.keras.layers.experimental.preprocessing.TextVectorization(
        standardize=s,
        max_tokens=max_features,
        output_mode='tf-idf')

    train_text = input_text.map(lambda x, y: x)
    vectorizer.adapt(train_text)

    return vectorizer

def generate_ngram_vectorizer(input_text, ngram, s):

    vectorizer = tf.keras.layers.experimental.preprocessing.TextVectorization(
        standardize=s,
        ngrams=ngram,
        max_tokens=max_features,
        output_mode='int',
        output_sequence_length=sequence_length)

    train_text = input_text.map(lambda x, y: x)
    vectorizer.adapt(train_text)

    return vectorizer

def pre_process_data(raw_training_data, raw_validation_data, raw_testing_data, vectorizer):
 
    #vectorizer = generate_vectorizer(raw_training_data)
    
    def vectorize_text(text, label):
        text = tf.expand_dims(text, -1)
        return vectorizer(text), label

    train_ds = raw_training_data.map(vectorize_text)
    val_ds = raw_validation_data.map(vectorize_text)
    test_ds = raw_testing_data.map(vectorize_text)

    return train_ds, val_ds, test_ds, vectorizer

#get glove files from https://www.kaggle.com/datasets/anindya2906/glove6b
glove_dir = '../input/glove6b'

def get_glove_embedding(vectorizer):
    
    voc = vectorizer.get_vocabulary()
    word_index = dict(zip(voc, range(len(voc))))
    
    embeddings_index = {}
    for f_path in os.listdir(glove_dir):
        with open(glove_dir + "/" + f_path) as f:
            for line in f:
                word, coefs = line.split(maxsplit=1)
                coefs = np.fromstring(coefs, "f", sep=" ")
                embeddings_index[word] = coefs

    print("Found %s word vectors." % len(embeddings_index))
    
    num_tokens = len(voc) + 2
    embedding_dim = 100
    hits = 0
    misses = 0

   # Prepare embedding matrix
    embedding_matrix = np.zeros((num_tokens, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
        # Words not found in embedding index will be all-zeros.
        # This includes the representation for "padding" and "OOV"
            embedding_matrix[i] = embedding_vector
            hits += 1
        else:
            misses += 1
    print("Converted %d words (%d misses)" % (hits, misses))
    
    embedding_layer = Embedding(
        num_tokens,
        embedding_dim,
        embeddings_initializer=keras.initializers.Constant(embedding_matrix),
        trainable=False,
    )
    
    return embedding_layer
    
def create_model(train_ds, val_ds, test_ds, use_glove):

    embedding_dim = 16

    # model = tf.keras.Sequential([
    #     tf.keras.layers.Embedding(max_features + 1, embedding_dim),
    #     tf.keras.layers.Dropout(0.2),
    #     tf.keras.layers.GlobalAveragePooling1D(),
    #     tf.keras.layers.Dropout(0.2),
    #     tf.keras.layers.Dense(1)])

    model = tf.keras.Sequential()

    embedding = tf.keras.layers.Embedding(max_features + 1, embedding_dim)
    if use_glove:
        embedding = use_glove
    model.add(embedding)
    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.LSTM(units=64))
    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Dense(units=32))
    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Dense(units=1))

    # model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    #               optimizer='adam',
    #               metrics=tf.metrics.BinaryAccuracy(threshold=0.0))

    model.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=tf.metrics.BinaryAccuracy(threshold=0.0))

    epochs = 10

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs)

    # loss, accuracy = model.evaluate(test_ds)
    #
    # print("Loss: ", loss)
    # print("Accuracy: ", accuracy)

    return model


def save_model(model, vectorizer):
    model.save("model.h5")

    pickle.dump({'config': vectorizer.get_config(),
                 'weights': vectorizer.get_weights()}
                , open("vectorizer.pkl", "wb"))


def load_model():
    model = tf.keras.models.load_model("model.h5")

    raw_pkl = pickle.load(open("vectorizer.pkl", "rb"))
    vectorizer = tf.keras.layers.experimental.preprocessing.TextVectorization.from_config(raw_pkl['config'])
    # You have to call `adapt` with some dummy data (BUG in Keras)
    # vectorizer.adapt(tf.data.Dataset.from_tensor_slices(["xyz"]))
    vectorizer.set_weights(raw_pkl['weights'])

    return model, vectorizer


def run_sentimental_analysis(model, vectorize_layer):

    export_model = tf.keras.Sequential([
        vectorize_layer,
        model,
        tf.keras.layers.Activation('sigmoid')
    ])

    export_model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), optimizer="adam", metrics=['accuracy']
    )

    while True:
        print("Review:")
        review = input()

        print(export_model.predict([review]))


def main():

    if IS_TRAIN:
        raw_training_data, raw_validation_data, raw_testing_data = load_data()
        
        #choose your desired preprocessing steps by uncommenting 
        #default
        chosen_vectorizer = generate_vectorizer(raw_training_data) 
        
        #tfidf options
        #chosen_vectorizer = generate_tfidf_vectorizer(raw_training_data, 'lower_and_strip_punctuation') #tfidf + lower_and_strip_punctuation      
        #chosen_vectorizer = generate_tfidf_vectorizer(raw_training_data,None) #tfidf
        
        #ngram options
        n = 2 #change the no. of ngrams here
        #chosen_vectorizer = generate_ngram_vectorizer(raw_training_data, n, 'lower_and_strip_punctuation') #ngram + lower_and_strip_punctuation
        #chosen_vectorizer = generate_ngram_vectorizer(raw_training_data, n, None) #ngram
        
        
        train_ds, val_ds, test_ds, vectorizer = pre_process_data(raw_training_data, raw_validation_data,
                                                                 raw_testing_data, chosen_vectorizer)
        use_glove = get_glove_embedding(vectorizer)
        #use_glove = None
        
        model = create_model(train_ds, val_ds, test_ds, use_glove)

        save_model(model, vectorizer)

    model, vectorizer = load_model()

    run_sentimental_analysis(model, vectorizer)

if __name__ == "__main__":
    main()
