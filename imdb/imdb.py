import tensorflow as tf
import pickle

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


def pre_process_data(raw_training_data, raw_validation_data, raw_testing_data):

    vectorizer = generate_vectorizer(raw_training_data)

    def vectorize_text(text, label):
        text = tf.expand_dims(text, -1)
        return vectorizer(text), label

    train_ds = raw_training_data.map(vectorize_text)
    val_ds = raw_validation_data.map(vectorize_text)
    test_ds = raw_testing_data.map(vectorize_text)

    return train_ds, val_ds, test_ds, vectorizer


def create_model(train_ds, val_ds, test_ds):

    embedding_dim = 16

    # model = tf.keras.Sequential([
    #     tf.keras.layers.Embedding(max_features + 1, embedding_dim),
    #     tf.keras.layers.Dropout(0.2),
    #     tf.keras.layers.GlobalAveragePooling1D(),
    #     tf.keras.layers.Dropout(0.2),
    #     tf.keras.layers.Dense(1)])

    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Embedding(max_features + 1, embedding_dim))
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


if __name__ == "__main__":

    if IS_TRAIN:
        raw_training_data, raw_validation_data, raw_testing_data = load_data()

        train_ds, val_ds, test_ds, vectorizer = pre_process_data(raw_training_data, raw_validation_data,
                                                                 raw_testing_data)
        model = create_model(train_ds, val_ds, test_ds)

        save_model(model, vectorizer)

    model, vectorizer = load_model()

    run_sentimental_analysis(model, vectorizer)








