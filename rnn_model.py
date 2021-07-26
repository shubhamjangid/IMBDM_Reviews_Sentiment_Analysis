import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import tensorflow as tf 

dataset, info = tfds.load('imdb_reviews', with_info=True, as_supervised = True) 

train_dataset, test_dataset = dataset['train'], dataset['test']

buffer_size = 10000
batch_size = 64
train_dataset = train_dataset.shuffle(buffer_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

vocab_size = 1000
encoder = tf.keras.layers.experimental.preprocessing.TextVectorization(
    max_tokens = vocab_size
)
encoder.adapt(train_dataset.map(lambda text, label: text))

model = tf.keras.Sequential([
                             encoder,
                             tf.keras.layers.Embedding(input_dim = len(encoder.get_vocabulary()), output_dim = 64, mask_zero = True),
                             tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
                             tf.keras.layers.Dense(64, activation = 'relu'),
                             tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])
#print(model.summary())

history = model.fit(train_dataset, epochs=1,
                    validation_data=test_dataset,
                    validation_steps=30)

model.save("E:\Python_ML_DS\Flask\model")