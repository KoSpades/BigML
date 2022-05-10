import os
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

train_df = pd.read_csv('input/covidx-cxr2/train.txt', sep=" ", header=None)
train_df.columns = ['patient id', 'filename', 'class', 'data source']
train_df = train_df.drop(['patient id', 'data source'], axis=1)

test_df = pd.read_csv('input/covidx-cxr2/test.txt', sep=" ", header=None)
test_df.columns = ['id', 'filename', 'class', 'data source']
test_df = test_df.drop(['id', 'data source'], axis=1)

train_path = 'input/covidx-cxr2/train/'
test_path = 'input/covidx-cxr2/test/'

train_df, valid_df = train_test_split(train_df, train_size=0.9, random_state=0)

print(f"Negative and positive values of train: {train_df['class'].value_counts()}")
print(f"Negative and positive values of validation: {valid_df['class'].value_counts()}")
print(f"Negative and positive values of test: {test_df['class'].value_counts()}")

train_datagen = ImageDataGenerator(rescale=1./255.,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   vertical_flip=True)
test_datagen = ImageDataGenerator(rescale=1.0/255.)

train_gen = train_datagen.flow_from_dataframe(dataframe=train_df, directory=train_path, x_col='filename',
                                              y_col='class', target_size=(200,200), batch_size=64,
                                              class_mode='binary')
valid_gen = test_datagen.flow_from_dataframe(dataframe=valid_df, directory=train_path, x_col='filename',
                                             y_col='class', target_size=(200,200), batch_size=64,
                                             class_mode='binary')
test_gen = test_datagen.flow_from_dataframe(dataframe=test_df, directory=test_path, x_col='filename',
                                            y_col='class', target_size=(200,200), batch_size=64,
                                            class_mode='binary')

base_model = tf.keras.applications.ResNet50V2(weights='imagenet',
                                              input_shape=(200, 200, 3),
                                              include_top=False)
for layer in base_model.layers:
    layer.trainable = False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

callbacks = [
    tf.keras.callbacks.ModelCheckpoint("covid_classifier_model.h5", save_best_only=True, verbose=0),
    tf.keras.callbacks.EarlyStopping(patience=3, monitor='val_loss', verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
]

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_gen,
                    validation_data=valid_gen, epochs=3,
                    callbacks=[callbacks])

model.load_weights('./covid_classifier_model.h5')
model.evaluate(test_gen)
