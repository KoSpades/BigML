import pandas as pd
import numpy as np
import PIL.Image as Image
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

num_samples = 5000
train_size = 0.9

train_df = pd.read_csv('input/covidx-cxr2/train.txt', sep=" ", header=None)
train_df.columns = ['patient id', 'filename', 'class', 'data source']
train_df = train_df.drop(['patient id', 'data source'], axis=1)

train_df = train_df.sample(n=num_samples, random_state=0)

test_df = pd.read_csv('input/covidx-cxr2/test.txt', sep=" ", header=None)
test_df.columns = ['id', 'filename', 'class', 'data source']
test_df = test_df.drop(['id', 'data source'], axis=1)

train_path = 'input/covidx-cxr2/train/'
test_path = 'input/covidx-cxr2/test/'

train_df, valid_df = train_test_split(train_df, train_size=train_size, random_state=0)

print(f"Negative and positive values of train: {train_df['class'].value_counts()}")
print(f"Negative and positive values of validation: {valid_df['class'].value_counts()}")
print(f"Negative and positive values of test: {test_df['class'].value_counts()}")

# Now we create the train_data and train_label that will be used for ImageDataGenerator.flow
train_data = list()
train_label = list()

valid_data = list()
valid_label = list()

test_data = list()
test_label = list()

for _, row in train_df.iterrows():
    file_path = "input/covidx-cxr2/train/" + row["filename"]
    cur_image = Image.open(file_path).convert('RGB')
    image_resized = cur_image.resize((200, 200))
    img_data = np.array(image_resized)
    train_data.append(img_data)
    if row["class"] == "positive":
        train_label.append(1)
    else:
        train_label.append(0)

for _, row in valid_df.iterrows():
    file_path = "input/covidx-cxr2/train/" + row["filename"]
    cur_image = Image.open(file_path).convert('RGB')
    image_resized = cur_image.resize((200, 200))
    img_data = np.array(image_resized)
    valid_data.append(img_data)
    if row["class"] == "positive":
        valid_label.append(1)
    else:
        valid_label.append(0)

for _, row in test_df.iterrows():
    file_path = "input/covidx-cxr2/test/" + row["filename"]
    cur_image = Image.open(file_path).convert('RGB')
    image_resized = cur_image.resize((200, 200))
    img_data = np.array(image_resized)
    test_data.append(img_data)
    if row["class"] == "positive":
        test_label.append(1)
    else:
        test_label.append(0)

train_data = np.asarray(train_data).reshape(int(num_samples*train_size), 200, 200, 3)
print(train_data.shape)

valid_data = np.asarray(valid_data).reshape(num_samples-int(num_samples*train_size), 200, 200, 3)
print(valid_data.shape)

test_data = np.asarray(test_data).reshape(400, 200, 200, 3)
print(test_data.shape)

train_datagen = ImageDataGenerator(rescale=1./255.,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   vertical_flip=True)
test_datagen = ImageDataGenerator(rescale=1.0/255.)

train_gen = train_datagen.flow(train_data, train_label, batch_size=64)
valid_gen = test_datagen.flow(valid_data, valid_label, batch_size=64)
test_gen = test_datagen.flow(test_data, test_label, batch_size=64)

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
                    validation_data=valid_gen, epochs=1,
                    callbacks=[callbacks])

model.load_weights('./covid_classifier_model.h5')
model.evaluate(test_gen)
