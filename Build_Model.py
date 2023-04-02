import tensorflow as tf
import keras
from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications import InceptionV3

# Load pre-trained InceptionV3 model without the top layer
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the pre-trained layers
for layer in base_model.layers:
    layer.trainable = False

# Add new classification layers on top of the pre-trained layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(5, activation='softmax')(x)

# Compile the model
Retinopathy_model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)
optimizer=keras.optimizers.Adam(lr=0.001)
# optimizer = keras.optimizers.Adam(lr=0.01)
Retinopathy_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
