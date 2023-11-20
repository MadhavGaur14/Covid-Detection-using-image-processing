import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np

# Load the VGG16 model with pre-trained weights
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Create your model
model = Sequential()
model.add(base_model)
model.add(Flatten(name='flatten'))
model.add(Dense(512, activation='relu', name='fc1'))
model.add(Dropout(0.5, name='dropout1'))
model.add(Dense(128, activation='relu', name='fc2'))
model.add(Dropout(0.5, name='dropout2'))
model.add(Dense(2, activation='softmax', name='fc3'))

# Image data generator
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=15,
    fill_mode="nearest")

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
    'E:\Coding stuf\python\DiseaseDetection',
    target_size=(224, 224),
    batch_size=10,
    class_mode='categorical')

test_set = test_datagen.flow_from_directory(
    'E:\Coding stuf\python\DiseaseDetection',
    target_size=(224, 224),
    batch_size=10,
    class_mode='categorical')

# Set up the optimizer
# Set up the optimizer
opt = tf.keras.optimizers.Adam(learning_rate=0.0001)

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# Training phase
EPOCHS = 1
es = tf.keras.callbacks.EarlyStopping(patience=18)
chkpt = tf.keras.callbacks.ModelCheckpoint(filepath='best_model_todate', save_best_only=True, save_weights_only=False)
history = model.fit(training_set, steps_per_epoch=len(training_set), epochs=EPOCHS, validation_data=test_set, validation_steps=len(test_set), callbacks=[es, chkpt])

# Plot training history
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), history.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), history.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), history.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), history.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.jpg")

from sklearn.metrics import classification_report, confusion_matrix
# Evaluate the model on the test dataset
test_loss, test_accuracy = model.evaluate(test_set)

# Get the predictions for the test dataset
predictions = model.predict(test_set)

# Calculate the confusion matrix
confusion = confusion_matrix(test_set.classes, np.argmax(predictions, axis=-1))

# Generate a classification report
report = classification_report(test_set.classes, np.argmax(predictions, axis=-1), target_names=test_set.class_indices)

# Print the evaluation metrics
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)
print("Confusion Matrix:")
print(confusion)
print("Classification Report:")
print(report)
