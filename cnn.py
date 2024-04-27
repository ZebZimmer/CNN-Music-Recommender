import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import LambdaCallback
import matplotlib.pyplot as plt

from utilities.constants import(
    CNN_SAVED_MODEL_LOCATION
)

# A CNN that takes in a 128x259 mel-spectrogram and outputs a 50 dimensional latent factor vector
class LFV_CNN:
    def __init__(self, input_dim_x=128, input_dim_y=259):
        self.model = models.Sequential()
        self.model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(input_dim_x, input_dim_y, 1)))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))

        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(64, activation='relu'))

        # Output layer
        self.model.add(layers.Dense(50, activation='softmax'))

        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        self.validation_losses = []
        self.regular_losses = []

    def train(self, train_songs, train_labels, validation_songs, validation_labels, epochs=1):
        plt.figure(1)
        save_val_loss_callback = LambdaCallback(on_epoch_end=lambda epoch, logs: self.save_and_print_loss(epoch, logs))

        self.model.fit(train_songs, train_labels, epochs=epochs, validation_data=(validation_songs, validation_labels), callbacks=[save_val_loss_callback])
        self.model.save(f"{CNN_SAVED_MODEL_LOCATION}/CNN_TEST.keras")

        plt.plot(self.validation_losses, color="red", label="Validation Loss")
        plt.plot(self.regular_losses, color="blue", label="Regular Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

    def test(self, test_images, test_labels):
        test_loss, test_acc = self.model.evaluate(test_images, test_labels)
        print(f"Test accuracy: {test_acc}")
        print(f"Test loss: {test_loss}")

    def save_and_print_loss(self, epoch, logs):
        self.validation_losses.append(logs['val_loss'])
        self.regular_losses.append(logs['loss'])
        print(f"Epoch {epoch + 1}: Validation Loss: {logs['val_loss']}")
        print(f"{logs = }")