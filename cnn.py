import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import LambdaCallback
import matplotlib.pyplot as plt

from utilities.constants import(
    SAVED_MODEL_LOCATION
)

# A CNN that takes in a 128x259 mel-spectrogram and outputs a 50 dimensional latent factor vector
class LFV_CNN:
    def __init__(self, input_dim_x=128, input_dim_y=259):
        self.model = models.Sequential([
            layers.Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(input_dim_x, input_dim_y, 1)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),

            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),

            # Output layer
            layers.Dense(50, activation='linear')  # Adjust the activation based on your specific needs
        ])

        self.model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

        self.validation_losses = []
        self.regular_losses = []
        self.validation_accuracy = []
        self.regular_accuracy = []

    def train(self, train_songs, train_labels, validation_songs, validation_labels, epochs=1):
        plt.figure(1)
        save_val_loss_callback = LambdaCallback(on_epoch_end=lambda epoch, logs: self.save_and_print_loss(epoch, logs))

        self.model.fit(train_songs, train_labels, epochs=epochs, validation_data=(validation_songs, validation_labels), callbacks=[save_val_loss_callback])
        # self.model.save(f"{SAVED_MODEL_LOCATION}/CNN_Trained_MSE_167s_{epochs}e.keras")
        self.model.save(f"{SAVED_MODEL_LOCATION}/CNN_MSE_15557s_{epochs}e.keras")

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

        axes[0].plot(self.validation_losses, color="red", label="Validation Loss")
        axes[0].plot(self.regular_losses, color="blue", label="Regular Loss")
        axes[0].set_xlabel("Epochs")
        axes[0].set_ylabel("Loss")
        axes[0].legend()

        axes[1].plot(self.validation_accuracy, color="red", label="Validation Accuracy")
        axes[1].plot(self.regular_accuracy, color="blue", label="Regular Accuracy")
        axes[1].set_xlabel("Epochs")
        axes[1].set_ylabel("Accuracy")
        axes[1].legend()

        plt.tight_layout()
        plt.show()

    def test(self, test_images, test_labels):
        test_loss, test_acc = self.model.evaluate(test_images, test_labels)
        print(f"Test accuracy: {test_acc}")
        print(f"Test loss: {test_loss}")

    def save_and_print_loss(self, epoch, logs):
        self.validation_losses.append(logs['val_loss'])
        self.regular_losses.append(logs['loss'])
        self.validation_accuracy.append(logs['val_accuracy'])
        self.regular_accuracy.append(logs['accuracy'])
        # print(f"Epoch {epoch + 1}: Validation Loss: {logs['val_loss']}")
        # print(f"{logs = }")