
import numpy as np

from keras.models import Model, Sequential
from keras.layers import Input, Dense, GaussianNoise, Dense
from keras.optimizers import Adam

from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier

def create_denoising_autoencoder(input_shape: tuple, layer_size: int, noise_std_dev: float = 0.1) -> tuple:
    """
    Create a denoising autoencoder model using the given input shape and layer size.

    Parameters:
        input_shape (tuple): The shape of the input data.
        layer_size (int): The number of neurons in the encoding layer.
        noise_std_dev (float): The standard deviation of the noise to add to the input data.

    Returns:
        tuple: A tuple containing the denoising autoencoder model and the encoder model.

    """
    input_layer = Input(shape=(input_shape,))
    noise = GaussianNoise(noise_std_dev)(input_layer)
    
    encoded = Dense(layer_size, activation='relu')(noise)
    decoded = Dense(input_shape, activation='sigmoid')(encoded)

    autoencoder = Model(input_layer, decoded)
    encoder = Model(input_layer, encoded)

    return autoencoder, encoder

def train_stacked_denoising_autoencoder(X_train: np.ndarray, X_test: np.ndarray, num_layers: int, layer_sizes: list,
                                        learning_rate: float = 0.0001, epochs: int = 50, batch_size: int = 16) -> list:
    """
    Train a stacked denoising autoencoder model using the given training and testing data, number of layers, and layer sizes.

    Parameters:
        X_train (np.ndarray): The training data to use.
        X_test (np.ndarray): The testing data to use.
        num_layers (int): The number of layers in the stacked autoencoder.
        layer_sizes (list): A list of integers indicating the number of neurons in each layer of the stacked autoencoder.
        learning_rate (float): The learning rate to use for training.
        epochs (int): The number of epochs to train each autoencoder.
        batch_size (int): The batch size to use for training.

    Returns:
        list: A list of encoder models learned at each layer of the stacked autoencoder.

    """
    encoders = []
    autoencoders = []

    input_data = X_train
    input_shape = X_train.shape[1]

    validation_data = X_test

    # Train a denoising autoencoder for each layer of the stacked autoencoder
    for i in range(num_layers):

        # Create the denoising autoencoder
        autoencoder, encoder = create_denoising_autoencoder(input_shape, layer_sizes[i])

        # Compile the autoencoder model
        autoencoder.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')

        # Train the autoencoder on the input data
        autoencoder.fit(input_data, input_data, epochs=epochs, batch_size=batch_size, validation_data=(validation_data, validation_data))

        # Save the encoder model and autoencoder model for this layer
        encoders.append(encoder)
        autoencoders.append(autoencoder)

        # Generate the input data for the next layer by encoding the current input data
        input_data = encoder.predict(input_data)
        input_shape = layer_sizes[i]

        # Generate the validation data for the next layer by encoding the current validation data
        validation_data = encoder.predict(validation_data)

    # Return the list of encoder models learned at each layer
    return encoders

def create_svc_classifier(kernel: str, gamma: float, C: float) -> OneVsRestClassifier:
    """
    Create a support vector machine classifier with the specified kernel, gamma, and regularization parameter.

    Parameters:
        kernel (str): The kernel to use for the support vector machine.
        gamma (float): The kernel coefficient for 'rbf', 'poly', and 'sigmoid'.
        C (float): The regularization parameter.

    Returns:
        OneVsRestClassifier: The trained support vector machine classifier.

    """
    svc = SVC(kernel=kernel, gamma=gamma, C=C)

    ovr_svc = OneVsRestClassifier(svc)

    return ovr_svc

def create_sdae_svc(X_train, X_test, y_train, y_test, num_layers, layer_sizes, kernel, gamma, C, learning_rate=0.001, epochs=50, batch_size=16):
    
    encoders = train_stacked_denoising_autoencoder(X_train, X_test, num_layers, layer_sizes, learning_rate, epochs, batch_size)
    stacked_encoder = Sequential(encoders)

    classifier = create_svc_classifier(kernel, gamma, C)

    return stacked_encoder, classifier