
from keras.models import Model, Sequential
from keras.layers import Input, Dense, GaussianNoise, Dense
from keras.optimizers import Adam

from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier

def create_denoising_autoencoder(input_shape, layer_size, noise_std_dev=0.1):
    
    input_layer = Input(shape=(input_shape,))
    noise = GaussianNoise(noise_std_dev)(input_layer)
    
    encoded = Dense(layer_size, activation='relu')(noise)
    decoded = Dense(input_shape, activation='sigmoid')(encoded)

    autoencoder = Model(input_layer, decoded)
    encoder = Model(input_layer, encoded)

    return autoencoder, encoder

def train_stacked_denoising_autoencoder(X_train, X_test, num_layers, layer_sizes, learning_rate=0.0001, epochs=50, batch_size=16):
    
    encoders = []
    autoencoders = []

    input_data = X_train
    input_shape = X_train.shape[1]

    validation_data = X_test

    for i in range(num_layers):

        autoencoder, encoder = create_denoising_autoencoder(input_shape, layer_sizes[i])
        autoencoder.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')
        autoencoder.fit(input_data, input_data, epochs=epochs, batch_size=batch_size, validation_data=(validation_data, validation_data))

        encoders.append(encoder)
        autoencoders.append(autoencoder)

        input_data = encoder.predict(input_data)
        input_shape = layer_sizes[i]

        validation_data = encoder.predict(validation_data)

    return encoders

def create_svc_classifier(kernel, gamma, C):

    svc = SVC(kernel=kernel, gamma=gamma, C=C)

    ovr_svc = OneVsRestClassifier(svc)

    return ovr_svc

def create_sdae_svc(X_train, X_test, y_train, y_test, num_layers, layer_sizes, kernel, gamma, C, learning_rate=0.001, epochs=50, batch_size=16):
    
    encoders = train_stacked_denoising_autoencoder(X_train, X_test, num_layers, layer_sizes, learning_rate, epochs, batch_size)
    stacked_encoder = Sequential(encoders)

    classifier = create_svc_classifier(kernel, gamma, C)

    return stacked_encoder, classifier