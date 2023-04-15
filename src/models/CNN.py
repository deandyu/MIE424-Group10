
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dense, Flatten, LocallyConnected2D, BatchNormalization, Dropout
from keras.optimizers import Adam

def create_cnn(input_shape, num_classes):

    cnn = Sequential()

    cnn.add(Conv2D(64, kernel_size=(5,5), activation='relu', input_shape=input_shape))
    cnn.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))

    cnn.add(Conv2D(64, kernel_size=(5,5), activation='relu', padding='same'))
    cnn.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))

    cnn.add(Conv2D(64, kernel_size=(5,5), activation='relu', padding='same'))
    cnn.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))

    cnn.add(LocallyConnected2D(32, kernel_size=(3,3), activation='relu'))

    cnn.add(Flatten())
    cnn.add(Dense(num_classes, activation='softmax'))

    cnn.compile(loss='categorical_crossentropy', 
                optimizer=Adam(learning_rate=0.001), 
                metrics=['accuracy'])
    
    return cnn

def create_cnn_with_bn_dropout(input_shape, num_classes):

    cnn = Sequential()

    cnn.add(Conv2D(64, kernel_size=(5,5), activation='relu', input_shape=input_shape))
    cnn.add(BatchNormalization())
    cnn.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
    cnn.add(Dropout(0.3))

    cnn.add(Conv2D(64, kernel_size=(5,5), activation='relu', padding='same'))
    cnn.add(BatchNormalization())
    cnn.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
    cnn.add(Dropout(0.3))

    cnn.add(Conv2D(64, kernel_size=(5,5), activation='relu', padding='same'))
    cnn.add(BatchNormalization())
    cnn.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
    cnn.add(Dropout(0.3))

    cnn.add(LocallyConnected2D(32, kernel_size=(3,3), activation='relu'))
    cnn.add(BatchNormalization())

    cnn.add(Flatten())
    cnn.add(Dense(num_classes, activation='softmax'))

    cnn.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
    
    return cnn