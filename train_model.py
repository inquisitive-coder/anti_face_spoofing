from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from train import *
from tensorflow.keras.models import load_model
IMG_SIZE = 24
def load_pretrained_model():
    model = load_model('eye_status_classifier.h5')
    model.summary()
    return model
def save_model(model):
  model.save('eye_status_classifier.h5')
def train(train_generator, val_generator):
    STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
    STEP_SIZE_VALID=val_generator.n//val_generator.batch_size

    print('[LOG] Intialize Neural Network')

    model = Sequential()

    model.add(Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=(IMG_SIZE,IMG_SIZE,1)))
    model.add(MaxPooling2D())

    model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D())

    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D())

    model.add(Flatten())

    model.add(Dense(units=120, activation='relu'))

    model.add(Dense(units=84, activation='relu'))

    model.add(Dense(units=1, activation = 'sigmoid'))


    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=val_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=20
    )
    save_model(model)