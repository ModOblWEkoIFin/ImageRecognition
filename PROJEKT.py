# Wczytanie wymaganych bibliotek
import sys

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.models import Sequential
from keras.utils import to_categorical
from keras.optimizers import SGD

# Wczytanie danych CIFAR10
cifar10 = tf.keras.datasets.cifar10
(X_train, y_train), (X_val, y_val) = cifar10.load_data()

print(f'Zbiór uczący: {X_train.shape}, zbiór walidacyjny: {X_val.shape}')

firstFig = plt.imshow(X_train[0])
plt.colorbar(firstFig).remove()
plt.show()


def plot_digit(digit, dem=32, font_size=8, rgb_layer=0):
    max_ax = font_size * dem

    plt.figure(figsize=(10, 10))
    plt.xlim([0, max_ax])
    plt.ylim([0, max_ax])
    plt.axis('off')

    for idx in range(dem):
        for jdx in range(dem):
            t = plt.text(idx * font_size, max_ax - jdx * font_size,
                         # wyswietla wartości skladowej RGB podanej w rgb_layer
                         digit[jdx][idx][rgb_layer], fontsize=font_size,
                         color="#000000")
            c0 = digit[jdx][idx][0] / 255
            c1 = digit[jdx][idx][1] / 255
            c2 = digit[jdx][idx][2] / 255
            t.set_bbox(dict(facecolor=(c0, c1, c2), alpha=0.5,
                            edgecolor='#f1f1f1'))

    plt.show()


# wyswietlenie obrazu z wartosciami rgb dla poszczególnych warstw
plot_digit(X_train[0], rgb_layer=0)
plot_digit(X_train[0], rgb_layer=1)
plot_digit(X_train[0], rgb_layer=2)


# Zbior uzytych klas
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
# Definicja zmiennych na podstawie klas
class_quantity = len(class_names)
img_height = 32
img_width = 32
batch_size = 64

# Wydrukowanie przykladowych obrazow z nazwami klas
plt.figure(figsize=(14, 10))
for i in range(40):
    plt.subplot(5, 8, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_train[i], cmap=plt.cm.binary)
    plt.xlabel(np.array(class_names)[y_train[i]])
plt.show()

# Dostosowanie danych do trenowania
X_train = X_train.astype('float32') / 255
X_val = X_val.astype('float32') / 255

y_train = to_categorical(y_train, len(class_names))
y_val = to_categorical(y_val, len(class_names))


######################################
# MR: MODEL wersja 1: ok 50% accuracy na 10 epokach
######################################
# def define_model():
#     model = Sequential()

#     # Zawsze dla modelu nalezy zdefiniowac rozmiar pierwszego obrazu (rozmiar x, rozmiar y, oznaczenie RGB)
#     model.add(Flatten(input_shape=(img_height, img_width, 3)))

#     # Przy wartosciach warstw neuronowych 128/16/10 - wartosc oceny modelu to tylko 10%. Porownaj sobie teraz z  768/256/10. Model niemalze pokrywa sie z danymi testowymi.
#     # Moznaby jeszcze pomanipulowac warstosciami w relu, aby otrzymac jeszcze dokladniejszy
#     # Start building Sequiential model in keras. We will use 3 layer MLP model for modelling the dataset.
#     model.add(Dense(768, activation='relu')) #rectified linear unit activation function
#     model.add(Dense(256, activation='relu')) #rectified linear unit activation function
#     model.add(Dense(class_quantity, activation='softmax')) #softmax converts a vector of value to a probability distribution
#     model.compile(optimizer='adam',
#                 loss='categorical_crossentropy',
#                 metrics=['accuracy'])
#     model.summary()
#     return model

######################################
# MR: MODEL wersja 2:
######################################
def define_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu',
                     kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu',
                     kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu',
                     kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu',
                     kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu',
                     kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu',
                     kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))

    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    return model


######################################
# MR: MODEL wersja 3:
# dodany batch optimalization i dropouty
######################################


# def define_model():
#     model = Sequential()
#     model.add(Conv2D(32, (3, 3), activation='relu',
#                      kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
#     model.add(BatchNormalization())
#     model.add(Conv2D(32, (3, 3), activation='relu',
#                      kernel_initializer='he_uniform', padding='same'))
#     model.add(BatchNormalization())
#     model.add(MaxPooling2D((2, 2)))
#     model.add(Dropout(0.2))
#     model.add(Conv2D(64, (3, 3), activation='relu',
#                      kernel_initializer='he_uniform', padding='same'))
#     model.add(BatchNormalization())
#     model.add(Conv2D(64, (3, 3), activation='relu',
#                      kernel_initializer='he_uniform', padding='same'))
#     model.add(BatchNormalization())
#     model.add(MaxPooling2D((2, 2)))
#     model.add(Dropout(0.3))
#     model.add(Conv2D(128, (3, 3), activation='relu',
#                      kernel_initializer='he_uniform', padding='same'))
#     model.add(BatchNormalization())
#     model.add(Conv2D(128, (3, 3), activation='relu',
#                      kernel_initializer='he_uniform', padding='same'))
#     model.add(BatchNormalization())
#     model.add(MaxPooling2D((2, 2)))
#     model.add(Dropout(0.4))
#     model.add(Flatten())
#     model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
#     model.add(BatchNormalization())
#     model.add(Dropout(0.5))
#     model.add(Dense(10, activation='softmax'))
#     opt = SGD(lr=0.001, momentum=0.9)
#     model.compile(optimizer=opt, loss='categorical_crossentropy',
#                   metrics=['accuracy'])
#     model.summary()

#     return model







######################
# Trenowanie modelu
######################
model = define_model()

history = model.fit(X_train,
                    y_train,
                    epochs=10,  # dla 10 epok dostalem pod 70% accuracy
                    # epochs=200, # jak ktos chce moze sprawdzic czy bedzie lepiej dla 200 epok 
                    verbose=1,
                    batch_size=batch_size,
                    validation_data=(X_val, y_val)
                    )
# Wyłącza zbiór walidacyjny (None) a zamiast tego random ze zbioru uczącego
# Kamil: Gdybysmy chcieli podzielic zbiory na: treningowy i testowy. Np. po to aby po zakoczeniu porownac oba zbiory
# history = model.fit(X_train,
#                     y_train,
#                     epochs=10,
#                     verbose=1,
#                     batch_size=batch_size,
#                     validation_split=0.2
#                     )


# Rysowanie wykresu. Mozna dostosowac argumenty, aby wykres byl lepiej widoczny
def draw_curves(history, key1='accuracy', ylim1=(0.0, 2.00),
                key2='loss', ylim2=(0.0, 2.00)):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history[key1], "r--")
    plt.plot(history.history['val_' + key1], "g--")
    plt.ylabel(key1)
    plt.xlabel('Epoch')
    plt.ylim(ylim1)
    plt.legend(['train', 'test'], loc='best')

    plt.subplot(1, 2, 2)
    plt.plot(history.history[key2], "r--")
    plt.plot(history.history['val_' + key2], "g--")
    plt.ylabel(key2)
    plt.xlabel('Epoch')
    plt.ylim(ylim2)
    plt.legend(['train', 'test'], loc='best')

    plt.show()


# Rysowanie wykresu
draw_curves(history, key1='accuracy', ylim1=(0.0, 2.00),
            key2='loss', ylim2=(0.0, 2.00))

#####################
# Uzycie wytrzenowanego modelu do rozpoznania poszczegolnych zdjec
#####################
y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)


# rysowanie predykcji dla danego zdjęcia, ktore wskazalismy
def plot_value_img(i, predictions, true_label, img):
    predictions, true_label, img = predictions[i], true_label[i], img[i]
    predicted_label = np.argmax(predictions)
    true_value = np.argmax(true_label)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)

    plt.yticks(np.arange(len(class_names)), class_names)
    thisplot = plt.barh(range(10), predictions, color="gray")
    thisplot[predicted_label].set_color('r')
    thisplot[true_value].set_color('g')

    plt.subplot(1, 2, 2)

    plt.imshow(img, cmap=plt.cm.binary)
    if predicted_label == true_value:
        color = 'green'
    else:
        color = 'red'

    plt.xlabel(
        "{} {:2.0f}% ({})".format(
            class_names[predicted_label], 100 * np.max(predictions), class_names[true_value]),
        color=color)
    plt.show()


# menu do sprawdzania wyników
def test():
    while True:
        x = input(
            "Podaj numer sklasyfikowanego obrazu z zakresu (1,9999) lub 0 w celu zakonczenia programu: ")
        try:
            x = int(x)
            if x == 0:
                sys.exit("Zamykanie...")
            plot_value_img(x, y_val_pred, y_val, X_val)
        except Exception as ex:
            print(f'Niepoprawny argument: {x}')
            print(f'Treść błędu: {ex}')


test()
