# Wczytanie wymaganych bibliotek
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Flatten
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.layers import Dropout

# Wczytanie danych CIFAR10
cifar10 = tf.keras.datasets.cifar10
(X_train, y_train), (X_val, y_val) = cifar10.load_data()

print(f'Zbiór uczący: {X_train.shape}, zbiór walidacyjny: {X_val.shape}')

plt.figure(figsize=(7, 7))
plt.imshow(X_train[0], cmap=plt.cm.binary)
plt.colorbar()
plt.show()


def plot_digit(digit, dem=32, font_size=8):
    max_ax = font_size * dem

    fig = plt.figure(figsize=(10, 10))
    plt.xlim([0, max_ax])
    plt.ylim([0, max_ax])
    plt.axis('off')

# TODO do zmiany pasek odcienia szarosci generowany po prawej przy pierwszym obrazie
    for idx in range(dem):
        for jdx in range(dem):
            t = plt.text(idx * font_size, max_ax - jdx * font_size,
                         digit[jdx][idx], fontsize=font_size,
                         color="#000000")
            c = digit[jdx][idx] / 255.
            t.set_bbox(dict(facecolor=(c, c, c), alpha=0.5,
                            edgecolor='#f1f1f1'))

    plt.show()

# TBD
# plot_digit(X_train[0])

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize=(14, 10))
for i in range(40):
    plt.subplot(5, 8, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_train[i], cmap=plt.cm.binary)
    plt.xlabel(np.array(class_names)[y_train[i]])
plt.show()

# Przelicz wartosci do przedzialu 0-1
X_train = X_train.astype('float32') / 255.0
X_val = X_val.astype('float32') / 255.0

y_train = to_categorical(y_train, len(class_names))
y_val = to_categorical(y_val, len(class_names))

model = Sequential()

model.add(Flatten(input_shape=(32, 32, 3)))

model.add(Dense(3000, activation='relu'))
model.add(Dense(1000, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

history = model.fit(X_train,
                    y_train,
                    epochs=10,
                    verbose=1,
                    batch_size=256,
                    validation_data=(X_val, y_val)
                    )

history = model.fit(X_train,
                    y_train,
                    epochs=10,
                    verbose=1,
                    batch_size=256,
                    validation_split=0.2
                    )

# Zmienilem rozmiary wykresu aby wartosci mogly sie zmiescic
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

# Zmienilem rozmiary wykresu aby wartosci mogly sie zmiescic
draw_curves(history, key1='accuracy', ylim1=(0.0, 2.00),
            key2='loss', ylim2=(0.0, 2.00))

model_best = Sequential()
model_best.add(Flatten(input_shape=(32, 32, 3)))
model_best.add(Dense(128, activation='relu'))
model_best.add(Dropout(0.3))
model_best.add(Dense(64, activation='relu'))
model_best.add(Dropout(0.3))
model_best.add(Dense(32, activation='relu'))
model_best.add(Dropout(0.3))
model_best.add(Dense(10, activation='softmax'))

model_best.compile(optimizer='adam',
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])

EarlyStop = EarlyStopping(monitor='val_loss',
                          patience=5,
                          verbose=1)

history_best = model_best.fit(X_train,
                              y_train,
                              epochs=50,
                              verbose=1,
                              batch_size=1024,
                              validation_data=(X_val, y_val),
                              callbacks=[EarlyStop]
                              )

# Zmienilem rozmiary wykresu aby wartosci mogly sie zmiescic
draw_curves(history_best, key1='accuracy', ylim1=(0.0, 2.00),
            key2='loss', ylim2=(0.0, 2.00))

y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)

# rysowanie predykcji dla danego zdjęcia

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
        "{} {:2.0f}% ({})".format(class_names[predicted_label], 100 * np.max(predictions), class_names[true_value]),
        color=color)
    plt.show()


# menu do sprawdzania wyników
# TODO dodanie warunku gdy podana wartosc jest z poza zakresu

def test():
    while True:
        x = input("Podaj numer obrazu do sprawdzenia (1,9999) lub 0 w celu zakonczenia programu: ")
        x = int(x)
        if x == 0:
            sys.exit("Exiting...")
        plot_value_img(x, y_val_pred, y_val, X_val)


test()
