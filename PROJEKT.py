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

firstFig = plt.imshow(X_train[0])
plt.colorbar(firstFig).remove()
plt.show()


def plot_digit(digit, dem=32, font_size=8):
    max_ax = font_size * dem

    plt.figure(figsize=(10,10))
    plt.xlim([0, max_ax])
    plt.ylim([0, max_ax])
    plt.axis('off')

    for idx in range(dem):
        for jdx in range(dem):
            t = plt.text(idx * font_size, max_ax - jdx * font_size,
                         digit[jdx][idx][0], fontsize=font_size, # wyswietla wartości pierwszej skladowej RGB, jak chcemy wszystkie 3 to trzeba przerobic
                         color="#000000")
            c0 = digit[jdx][idx][0] / 255
            c1 = digit[jdx][idx][1] / 255
            c2 = digit[jdx][idx][2] / 255
            t.set_bbox(dict(facecolor=(c0, c1, c2), alpha=0.5,
                            edgecolor='#f1f1f1'))

    plt.show()

# wyswietlenie obrazu z wartosciami odcieni szarosci (raczej nie jest nam to potrzebne)
plot_digit(X_train[0])
# Zbior uzytych klas
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

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

# Przelicz wartosci do przedzialu 0-1
X_train = X_train.astype('float32') / 255
X_val = X_val.astype('float32') / 255

y_train = to_categorical(y_train, len(class_names))
y_val = to_categorical(y_val, len(class_names))

model = Sequential()

# Zawsze dla modelu nalezy zdefiniowac rozmiar pierwszego obrazu (rozmiar x, rozmiar y, oznaczenie RGB)
model.add(Flatten(input_shape=(32, 32, 3)))

# Przy wartosciach warstw neuronowych 128/16/10 - wartosc oceny modelu to tylko 10%. Porownaj sobie teraz z  768/256/10. Model niemalze pokrywa sie z danymi testowymi.
# Moznaby jeszcze pomanipulowac warstosciami w relu, aby otrzymac jeszcze dokladniejszy
# Start building Sequiential model in keras. We will use 3 layer MLP model for modelling the dataset.
model.add(Dense(768, activation='relu')) #rectified linear unit activation function
model.add(Dense(256, activation='relu')) #rectified linear unit activation function
model.add(Dense(10, activation='softmax')) #softmax converts a vector of value to a probability distribution

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# Dostosowanie danych treningowych do skonfigurowanej sieci neuronowej
history = model.fit(X_train,
                    y_train,
                    epochs=10,
                    verbose=1,
                    batch_size=256,
                    validation_data=(X_val, y_val)
                    )
# nie bardzo wiem po co to ponizej bylo, z doca widzę że wyłączało zbiór walidacyjny (None) a zamiast tego random ze zbioru uczącego
# Kamil: Gdybysmy chcieli podzielic zbiory na: treningowy i testowy. Np. po to aby po zakoczeniu porownac oba zbiory
# history = model.fit(X_train,
#                     y_train,
#                     epochs=10,
#                     verbose=1,
#                     batch_size=256,
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

# Rysowanie wykresu. Mozna dostosowac argumenty, aby wykres byl lepiej widoczny
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

# Rysowanie wykresu. Mozna dostosowac argumenty, aby wykres byl lepiej widoczny
draw_curves(history_best, key1='accuracy', ylim1=(0.0, 2.00),
            key2='loss', ylim2=(0.0, 2.00))

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
        "{} {:2.0f}% ({})".format(class_names[predicted_label], 100 * np.max(predictions), class_names[true_value]),
        color=color)
    plt.show()


# menu do sprawdzania wyników

def test():
    while True:
        x = input("Podaj numer obrazu do sprawdzenia (1,9999) lub 0 w celu zakonczenia programu: ")
        try:
            x = int(x)
            if x == 0:
                sys.exit("Exiting...")
            plot_value_img(x, y_val_pred, y_val, X_val)
        except Exception as ex:
            print(f'Niepoprawny argument: {x}')
            print(f'Treść błędu: {ex}')

test()
