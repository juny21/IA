import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator

# Carregar os dados do MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Pré-processamento dos dados
X_train = X_train.reshape((-1, 28, 28, 1)).astype('float32') / 255.0
X_test = X_test.reshape((-1, 28, 28, 1)).astype('float32') / 255.0

# Criação da função para construir o modelo
def create_model(num_filters, dropout_rate=None, batch_normalization=False):
    model = Sequential()
    model.add(Conv2D(filters=num_filters, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(units=128, activation='relu'))
    
    if dropout_rate:
        model.add(Dropout(dropout_rate))
    
    if batch_normalization:
        model.add(BatchNormalization())
    
    model.add(Dense(units=10, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# A. Melhor quantidade de camadas de convolução-pooling: 1
model = create_model(32)
history = model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
_, accuracy = model.evaluate(X_train, y_train, verbose=0)
print("A. Melhor quantidade de camadas de convolução-pooling: 1")
print("   Acurácia correspondente: {:.4f}".format(accuracy))

# B. Comparação de diferentes quantidades de feature maps (filters)
filters = [16, 32, 64]
accuracies = []

for num_filters in filters:
    model = create_model(num_filters)
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
    _, accuracy = model.evaluate(X_train, y_train, verbose=0)
    accuracies.append(accuracy)

best_num_filters = filters[np.argmax(accuracies)]
best_accuracy = np.max(accuracies)

print("B. Melhor quantidade de feature maps (filters): {}".format(best_num_filters))
print("   Acurácia correspondente: {:.4f}".format(best_accuracy))

# C. Comparação de diferentes quantidades de camadas densas
layers = [1, 2, 3]
accuracies = []

for num_layers in layers:
    model = create_model(best_num_filters)
    for _ in range(num_layers):
        model.add(Dense(units=128, activation='relu'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
    _, accuracy = model.evaluate(X_train, y_train, verbose=0)
    accuracies.append(accuracy)

best_num_layers = layers[np.argmax(accuracies)]
best_accuracy = np.max(accuracies)

print("C. Melhor quantidade de camadas densas: {}".format(best_num_layers))
print("   Acurácia correspondente: {:.4f}".format(best_accuracy))

# D. Comparação de diferentes porcentagens de dropout
dropout_rates = [0.0, 0.25, 0.5]
accuracies = []

for dropout_rate in dropout_rates:
    model = create_model(best_num_filters, dropout_rate=dropout_rate)
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
    _, accuracy = model.evaluate(X_train, y_train, verbose=0)
    accuracies.append(accuracy)

best_dropout_rate = dropout_rates[np.argmax(accuracies)]
best_accuracy = np.max(accuracies)

print("D. Melhor porcentagem de dropout: {}".format(best_dropout_rate))
print("   Acurácia correspondente: {:.4f}".format(best_accuracy))

# E. Aplicação de batch normalization
model = create_model(best_num_filters, dropout_rate=best_dropout_rate, batch_normalization=True)
history = model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
_, accuracy = model.evaluate(X_train, y_train, verbose=0)

print("E. Batch normalization")
print("   Acurácia correspondente: {:.4f}".format(accuracy))

# F. Aplicação de data augmentation
datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.1)
datagen.fit(X_train)

model = create_model(best_num_filters, dropout_rate=best_dropout_rate, batch_normalization=True)
history = model.fit(datagen.flow(X_train, y_train, batch_size=32), steps_per_epoch=len(X_train) / 32, epochs=10, verbose=0)
_, accuracy = model.evaluate(X_train, y_train, verbose=0)

print("F. Data augmentation")
print("   Acurácia correspondente: {:.4f}".format(accuracy))

# Avaliação no conjunto de dados de teste
_, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print("Acurácia no conjunto de dados de teste: {:.4f}".format(test_accuracy))
