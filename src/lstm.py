import numpy as np
import tensorflow as tf
 
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
 
from keras.optimizers import RMSprop


def read_data(file = 'src/input.txt'):
    with open(file, 'r', encoding="utf-8") as file:
        text = file.read()
    
    words = text.split() 
    vocabulary = sorted(list(set(words)))

    print(f"\nДлина текста {len(vocabulary)}\n")

    return words, vocabulary

def generate_data_educ(words, vocabulary, max_length = 4):
    sentences = []
    next_words = []
    for i in range(0, len(words) - max_length, 1):
        sentences.append(words[i:i + max_length]) 
        next_words.append(words[i + max_length])  
    
    word_to_index = {word: i for i, word in enumerate(vocabulary)}

    X = np.zeros((len(sentences), max_length, len(vocabulary)), dtype = np.bool_)
    y = np.zeros((len(sentences), len(vocabulary)), dtype = np.bool_)
    for i, sentence in enumerate(sentences):
        for t, word in enumerate(sentence):
            X[i, t, word_to_index[word]] = 1
        y[i, word_to_index[next_words[i]]] = 1

    return X, y

def gen_model(vocabulary, max_length):
    model = Sequential()
    model.add(LSTM(256, input_shape =(max_length, len(vocabulary))))
    model.add(Dense(len(vocabulary)))
    model.add(Activation('softmax'))
    optimizer = RMSprop(learning_rate = 0.01)
    model.compile(loss ='categorical_crossentropy', optimizer = optimizer)

    return model

def educ_model(model, X, y):
    model.fit(X, y, batch_size = 128, epochs = 30)

def load_model(file, vocabulary, max_length):
    model = gen_model(vocabulary, max_length)
    model.load_weights(file)
    return model

def save_model(model, file):
    model.save_weights(file)

def sample_index(preds, temperature = 1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_text(model, start, max_length, diversity, length, vocabulary):
    if(max_length != len(start)):
        return ''
    
    generated = ''
    sentence = start
    for str_s in sentence:
        generated += str_s + " " 

    index_to_word = {i: word for i, word in enumerate(vocabulary)}
    word_to_index = {word: i for i, word in enumerate(vocabulary)}

    for i in range(length):
            x_pred = np.zeros((1, max_length, len(vocabulary)))
            for t, word in enumerate(sentence):
                x_pred[0, t, word_to_index[word]] = 1.
 
            preds = model.predict(x_pred, verbose = 0)[0]
            next_index = sample_index(preds, diversity)
            next_word = index_to_word[next_index]
 
            generated += ' ' + next_word
            sentence = sentence[1:] 
            sentence.append(next_word)

    return generated

def inside_menu(model, vocabulary, len_chair):
    while True:
        print("1. Сгенерировать текст\n2. Сохранить модель\n3. Выход\n")

        choice = input("Выбор:")
        if choice == '1':
            length = int(input("Длина текста:"))
            start = input("Начальное словосочетание:")
            start = start.split()
            if len(start) != len_chair:
                print("Неверная длина префикса")
                return

            diversity = float(input("Креативность текста:"))

            text = generate_text(model, start, len_chair, diversity, length, vocabulary)

            with open('../result/gen.txt', 'w', encoding='utf-8') as file:
                file.write(text)
        elif choice == '2':
            file = input("Фаил:")
            save_model(model, file)
        else:
            return


if __name__ == "__main__":
    while True:
        print("1. Создать модель\n2. Загрузить модель\n3. Выход\n")

        choice = input("Выбор:")
        if choice == '1':
            input_file = input("Фаил:")
            words, vocabulary = read_data(input_file)

            len_chair = int(input("Длина префикса:"))
            model = gen_model(vocabulary, len_chair)

            X, y = generate_data_educ(words, vocabulary, len_chair)
            educ_model(model, X, y)

            inside_menu(model, vocabulary, len_chair)
        elif choice == '2':
            input_file = input("Фаил данных:")
            words, vocabulary = read_data(input_file)
            len_chair = int(input("Длина префикса:"))

            wed_file = input("Фаил весов:")
            model = load_model(wed_file, vocabulary, len_chair)

            inside_menu(model, vocabulary, len_chair)
        else:
            break








