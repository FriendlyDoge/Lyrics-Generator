__author__ = 'Maifriende & Gabriel'

import numpy.lib.polynomial as nppol
from pickle import dump
from keras.preprocessing.text import Tokenizer
from keras.utils import  to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding

def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text

in_filename = 'musicas_sequencias.txt'
doc = load_doc(in_filename)
lines = doc.split('\n')

#mapear palavra para inteiro
tokenizer = Tokenizer()                             #from keras
tokenizer.fit_on_texts(lines)                       #treinando Tokenizer para texto completo
sequences = tokenizer.texts_to_sequences(lines)     #selecionando palavras unicas e convertendo para integer
vocab_size = len(tokenizer.word_index) + 1          #word_index para acessar integer correspondente da word

#entrada (X) e saida (y)
sequences = nppol.array(sequences)
X, y = sequences[:,:-1], sequences[:,-1]
y = to_categorical(y, num_classes=vocab_size)       #keras cria one hot encode (0's e 1)
                                                    #1 para indicar a posição da palavra
                                                    #modelo aprende a prever a probabilidade de distribuicao para a proxima palavra
seq_length = X.shape[1]                             #numero de colunas da entrada = tamanho da sequencia (100)

#modelo
tam_embedding = 50     #tamanho do espaço do vetor de embedding
tam_batch = 128
tam_epochs = 20
tam_celulas = 400
tam_neuronios = 100
model = Sequential()
model.add(Embedding(vocab_size, tam_embedding, input_length=seq_length))
model.add(LSTM(tam_celulas, return_sequences=True))
model.add(LSTM(tam_celulas))
model.add(Dense(tam_neuronios, activation='relu'))  #funcao de ativacao Relu para interpretar as caracteristicas extraidas da sequencia
                                                    #camada de saida preve a proxima palavra com um vetor unico do tamanho do vocabulario
                                                    # com a propabilidade de cada palavra nele.
model.add(Dense(vocab_size, activation='softmax'))  #funcao de ativacao Softmax para garantir que as saidas tenham as caracteristicas das
                                                    # probabilidades normalizadas
print(model.summary())
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
                #to fit the model               #implementacao para gradiente mini-batch eficiente #acuracia usada como metrica
model.fit(X, y, batch_size=tam_batch, epochs=tam_epochs)
model.save('model.h5')                              #h5: keras format
dump(tokenizer, open('tokenizer.pkl', 'wb'))        #Pickle: salva tokenizer como mapeamento de word para integer
