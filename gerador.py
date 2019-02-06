__author__ = 'Maifriende & Gabriel'

from random import randint
from pickle import load
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text

def generate_seq(model, tokenizer, seq_length, seed_text, n_words):
    result = list()
    in_text = seed_text
    for _ in range(n_words):
        encoded = tokenizer.texts_to_sequences([in_text])[0]                    #transformar de texto para integer
        encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre') #truncate para evitar entrada muito longa
        yhat = model.predict_classes(encoded, verbose=0)                        #preve probabilidade pra cada palavra
                                                                                # e retorna o index da palavra com maior probabilidade
        out_word = ''
        for word, index in tokenizer.word_index.items():                        #olhar index no mapeamento de Tokenizer para pegar palavra associada ao index
            if index == yhat:
                out_word = word
                break
        in_text += ' '+out_word
        result.append(out_word)
    return ' '.join(result)

in_filename = 'musicas_sequencias.txt'
doc = load_doc(in_filename)
lines = doc.split('\n')
seq_length = len(lines[0].split()) - 1
model = load_model('model.h5')                  #load model from keras
tokenizer = load(open('tokenizer.pkl', 'rb'))   #load from pickle
seed_text = lines[randint(0, len(lines))]       #parte aletoria do texto como semente
print(seed_text + '\n')
num_palavras = 100
generated = generate_seq(model, tokenizer, seq_length, seed_text, num_palavras)
print(generated)
