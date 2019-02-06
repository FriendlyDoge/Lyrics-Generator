__author__ = 'Maifriende & Gabriel'

import string

def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text

def clean_doc(doc):     #tirar pontuacao e maiuscula
    doc = doc.replace('-', ' ')
    tokens = doc.split()  #separa por espaço
    table = str.maketrans('', '', string.punctuation) #remove pontuacao
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()] #remove nao alfabeticos
    tokens = [word.lower() for word in tokens]
    return tokens

def save_doc(lines, filename):  #salvar em formato de token
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()

#carrega musicas
in_filename = 'musicas_marisa.txt'
doc = load_doc(in_filename)

#limpa texto e coloca em token
tokens = clean_doc(doc)

#organizar em sequencias
length = 100 + 1
sequences = list()
for i in range(length, len(tokens)):
    seq = tokens[i-length:i]
    line = ' '.join(seq)
    sequences.append(line)
out_filename = 'musicas_sequencias.txt'
save_doc(sequences, out_filename)

