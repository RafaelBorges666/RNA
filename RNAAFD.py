# Progeto: RNAAFD
# Objetivo: Percorrer um automato (AFD) usando RNA
# Autores: Rafael Borges.

import numpy as np
import warnings as wr
from sklearn import neural_network as nn
from sklearn.exceptions import ConvergenceWarning

# taxa de aprendizado
lr = 0.05

# dataset de treino

# o = 00
# g = 01
# c = 11
# a = 10 

#goagcog

                        #q0,q1,q2,q3,q4,q5,q6,q7,q8,q9
treinoEntrada = np.array([
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,1], #q0 - g - q1
                        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,0], #q1 - o - q2
                        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1,0], #q2 - a - q4
                        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,1], #q4 - g - q5
                        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,1], #q5 - c - q7
                        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,0], #q7 - o - q8
                        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,1], #q8 - g - q9                       
                       ])

treinoSaida = np.array([
                        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,0], #q0 - g - q1
                        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1,0], #q1 - o - q2
                        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,1], #q2 - a - q4
                        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,1], #q4 - g - q5
                        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,0], #q5 - c - q7
                        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,1], #q7 - o - q8
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,1], #q8 - g - q9                       
                       ])

teste = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,1]]) #q0 - g - q1
         
mlp = nn.MLPClassifier(hidden_layer_sizes=(40,), max_iter=128, alpha=1e-4, solver='sgd', verbose=10, random_state=1, learning_rate_init=lr)

# treino
print('#################### EXECUCAO ####################')
print('Treinamento') 
with wr.catch_warnings():
    wr.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
    mlp.fit(treinoEntrada, treinoSaida)

# teste
print('Testes') 
Y = mlp.predict(treinoEntrada)

# resultado 
print('Resultado procurado') 
print(treinoSaida)
print("Score de treino: %f" % mlp.score(treinoEntrada, treinoSaida))
print('Resultado encontrado') 
print(Y)

sumY = [sum(Y[i]) for i in range(np.shape(Y)[0])] # saida
sumT = [sum(treinoSaida[i]) for i in range(np.shape(treinoSaida)[0])] # target

print('Comparacao de resultados') 
print(np.logical_xor(sumY, sumT))
print('\n')

print('Percorrendo automato \n')

Y = mlp.predict(teste)
print(Y)
print('\n')

for i in range(2,8):
    Y = mlp.predict(Y)
    print(Y)
    print('\n')
