from Datos import Datos
from EstrategiaParticionado import ValidacionCruzada
from EstrategiaParticionado import ValidacionSimple
from Clasificador import ClasificadorNaiveBayes
from Clasificador import ClasificadorVecinosProximos
#from sklearn import preprocessing
#from sklearn import cross_validation
#from sklearn.naive_bayes import MultinomialNB
#from sklearn.naive_bayes import GaussianNB
import numpy as np
import matplotlib.pyplot as plt
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.linear_model import LogisticRegression
from Clasificador import ClasificadorRegLog
from Clasificador import AlgoritmoGenetico

from copy import deepcopy


dat = Datos("ejemplo5.data", True)
dat2 = deepcopy(dat)
val =  ValidacionSimple()
# Instancia el algoritmo con el que quieres clasificar
clas = AlgoritmoGenetico()

#---------AG PROPIO-----------#

errores = clas.validacion(val, dat, clas)
print(str(clas.generaciones) + " Generaciones con " + str(clas.poblacion) + " individuos y numero de reglas " + str(clas.num_reglas))

print ("Algoritmo Genetico, tasa de error:", errores[0])

print "Mejor individuo (1 regla):"
print(clas.mejor_individuo)

print "Fitness medio por generacion:"
print(clas.fitness_medio_gen)

print "Diccionario:"

print(dat.diccionarios)

print "Fitness medio de la generacion:"
plt.plot(clas.fitness_medio_gen)
plt.show()

print "Fitness del mejor individuo:"
plt.plot(clas.fitness_mejores_individuos)
plt.show()

#---------PLOT GRAFICA-----------#

#ii = val.particiones[-1].indicesTrain
#plotModel(dat.datos[ii,0],dat.datos[ii,1],dat.datos[ii,-1]!=0,clas, "Frontera", dat.diccionario
