from abc import ABCMeta,abstractmethod
from numpy import random
import numpy as np
from Datos import Datos


class Particion():
  
  indicesTrain=[]
  indicesTest=[]
  
  def __init__(self):
    self.indicesTrain=[]
    self.indicesTest=[]

#####################################################################################################

class EstrategiaParticionado(object):
  
  # Clase abstracta
  __metaclass__ = ABCMeta
  
  # Atributos: deben rellenarse adecuadamente para cada estrategia concreta
  nombreEstrategia="null"
  numeroParticiones=0
  particiones=[]
  
  
  
  @abstractmethod
  # TODO: esta funcion deben ser implementadas en cada estrategia concreta  
  def creaParticiones(self,datos,seed=None, porcentaje=0.5):



    pass
  

#####################################################################################################

class ValidacionSimple(EstrategiaParticionado):
  
  
  # Crea particiones segun el metodo tradicional de division de los datos segun el porcentaje deseado.
  # Devuelve una lista de particiones (clase Particion)
  # TODO: implementar


  def creaParticiones(self,datos,seed=None, porcentaje=0.5):

    random.seed(seed)

    self.nombreEstrategia = "Simple"
    self.numeroParticiones = 2
    particion_aux = Particion()
    array_aux = np.array(())

    array_aux = random.permutation((len(datos.datos)))

    particion_aux.indicesTrain = array_aux[0:int((len(array_aux)*porcentaje))-1]
    particion_aux.indicesTest = array_aux[int((len(array_aux)*porcentaje)):len(array_aux)]

    self.particiones.append(particion_aux)
    pass

      

#####################################################################################################
class ValidacionCruzada(EstrategiaParticionado):
  # Crea particiones segun el metodo de validacion cruzada.
  # El conjunto de entrenamiento se crea con las nfolds-1 particiones
  # y el de test con la particion restante
  # Esta funcion devuelve una lista de particiones (clase Particion)
  # TODO: implementar
  def creaParticiones(self,datos,seed=None, numParticiones=5):
    random.seed(seed)

    self.nombreEstrategia = "Cruzada"
    self.numeroParticiones = numParticiones
    array_aux = np.array(())

    array_aux = random.permutation((len(datos.datos)))

    for i in range(0, self.numeroParticiones):
        particion_aux = Particion()
        if (i==0):
            particion_aux.indicesTest = array_aux[0: (len(array_aux)/numParticiones)-1]
            particion_aux.indicesTrain = array_aux[(len(array_aux)/numParticiones):len(array_aux)]
        elif (i == numParticiones - 1):
            particion_aux.indicesTest = array_aux[(numParticiones - 1) * (len(array_aux) / numParticiones): len(array_aux)]
            particion_aux.indicesTrain = array_aux[0: (numParticiones - 1) * (len(array_aux) / numParticiones) - 1]
        else:
            particion_aux.indicesTest = array_aux[i * len(array_aux) / numParticiones:((i + 1) * len(array_aux) / numParticiones) - 1]
            particion_aux.indicesTrain = np.concatenate((array_aux[0:(i * len(array_aux) / numParticiones) - 1], array_aux[((i + 1) * len(array_aux) / numParticiones):len(array_aux)]))
        self.particiones.append(particion_aux)
    pass

