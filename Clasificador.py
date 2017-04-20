from abc import ABCMeta, abstractmethod
import numpy as np
from math import log10
from math import sqrt
from math import fabs
from random import random
from random import randint
from random import choice
from copy import deepcopy
from scipy.stats import norm


class Clasificador(object):
    # Clase abstracta
    __metaclass__ = ABCMeta

    # Metodos abstractos que se implementan en casa clasificador concreto
    @abstractmethod
    # TODO: esta funcion deben ser implementadas en cada clasificador concreto
    # datosTrain: matriz numpy con los datos de entrenamiento
    # atributosDiscretos: array bool con la indicatriz de los atributos nominales
    # diccionario: array de diccionarios de la estructura Datos utilizados para la codificacion
    # de variables discretas
    def entrenamiento(self, datosTrain, atributosDiscretos, diccionario):
        pass

    @abstractmethod
    # TODO: esta funcion deben ser implementadas en cada clasificador concreto
    # devuelve un numpy array con las predicciones
    def clasifica(self, datosTest, atributosDiscretos, diccionario):
        pass

    # Obtiene el numero de aciertos y errores para calcular la tasa de fallo
    # TODO: implementar
    def error(self, datos, pred):
        nAciertos = 0
        nFallos = 0
        lineas = datos.shape[0]
        lineas2 = pred.shape[0]
        if (lineas != lineas2): return -1
        for i in range(0, lineas - 1):
            if (datos[i] == pred[i]):
                nAciertos = nAciertos + 1
            else:
                nFallos = nFallos + 1
        return nFallos / float(nFallos + nAciertos)

    # Realiza una clasificacion utilizando una estrategia de particionado determinada
    # TODO: implementar esta funcion
    def validacion(self, particionado, dataset, clasificador, apartado3=False, apartado4=False, seed=None):
        particionado.creaParticiones(dataset)

        datos_train = np.array(())
        datos_test = np.array(())

        errores = []

        if particionado.nombreEstrategia == 'Simple':
            datos_train = dataset.extraeDatos(particionado.particiones[0].indicesTrain)
            datos_test = dataset.extraeDatos(particionado.particiones[0].indicesTest)

            clasificador.entrenamiento(datos_train, dataset.nominalAtributos, dataset.diccionarios)
            pred = clasificador.clasifica(datos_test, dataset.nominalAtributos, dataset.diccionarios)
            #print(pred)
            errores.append(self.error(datos_test[:, datos_test.shape[1] - 1], pred))

        elif particionado.nombreEstrategia == 'Cruzada':
            for i in range(0, particionado.numeroParticiones):
                datos_train = dataset.extraeDatos(particionado.particiones[i].indicesTrain)
                datos_test = dataset.extraeDatos(particionado.particiones[i].indicesTest)
                clasificador.entrenamiento(datos_train, dataset.nominalAtributos, dataset.diccionarios)
                pred = clasificador.clasifica(datos_test, dataset.nominalAtributos, dataset.diccionarios)

                errores.append(self.error(datos_test[:, datos_test.shape[1] - 1], pred))

        return errores, pred, datos_test[:, datos_test.shape[1] - 1]

    def score(self, datosTest, atributosDiscretos, diccionario):
        scores = np.zeros((len(datosTest), len(diccionario[-1])))
        preds = map(lambda x : int(x), self.clasifica(datosTest, atributosDiscretos, diccionario))
        scores[range(datosTest.shape[0]),preds] = 1.0
        return scores

#############################################################################

class ClasificadorAPriori(Clasificador):
    mayoritaria = 0

    # TODO: implementar
    def entrenamiento(self, datostrain, atributosDiscretos=None, diccionario=None):
        # if (diccionario == None):
        #    return -1

        # Obtener la clase mayoritaria de los datos
        sh = datostrain.shape[1]
        self.mayoritaria = np.argmax(np.bincount(datostrain[:, sh - 1].astype(int)))
        pass



        # TODO: implementar

    def clasifica(self, datostest, atributosDiscretos=None, diccionario=None):
        # Asignar la clase mayoritaria a todos los datos
        datospred = np.copy(datostest)
        num_filas, num_cols = datostest.shape
        for i in range(0, num_filas):
            datospred[i, num_cols - 1] = self.mayoritaria

        return datospred[:, num_cols - 1]


##############################################################################

class ClasificadorNaiveBayes(Clasificador):
    arrayprobclases = []
    medias = []
    desviaciones = []
    tamanioclase = 0
    arrayclases = []
    laplace = False

    # TODO: implementar
    def entrenamiento(self, datostrain, atributosDiscretos, diccionario):

        arrayProbabilidades = []
        diccionarioclases = {}

        for i in range(0, diccionario[diccionario.__len__() - 1].__len__()):  # iteramos tantas veces como clases existan
            array_aux = []
            for j in range(0, datostrain.shape[0]):
                if (datostrain[j, atributosDiscretos.__len__() - 1] == i):
                    array_aux.append(j)
            diccionarioclases[i] = array_aux[:]  # Hay tantos diccionarios como clases. El diccionario de clase i tiene los indices de filas donde aparece la clase i
            self.arrayprobclases.append(array_aux.__len__() / float(datostrain.shape[0]))
        self.tamanioclase = diccionarioclases.__len__()
        for i in range(0, diccionario[diccionario.__len__() - 1].__len__()):  # volvemos a iterar por tantas clases existan
            datostrain_aux = datostrain[diccionarioclases[i]]  # recorremos todas las filas donde aparece clase j
            mediasaux = []
            desviacionesaux = []
            for j in range(0, diccionario.__len__() - 1):  # ahora iteramos por cols
                if (atributosDiscretos[j] == True):
                    valoresposibles, contador = np.unique(datostrain_aux[:, j], return_counts=True)
                    suma = np.sum(contador)
                    diccionario_aux = {}
                    for k in diccionario[j].values():
                        diccionario_aux[k] = 0
                    for l in range(0, valoresposibles.shape[0]):  # insertamos posibles valores de cada col en el diccionario de probabilidades
                        valor = (contador[l] / float(suma))
                        diccionario_aux[int(valoresposibles[l])] = valor  # a cada posible valor de columna k le corresponde una prob de clase l
                    arrayProbabilidades.insert(j, diccionario_aux.copy())
                    mediasaux.append(-1)
                    desviacionesaux.append(-1)
                else:
                    arrayProbabilidades.append({})
                    media = np.array(datostrain_aux[:, j]).astype(np.float)
                    mediasaux.append(np.mean(media))
                    if (self.laplace == False):
                        desviacionesaux.append(np.std(media))
                    else:
                        desviacionesaux.append(np.std(media) + 1e-6)
            self.desviaciones.append(desviacionesaux)
            self.medias.append(mediasaux)
            self.arrayclases.append(arrayProbabilidades[:])
        pass

    # TODO: implementar
    def clasifica(self, datostest, atributosDiscretos, diccionario):
        probabilidadrow = []
        predicciones = []

        datostest_aux = np.copy(datostest)

        for i in range(0, datostest_aux.shape[0]):  # iteramos num filas
            k = 0
            for arrayProbabilidades in self.arrayclases:  # iteramos numero de clases
                productorio = 1
                for j in range(0, diccionario.__len__()-1):  # iteramos n columnas
                    if atributosDiscretos[j] == True:
                        int_aux = int(datostest_aux[i, j])
                        productorio = productorio * arrayProbabilidades[j][int_aux]
                    else:
                        normal = norm.pdf(datostest_aux[i, j], self.medias[k][j], self.desviaciones[k][j])
                        productorio = productorio * normal
                probclase = self.arrayprobclases[self.arrayclases.index(arrayProbabilidades)]
                productorio = productorio * probclase
                probabilidadrow.append(productorio)
                k = k + 1
            ##predicciones.append(probabilidadrow.index(max(probabilidadrow)))
            valor = probabilidadrow.index(max(probabilidadrow))
            datostest_aux[i, diccionario.__len__() - 1] = valor
            probabilidadrow = []
        self.arrayclases = []
        return datostest_aux[:, diccionario.__len__() - 1]


#############################################################################

class ClasificadorVecinosProximos(Clasificador):
    medias_train = []
    desvs_train = []

    medias_test = []
    desvs_test = []

    k_vecinos = 11

    datostrain = np.array(())
    # TODO: implementar
    def entrenamiento(self, datostrain, atributosDiscretos=None, diccionario=None):

        self.datostrain = datostrain

        for i in range(0, datostrain.shape[1] - 1):
            self.medias_train.append(np.mean(datostrain[:, i]))
            self.desvs_train.append(np.std(datostrain[:, i]))

        for dato in datostrain:
            for i in range(0, len(dato) - 1):
                dato[i] = (dato[i] - self.medias_train[i]) / self.desvs_train[i]



    def clasifica(self, datostest, atributosDiscretos=None, diccionario=None):
        arrayClasifica = []
        distancias = np.array(())

        for i in range(0, datostest.shape[1] -1):
            self.medias_test.append(np.mean(datostest[:, i]))
            self.desvs_test.append(np.std(datostest[:, i]))
        datostest_aux = np.copy(datostest)

        for dato in datostest_aux:
            for i in range(0, len(dato) - 1):
                dato[i] = (dato[i] - self.medias_test[i]) / self.desvs_test[i]

        for i in range(0, datostest_aux.shape[0]):
            for j in range(0, self.datostrain.shape[0]):
                distancias_puntos = []
                for k in range (0, datostest.shape[1] -1):
                    distancias_puntos.append(abs(datostest_aux[i][k] - self.datostrain[j][k]) ** 2)
                if (j == 0):
                    distancias_aux = np.array(())
                    array_aux = np.array([sqrt(sum(distancias_puntos)), self.datostrain[j][self.datostrain.shape[1] - 1]])
                    distancias_aux = np.append(distancias_aux, array_aux)
                else:
                    array_aux = np.array([sqrt(sum(distancias_puntos)), self.datostrain[j][self.datostrain.shape[1] - 1]])
                    distancias_aux = np.vstack([distancias_aux, array_aux])
            distancias_aux = distancias_aux[np.lexsort(np.fliplr(distancias_aux).T)]
            clases = distancias_aux[:self.k_vecinos][:, 1]
            clases_u, num = np.unique(clases, return_counts=True)
            n_max = max(num)
            mayor = np.where(num == n_max)
            datostest_aux[i][datostest_aux.shape[1] - 1] = clases_u[mayor[0][0]]

        return datostest_aux[:, datostest_aux.shape[1] - 1]

        pass


##############################################################################


class ClasificadorMulticlase(Clasificador):
    def __init__(self, clasificadorbase):
        self.clasificadorbase = ClasificadorRegLog()
        self.clasificadores = []

    def entrenamiento(self, datostrain, atributosDiscretos, diccionario):

        # se van diferentes labels en funcion de la estrategia multiclase
        n_classes = len(diccionario[-1])
        self.clasificadores = []
        ovadiccionario = deepcopy(diccionario)
        ovadiccionario[-1] = {'-': 0, '+': 1}


        for i in range(n_classes):
            new_y = np.zeros((datostrain.shape[0], 1))
            new_y[datostrain[:, -1] == i, :] = 1
            self.clasificadores.append(deepcopy(self.clasificadorbase))
            self.clasificadores[i].entrenamiento(np.append(datostrain[:, :-1], new_y, axis=1), atributosDiscretos,
                                                 ovadiccionario)

    def clasifica(self, datostest, atributosDiscretos, diccionario):

        scores = np.zeros((datostest.shape[0], len(self.clasificadores)))
        ovadiccionario = deepcopy(diccionario)
        ovadiccionario[-1] = {'-': 0, '+': 1}

        # evaluar el score para cada clasificador one-versus-all
        for i, c in enumerate(self.clasificadores):
            scores[:, i] = c.score(datostest, atributosDiscretos, ovadiccionario)[:, 1]

        # se predice como aquella clase con mas confianza
        preds = np.argmax(scores, axis=1)
        return preds


# noinspection PyTypeChecker
class ClasificadorRegLog(Clasificador):
    # TODO: implementar

    epocas = 10
    cteApr = 1.5
    wFinal = np.array(())

    def entrenamiento(self, datostrain, atributosDiscretos=None, diccionario=None):

        nrows, ncol = datostrain.shape
        w = np.array(())

        #Generacion de vector aleatorio de <numero atributos+1> elementos
        for i in range(0, ncol):
            w = np.append(w, random()*2 -1)

        for j in range(0, self.epocas):
            for i in range(0, nrows):
                x = np.array(())
                x = np.append(x, 1)
                for k in range(1, ncol):
                    x = np.append(x, float(datostrain[i][k-1]))
                aux = 0
                aux = np.dot(np.transpose(w), x) #pescalar
                e = (np.exp(-aux))
                aux = 1/(1 + e)
                for k in range(0, ncol):
                    w[k] = w[k] - self.cteApr*(aux - datostrain[i, ncol-1])*x[k]

        self.wFinal = w


    def clasifica(self, datostest, atributosDiscretos=None, diccionario=None):

        nrows, ncol = datostest.shape
        pred = np.copy(datostest)
        i=0
        for fila in pred:

            x = np.array(())

            x = np.append(x, 1)
            for i in range (1, ncol):
                x = np.append(x, float(fila[i-1]))

            aux = np.dot(np.transpose(self.wFinal),x)
            e = (np.exp(-aux))
            aux = 1 / (1 + e)


            if (aux < 0.5):
                fila[ncol-1] = 0
            else:
                fila[ncol-1] = 1

        return pred[:, pred.shape[1] - 1]

        pass

##############################################################################

class AlgoritmoGenetico(Clasificador):

    probCruce = 0.6
    probMutacion = 0.005
    probElitismo = 0.05
    poblacion = 500
    generaciones = 100 # 10 individuos y generaciones, por ejemplo
    fitness_medio_gen = []
    fitness_mejores_individuos = []

    num_reglas = 2
    atributos = np.array(())

    def entrenamiento(self, datostrain, atributosDiscretos=None, diccionario=None):

        Poblacion = np.array(())

        #Poblacion aleatoria
        bits = 0
        nrows, ncols = datostrain.shape
        Poblacion = []
        mejor_fitness = 0
        mejor_individuo = []
        fitnesses = []
        condicionParada = False
        lista_fitness = np.array(())


        for i in range(0, ncols - 1):
            values = np.unique(datostrain[:, i])
            self.atributos = np.append(self.atributos, len(values)).astype(int)
        #Mas un bit para la clase
        bits = np.sum(self.atributos).astype(int)

        bits += 1

        for i in range(0, self.poblacion):
            Individuo_aleatorio = []
            for k in range(0, self.num_reglas):
                regla_n = np.array(())
                for j in range (0, bits):
                    bit = randint(0, 1)
                    regla_n = np.append(regla_n, bit).astype(int)
                Individuo_aleatorio.append(regla_n)
            Poblacion.append(Individuo_aleatorio)

        #Seleccion de progenitores

        fitness_individuos = []
        ind = 0
        for individuo in Poblacion:
            prediccion = np.array(())
            for row in range(0, nrows):
                num_r = 0
                # para cada ejemplo usamos todas las reglas hasta que una lo clasifique
                for regla in individuo:
                    valores = []
                    clase = regla[-1]
                    index = 0
                    for k in range(0, self.atributos.__len__()):
                        regla_atr = regla[index:index + self.atributos[k]]
                        valores.append(np.array((np.where(regla_atr == 1))))
                        index += self.atributos[k]
                    conclusion = True
                    for col in range(0, ncols - 1):
                        dato = datostrain[row][col]
                        vals = valores[col]
                        if dato not in vals and conclusion == True:
                            conclusion = False
                            # break
                    if conclusion == True:
                        prediccion = np.append(prediccion, clase)
                        # ya hemos clasificado, asi que salimos del bucle de las reglas: primera regla que clasifique
                        break
                    elif conclusion == False and (self.num_reglas == 1 or num_r == len(individuo) - 1):
                        # Si no se cumple ninguna regla, se coge la contraria a la ultima
                        if clase == 1:
                            prediccion = np.append(prediccion, 0)
                        else:
                            prediccion = np.append(prediccion, 1)
                    num_r += 1

            fitness = self.error(datostrain[:,- 1], prediccion)
            fitness = 1 - fitness
            fitnesses.append(fitness)
            lista_aux = []
            lista_aux.append(ind)
            lista_aux.append(fitness)
            if (ind == 0):
                lista_fitness = lista_aux[:]
                lista_aux = []
            else:
                lista_fitness = np.vstack([lista_fitness, lista_aux])
                lista_aux = []

            if fitness > mejor_fitness:
                mejor_fitness = fitness
                mejor_individuo = individuo

            ind += 1

        #fitness_individuos.append(reduce(lambda x, y: x + y, fitnesses) / len(fitnesses))
        self.fitness_mejores_individuos.append(mejor_fitness)
        probs_fitnesses = []
        for i in range(0, len(fitnesses)):
            probs_fitnesses.append(fitnesses[i]/sum(fitnesses))

        # Guardamos fitness medio de la primera generacion

        self.fitness_medio_gen.append(sum(fitnesses)/self.poblacion)

        gens = 0
        while gens < self.generaciones and mejor_fitness < 0.95:
            # Seleccion progenitores
            p_prima = []
            for i in range(0, self.poblacion):
                rands = np.array(range(0, self.poblacion))
                ind_aleat = np.random.choice(rands, p=probs_fitnesses)
                p_prima.append(Poblacion[ind_aleat])

            # Recombinacion en un punto

            for i in range(0, len(p_prima)/2):
                punto_cruce = randint(1, bits - 1)
                prob = random()
                if prob < self.probCruce:
                    rand1 = randint(0, self.num_reglas -1)
                    rand2 = randint(0, self.num_reglas - 1)
                    padre = p_prima[i][rand1]
                    padre2 = p_prima[(len(p_prima)/2)+i][rand2]
                    vastago1 = np.concatenate((padre[0:punto_cruce], padre2[punto_cruce:]))
                    vastago2 = np.concatenate((padre2[0:punto_cruce], padre[punto_cruce:]))
                    p_prima[i][rand1] = vastago1
                    p_prima[(len(p_prima) / 2) + i][rand2] = vastago2

            # Mutacion

            for i in range(0, len(p_prima)):
                prob = random()
                if prob < self.probMutacion:
                    bit = randint(0,bits-1)
                    ele = p_prima[i][randint(0,self.num_reglas-1)]
                    if ele[bit] == 1:
                        ele[bit] = 0
                    else:
                        ele[bit] = 1

            # Elitismo

            num_individuos_elite = int(self.poblacion*self.probElitismo)
            aux2 = np.fliplr(lista_fitness)

            lista = np.sort(aux2[:,0])

            elite = lista[len(lista)-num_individuos_elite:len(lista)]

            for i in range (0, len(elite)):
                for ind in lista_fitness:
                    if ind[1] == elite[i]:
                        p_prima.pop(len(p_prima) - i - 1)
                        p_prima.append(Poblacion[int(ind[0])])


            # for i in range(0, num_individuos_elite):



            # Seleccion
            #Clasificamos los progenitores

            fitnesses = []
            fitness_individuos = []
            ind = 0

            for individuo in p_prima:
                prediccion = np.array(())
                for row in range(0, nrows):
                    num_r = 0
                    # para cada ejemplo usamos todas las reglas hasta que una lo clasifique
                    for regla in individuo:
                        valores = []
                        clase = regla[-1]
                        index = 0
                        for k in range(0, self.atributos.__len__()):
                            regla_atr = regla[index:index + self.atributos[k]]
                            valores.append(np.array((np.where(regla_atr == 1))))
                            index += self.atributos[k]
                        conclusion = True
                        for col in range(0, ncols - 1):
                            dato = datostrain[row][col]
                            vals = valores[col]
                            if dato not in vals and conclusion == True:
                                conclusion = False
                                # break
                        if conclusion == True:
                            prediccion = np.append(prediccion, clase)
                            # ya hemos clasificado, asi que salimos del bucle de las reglas: primera regla que clasifique
                            break
                        elif conclusion == False and (self.num_reglas == 1 or num_r == len(individuo) - 1):
                            # Si no se cumple ninguna regla, se coge la contraria a la ultima
                            if clase == 1:
                                prediccion = np.append(prediccion, 0)
                            else:
                                prediccion = np.append(prediccion, 1)
                        num_r += 1

                fitness = self.error(datostrain[:, - 1], prediccion)
                fitness = 1 - fitness
                fitnesses.append(fitness)
                lista_aux = []
                lista_aux.append(ind)
                lista_aux.append(fitness)
                if (ind == 0):
                    lista_fitness = lista_aux[:]
                    lista_aux = []
                else:
                    lista_fitness = np.vstack([lista_fitness, lista_aux])
                    lista_aux = []

                if fitness > mejor_fitness:
                    mejor_fitness = fitness
                    mejor_individuo = individuo

                ind += 1

            self.fitness_mejores_individuos.append(mejor_fitness)
            #fitness_individuos.append(reduce(lambda x, y: x + y, fitnesses) / len(fitnesses))

            probs_fitnesses = []
            for i in range(0, len(fitnesses)):
                probs_fitnesses.append(fitnesses[i] / sum(fitnesses))

            self.fitness_medio_gen.append(sum(fitnesses)/self.poblacion)

            Poblacion = p_prima

            gens += 1

        self.mejor_individuo = mejor_individuo





    def clasifica(self, datostest, atributosDiscretos=None, diccionario=None):

        nrows, ncols = datostest.shape
        prediccion = np.array(())
        ultima_regla = self.mejor_individuo[:][-1]

        for row in range(0, nrows):
            num_r = 0
            #para cada ejemplo usamos todas las reglas hasta que una lo clasifique
            for regla in self.mejor_individuo:
                valores = []
                clase = regla[-1]
                index = 0
                for k in range(0, self.atributos.__len__()):
                    regla_atr = regla[index:index + self.atributos[k]]  # CAMBIAR EL INDIVIDUO[0] POR INDIVIDUO[NUM_REGLA]
                    valores.append(np.array((np.where(regla_atr == 1))))
                    index += self.atributos[k]
                conclusion = True
                for col in range(0, ncols - 1):
                    dato = datostest[row][col]
                    vals = valores[col]
                    if dato not in vals and conclusion == True:
                        conclusion = False
                        # break
                if conclusion == True:
                    prediccion = np.append(prediccion, clase)
                    # ya hemos clasificado, asi que salimos del bucle de las reglas
                    break
                elif conclusion == False and (self.num_reglas == 1 or num_r == len(self.mejor_individuo)-1):
                    if clase == 1:
                        prediccion = np.append(prediccion, 0)
                    else:
                        prediccion = np.append(prediccion, 1)
                num_r += 1

        return prediccion


class ClasificadorEnsemble(Clasificador):

    clasificadores = [ClasificadorNaiveBayes(), ClasificadorMulticlase(ClasificadorRegLog), ClasificadorVecinosProximos()]

    def entrenamiento(self, datostrain, atributosDiscretos=None, diccionario=None):

        for cl in self.clasificadores:
            cl.entrenamiento(datostrain, atributosDiscretos, diccionario)

    def clasifica(self, datostest, atributosDiscretos=None, diccionario=None):

        nrows, ncols = datostest.shape
        preds = []
        ret = []
        ponderaciones = []
        for cl in self.clasificadores:
            preds.append(cl.clasifica(datostest, atributosDiscretos, diccionario))

        for i in range(0, nrows):
            ponderaciones = []
            for pred in preds:
                ponderaciones.append(pred[i])

            c, n = np.unique(ponderaciones, return_counts=True)

            ind = np.argmax(n)
            ret = np.append(ret, c[ind])
        return ret

pass
