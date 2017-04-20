import numpy as np


class Datos(object):
    supervisado = True
    TiposDeAtributos = ('Continuo', 'Nominal')
    tipoAtributos = []
    nombreAtributos = []
    nominalAtributos = []
    datos = np.array(())
    # Lista de diccionarios. Uno por cada atributo.
    diccionarios = []

    # TODO: procesar el fichero para asignar correctamente las variables supervisado, tipoAtributos, nombreAtributos,nominalAtributos, datos y diccionarios
    def __init__(self, nombreFichero, sup):
        with open(nombreFichero, "r") as f:

            lista_aux = []
            diccionario_aux = {}
            lineas = int(f.readline())
            array_aux = np.array(()).astype(int)
            self.supervisado=sup
            self.nombreAtributos = f.readline().split(',')
            self.tipoAtributos = f.readline().split(',')

            pos = f.tell()

            for i in range(0,self.nombreAtributos.__len__()):
                if (self.tipoAtributos[i].strip('\r\n') == 'Nominal'):
                    self.nominalAtributos.append(True)
                else:
                    self.nominalAtributos.append(False)

            for i in range(0, self.nominalAtributos.__len__()):
                diccionario_aux.clear()
                lista_aux[:] = []

                if (self.nominalAtributos[i] == True):
                    f.seek(pos)

                    for j in range(1, lineas):
                        linea_aux = f.readline().split(',')
                        if (lista_aux.__contains__(linea_aux[i]) == False):
                            lista_aux.append(linea_aux[i])
                    sorted_lista_aux = sorted(lista_aux)
                    for k, val in enumerate(sorted_lista_aux):
                        diccionario_aux[val.strip('\r\n')] = int(k)
                    self.diccionarios.append(diccionario_aux.copy())
                else:
                    self.diccionarios.append(diccionario_aux.copy())

            f.seek(pos)

            for j in range(0, lineas):
                array_aux = np.array(())
                linea_aux = f.readline().split(',')
                array_aux = array_aux.astype(int)
                for i in range(0, self.nombreAtributos.__len__()):
                    if (self.nominalAtributos[i] == True):
                        #array_aux.append(self.diccionarios.__getitem__(i)[linea_aux[i]])
                        array_aux = np.append(array_aux, self.diccionarios.__getitem__(i)[linea_aux[i].strip('\r\n')])
                        #array_aux. self.diccionarios.__getitem__(i)[linea_aux[i]])
                    else:
                        #array_aux.append(linea_aux[i])
                        array_aux = np.append(array_aux, float(linea_aux[i]))
                if (j==0):
                    self.datos = array_aux
                else:
                    self.datos = np.vstack([self.datos, np.copy(array_aux)])

    # TODO: hacer en las proximas practicas
    def extraeDatos(self, idx):
        return self.datos[idx]
        #ret = np.array(())

        #for i in range(0, len(idx)):
         #   ret = np.append(ret, self.datos[i])

        #return ret

