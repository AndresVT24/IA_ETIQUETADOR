__authors__ = ['1782649', '1745722', '1691333']
__group__ = '10'

import numpy as np
import utils

import math


class KMeans:

    def __init__(self, X, K=1, options=None):
        """
         Constructor of KMeans class
             Args:
                 K (int): Number of cluster
                 options (dict): dictionary with options
            """
        self.num_iter = 0 # numero de iteraciones a 0, pero nosotros no elegimos que numero de iteraciones realizara
        self.K = K # Number of clusters
        self._init_X(X) # matriz de puntos de tamaño N x D (o PxD)
        self._init_options(options)  # DICT options

    # kmeans trabaja con una matriz de dos dimensiones NxD
    # N = número de puntos
    # D = número de características de cada punto
    
    # Funció que rep com a entrada una matriu de punts X i fa diverses coses
    def _init_X(self, X):
        
        """
        ---- Con P numero de puntos y D numero de caracteristicas
        Initialization of all pixels, sets X as an array of data in vector form (PxD)
            Args:
                X (list or np.array): list(matrix) of all pixel values
                if matrix has more than 2 dimensions, the dimensionality of the sample space is the length of the last dimension
        """
        X = np.array(X, dtype=float) # al convertirlo a numpy array directamente es float
        # kmeans quiere una imagen cono una tabla de puntos donde filas son puntos y columnas son caracteristicas
        
        if X.ndim > 2:
            X = X.reshape(-1, X.shape[-1]) #X.shape[-1] = 3 columnas, numpy hace 1800 / 3 = 600 | 20*30*3 = 1800 valores
        
        self.X = X

    # Inicializacion de la variable opciones
    """
        cómo inicializar los centroides
        si mostrar información por pantalla
        cuándo considerar que ha convergido
        cuántas iteraciones máximas permitir
        qué criterio usar para evaluar el resultado
    """
    def _init_options(self, options=None):
        """
        Initialization of options in case some fields are left undefined
        Args:
            options (dict): dictionary with options
        """
        if options is None:
            options = {}
            
        if 'km_init' not in options:
            options['km_init'] = 'first'
        if 'verbose' not in options:
            options['verbose'] = False
        if 'tolerance' not in options:
            options['tolerance'] = 0
        if 'max_iter' not in options:
            options['max_iter'] = np.inf
        if 'fitting' not in options:
            options['fitting'] = 'WCD'  # within class distance.

        # If your methods need any other parameter you can add it to the options dictionary
        self.options = options

    # Funció que inicialitza les variables de classe centroids K x D, i old_centroids K x D.
    def _init_centroids(self):
        
        centroids = []
        
        # L’opció ‘first’ assigna als centroides als primers K punts de la imatge X que siguin diferents entre ells.
        contador = 0
        i = 0
        if self.options['km_init'].lower() == 'first':
    
            distancia_minima = 30

            while contador < self.K and i < len(self.X):
                punto = self.X[i]

                repetido = False
                
                for c in centroids:
                    if euclidean_dist(punto, c) < distancia_minima:
                        repetido = True

                if not repetido:
                    centroids.append(punto)
                    contador += 1

                i += 1

            # Si con distancia_minima no hemos llegado a K,
            # rellenamos con puntos diferentes exactos.
            i = 0
            
            while contador < self.K and i < len(self.X):
                punto = self.X[i]

                repetido = False
                
                for c in centroids:
                    if np.array_equal(punto, c):
                        repetido = True

                if not repetido:
                    centroids.append(punto)
                    contador += 1

                i += 1
        # L’opció ’random’ triara, de forma que no es repeteixin, centroides a l’atzar    
        elif self.options['km_init'].lower() == 'random':
            unique_points = np.unique(self.X, axis=0)
            indices = np.random.choice(len(unique_points), self.K, replace=False)
            centroids = unique_points[indices]
        # L’opció ’custom’ podrà seguir qualsevol altra política de selecció inicial de centroides que vosaltres considereu    
        # --> punts distribuïts sobre la diagonal del hipercub de les dades?
        elif self.options['km_init'].lower() == 'custom':
            minimo = np.min(self.X, axis=0)
            maximo = np.max(self.X, axis=0)
            centroids = np.linspace(minimo, maximo, self.K)

        self.centroids = np.array(centroids, dtype=float)
        self.old_centroids = np.zeros_like(self.centroids) # sirve para despues comparar y saber si han convergido, se inicializa con la forma de los centroides como primera instancia

    """Funció que per a cada punt de la imatge X, assigna quin és el centroide més
    proper i ho guarda a la variable de la classe KMeans: self.labels"""
    def get_labels(self):
        
        distancias = distance(self.X, self.centroids)
        llista_minims = []
        
        for i, punto in enumerate(self.X):
            minim = distancias[i,0]
            centroide_minim = 0
            for j, centroide in enumerate(self.centroids):
                if distancias[i,j] < minim:
                    minim = distancias[i,j]
                    centroide_minim = j
            llista_minims.append(centroide_minim)

        self.labels = np.array(llista_minims) # guardo qué centroide tiene la distancia mínima.

    # calcula els nous centroides
    def get_centroids(self):

        self.old_centroids = self.centroids.copy()
        new_centroids = np.zeros_like(self.centroids)
        
        for k in range(self.K): # para cada cluster
            puntos_cluster = []

            for i in range(len(self.X)): # mira los labels que son iguales al cluster de la iteracion
                if self.labels[i] == k:
                    puntos_cluster.append(self.X[i])
                    
            media = np.mean(puntos_cluster, axis=0) # axis 0 calcula media por columnas
            new_centroids[k] = media
        
        self.centroids = new_centroids
        
    """
        Checks if there is a difference between current and old centroids
    """
    def converges(self):
        
        tolerance = self.options['tolerance'] # si es 0 pues es que quieren todos los centroides exactamente iguales

        for i in range(self.centroids.shape[0]):          # recorre centroides "filas"
            for j in range(self.centroids.shape[1]):      # recorre características RGB "columnas"
                diferencia = abs(self.centroids[i, j] - self.old_centroids[i, j])

                if diferencia > tolerance:
                    return False

        return True

    def fit(self):
        """
        Runs K-Means algorithm until it converges or until the number of iterations is smaller
        than the maximum number of iterations.
        """
        self.num_iter = 0
        self._init_centroids()
        
        while self.num_iter < self.options['max_iter']:
            # Asigna cada punto al centroide más cercano
            self.get_labels()

            # Guarda centroides antiguos y calcula los nuevos
            self.get_centroids()

            # Contamos una iteración completa
            self.num_iter += 1

            # Si los centroides ya no cambian, paramos
            if self.converges():
                break
        

    # DISTANCIA INTRA-CLASS -> a clusters mas compactos mejor
    def withinClassDistance(self):
        """
         returns the within class distance of the current clustering
        """
        suma = 0

        for i in range(len(self.X)):
            label = self.labels[i]
            punto = self.X[i]
            centroide = self.centroids[label]

            distancia = euclidean_dist(punto, centroide)
            suma += distancia ** 2

        self.WCD = suma / len(self.X)

        return self.WCD

    # Trobar la millor k
    def find_bestK(self, max_K):
        """
         sets the best k analysing the results up to 'max_K' clusters
        """
        llista_K = []
        llista_WCD = []
        llindar = 20  # si la mejora es menor al 20%, paramos

        # Probamos diferentes valores de K
        for k in range(2, max_K + 1): # va del 2 al max_k
            self.K = k
            self.num_iter = 0
            
            self.fit()
            wcd = self.withinClassDistance()

            llista_K.append(k)
            llista_WCD.append(wcd)

        # Si no encontramos una K mejor, nos quedamos con max_K
        bestK = max_K

        i = 1
        trobat = False

        while i < len(llista_WCD) and not trobat:
            WCD_anterior = llista_WCD[i - 1]
            WCD_actual = llista_WCD[i]

            percent_DEC = 100 * WCD_actual / WCD_anterior
            millora = 100 - percent_DEC

            if millora < llindar: # si la mejora es mejor que el anterior por ejemplo un 11% mejor pues me quedo con esa k
                bestK = llista_K[i - 1]
                trobat = True

            i += 1

        self.K = bestK

        return self.K

""" Funció que pren com a entrada la imatge X (N × D) i els centroides C (K × D),
    i calcula la distància euclidiana entre cada punt de la imatge amb cada centroide, i ho
    retorna en forma d’una matriu de dimensió N × K """
    
def euclidean_dist(x, y):
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)

    return np.sqrt(np.sum((x - y) ** 2))

def distance(X, C):
    """
    Calculates the distance between each pixel and each centroid
    Args:
        X (numpy array): PxD 1st set of data points (usually data points)
        C (numpy array): KxD 2nd set of data points (usually cluster centroids points)

    Returns:
        dist: PxK numpy array position ij is the distance between the
        i-th point of the first set an the j-th point of the second set
    """
    
    X = np.array(X, dtype=float)
    C = np.array(C, dtype=float)

    dist = np.zeros((X.shape[0], C.shape[0]))

    for j, centroide in enumerate(C):
        diferencias = X - centroide
        cuadrados = diferencias ** 2
        suma = np.sum(cuadrados, axis=1)
        dist[:, j] = np.sqrt(suma)

    return dist


def get_colors(centroids):
    """
    for each row of the numpy matrix 'centroids' returns the color label following the 11 basic colors as a LIST
    Args:
        centroids (numpy array): KxD 1st set of data points (usually centroid points)

    Returns:
        labels: list of K labels corresponding to one of the 11 basic colors
    """

    probabilidades = utils.get_color_prob(centroids)
    labels = []

    for i in range(probabilidades.shape[0]): # por cada fila
        indice_color = np.argmax(probabilidades[i]) # recojo el indice del color con la probabiliad mas alta
        color = str(utils.colors[indice_color]) # recupero el color que es colors = np.array(['Red', 'Orange', 'Brown', 'Yellow', 'Green', 'Blue', 'Purple', 'Pink', 'Black', 'Grey', 'White'])
        labels.append(color)

    return labels