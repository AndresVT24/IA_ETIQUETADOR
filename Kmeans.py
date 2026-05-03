__authors__ = ['1782649', '1745722', '1691333']
__group__ = '10'

import numpy as np
import utils


class KMeans:

    def __init__(self, X, K=1, options=None):
        """
         Constructor of KMeans class
             Args:
                 K (int): Number of cluster
                 options (dict): dictionary with options
            """
        self.num_iter = 0
        self.K = K
        self._init_X(X)
        self._init_options(options)  # DICT options

    #############################################################
    ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
    #############################################################

    def _init_X(self, X):
        """Initialization of all pixels, sets X as an array of data in vector form (PxD)
            Args:
                X (list or np.array): list(matrix) of all pixel values
                    if matrix has more than 2 dimensions, the dimensionality of the sample space is the length of
                    the last dimension
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        X = np.array(X, dtype=float)
        if X.ndim > 2:
            self.X = X.reshape(-1, X.shape[-1])
        else:
            self.X = X

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

        #############################################################
        ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
        #############################################################

    def _init_centroids(self):
        """
        Initialization of centroids
        """

        if self.options['km_init'].lower() == 'first':
            unique, seen = [], set()
            for row in self.X:
                key = tuple(row)
                if key not in seen:
                    seen.add(key)
                    unique.append(row.copy())
                    if len(unique) == self.K:
                        break
            self.centroids = np.array(unique)
        elif self.options['km_init'].lower() == 'random':
            indices = np.random.choice(self.X.shape[0], self.K, replace=False)
            self.centroids = self.X[indices].copy()
        elif self.options['km_init'].lower() == 'custom':
            indices = np.linspace(0, self.X.shape[0] - 1, self.K, dtype=int)
            self.centroids = self.X[indices].copy()

        self.old_centroids = np.zeros_like(self.centroids)

    def get_labels(self):
        """
        Calculates the closest centroid of all points in X and assigns each point to the closest centroid
        """
        self.labels = np.argmin(distance(self.X, self.centroids), axis=1)

    def get_centroids(self):
        """
        Calculates coordinates of centroids based on the coordinates of all the points assigned to the centroid
        """
        self.old_centroids = self.centroids.copy()
        new_centroids = []
        for k in range(self.K):
            pts = self.X[self.labels == k]
            new_centroids.append(pts.mean(axis=0) if len(pts) > 0 else self.old_centroids[k])
        self.centroids = np.array(new_centroids)

    def converges(self):
        """
        Checks if there is a difference between current and old centroids
        """
        return np.allclose(self.centroids, self.old_centroids, atol=self.options['tolerance'])

    def fit(self):
        """
        Runs K-Means algorithm until it converges or until the number of iterations is smaller
        than the maximum number of iterations.
        """
        self.num_iter = 0
        self._init_centroids()
        while not self.converges() and self.num_iter < self.options['max_iter']:
            self.get_labels()
            self.get_centroids()
            self.num_iter += 1

    def withinClassDistance(self):
        """
         returns the within class distance of the current clustering
        """

        dists = distance(self.X, self.centroids)
        return np.mean(np.min(dists, axis=1) ** 2)

    def find_bestK(self, max_K):
        """
         sets the best k analysing the results up to 'max_K' clusters
        """
        self.K = 2
        self.fit()
        prev_wcd = self.withinClassDistance()

        for k in range(3, max_K + 1):
            self.K = k
            self.fit()
            curr_wcd = self.withinClassDistance()
            if curr_wcd / prev_wcd > 0.8:
                self.K = k - 1
                self.fit()
                return
            prev_wcd = curr_wcd


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

    return np.sqrt(np.sum((X[:, np.newaxis, :] - C[np.newaxis, :, :]) ** 2, axis=2))


def get_colors(centroids):
    """
    for each row of the numpy matrix 'centroids' returns the color label following the 11 basic colors as a LIST
    Args:
        centroids (numpy array): KxD 1st set of data points (usually centroid points)

    Returns:
        labels: list of K labels corresponding to one of the 11 basic colors
    """

    color_probs = utils.get_color_prob(centroids)
    return list(utils.colors[np.argmax(color_probs, axis=1)])
