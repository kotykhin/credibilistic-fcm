from copy import copy
import numpy as np
import math
import random
import logging

from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler

SMALL_VALUE = 0.00001


class FCM:
    """
        This algorithm is from the paper
        "FCM: The fuzzy c-means clustering algorithm" by James Bezdek
        Here we will use the Euclidean distance

        Pseudo code:
        1) Fix c, m, A
        c: n_clusters
        m: 2 by default
        A: we are using Euclidean distance, so we don't need it actually
        2) compute the means (cluster centers)
        3) update the membership matrix
        4) compare the new membership with the old one, is difference is less than a threshold, stop. otherwise
            return to step 2)
    """

    def __init__(self, n_clusters=2, m=2, max_iter=50):
        self.n_clusters = n_clusters
        self.n_features = None
        self.cluster_centers_ = None
        self.n_points = None
        self.u = None  # The membership
        self.m = m  # the fuzziness, m=1 is hard not fuzzy. see the paper for more info
        self.max_iter = max_iter
        self.logger = logging.getLogger(__name__)
        self.logger.addHandler(logging.NullHandler())

    def get_logger(self):
        return self.logger

    def set_logger(self, tostdout=False, logfilename=None, level=logging.WARNING):
        if tostdout:
            self.logger.addHandler(logging.StreamHandler())
        if logfilename:
            self.logger.addHandler(logging.FileHandler(logfilename))
        if level:
            self.logger.setLevel(level)

    def compute_cluster_centers(self, X, update_func=None):
        """
        :param X:
        :return:

        vi = (sum of membership for cluster i ^ m  * x ) / sum of membership for cluster i ^ m  : for each cluster i

        """
        centers = []
        for c in range(self.n_clusters):
            sum1_vec = np.zeros(self.n_features)
            sum2_vec = 0.0
            for i in range(self.n_points):
                interm1 = (self.u[i][c] ** self.m)
                interm2 = interm1 * X[i]
                sum1_vec += interm2
                sum2_vec += interm1
                if np.any(np.isnan(sum1_vec)):
                    self.logger.debug("compute_cluster_centers> interm1 %s" % str(interm1))
                    self.logger.debug("compute_cluster_centers> interm2 %s" % str(interm2))
                    self.logger.debug("compute_cluster_centers> X[%d] %s" % (i, str(X[i])))
                    self.logger.debug("compute_cluster_centers> loop sum1_vec %s" % str(sum1_vec))
                    self.logger.debug("compute_cluster_centers> loop sum2_vec %s" % str(sum2_vec))
                    self.logger.debug("X: [%d] %s" % (i-1, X[i-1]))
                    self.logger.debug("X: [%d] %s" % (i+1, X[i+1]))
                    self.logger.debug("X: ")
                    self.logger.debug(X)
                    raise Exception("There is a nan in compute_cluster_centers method if")
            if sum2_vec == 0:
                sum2_vec = 0.000001
            centers.append(sum1_vec/sum2_vec)
        self.cluster_centers_ = np.array(centers)
        return centers

    def distance_squared(self, x, c):
        """
        Compute the Euclidean distance
        :param x: is a single point from the original data X
        :param c: is a single point that represent a center or a cluster
        :return: the distance
        """
        sum_of_sq = 0.0
        for i in range(len(x)):
            sum_of_sq += (x[i]-c[i]) ** 2
        return sum_of_sq

    def compute_membership_single(self, X, datapoint_idx, cluster_idx):
        """
        :param datapoint_idx:
        :param cluster_idx:
        :return: return computer membership for the given ids
        """
        clean_X = X
        d1 = self.distance_squared(clean_X[datapoint_idx], self.cluster_centers_[cluster_idx])
        sum1 = 0.0
        for c in self.cluster_centers_:  # this is to compute the sigma
            d2 = self.distance_squared(c, clean_X[datapoint_idx])
            if d2 == 0.0:
                d2 = SMALL_VALUE
            sum1 += (d1/d2) ** (1.0/(self.m-1))
            if np.any(np.isnan(sum1)):
                self.logger.debug("nan is found in compute_membership_single")
                self.logger.debug("d1: %s" % str(d1))
                self.logger.debug("sum1: %s" % str(sum1))
                self.logger.debug("d2: %s" % str(d2))
                self.logger.debug("c: %s" % str(c))
                self.logger.debug("X[%d] %s" % (datapoint_idx, str(clean_X[datapoint_idx])))
                self.logger.debug("centers: %s" % str(self.cluster_centers_))
                raise Exception("nan is found in computer_memberhip_single method in the inner for")
        if sum1 == 0:  # because otherwise it will return inf
            return 1.0 - SMALL_VALUE
        if np.any(np.isnan(sum1 ** -1)):
            self.logger.debug("nan is found in compute_membership_single")
            self.logger.debug("d1: %s" % str(d1))
            self.logger.debug("sum1: %s" % str(sum1))
            self.logger.debug("X[%d] %s" % (datapoint_idx, str(clean_X[datapoint_idx])))
            self.logger.debug("centers: %s" % str(self.cluster_centers_))
            raise Exception("nan is found in computer_memberhip_single method")
        return sum1 ** -1

    def update_membership(self, X):
        """
        update the membership matrix
        :param X: data points
        :return: nothing

        For performance, the distance can be computed once, before the loop instead of computing it every time
        """
        for i in range(self.n_points):
            for c in range(len(self.cluster_centers_)):
                self.u[i][c] = self.compute_membership_single(X, i, c)

    def fit(self, X: np.ndarray):
        """
        :param X
        :return: self
        """
        self.n_points = X.shape[0]
        self.n_features = X.shape[1]
        self.random_cluster_centers()
        self.u = np.zeros((self.n_points, self.n_clusters))
        centers_history = []
        centers_history.append(self.cluster_centers_.copy())
        i = 0
        while not self.check_error(centers_history):
            self.update_membership(X)
            self.compute_cluster_centers(X)
            centers_history.append(self.cluster_centers_.copy())
            self.logger.info("updated membership is: ")
            self.logger.info(self.u)
            self.logger.info("updated cluster centers are: ")
            self.logger.info(self.cluster_centers_)
            i += 1
        self.logger.info(f'took {i} iterations')
        return self

    def check_error(self, centers_history):
        if len(centers_history) < 2:
            return False
        last_centers = self.cluster_centers_
        pre_last_centers = centers_history[-2]
        for last_cluser_center, prelast_cluster_center in zip(last_centers, pre_last_centers):
            for i in range(len(last_centers)):
                if math.fabs(last_cluser_center[i] - prelast_cluster_center[i]) > 0.0001:
                    return False
        return True


    def random_cluster_centers(self):
        centers = []
        for i in range(self.n_clusters):
            coordinates = []
            for c in range(self.n_features):
                point = random.uniform(-1, 1)
                coordinates.append(point)
            centers.append(coordinates)
        self.cluster_centers_ = np.array(centers)
        return centers

    def credibilistic_recalculation(self):
        max_credibiliscic = self.u.max()
        for i in range(len(self.u)):
            for j in range(len(self.u[i])):
                self.u[i][j] = (self.u[i][j] + 1 - max_credibiliscic) / 2

def normalize_input_data(data: DataFrame):
    new_data = copy(data)
    # clear any string features, remain only numeric 
    for col_name in new_data.columns:
        if any(type(value) is str for value in data[col_name]):
            new_data.drop(col_name, axis = 1, inplace = True)
    # delete id if exist
    new_data.drop('Id', axis=1, inplace=True, errors='ignore')
    # make normalization in range -1 to 1
    for feature_name in new_data.columns:
        column_feature = new_data[[feature_name]]
        scaler = MinMaxScaler((-1, 1)).fit_transform(column_feature)
        new_data.drop(column_feature, axis = 1, inplace = True)
        new_data[feature_name] = scaler.flatten()
    return new_data
