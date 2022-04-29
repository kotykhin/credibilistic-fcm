import pandas as pd
import logging
import sys
import os
sys.path.append(os.path.join(sys.path[0], '../'))

from fuzzycmeans.fuzzy_clustering import FCM, normalize_input_data
from fuzzycmeans.visualization import draw_model_2d


def example():
    df = pd.read_csv('Iris.csv')
    normal_data = normalize_input_data(df)
    headers = normal_data.columns
    test_data = normal_data.to_numpy()
    fcm = FCM(n_clusters=3)
    fcm.set_logger(tostdout=True, level=logging.DEBUG)
    fcm.fit(test_data)
    print("\n\Original data")
    print(df)
    print("\n\nNormalized data")
    print(normal_data)
    print("\n\nFCM membership matrix")
    print(fcm.u)
    print("\nCluster centers after FCM")
    print(fcm.cluster_centers_)
    fcm.credibilistic_recalculation()
    fcm.compute_cluster_centers(test_data)

    print("\n\nCredibilistic predicted membership")
    print(fcm.u)
    print("\nCluster centers after credibilistic")
    print(fcm.cluster_centers_)

    draw_model_2d(fcm, test_data, headers)


example()
