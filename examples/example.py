import pandas as pd
import numpy as np
import logging
import sys
import os
sys.path.append(os.path.join(sys.path[0], '../'))

from fuzzycmeans.fuzzy_clustering import FCM, normalize_input_data
from fuzzycmeans.visualization import draw_model_2d


def example():
    df = pd.read_csv('top10s_small.csv', sep=';')
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
    print("\nCluster centers after FCM")
    print(fcm.cluster_centers_)
    draw_model_2d(fcm, test_data,'Fcm', headers)
    fcm.credibilistic_recalculation()
    fcm.compute_cluster_centers(test_data)

    df["Cluster"] = [np.argmax(row) + 1 for row in fcm.u]
    print("\n\Original data with result")
    print(df)
    print("\nCluster centers after credibilistic")
    print(fcm.cluster_centers_)

    draw_model_2d(fcm, test_data, 'Credibilistic', headers)


example()
