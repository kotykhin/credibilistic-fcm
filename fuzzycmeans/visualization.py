import base64
from io import BytesIO
from matplotlib import pyplot as plt
from itertools import combinations

def draw_model_2d(fcm, test_data, name, headers):
    rgb_spectre = fcm.u.tolist()
    for i in range(len(rgb_spectre)):
        if len(rgb_spectre) == 2:
            rgb_spectre[i].append(0)
    comb = list(combinations(range(0, len(fcm.cluster_centers_[0])), 2))
    transponate_input = test_data.transpose()
    all_combination_features = list(combinations(transponate_input, 2))
    fig, axes = plt.subplots(len(all_combination_features), 2, figsize=(16, 5 * len(all_combination_features)))
    plt.subplots_adjust(top=0.99) 
    fig.suptitle(f'{name} fuzzy clustering result', fontsize=16, y=1)
    i = 0 
    for features_to_compare, centers_i in zip(all_combination_features, comb):
        axes[i, 0].scatter(features_to_compare[:][0], features_to_compare[:][1], alpha=1)
        axes[i, 0].set_ylabel(headers[centers_i[0]])
        axes[i, 0].set_xlabel(headers[centers_i[1]])
        axes[i, 1].scatter(features_to_compare[:][0], features_to_compare[:][1], c = rgb_spectre, alpha=1)
        axes[i, 1].set_ylabel(headers[centers_i[0]])
        axes[i, 1].set_xlabel(headers[centers_i[1]])
        for cluster in fcm.cluster_centers_:
            axes[i, 1].scatter(cluster[centers_i[0]], cluster[centers_i[1]], marker="+", s=150, c='b')
        i += 1
    tmpfile = BytesIO()
    fig.savefig(tmpfile, format='png')
    encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')

    html = '<img src=\'data:image/png;base64,{}\'>'.format(encoded)

    with open(f'{name}.html','w') as f:
        f.write(html)
