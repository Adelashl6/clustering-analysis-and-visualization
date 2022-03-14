from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import hdbscan
from sklearn.metrics import silhouette_score
from MulticoreTSNE import MulticoreTSNE as TSNE
import plotly.graph_objects as go
import pickle
import os


def apply_HDBSCAN(data, args):
    cluster = hdbscan.HDBSCAN(min_cluster_size=50, min_samples=10)
    labels = cluster.fit_predict(data)
    clusters_tsne_scale = pd.DataFrame({'clusters': labels})
    clusters_tsne_scale.to_csv(os.path.join(args.result_path, 'hdbscan_disinfo_blog.csv'), index=False)
    return clusters_tsne_scale


def apply_tsne(data, args):
    tsne = TSNE(n_components=args.n_component, verbose=1, perplexity=80, n_iter=2000, learning_rate=200, n_jobs=32)
    tsne_scale_results = tsne.fit_transform(data)
    tsne_scale_results = pd.DataFrame(tsne_scale_results, columns=['tsne1', 'tsne2'])

    return tsne_scale_results


def apply_kmeans(data, args):

    '''
    # elbow method to determine the number of K-means cluster
    Sum_of_squared_distances = []
    K = range(1, 20)
    for k in K:
        km = KMeans(n_clusters=k)
        km = km.fit(data)
        Sum_of_squared_distances.append(km.inertia_)

    plt.plot(K, Sum_of_squared_distances, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Sum_of_squared_distances')
    plt.title('Elbow Method For Optimal k')
    plt.show()
    '''
    kmeans_tsne_scale = KMeans(n_clusters=6, n_init=500, max_iter=1000, init='k-means++', random_state=42).fit(data)
    sil_score = silhouette_score(data, kmeans_tsne_scale.labels_, metric='euclidean')
    print('KMeans tSNE Scaled Silhouette Score: ', sil_score)

    label = kmeans_tsne_scale.labels_
    clusters_tsne_scale = pd.concat([data, pd.DataFrame({'tsne_clusters':label})], axis=1)
    clusters_tsne_scale.to_csv(os.path.join(args.result_path, 'kmeans_disinfo.csv'), index=False)
    return clusters_tsne_scale


def plot_kmeans_2D(data, args):
    plt.figure(figsize=(60, 60))
    sns.scatterplot(data.iloc[:, 2], data.iloc[:, 3], hue=data.iloc[:, 1].values, palette='Set1', s=100, alpha=0.4)
    plt.title('Kmeans Cluster (21) Derived from t-SNE')
    plt.legend()
    plt.savefig(os.path.join(args.result_path, 'tsne_disinfo.png'))
    # plt.show()


def plot_kmeans_3D(data):
    scene = dict(xaxis=dict(title='tsne1'), yaxis=dict(title='tsne2'),  zaxis=dict(title='tsne3'))
    trace = go.Scatter3d(x=data.iloc[:, 0], y=data.iloc[:, 1],
                         z=data.iloc[:, 2], mode='markers',
                         marker=dict(color=data.iloc[:, -1].values,
                                     colorscale='Viridis', size=3, line=dict(color='yellow', width=5)))
    layout = go.Layout(margin=dict(l=200, r=200), scene=scene, height=1000, width=1500)
    data = [trace]
    fig = go.Figure(data=data, layout=layout)
    fig.show()


def main(args):
    # read data
    embeddings = pickle.load(open(args.data_path, 'rb'))
    id = embeddings['id'].values
    data = np.array(embeddings['embeddings'].values.tolist())
    del embeddings

    column_names = ['feature_{}'.format(i) for i in range(768)]
    data = pd.DataFrame(data, columns=column_names)
    print(data.head(5))

    # normalize standardization
    scaler = StandardScaler()
    scaler.fit(data)

    X_scale = scaler.transform(data)
    data_scale = pd.DataFrame(X_scale, columns=data.columns)
    print(data_scale.head(5))

    # apply kmeans
    clusters_scale = apply_HDBSCAN(data_scale, args)

    # apply t-sne for dimension reductivity
    labels = clusters_scale['clusters'].values
    filtered_index = (labels > -1)

    id = id[filtered_index]
    data_scale = data_scale[filtered_index]
    # chunk_num = chunk_num[filtered_index]
    blog_labels = labels[filtered_index]

    clusters_tsne_results = apply_tsne(data_scale, args)
    clusters_tsne_results = pd.concat([pd.DataFrame({'id': id.tolist(),
                                                     'tsne_clusters': blog_labels}), clusters_tsne_results], axis=1)
    clusters_tsne_results.to_csv(os.path.join(args.result_path, 'tsne_disinfo.csv'), index=False)

    # visualize cluster in 2D using T-SNE
    plot_kmeans_2D(clusters_tsne_results, args)

    # visualize cluster in 3D using T-SNE
    # plot_kmeans_3D(clusters_tsne_results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='apply t-SNE for dimension reduction and '
                                                 'visualize the data by K-means clustering')
    parser.add_argument('--data_path', type=str, default='./ukraine_blog_new/Disinfo_head_blogs_emb.pkl')
    parser.add_argument('--result_path', type=str, default='./ukraine_blog_new/result1')
    parser.add_argument('--n_component', type=int, default=2, help='number of embedded dimensions using T-SNE')

    args = parser.parse_args()
    main(args)

