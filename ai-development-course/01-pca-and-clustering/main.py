import numpy as np
import matplotlib.pyplot as plt
from math import sin

def generate_data(counts=130, std=10.0):
    centers = np.array([[150, 150],
                        [200, 200.0],
                        [100, 200.0]], dtype=float)
    X_list = []
    for c in centers:
        pts = np.random.normal(loc=c, scale=std, size=(counts, 2))
        X_list.append(pts)
    X = np.vstack(X_list)
    return X, centers

def dimension_expansion(X):
    N = len(X)
    new_cords = []
    for i in range(N):
        x1, x2 = X[i][0], X[i][1]
        new_cords.append([x1 + x2, np.log(x1) + x2, sin(x1 * x2)])
    X_expanded = np.hstack([X, np.vstack(new_cords)])
    return X_expanded


def PCA(X):
    X_transformed = X.copy()
    for col in range(X.shape[1]):
        n_th_column = X_transformed[:, col]
        mean_val = n_th_column.mean()
        std_val = np.std(n_th_column)
        if std_val > 0:
            X_transformed[:, col] = (n_th_column - mean_val) / std_val
        else:
            X_transformed[:, col] = 0

    cov_matrix = (X_transformed.T @ X_transformed) / (X_transformed.shape[0] - 1)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    sorted_eig_ind = np.argsort(eigenvalues)[::-1]
    v1, v2 = eigenvectors[:, sorted_eig_ind[0]], eigenvectors[:, sorted_eig_ind[1]]
    Z_new = X_transformed @ np.hstack([v1.reshape(-1, 1), v2.reshape(-1, 1)])
    return Z_new


def k_mean(Z, k):
    eps = 0.005
    max_iter = 300
    N_total = Z.shape[0]
    rand_indices = np.random.choice(N_total, k, replace=False)
    centers = Z[rand_indices]
    diff = float('inf')
    iteration = 0
    while diff > eps and iteration < max_iter:
        iteration += 1
        clusters = [[] for _ in range(k)]
        for i in range(N_total):
            distances = [np.linalg.norm(centers[j] - Z[i]) for j in range(k)]
            cluster_ind = np.argmin(distances)
            clusters[cluster_ind].append(Z[i])

        old_centers = centers.copy()
        centers = []
        for j in range(k):
            if len(clusters[j]) > 0:
                centers.append(np.mean(clusters[j], axis=0))
            else:
                centers.append(Z[np.random.randint(0, N_total)])
        centers = np.array(centers, dtype=float)
        diff = np.sqrt(np.mean(np.linalg.norm(centers - old_centers, axis=1)**2))
    clusters = [np.vstack(c) for c in clusters]
    return clusters, centers


def elbow_method(Z, K_values=range(1, 7)):
    sko_list = []
    for k in K_values:
        clusters, centers = k_mean(Z, k)
        sko = 0
        for i in range(len(centers)):
            if len(clusters[i]) > 0:
                sko += np.sum(np.linalg.norm(clusters[i] - centers[i], axis=1) ** 2)
        sko_list.append(sko)

    return sko_list, K_values

def main():
    X, real_centers = generate_data()
    X_expanded = dimension_expansion(X)
    Z_new = PCA(X_expanded)
    sko_list, K_values = elbow_method(Z_new, K_values=range(1, 10))
    clusters, centers = k_mean(Z_new, 3)

    sko_diff = [sko_list[i] - sko_list[i+1] for i in range(len(sko_list)-1)]
    sko_div = [sko_diff[i] / sko_diff[i+1] for i in range(len(sko_diff)-1)]
    print(f"Optimal count of clusters = {np.argmax(sko_div)+2}")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = ['blue', 'red', 'green']

    for i in range(3):
        axes[0].scatter(X[:, 0], X[:, 1], marker='X', s=25,color=colors[0])
    axes[0].scatter(real_centers[:, 0], real_centers[:, 1], marker='o', s=80,color='yellow', edgecolor='black', linewidths=1, label='Исходные центры')
    axes[0].set_title('Исходные данные')
    axes[0].set_xlabel('X1')
    axes[0].set_ylabel('X2')
    axes[0].axis('equal')
    axes[0].grid(True, alpha=0.35)
    axes[0].legend(loc='best')

    for i in range(3):
        if len(clusters[i]) > 0:
            axes[1].scatter(clusters[i][:, 0], clusters[i][:, 1], marker='X', s=25,
                            color=colors[i], label=f'Кластер {i}')
    axes[1].scatter(centers[:, 0], centers[:, 1], marker='o', s=80, color='yellow',
                    edgecolor='black', linewidths=1, label='Центры кластеров')
    axes[1].set_title('Результат кластеризации k-means (k=3)')
    axes[1].set_xlabel('Главная компонента 1')
    axes[1].set_ylabel('Главная компонента 2')
    axes[1].axis('equal')
    axes[1].grid(True, alpha=0.35)
    axes[1].legend(loc='best')

    axes[2].plot(list(K_values), sko_list, marker='o')
    axes[2].set_xlabel('k (число кластеров)')
    axes[2].set_ylabel('SSE (сумма квадратов расстояний)')
    axes[2].set_title('Метод локтя')
    axes[2].grid(True)
    axes[2].axvline(x=3, color='red', linestyle='--',label=f'Оптимальное k = {3}')
    axes[2].legend()

    plt.tight_layout()
    plt.show()

main()
