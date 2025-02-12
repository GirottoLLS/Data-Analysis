import pandas as pd
from sklearn import datasets
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import scipy.cluster.hierarchy as sch
from yellowbrick.cluster import KElbowVisualizer
from scipy.stats import zscore
from yellowbrick.cluster import SilhouetteVisualizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import seaborn as sns
import pingouin as pg
import plotly.express as px 
import plotly.io as pio
pio.renderers.default='browser'


#%% Carregamento e padronicação de dados

# Primeiro vamos carregar nossos dados
dados = pd.read_excel('RadicularesMediaAbsoluta.xlsx')
print(dados.info())

# Vamos retirar a coluna com os nomes dos tratamentos
dados_cluster = dados.drop(columns=['Especie'])
tab_descritivas = dados_cluster.describe().T

# Vamos padronizar nossos dados
# Segurança
# Utilizamos o ZScore
dados_pad = dados_cluster.apply(zscore, ddof=1)
print(round(dados_pad.mean(), 3))
print(round(dados_pad.std(), 3))

#%% CLusterização hierárquica

# Vamos criar um cluster hierárquico com métrica de distância euclidiana
# Método de encadeamento simples
plt.figure(figsize=(16,8))
dend_sing = sch.linkage(dados_pad, method = 'single', metric = 'euclidean')
dendrogram_s = sch.dendrogram(dend_sing, color_threshold = 2, labels = list(dados.Especie))

plt.xlabel('Tratamentos', fontsize=16)
plt.ylabel('Distância euclidiana', fontsize=16)
plt.axhline(y = 2, color = 'red', linestyle = '--')
plt.show()

# Vamos criar um cluster hierárquico com métrica de distância euclidiana
# Método de encadeamento completo
plt.figure(figsize=(16,8), dpi = 300)
plt.gca().spines['top'].set_color('black')
plt.gca().spines['right'].set_color('black')
plt.gca().spines['left'].set_color('black')
plt.gca().spines['bottom'].set_color('black')
dend_sing = sch.linkage(dados_pad, method = 'complete', metric = 'euclidean')
dendrogram_s = sch.dendrogram(dend_sing, color_threshold = 2, labels = list(dados.Especie))
plt.grid(False)
plt.xlabel('Espécies', fontsize=16, labelpad=20)
plt.ylabel('Distância euclidiana', fontsize=16)
plt.axhline(y = 2, color = 'red', linestyle = '--')
plt.show()

# Vamos criar um cluster hierárquico com métrica de distância euclidiana
# Método de encadeamento médio
plt.figure(figsize=(16,8))
dend_sing = sch.linkage(dados_pad, method = 'average', metric = 'euclidean')
dendrogram_s = sch.dendrogram(dend_sing, color_threshold = 2, labels = list(dados.Especie))

plt.xlabel('Tratamentos', fontsize=16)
plt.ylabel('Distância euclidiana', fontsize=16)
plt.axhline(y = 2, color = 'red', linestyle = '--')
plt.show()

#%% Testes de número de clusteres para k-means

# FOrma número 1 de utilizar o Método do Cotovelo
elbow = []
K = range(1,9) # ponto de parada pode ser parametrizado manualmente
for k in K:
    kmeanElbow = KMeans(n_clusters=k, init='random', random_state=100).fit(dados_pad)
    elbow.append(kmeanElbow.inertia_)
    
plt.figure(figsize=(16,8))
plt.plot(K, elbow, marker='o')
plt.xlabel('Nº Clusters', fontsize=16)
plt.xticks(range(1,11)) # ajustar range
plt.ylabel('WCSS', fontsize=16)
plt.title('Método de Elbow', fontsize=16)
plt.show()

# Forma número 2 de utilizar o método do cotovelo
km = KMeans(random_state=42)
visualizer = KElbowVisualizer(km, k=(2,9))
visualizer.fit(dados_pad)        
visualizer.show()

# Método alternativo para seleção do número de clsuters
# Método da Silhueta
    # Cada silhueta é um cluster
    # A linha tracejada marca a média do valor de silhueta, quanto maior melhor
    # Não pode existir grande variação no tamanho de um cluster/silhueta
    # Nenhuma silhueta pode estar inteira abaixo da linha tracejada
fig, ax = plt.subplots(3, 2, figsize=(15,8))
for i in [2, 3, 4, 5, 6, 7]:
    km = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=100, random_state=42)
    q, mod = divmod(i, 2)
    visualizer = SilhouetteVisualizer(km, colors='yellowbrick', ax=ax[q-1][mod])
    visualizer.fit(dados_pad) 
#%% k-means

# Agora vamos criar um cluster não hierárquico
# K-means
# Como n-clusters vamos utilizar o que obtivemos nos clusters hierárquicos
kmeans_final = KMeans(n_clusters = 3, init = 'random', random_state=100).fit(dados_pad)

# Vamos adicionar o resultado nos dataframes que carregamos e modificamos
kmeans_clusters = kmeans_final.labels_
dados_cluster['cluster_kmeans'] = kmeans_clusters
dados_pad['cluster_kmeans'] = kmeans_clusters
dados['cluster_kmeans'] = kmeans_clusters
dados_cluster['cluster_kmeans'] = dados_cluster['cluster_kmeans'].astype('category')
dados_pad['cluster_kmeans'] = dados_pad['cluster_kmeans'].astype('category')
dados['cluster_kmeans'] = dados['cluster_kmeans'].astype('category')

#%% Teste F de variância

# Vamos fazer um teste F de variância
# ANOVA One-way
# Testar se as variáveis influenciaram os clusters individualmente pelo valor de p
# Podemos comparar o quanto cada uma contribuíu pros clusters pelo valor de F
pg.anova(dv='Um', 
         between='cluster_kmeans', 
         data=dados_pad,
         detailed=True).T
pg.anova(dv='Dois', 
         between='cluster_kmeans', 
         data=dados_pad,
         detailed=True).T
pg.anova(dv='Tres', 
         between='cluster_kmeans', 
         data=dados_pad,
         detailed=True).T
pg.anova(dv='Quatro', 
         between='cluster_kmeans', 
         data=dados_pad,
         detailed=True).T
pg.anova(dv='Cinco', 
         between='cluster_kmeans', 
         data=dados_pad,
         detailed=True).T

# Vamos fazer uma figura 3d para visualizar os clusters
fig = px.scatter_3d(dados_cluster, 
                    x='Um', 
                    y='Tres', 
                    z='Cinco',
                    color='cluster_kmeans')
fig.show()

#%% Estatística dos clusteres

# Agora vamos explorar os clusters com estatística descritiva baseada neles
analise_dados = dados.drop(columns=['Especie']).groupby(by=['cluster_kmeans'])
analise_dados_desc = analise_dados.describe().T
analise_dados_media = analise_dados.mean().T

#%% Plotagem dos dados com os centróides

# Vamos trabalhar com um gráfico de 2 dimensões
# Selecionamos quais vão ser estas dimensões nos eixos X e Y
dimensao_x = dados_pad.iloc[:, 2]  # Dimensão 1
dimensao_y = dados_pad.iloc[:, 4]  # Dimensão 2

# Vamos buscar os valores dos centróides
# Nos atentamos às dimensões que estamos utilizando (apenas 2)
# Selecionamos apenas valores de dimensões correspondentes
kmeans_final.cluster_centers_cortado = kmeans_final.cluster_centers_[:, [2,4]]
centroids = kmeans_final.cluster_centers_cortado

# Buscamos os integrantes dos clusteres
clusters = kmeans_final.fit_predict(dados_pad)

# Marcadores customizados para os centróides
# Nomes específicos para os pontos do scatterplot
marcadores = ['o', 's', '^']
dot_names = dados['Especie'].tolist()

# Criamos o plot
fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
sns.scatterplot(x=dimensao_x, y=dimensao_y, facecolor='white', edgecolor='black', s=50, legend=None)

# Plotamos os centróides
# Adicionamos os marcadores customizados aos centróides
for i, centroid in enumerate(centroids):
    marker = marcadores[i % len(marcadores)]
    plt.scatter(centroid[0], centroid[1], color='black', marker=marker, s=180)

# Nomeamos os pontos individuais
for i, (x, y) in enumerate(zip(dimensao_x, dimensao_y)):
    if x > 1 and y > 1:
        x_offset = -0.4
        y_offset = -0.09
    elif x < -1.2 and y < -0.8:
        x_offset = -0.09
        y_offset = -0.09
    else:
        x_offset = 0
        y_offset = 0.05
    plt.text(x + x_offset, y + y_offset, dot_names[i], fontsize=12, color='black')

# Tracejamos linhas dos integrantes aos seus centróides
for i, sample in enumerate(zip(dimensao_x, dimensao_y)):
    plt.plot([sample[0], centroids[clusters[i], 0]], [sample[1], centroids[clusters[i], 1]], 
             color='black', linestyle='--', alpha=0.6)

# Legendas dos eixos
plt.xlabel('Dimension one', fontsize=12)
plt.ylabel('Dimension two', fontsize=12)

# Remover grade e adicionar linhas nos valores de Y e X no 0
plt.grid(False)
ax.axhline(0, color='black', linestyle='-', linewidth=1)
ax.axvline(0, color='black', linestyle='-', linewidth=1)

# Linhas pretas na caixa do plot
plt.gca().spines['top'].set_color('black')
plt.gca().spines['right'].set_color('black')
plt.gca().spines['bottom'].set_color('black')
plt.gca().spines['left'].set_color('black')


# Legenda customizada ao plot
legend_elements = [
    mlines.Line2D([], [], color='black', marker='o', linestyle='None', markersize=10, label='Cluster 1'),
    mlines.Line2D([], [], color='black', marker='^', linestyle='None', markersize=10, label='Cluster 2'),
    mlines.Line2D([], [], color='black', marker='s', linestyle='None', markersize=10, label='Cluster 3'),]
ax.legend(handles=legend_elements, loc='upper left', fontsize=12, edgecolor='black', framealpha=1.0)

# Mostrar o plot
plt.tight_layout()
plt.show()
