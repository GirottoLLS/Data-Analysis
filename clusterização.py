import pandas as pd
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import scipy.cluster.hierarchy as sch
from yellowbrick.cluster import KElbowVisualizer
from scipy.stats import zscore
from yellowbrick.cluster import SilhouetteVisualizer
import seaborn as sns
import pingouin as pg
import plotly.express as px 
import plotly.io as pio
pio.renderers.default='browser'

#%% 1.0 Carregamento de dados e padronização

# Carregamento de dados
dados = pd.read_excel('RadicularesMediaAbsoluta.xlsx')
print(dados.info())

# Criamos um objeto sem a coluna "Especie" que possui apenas strings
# Visualizamos os dados em uma tabela descritiva
dados_cluster = dados.drop(columns=['Especie'])
tab_descritivas = dados_cluster.describe().T

# Padronizamos os dados utilizando o Zscore
# Segurança caso existam dados muito distintos entre si
dados_pad = dados_cluster.apply(zscore, ddof=1)
print(round(dados_pad.mean(), 3))
print(round(dados_pad.std(), 3))

#%% 2.0 Clusterização hierárquica

# Criamos clusteres hierárquico com métrica de distância euclidiana
# Três tipos de medidas de encadeamento podem ser utilizados
# Ajustamos o threshold com base na distância entre os grupos

# Método de encadeamento simples
plt.figure(figsize=(16,8))
dend_sing = sch.linkage(dados_pad, method = 'single', metric = 'euclidean')
dendrogram_s = sch.dendrogram(dend_sing, color_threshold = 2, labels = list(dados.Especie))

plt.grid(False)
plt.xlabel('Tratamentos', fontsize=16)
plt.ylabel('Distância euclidiana', fontsize=16)
plt.axhline(y = 2, color = 'red', linestyle = '--')
plt.show()

# Método de encadeamento completo
plt.figure(figsize=(16,8), dpi = 300)
plt.gca().spines['top'].set_color('black')
plt.gca().spines['right'].set_color('black')
plt.gca().spines['left'].set_color('black')
plt.gca().spines['bottom'].set_color('black')
dend_sing = sch.linkage(dados_pad, method = 'complete', metric = 'euclidean')
dendrogram_s = sch.dendrogram(dend_sing, color_threshold = 2, labels = list(dados.Especie))

plt.grid(False)
plt.xlabel('Tratamentos', fontsize=16, labelpad=20)
plt.ylabel('Distância euclidiana', fontsize=16)
plt.axhline(y = 2, color = 'red', linestyle = '--')
plt.show()

# Método de encadeamento médio
plt.figure(figsize=(16,8))
dend_sing = sch.linkage(dados_pad, method = 'average', metric = 'euclidean')
dendrogram_s = sch.dendrogram(dend_sing, color_threshold = 2, labels = list(dados.Especie))

plt.grid(False)
plt.xlabel('Tratamentos', fontsize=16)
plt.ylabel('Distância euclidiana', fontsize=16)
plt.axhline(y = 2, color = 'red', linestyle = '--')
plt.show()

#%% 3.0 Definição do número ótimo de clusters

# k-means requer número de clusters definidos antes da criação dos grupos
# Métodos do "cotovelo" e "silhueta" são utilizados
# Consistem em correr k-means com diversas possibilidades de cluster diferentes


# Forma número 1 de utilizar o Método do Cotovelo
# ponto de parada pode ser parametrizado manualmente
# Produz um gráfico para análise visual do "k" ótimo
elbow = []
K = range(1,9) 
for k in K:
    kmeanElbow = KMeans(n_clusters=k, init='random', random_state=100).fit(dados_pad)
    elbow.append(kmeanElbow.inertia_)
    
plt.figure(figsize=(16,8))
plt.plot(K, elbow, marker='o')
plt.xlabel('Nº Clusters', fontsize=16)
plt.xticks(range(1,11)) # ajustar range
plt.ylabel('WCSS', fontsize=16)
plt.title('Elbow Method', fontsize=16)
plt.show()

# Forma número 2 de utilizar o método do cotovelo
# Mais intensa em questão de processamento
# Trás automaticamente medidas de "k" ótimas e seu score associado
km = KMeans(random_state=42)
visualizer = KElbowVisualizer(km, k=(2,9))
visualizer.fit(dados_pad)        
visualizer.show()

# Método da Silhueta
# Cada silhueta é um cluster
# A linha tracejada marca a média do valor de silhueta,
    # quanto maior melhor
# Não pode existir grande variação no tamanho de um cluster/silhueta
    # Clusteres estão desiguais
# Nenhuma silhueta pode estar inteira abaixo da linha tracejada
    # Clusteres não trazem informações
fig, ax = plt.subplots(3, 2, figsize=(15,8))
for i in [2, 3, 4, 5, 6, 7]:
    km = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=100, random_state=42)
    q, mod = divmod(i, 2)
    visualizer = SilhouetteVisualizer(km, colors='yellowbrick', ax=ax[q-1][mod])
    visualizer.fit(dados_pad) 
#%% 4.0 K-means

# Produzimos uma clusterização por k-means
# Como "n-clusters" utilizamos medidas de cotovelo e silhueta
# Consideramos medidas visuais de clusteres hierárquicos
kmeans_final = KMeans(n_clusters = 3, init = 'random', random_state=100).fit(dados_pad)

# Adicionamos o resultado nos dataframes que carregamos e modificamos
kmeans_clusters = kmeans_final.labels_
dados_cluster['cluster_kmeans'] = kmeans_clusters
dados_pad['cluster_kmeans'] = kmeans_clusters
dados['cluster_kmeans'] = kmeans_clusters
dados_cluster['cluster_kmeans'] = dados_cluster['cluster_kmeans'].astype('category')
dados_pad['cluster_kmeans'] = dados_pad['cluster_kmeans'].astype('category')
dados['cluster_kmeans'] = dados['cluster_kmeans'].astype('category')

#%% 5.0 Teste F de variância (ANOVA One-way)

# Variáveis (tratamentos) influenciaram os clusters individualmente?
    # Valor de P
# O quanto cada variável (tratemento) contribuiu para os clusters?
    # Valor de F
variables = ['Um', 'Dois', 'Tres', 'Quatro', 'Cinco']
anova_results = [pg.anova(dv=var, between='cluster_kmeans', data=dados_pad, detailed=True).assign(Variable=var) for var in variables]

# Salvamos os resultados em um dataframe
df_anova = pd.concat(anova_results).set_index('Variable')
print(df_anova)

# Figura 3d para visualizar os clusters
fig = px.scatter_3d(dados_cluster, 
                    x='Um', 
                    y='Tres', 
                    z='Cinco',
                    color='cluster_kmeans')
fig.show()

#%% 6.0 Estatística dos clusteres

# Estatística descritiva dos clusteres formados
analise_dados = dados.drop(columns=['Especie']).groupby(by=['cluster_kmeans'])
analise_dados_desc = analise_dados.describe().T
analise_dados_media = analise_dados.mean().T

#%% 7.0 Plotagem 2D dos dados com os centróides (k-means)

# Selecionamos as 2 dimensões nos eixos X e Y
dimensao_x = dados_pad.iloc[:, 2]  # Dimensão 1 como x
dimensao_y = dados_pad.iloc[:, 4]  # Dimensão 2 como y

# Vamos buscar os valores dos centróides
# Selecionamos apenas valores de dimensões correspondentes
kmeans_final.cluster_centers_cortado = kmeans_final.cluster_centers_[:, [2,4]]
centroides = kmeans_final.cluster_centers_cortado

# Buscamos os integrantes dos clusteres
clusters = kmeans_final.fit_predict(dados_pad)

# Marcadores customizados para os centróides
# Nomes específicos (espécies) para os pontos do scatterplot
marcadores = ['o', 's', '^']
nomes = dados['Especie'].tolist()

# Criamos o scatterplot Seaborn com dimensões e resolução definidas
fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
sns.scatterplot(x=dimensao_x, y=dimensao_y, facecolor='white', edgecolor='black', s=50, legend=None)

# Plotamos os centróides
# Adicionamos os marcadores customizados aos centróides
for i, centroid in enumerate(centroides):
    marker = marcadores[i % len(marcadores)]
    plt.scatter(centroid[0], centroid[1], color='black', marker=marker, s=180)

# Nomeamos os pontos individuais
# Ajustamos as posições com base na plotagem
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
    plt.text(x + x_offset, y + y_offset, nomes[i], fontsize=12, color='black')

# Tracejamos linhas de cada integrante aos seus repetitivos centróides
for i, sample in enumerate(zip(dimensao_x, dimensao_y)):
    plt.plot([sample[0], centroides[clusters[i], 0]], [sample[1], centroides[clusters[i], 1]], 
             color='black', linestyle='--', alpha=0.6)

# Legendas dos eixos
# Remover grade
# Adicionar linhas nos valores de Y,X = 0
plt.xlabel('Dimension one', fontsize=12)
plt.ylabel('Dimension two', fontsize=12)
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
