import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

#%% 1.0 Leitura de banco de dados e ajuste de nomes

# Ler arquivo
df = pd.read_excel('pigmentosR.xlsx')

# Criamos objetos com nomes de espécies e tratamentos
especies = ['L. sativa', 'B. pilosa', 'G. ulmifolia', 'M. chamomilla']
tratamentos = ['0', '0.125', '0.25', '0.5', '1', '2']


#%% 2.0 Dados a serem plotados

# Dicionário para armazenar as listas dos primeiros 5 valores em listas
# Utilizamos todas as colunas
# Estes são valores de comprimentos
column_data = {}
for column in df.columns:
    column_data[column] = df[column].tolist()[:6]

# Salvamos os valores de forma concatenada na ordem que vamos utilizá-los
valoresA = (column_data['L. sativa A'] + column_data['L. sativa B'] + column_data['L. sativa C'] +
            column_data['B. pilosa A'] + column_data['B. pilosa B'] + column_data['B. pilosa C'] +
            column_data['G. ulmifolia A'] + column_data['G. ulmifolia B'] + column_data['G. ulmifolia C'] +
            column_data['M. chamomilla A'] + column_data['M. chamomilla B'] + column_data['M. chamomilla C'])

# Dicionário para armazenar listas dos valores 6 a 10
# Estes são os valores do Pós Teste de Tukey
tukeyA = {}
for column in df.columns:
    tukeyA[column] = df[column].tolist()[6:12]

valoresTA = (tukeyA['L. sativa A'] + tukeyA['L. sativa B'] + tukeyA['L. sativa C'] +
             tukeyA['B. pilosa A'] + tukeyA['B. pilosa B'] + tukeyA['B. pilosa C'] +
             tukeyA['G. ulmifolia A'] + tukeyA['G. ulmifolia B'] + tukeyA['G. ulmifolia C'] +
             tukeyA['M. chamomilla A'] + tukeyA['M. chamomilla B'] + tukeyA['M. chamomilla C'])

# Dicionário para armazenar as listas dos valores 11 a 15
# Estes são os dados de erros
uncertaintyA = {}
for column in df.columns:
    uncertaintyA[column] = df[column].tolist()[12:18]
    
yerr = (uncertaintyA['L. sativa A'] + uncertaintyA['L. sativa B'] + uncertaintyA['L. sativa C'] +
        uncertaintyA['B. pilosa A'] + uncertaintyA['B. pilosa B'] + uncertaintyA['B. pilosa C'] +
        uncertaintyA['G. ulmifolia A'] + uncertaintyA['G. ulmifolia B'] + uncertaintyA['G. ulmifolia C'] +
        uncertaintyA['M. chamomilla A'] + uncertaintyA['M. chamomilla B'] + uncertaintyA['M. chamomilla C'])

#%% 3.0 Preparação e plotagem

# Definimos as posições no eixo X
# Depois determinamos as espessuras das barras
x = np.arange(len(tratamentos))
width = 0.25

# Preparamos elementos para a criação de uma caixa de legenda
# Esta é customizada em cor, formato e label
legend_elements = [
    mpatches.Rectangle((0, 0), 1, 1, facecolor='black', edgecolor='black', label='Chlorophyll A'),
    mpatches.Rectangle((0, 0), 1, 1, facecolor='dimgray', edgecolor='black', label='Chlorophyll B'),
    mpatches.Rectangle((0, 0), 1, 1, facecolor='lightgray', edgecolor='black', label='Carotenoids')
]

# Salvamos o padrão de coloração a ser utilizado
cores = ['black','dimgray','lightgray']

# Definimos a figura com 4 subplots no padrão 2,2
# Tamanho e resolução da figura definidos
# Subplots vão compartilhar os eixos X e Y entre si
# Ajustamos os eixos 'axes' em objetos de uma dimensão ao invés de 2 (por causa do formato 2,2)
fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=300, sharex=True, sharey=True)
axes = axes.flatten()

# Vamos fazer em loop todos os subplots
# Cada um com seu valor para 'barra', 'erro' e 'título'
# Cada subplot terá dados de uma espécie
for i, sp in enumerate(especies):
    ax = axes[i]
    
    # Extraímos os valores por espécie
    values_A = column_data[f'{sp} A']
    values_B = column_data[f'{sp} B']
    values_C = column_data[f'{sp} C']
    
    # Extraímos os erros por espécie
    errosA = uncertaintyA[f'{sp} A']
    errosB = uncertaintyA[f'{sp} B']
    errosC = uncertaintyA[f'{sp} C']
    
    # Extraímos as letras do Teste de Tukey por espécie
    tukey_A = tukeyA[f'{sp} A']
    tukey_B = tukeyA[f'{sp} B']
    tukey_C = tukeyA[f'{sp} C']
    
    # Plotamos as barras 
    # Utilizamos o valor de espessura para determinar posição das barras de "A", "B" e "C"
    ax.bar(x - width, values_A, width, label=f'{sp} A', yerr = errosA, alpha=0.7,color=cores[0])
    ax.bar(x, values_B, width, label=f'{sp} B', yerr = errosB, alpha=0.7,color=cores[1])
    ax.bar(x + width, values_C, width, label=f'{sp} C', yerr = errosC, alpha=0.7,color=cores[2])
    
    # Adicionar as letras acima das barras
    # Consideramos os erros para evitar a sobreposição destes com as letras
    
    for j in range(len(tratamentos)):
        ax.text(x[j] - width, values_A[j] + errosA[j] + 0.35, tukey_A[j], ha='center', fontsize=14, fontweight='bold')
        ax.text(x[j], values_B[j] + errosB[j] + 0.35, tukey_B[j], ha='center', fontsize=14, fontweight='bold')
        ax.text(x[j] + width, values_C[j] + errosC[j] + 0.35, tukey_C[j], ha='center', fontsize=14, fontweight='bold')
    
    # Adicionamos o nome da espécie plotada em cada subplot
    # Ajustamos parâmetros de eixo
    ax.text(0.98, 0.95, f'{sp}', transform=ax.transAxes, fontsize=18, fontweight='bold',
            va='top', ha='right')
    ax.tick_params(axis='y', labelsize=15)
    plt.ylim(0, 12)
    ax.set_xticks(x)
    ax.set_xticklabels(tratamentos,fontsize=15)
    
    # Adicionamos a legenda customizada que criamos previamente
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 0.85),
           ncol=1, fontsize=12, frameon=False, )

# Adicionamos títulos dos eixos
# Criamos a figura
fig.supylabel("Mass (mg/g)", fontsize=19)
fig.supxlabel("Treatment (mg/mL)", fontsize=19)
plt.tight_layout()
plt.show()

