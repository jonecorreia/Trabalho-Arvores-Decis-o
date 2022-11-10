#importando as bibliotecas que serão utilizadas
from sklearn.datasets import load_wine
import pandas as pd
import graphviz 
from sklearn.tree import DecisionTreeClassifier,plot_tree, export_graphviz, export_text
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# importando os dados do toy dataset wine
x,y = load_wine(return_X_y =True, as_frame=True)

# separando em treino e teste
x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=42)

#instanciando a árvore de decisão
tree = DecisionTreeClassifier(random_state=42) # utilizando uma decision tree com os parâmetros default

# treinando a árvore de decisão nos dados
tree.fit(x_train, y_train)

# criando o fig e o axes 
fig, ax = plt.subplots(figsize=(8,6),dpi=92)

#criando o plot
plot_tree(tree,ax=ax)

#plotando o gráfico
plt.tight_layout();

# criando o fig e o axes 
fig, ax = plt.subplots(figsize=(12,8),dpi=92)

#criando o plot
plot_tree(tree, # a decision tree que será plotada
          feature_names = x_train.columns, # trará o nome das features utilizadas na primeira linha de cada nó
          ax=ax, # plotamos no axes criado do matplotlib
          precision=1, # precisão dos valores numéricos
          max_depth=2, #  escolhemos a profundidade da árvore
          # proportion = True # retorna a proporção dos valores das amostras
          #fontsize = 20 # mudar o tamanho da fonte
        )

#plotando o gráfico
plt.tight_layout();

tree = DecisionTreeClassifier(
criterion='gini', # gini vem por default, mas podemos optar por entropy
splitter='best', # a estratégia utilizada para fazer a separação de cada nó
# ela também pode ser feita de forma randômica utilizando 'random'
max_depth= None, # a máxima profundida que sua árvore de decisão pode ter
# se for None ela vai buscar a máxima pureza possível

min_samples_split = 2, # o mínimo de registros necessários para que uma separação seja feita
min_samples_leaf = 1, # o mínimo de registros necessários em cada nós-folha (veja a primeira imagem)
max_features = None, # o número de atributos que será considerado durante o split
# None -> seleciona todos os atributos, 'sqrt' -> raiz quadrada do número dos atributos, 'log2' -> log de base 2 do número de atributos

max_leaf_nodes=None, # a quantidade máxima de nós-folha que a árvore pode ter
# se for None ele não limitará o número de nós-folha

min_impurity_decrease=0.0, # o split irá ocorrer em cada nó se o decréscimo da impureza foi maior ou igual a este valor
random_state= 42, # permite o notebook ser reproduzível             
# veja mais parâmetros na documentação oficial!
)

# treinando a árvore de decisão nos dados
tree.fit(x_train, y_train)

#criando um dicionário com as variáveis
dic = {'score':tree.feature_importances_,'features':x_train.columns}

#criando um dataframe com os dados
df = pd.DataFrame(dic).sort_values('score',ascending=False)

#instanciando o figure o e axes no matplotlib
fig, ax  = plt.subplots(figsize=(10,6), dpi=92)

#fazendo o gráfico
df.sort_values('score',ascending=True).plot(kind='barh',x='features',y='score',ax=ax)
ax.set_frame_on(False)
ax.set_title('Avaliando a importância de cada feature', # texto do título
             loc='left', # posicionamento do título no Axes
             pad=20, # Distanciamento do título com outros objetos
             fontdict={'fontsize':20}, # Tamanho da fonte utilizado
             color='#3f3f4e') # cor da fonte em hexadecimal
ax.tick_params(axis='both', # escolhendo os ticks do eixo x
               length=0, # colocamos os ticks de tamanho zero, compare com os desenhos de cima
               labelsize=12, # tamanho da fonte para os eixos
               colors='dimgrey') # cor da fonte para o eixo x
ax.set_ylabel(None)
ax.legend().remove()
ax.grid(linestyle='--',lw=0.3,aa=True)

#plotando o gráfico
plt.tight_layout();

# exportando uma árvore de decisão em DOT format
export_graphviz(tree, # árvore de decisão a ser exportada
                out_file = 'arvore.dot',  # nome do arquivo output
                feature_names = x_train.columns, # retornar com as features
                precision=2, # número de digitos da precisão
                filled = True, # pinta os nós de acordo com cada classe
                class_names=['class_0','class_1','class_2']) # lista do nome das classes
                #special_characters=True, # 
                #label = 'root', # as informações das features aparecem apenas na raíz
                #leaves_parallel = True) # faz com que as folhas fiquem paralelas
                #rotate = True); # retorna a árvore na horizontal
                
                # exportando uma árvore de decisão em DOT format
dot_data = export_graphviz(tree, # árvore de decisão a ser exportada
                out_file = None,  # nome do arquivo output
                feature_names = x_train.columns, # retornar com as features
                precision=2, # número de digitos da precisão
                filled = True, # -
                class_names=['class_0','class_1','class_2']) # lista do nome das classes
                #special_characters=True, # 
                #label = 'root', # as informações das features aparecem apenas na raíz
                #leaves_parallel = True) # faz com que as folhas fiquem paralelas
                #rotate = True); # retorna a árvore na horizontal

graph = graphviz.Source(dot_data)

print(export_text(tree, # árvore de decisão treinada
                  show_weights=True, # retorna o número de amostras de cada classe
                  spacing=5)) # dá um espaçamento entre os nós