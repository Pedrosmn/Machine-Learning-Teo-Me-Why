#%%

import pandas as pd

df = pd.read_excel('data/dados_cerveja_nota.xlsx')
df.head()

# %%

feature = 'cerveja'
target = 'nota'

X = df[[feature]]
y = df[target]

# %%

from sklearn import linear_model
from sklearn import tree

linear = linear_model.LinearRegression()
linear.fit(X, y)
predict_linear = linear.predict(X.drop_duplicates())

arvore_full = tree.DecisionTreeRegressor(random_state=42)
arvore_full.fit(X, y)
predict_arvore_full = arvore_full.predict(X.drop_duplicates())

arvore_d2 = tree.DecisionTreeRegressor(random_state=42, max_depth=2)
arvore_d2.fit(X, y)
predict_arvore_d2 = arvore_d2.predict(X.drop_duplicates())

# %%

import matplotlib.pyplot as plt

plt.plot(X['cerveja'], y, 'o')
plt.grid()
plt.xlabel('Cerveja')
plt.ylabel('Nota')
plt.title('Relação Cerveja vs Nota')

plt.plot(X.drop_duplicates()['cerveja'], predict_linear)
plt.plot(X.drop_duplicates()['cerveja'], predict_arvore_full)
plt.plot(X.drop_duplicates()['cerveja'], predict_arvore_d2)

plt.legend(['Observado',
           'Regressão Linear',
           'Árvore Total',
           'Árvore Max Depth = 2'])
