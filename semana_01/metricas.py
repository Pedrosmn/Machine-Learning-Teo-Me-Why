# %%
import pandas as pd

df = pd.read_csv('data/dados_comunidade.csv')

df.head()

# %%
features_dummies = [
    'Como conheceu o Téo Me Why?',
    'Quantos cursos acompanhou do Téo Me Why?',
    'Estado que mora atualmente',
    'Área de Formação',
    'Tempo que atua na área de dados',
    'Posição da cadeira (senioridade)'
]

features_num = [
    'Curte games?',
    'Curte futebol?',
    'Curte livros?',
    'Curte jogos de tabuleiro?',
    'Curte jogos de fórmula 1?',
    'Curte jogos de MMA?',
    'Idade'
]

# %%

df_dummies = (pd.get_dummies(df[features_dummies])).astype(int)

df_num = df[features_num]
df_num = df_num.replace({
    'Sim':1,
    'Não':0
    })

# %%

df_analise = pd.DataFrame()
df_analise = pd.concat([df_num, df_dummies], axis=1)
df_analise['pessoa feliz'] = df['Você se considera uma pessoa feliz?']

df_analise['pessoa feliz'] = df_analise['pessoa feliz'].replace({
    'Sim':1,
    'Não':0
    })

# %%
features = df_analise.columns.to_list()[:-1]
X = df_analise[features]
y = df_analise['pessoa feliz']

# %% 

from sklearn import tree
from sklearn import linear_model
from sklearn import naive_bayes

arvore = tree.DecisionTreeClassifier(random_state=42, min_samples_leaf=6)
arvore.fit(X=X, y=y)

reg = linear_model.LogisticRegression(penalty=None, fit_intercept=True, max_iter=1000)
reg.fit(X=X, y=y)

naive = naive_bayes.GaussianNB()
naive.fit(X=X, y=y)


# %%

arvore_pred = arvore.predict(X=X)
arvore_proba = arvore.predict_proba(X=X)[:,1]

reg_pred = reg.predict(X=X)
reg_proba = reg.predict_proba(X=X)[:,1]

naive_pred = naive.predict(X=X)
naive_proba = naive.predict_proba(X=X)[:,1]


# %%

df_pred = pd.DataFrame()
df_pred['pessoa feliz'] = df_analise['pessoa feliz']

df_pred['arvore_pred'] = arvore_pred
df_pred['arvore_proba'] = arvore_proba

df_pred['reg_pred'] = reg_pred
df_pred['reg_proba'] = reg_proba

df_pred['naive_pred'] = naive_pred
df_pred['naive_proba'] = naive_proba

# %%

from sklearn import metrics

arvore_acc = metrics.accuracy_score(y_true=df_pred['pessoa feliz'], y_pred=df_pred['arvore_pred'])
arvore_prec = metrics.precision_score(y_true=df_pred['pessoa feliz'], y_pred=df_pred['arvore_pred'])
arvore_recall = metrics.recall_score(y_true=df_pred['pessoa feliz'], y_pred=df_pred['arvore_pred'])
arvore_roc = metrics.roc_curve(df_pred['pessoa feliz'], df_pred['arvore_proba'])
arvore_auc = metrics.roc_auc_score(df_pred['pessoa feliz'], df_pred['arvore_proba'])

reg_acc = metrics.accuracy_score(y_true=df_pred['pessoa feliz'], y_pred=df_pred['reg_pred'])
reg_prec = metrics.precision_score(y_true=df_pred['pessoa feliz'], y_pred=df_pred['reg_pred'])
reg_recall = metrics.recall_score(y_true=df_pred['pessoa feliz'], y_pred=df_pred['reg_pred'])
reg_roc = metrics.roc_curve(df_pred['pessoa feliz'], df_pred['reg_proba'])
reg_auc = metrics.roc_auc_score(df_pred['pessoa feliz'], df_pred['reg_proba'])

naive_acc = metrics.accuracy_score(y_true=df_pred['pessoa feliz'], y_pred=df_pred['naive_pred'])
naive_prec = metrics.precision_score(y_true=df_pred['pessoa feliz'], y_pred=df_pred['naive_pred'])
naive_recall = metrics.recall_score(y_true=df_pred['pessoa feliz'], y_pred=df_pred['naive_pred'])
naive_roc = metrics.roc_curve(df_pred['pessoa feliz'], df_pred['naive_proba'])
naive_auc = metrics.roc_auc_score(df_pred['pessoa feliz'], df_pred['naive_proba'])

# %%

import matplotlib.pyplot as plt

plt.plot(arvore_roc[0], arvore_roc[1], 'o-')
plt.plot(naive_roc[0], naive_roc[1], 'o-')
plt.plot(reg_roc[0], reg_roc[1], 'o-')
plt.grid(True)
plt.title("ROC Curve")
plt.xlabel("1 - Especificidade")
plt.ylabel("Recall")

plt.legend([
    f"Árvore: {arvore_auc:.2f}",
    f"Naive: {naive_auc:.2f}",
    f"Reg Log.: {reg_auc:.2f}",
])

# %%