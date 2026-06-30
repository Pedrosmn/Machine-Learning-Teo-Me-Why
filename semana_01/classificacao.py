# %%

import pandas as pd

df = pd.read_excel('data/dados_cerveja_nota.xlsx')
df

# %%

df['aprovado'] = (df['nota'] >= 5).astype(int)
df

# %%

from sklearn import linear_model

reg_log = linear_model.LogisticRegression(fit_intercept=True, penalty=None)
reg_log.fit(df[['cerveja']], df['aprovado'])

reg_log_predict = reg_log.predict(df[['cerveja']].drop_duplicates())
reg_log_pred_proba = reg_log.predict_proba(df[['cerveja']].drop_duplicates())[:,1]

# %%

from sklearn import tree

decision_tree = tree.DecisionTreeClassifier(max_depth=2, random_state=42)
decision_tree.fit(df[['cerveja']], df['aprovado'])

tree_predict = decision_tree.predict(df[['cerveja']].drop_duplicates())
tree_pred_proba = decision_tree.predict_proba(df[['cerveja']].drop_duplicates())[:,1]

# %%

from sklearn import naive_bayes

nb = naive_bayes.GaussianNB()
nb.fit(df[['cerveja']], df['aprovado'])

nb_predict = nb.predict(df[['cerveja']].drop_duplicates())
nb_pred_proba = nb.predict_proba(df[['cerveja']].drop_duplicates())[:,1]

# %%

import matplotlib.pyplot as plt

plt.plot(df['cerveja'], df['aprovado'], "o", color='royalblue')
plt.grid(True)
plt.title('Cerveja x Aprovado')
plt.xlabel('Cerveja')
plt.ylabel('Aprovado')

plt.plot(df['cerveja'].drop_duplicates(), reg_log_predict, color="darkgreen")
plt.plot(df['cerveja'].drop_duplicates(), reg_log_pred_proba, color="lightgreen")

plt.plot(df['cerveja'].drop_duplicates(), tree_predict, color="red")
plt.plot(df['cerveja'].drop_duplicates(), tree_pred_proba, color="yellow")

plt.plot(df['cerveja'].drop_duplicates(), nb_predict, color="teal")
plt.plot(df['cerveja'].drop_duplicates(), nb_pred_proba, color="cyan")

plt.hlines(0.5, xmin=1, xmax=9, colors='black', linestyles='--')

plt.legend(['Observado',
            'Reg Log Predict',
            'Reg Log Proba',
            'Tree Predict',
            'Tree, Proba',
            'NB Predict',
            'NB Proba'])


# %%
reg_log_pred_proba