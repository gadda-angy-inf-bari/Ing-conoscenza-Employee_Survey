from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

import pandas as pd
from prova_preprocessing2 import preprocess_data, preprocess_scalable_models, preprocess_tree_models, split_dataset

# Supponiamo che tu abbia un dataset preprocessato X (features) e y (target)
# Sostituisci X e y con i tuoi dati
# Caricare il dataset
data = pd.read_csv("employee_survey.csv")

data.drop(['EmpID'], axis=1, inplace=True)
data = data.dropna()
data = data.drop_duplicates()


# Preprocessing dei dati
data, cat_columns, num_columns, bool_columns = preprocess_data(data)

# Verifica colonne presenti dopo la lettura del file
print("Colonne del dataset caricato:", data.columns)  # Debug per controllare le colonne dopo la lettura

data_tree = preprocess_tree_models(data, cat_columns)
data_other = preprocess_scalable_models(data, cat_columns, num_columns, bool_columns)


stress_col = "JobSatisfaction" #"Stress" 

X = data_tree.drop([stress_col], axis=1)
y = data_tree[stress_col]

# Suddivisione del dataset in training set e test set
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = split_dataset(X, y)

# Inizializzazione del classificatore GaussianNB
model = GaussianNB()

# Definizione della griglia degli iperparametri da testare
param_grid = {
    'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4]  # Valori tipici per il smoothing
}

# Inizializzazione di GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=5, verbose=1)

# Addestramento del modello
#model.fit(X_train, y_train)

# Esecuzione della ricerca
grid_search.fit(X_train, y_train)

# Miglior modello trovato da GridSearch
best_model = grid_search.best_estimator_
print(f"Miglior iperparametro: {grid_search.best_params_}")

# Predizione sui dati di test
#y_pred = model.predict(X_test)

# Predizione sui dati di test con il modello ottimizzato
y_pred = best_model.predict(X_test)

# Calcolo delle metriche di valutazione
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')  # Usa 'weighted' per ponderare le classi
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Stampa dei risultati
print("Metriche di valutazione:")
print(f"Accuratezza: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")

# Report di classificazione dettagliato
print("\nReport di classificazione completo:")
print(classification_report(y_test, y_pred))
