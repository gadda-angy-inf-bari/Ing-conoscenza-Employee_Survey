from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import numpy as np

import pandas as pd
#from prova_preprocessing2T import preprocess_data, preprocess_scalable_models, preprocess_tree_models, split_dataset
from preprocessing_data import load_data, preprocess_data, preprocess_scalable_models, preprocess_tree_models, split_dataset


def main():
    # Caricare il dataset
    data = load_data()


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

    # Esecuzione della ricerca
    grid_search.fit(X_train, y_train)

    # Miglior modello trovato da GridSearch
    best_model = grid_search.best_estimator_
    print(f"Miglior iperparametro: {grid_search.best_params_}")

    # Implementazione manuale della cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_accuracies = []
    cv_precisions = []
    cv_recalls = []
    cv_f1s = []

    y_pred_cv = []
    y_true_cv = []

    for train_index, val_index in kf.split(X_train):
        X_ktrain, X_kval = X_train.iloc[train_index], X_train.iloc[val_index]
        y_ktrain, y_kval = y_train.iloc[train_index], y_train.iloc[val_index]

        # Addestramento del modello sul fold corrente
        best_model.fit(X_ktrain, y_ktrain)

        # Predizione sul validation set
        y_kpred = best_model.predict(X_kval)

        # Accumula predizioni e vere etichette
        y_pred_cv.extend(y_kpred)
        y_true_cv.extend(y_kval)

        # Calcolo delle metriche
        cv_accuracies.append(accuracy_score(y_kval, y_kpred))
        cv_precisions.append(precision_score(y_kval, y_kpred, average='weighted'))
        cv_recalls.append(recall_score(y_kval, y_kpred, average='weighted'))
        cv_f1s.append(f1_score(y_kval, y_kpred, average='weighted'))

    # Accumulazione delle metriche di valutazione
    print("\nMetriche cross-validation:")
    print(f"Accuratezza media (CV): {np.mean(cv_accuracies):.2f} (+/- {np.std(cv_accuracies):.2f})")
    print(f"Precisione media (CV): {np.mean(cv_precisions):.2f} (+/- {np.std(cv_precisions):.2f})")
    print(f"Recall media (CV): {np.mean(cv_recalls):.2f} (+/- {np.std(cv_recalls):.2f})")
    print(f"F1-Score medio (CV): {np.mean(cv_f1s):.2f} (+/- {np.std(cv_f1s):.2f})")

    # Addestramento finale del modello migliore
    best_model.fit(X_train, y_train)

    # Test finale sul modello migliore
    y_pred = best_model.predict(X_test)

    # Calcolo delle metriche di valutazione finali
    final_accuracy = accuracy_score(y_test, y_pred)
    final_precision = precision_score(y_test, y_pred, average='weighted')
    final_recall = recall_score(y_test, y_pred, average='weighted')
    final_f1 = f1_score(y_test, y_pred, average='weighted')

    # Stampa dei risultati finali
    print("\nMetriche finali sul test set:")
    print(f"Test Accuracy: {final_accuracy:.2f}")
    print(f"Test precision: {final_precision:.2f}")
    print(f"Test recall: {final_recall:.2f}")
    print(f"Test f1_score: {final_f1:.2f}")

    # Report di classificazione dettagliato
    print("\nReport di classificazione completo:")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()