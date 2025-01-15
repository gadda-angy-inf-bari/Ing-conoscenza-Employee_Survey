import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, classification_report
import numpy as np

#from prova_preprocessing2T import preprocess_data, preprocess_scalable_models, preprocess_tree_models, split_dataset
from preprocessing_data import load_data, preprocess_data, preprocess_scalable_models, preprocess_tree_models, split_dataset
import os


def main():
    data = load_data()
    

    # Preprocessing dei dati
    data, cat_columns, num_columns, bool_columns = preprocess_data(data)

    # Verifica colonne presenti dopo la lettura del file
    print("Colonne del dataset caricato:", data.columns)  # Debug per controllare le colonne dopo la lettura

    data_tree = preprocess_tree_models(data, cat_columns)
    data_other = preprocess_scalable_models(data, cat_columns, num_columns, bool_columns)


    stress_col = "Stress" #"JobSatisfaction" 

    X = data_tree.drop([stress_col], axis=1)
    y = data_tree[stress_col]

    # Train-test split
    X_train, X_test, y_train, y_test = split_dataset(X, y)

    # Definire i modelli e iperparametri per la ricerca
    models = {
        "DecisionTree": DecisionTreeClassifier(random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42),
        "RandomForest": RandomForestClassifier(random_state=42),
        'Neural Network': MLPClassifier(random_state=42)
    }

    param_grids = {
        "DecisionTree": {
            "criterion": ['gini', 'entropy', 'log_loss'],
            "max_depth": [2, 3, 4, 5, 10, 20],
            "min_samples_split": [2, 3, 4, 5, 10, 20],
        },
        'Logistic Regression': {
            'penalty': ['l1', 'l2', 'elasticnet', 'none'],
            'C': [0.01, 0.1, 1, 10, 100, 1000],
            'max_iter': [100, 200, 500, 1000],
        },
        "RandomForest": {
            "n_estimators": [10, 50, 100, 200, 500, 1000],
            "max_depth": [1, 2, 5, 10, 20],
            "criterion": ['gini', 'entropy'],
        },
        'Neural Network': {
            'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
            'learning_rate_init': [0.001, 0.01],
            'max_iter': [200, 300, 500]
        }
    }

    # Cross-validation e ricerca iperparametri
    best_models = {}
    results = []

    # crere l'oggetto KFold applicando la regola di Sturges
    sturges = int(1 + np.log(len(X)))
    kf = KFold(n_splits=sturges, shuffle=True, random_state=42)

    fold = 0
    aucs = []
    scores = []  # Lista per raccogliere i punteggi
    std_dev = {}

    for model_name, model in models.items():
        print(f"Running GridSearchCV for {model_name}...")
        grid_search = GridSearchCV(model, param_grids[model_name], cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        print(f"Migliori iperparametri per {model_name}:", grid_search.best_params_)
        print("Miglior score:", grid_search.best_score_)

        # Memorizza il migliore modello e i risultati
        best_model = grid_search.best_estimator_
        best_models[model_name] = best_model
        results.append({
            "model": model_name,
            "best_params": grid_search.best_params_,
            "best_score": grid_search.best_score_
        })
        os.system("pause")   



        # LOOP DI CROSS-VALIDAZIONE
        # kf.split() genera gli indici e i valori per le rispettive X e y, 
        # crea il set di validazione su cui testare il modello nello split.
        for train_idx, val_idx, in kf.split(X, y):
            X_tr = X.iloc[train_idx] 
            y_tr = y.iloc[train_idx]
            
            X_val = X.iloc[val_idx]
            y_val = y.iloc[val_idx]
            
            # addestrare un modello
            best_model.fit(X_tr, y_tr)
            # eseguire la predizione e salvare in una lista
            pred = best_model.predict(X_val)

            acc_score = accuracy_score(y_val, pred)

            
            # Probabilità predette sul miglior modello
            if hasattr(best_model, "predict_proba"):
                pred_prob = best_model.predict_proba(X_val)
                print("Forma di pred_prob:", pred_prob.shape)
            else:
                raise ValueError("Il modello non supporta predict_proba")

            # Calcolo AUC
            if len(np.unique(y)) == 2:  # Caso binario
                auc_score = roc_auc_score(y_val, pred_prob[:, 1])
            else:  # Caso multi-classe
                auc_score = roc_auc_score(y_val, pred_prob, multi_class='ovr')


            print(f"======= Fold {fold} ========")
            print(
                f"Accuracy on the validation set is {acc_score:0.4f} and AUC is {auc_score:0.4f}"
            )
            # aggiorniamo il valore di fold così possiamo stampare il progresso
            fold += 1
            aucs.append(auc_score)


            # Calcolare il punteggio per questo fold (ad esempio, accuracy)
            score = best_model.score(X_val, y_val)  # Puoi usare anche altre metriche come accuracy o f1
            scores.append(score)
        
        general_auc_score = np.mean(aucs)
        print(f'\nOur out of fold AUC score is {general_auc_score:0.4f}')

        # calcolare la deviazione standard dei punteggi ottenuti
        std_dev[model_name] = np.std(scores)

        print(f"La deviazione standard di {model_name} dei punteggi è: {std_dev[model_name]}")

        
        os.system("pause")    
    
    plt.plot(["DecisionTree", "Logistic Regression", "RandomForest", 'Neural Network'],
            [std_dev["DecisionTree"], std_dev["Logistic Regression"], std_dev["RandomForest"], std_dev['Neural Network']])
    plt.title("Standard deviation")
    plt.ylabel("Standard deviation value")
    plt.xlabel("Classifiers")
    plt.show()
    print("\nStandard deviation for DecisionTree:", std_dev["DecisionTree"])
    print("\nStandard deviation for Logistic Regression:", std_dev["Logistic Regression"])
    print("\nStandard deviation for RandomForest:", std_dev["RandomForest"])
    print("\nStandard deviation for Neural Network:", std_dev["Neural Network"])
    os.system("pause") 

    # Valutare del modello appreso sui test set
    final_results = []
    metrics = {}
    for model_name, best_model in best_models.items():
        y_pred = best_model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        test_precision = precision_score(y_test, y_pred, average='macro')
        test_recall = recall_score(y_test, y_pred, average='macro')
        test_f1_score = f1_score(y_test, y_pred, average='macro')
        final_results.append({
            "model": model_name,
            "test_accuracy": test_accuracy,
            "test_precision": test_precision,
            "test_recall": test_recall,
            "test_f1_score": test_f1_score,
            "classification_report": classification_report(y_test, y_pred, output_dict=True)
        })
        
        # Calcolo delle metriche
        metrics[model_name] = {
            'Accuracy': test_accuracy,
            'Precision': test_precision,
            'Recall': test_recall,
            'F1 Score': test_f1_score
        }
        classes_to_keep = np.unique(y_test)  # Mantieni solo le classi con dati reali
        print(f"Model: {model_name} | Test Accuracy: {test_accuracy} \n\t \
              Test precision: {test_precision} \n\t \
                Test recall: {test_recall} \n\t \
                    Test f1_score: {test_f1_score} \n\t")
        print(classification_report(y_test, y_pred, labels=classes_to_keep, zero_division=0))
        os.system("pause") 

    # Stampare il modello migliore
    best_model_result = max(final_results, key=lambda x: x['test_accuracy'])
    print(f"\nBest Model: {best_model_result['model']} with Test Accuracy: {best_model_result['test_accuracy']}")

    os.system("pause") 


    # Visualizzare i risultati delle metriche
    metrics_df = pd.DataFrame(metrics).T
    metrics_df.plot(kind='bar', figsize=(10, 6))
    plt.title('Confronto delle Metriche dei Modelli')
    plt.ylabel('Valore')
    plt.xlabel('Modelli')
    plt.xticks(rotation=45)
    plt.legend(loc='best')
    plt.show()




if __name__ == "__main__":
    main()
