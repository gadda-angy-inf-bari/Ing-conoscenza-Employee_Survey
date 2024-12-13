import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.preprocessing import LabelEncoder

import seaborn as sns
import numpy as np
from sklearn.svm import SVC

from prova_preprocessing2 import preprocess_data, preprocess_scalable_models, preprocess_tree_models, split_dataset
import os



def grid_search_cross_validation(X, y, param_grid, k=5):
    """
    Esegue la Grid Search con cross-validation su tutto il dataset, iterando su ogni fold.
    
    Parametri:
    - X: array o DataFrame delle feature.
    - y: array o serie dei target.
    - param_grid: dizionario con i parametri da testare.
    - k: numero di fold per la cross-validation.
    
    Ritorna:
    - Combinazione di parametri ottimali.
    - Migliore score medio sui fold.
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    best_score = -np.inf
    best_params = None
    
    # Itera su tutte le combinazioni di parametri
    for n_estimators in param_grid['n_estimators']:
        for max_depth in param_grid['max_depth']:
            for min_samples_split in param_grid['min_samples_split']:
                print(f"Testing parameters: n_estimators={n_estimators}, max_depth={max_depth}, min_samples_split={min_samples_split}")
                scores = []
                
                # Cross-validation sui fold
                for train_index, test_index in kf.split(X):
                    #X_train, X_test = X[train_index], X[test_index]
                    #y_train, y_test = y[train_index], y[test_index]
                    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                    
                    # Addestra il modello con i parametri attuali
                    model = RandomForestClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        random_state=42
                    )
                    model.fit(X_train, y_train)
                    
                    # Calcola lo score sul fold corrente
                    y_pred = model.predict(X_test)
                    score = accuracy_score(y_test, y_pred)
                    scores.append(score)
                
                # Calcola lo score medio per questa combinazione di parametri
                mean_score = np.mean(scores)
                print(f"Mean score: {mean_score:.4f}")
                
                # Aggiorna i migliori parametri se necessario
                if mean_score > best_score:
                    best_score = mean_score
                    best_params = {
                        'n_estimators': n_estimators,
                        'max_depth': max_depth,
                        'min_samples_split': min_samples_split
                    }
    
    return best_params, best_score


def main():
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


    stress_col = "Stress" #"JobSatisfaction" 

    X = data_tree.drop([stress_col], axis=1)
    y = data_tree[stress_col]

    X_train, X_test, y_train, y_test = split_dataset(X, y)


    train_accs = []
    test_accs = []


    # inizializziamo un loop dove cambieremo il valore di max depth, partendo da 1 a 25
    for depth in range(1, 25):
        clf = DecisionTreeClassifier(max_depth=depth)
        clf.fit(X_train, y_train)
        train_predictions = clf.predict(X_train)
        test_predictions = clf.predict(X_test)
        
        train_acc = accuracy_score(y_train, train_predictions)
        test_acc = accuracy_score(y_test, test_predictions)
        
        # inseriamo in liste vuote le nostre accuracies
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        
    # visualizziamo i dati
    plt.figure(figsize=(10, 5))
    sns.set_style('whitegrid')
    plt.plot(train_accs, label='train accuracy')
    plt.plot(test_accs, label='test accuracy')
    plt.legend(loc='upper left', prop={'size': 15})
    plt.xticks(range(0, 26, 5))
    plt.xlabel('max_depth', size=20)
    plt.ylabel('accuracy', size=20)
    plt.show()


    #import numpy as np
    #import pandas as pd
    #from sklearn.tree import DecisionTreeClassifier
    #from sklearn.model_selection import train_test_split, GridSearchCV
    #from sklearn.metrics import accuracy_score, classification_report
    #import seaborn as sns
    #import matplotlib.pyplot as plt

    # Dataset di esempio (sostituire con il proprio dataset)
    #X = np.random.rand(500, 10)  # 500 campioni, 10 feature
    #y = np.random.randint(0, 2, 500)  # Binario (0/1)

    # Step 1: Train-test split
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 2: Definizione dei modelli e iperparametri per la ricerca
    models = {
        "DecisionTree": DecisionTreeClassifier(random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42),
        "RandomForest": RandomForestClassifier(random_state=42),
        #"SVC": SVC(random_state=42),
        'Neural Network': MLPClassifier(random_state=42)
    }

    param_grids = {
        "DecisionTree": {
            "criterion": ['gini', 'entropy', 'log_loss'],
            "max_depth": [2, 3, 4, 5, 10, 20],
            "min_samples_split": [2, 3, 4, 5, 10, 20],
            #"min_samples_leaf": [1, 2, 4, 5, 10], #[1, 2, 4, 5, 10, 50],
            #'max_features': ['auto', 'sqrt', 'log2'], 
            #'class_weight': [None, 'balanced'], 
        },
        'Logistic Regression': {
            'penalty': ['l1', 'l2', 'elasticnet', 'none'],
            'C': [0.01, 0.1, 1, 10, 100, 1000],
            #'solver': ['liblinear', 'saga', 'lbfgs', 'newton-cg'],
            'max_iter': [100, 200, 500, 1000],
            #'l1_ratio': [0.1, 0.5, 0.9]
        },
        "RandomForest": {
            "n_estimators": [10, 50, 100, 200, 500, 1000],
            "max_depth": [1, 2, 5, 10, 20], #[None, 10, 20],
            #"min_samples_split": [2, 5, 10], 
            #"max_features": ['sqrt', 'log2'], 
            "criterion": ['gini', 'entropy'],
            #"bootstrap": [True, False]
        },
        #"SVC": {
            #"C": [0.01, 0.1, 1],
            #"kernel": ['linear', 'rbf', 'sigmoid'], #['linear', 'rbf', 'poly', 'sigmoid'],
            #"gamma": ["scale", "auto"], #["scale", "auto", 0.1, 0.01, 0.001],
            #'degree': [0, 1, 2, 3, 4, 5],
            #'class_weight': [None, 'balanced']
        #},
        'Neural Network': {
            'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
            #'activation': ['relu', 'tanh', 'logistic'],
            #'solver': ['adam', 'sgd', 'lbfgs'],
            #'alpha': [0.0001, 0.001, 0.01, 0.1, 1],
            #'learning_rate': ['constant', 'invscaling', 'adaptive'],
            'learning_rate_init': [0.001, 0.01],
            'max_iter': [200, 300, 500]
        }
    }

    # Step 3: Cross-validation e ricerca iperparametri
    best_models = {}
    results = []

     # creiamo l'oggetto KFold applicando la regola di Sturges
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
        #best_models[model_name] = grid_search.best_estimator_
        best_model = grid_search.best_estimator_
        best_models[model_name] = best_model
        results.append({
            "model": model_name,
            "best_params": grid_search.best_params_,
            "best_score": grid_search.best_score_
        })
        os.system("pause")   



        # QUESTO È IL LOOP DI CROSS-VALIDAZIONE! 
        for train_idx, val_idx, in kf.split(X, y):
            # l'oggetto kf genera gli indici e i valori per le rispettive X e y, creando il set di validazione su cui testare il modello nello split.
            X_tr = X.iloc[train_idx] 
            y_tr = y.iloc[train_idx]
            
            X_val = X.iloc[val_idx]
            y_val = y.iloc[val_idx]
            
            # qui addestriamo il modello
            #clf = DecisionTreeClassifier(max_depth=5)
            #clf.fit(X_tr, y_tr)
            best_model.fit(X_tr, y_tr)
            # creiamo le predizioni e salviamo lo score nella lista aucs
            #pred = clf.predict(X_val)
            pred = best_model.predict(X_val)

            #pred_prob = clf.predict_proba(X_val)[:, 1]
            acc_score = accuracy_score(y_val, pred)

            
            # Probabilità predette clf -> best_model
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

            #auc_score = roc_auc_score(y_val, pred_prob, multi_class='ovr')

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

        # Ora calcola la deviazione standard dei punteggi ottenuti
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

    # Step 4: Valutazione sui test set
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

    # Step 5: Selezione del modello migliore
    best_model_result = max(final_results, key=lambda x: x['test_accuracy'])
    print(f"\nBest Model: {best_model_result['model']} with Test Accuracy: {best_model_result['test_accuracy']}")

    os.system("pause") 





    # Visualizzazione dei risultati delle metriche
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
