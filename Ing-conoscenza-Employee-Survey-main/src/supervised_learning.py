import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder

def preprocess_data(data):
    # Rimozione colonne non necessarie
    if 'EmpID' in data.columns:
        data.drop(['EmpID'], axis=1, inplace=True)

    # Verifichiamo che la colonna 'Stress' esista prima di ogni modifica
    print("Colonne prima del preprocessing:", data.columns)

    # Convertire 'Stress' in categoria
    if 'Stress' in data.columns:
        data['Stress'] = data['Stress'].astype('category')
    else:
        print("Attenzione: 'Stress' non trovata nel dataset.")

    # Separare le colonne numeriche e categoriali
    numerical_columns = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()

    # Rimuoviamo 'Stress' temporaneamente dalle colonne categoriali per non codificarla
    if 'Stress' in categorical_columns:
        categorical_columns.remove('Stress')

    # Codifica delle variabili categoriali
    data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)
    
    # Riconfermiamo che Stress è ancora nel dataset
    print("Colonne dopo il preprocessing:", data.columns)

    return data

def plot_kfold_cross_validation(models, X_train, y_train, kfold):
    
    for model_name, (model, param_name, param_range) in models.items():
        train_errors = []
        val_errors = []

        # Convertiamo i parametri in valori scalari per il grafico se necessario
        if model_name == 'Neural Network':
            param_range = [sum(p) for p in param_range]  # Convertiamo tuple in somma di neuroni

        
        for param_value in param_range:
            # Imposta il parametro chiave del modello corrente
            model.set_params(**{param_name: param_value})
            
            train_error, val_error = 0, 0
            for train_idx, val_idx in kfold.split(X_train):
                X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
                
                model.fit(X_train_fold, y_train_fold)
                
                train_error += 1 - accuracy_score(y_train_fold, model.predict(X_train_fold))
                val_error += 1 - accuracy_score(y_val_fold, model.predict(X_val_fold))
            
            # Calcoliamo l'errore medio per training e validation
            train_errors.append(train_error / kfold.n_splits)
            val_errors.append(val_error / kfold.n_splits)
        
        # Traccia il grafico degli errori
        plt.figure(figsize=(8, 5))
        plt.plot(param_range, train_errors, label=f'{model_name} - Errore su Training Set')
        plt.plot(param_range, val_errors, label=f'{model_name} - Errore su Validation Set')

        plt.xlabel(f'Valore di {param_name} per {model_name}')
        plt.ylabel('Errore')
        plt.title(f'K-Fold Cross Validation - {model_name}')
        plt.legend()
        plt.show()

def main():
    # Caricare il dataset
    data = pd.read_csv('employee_survey.csv')

    data = data.dropna()
    data = data.drop_duplicates()

    # Preprocessing dei dati
    data = preprocess_data(data)

    # Verifica colonne presenti dopo la lettura del file
    print("Colonne del dataset caricato:", data.columns)  # Debug per controllare le colonne dopo la lettura

    # Definire la variabile target e le feature
    stress_col = 'Stress'
    
    if stress_col not in data.columns:
        print(f"Errore: La colonna '{stress_col}' non è presente nel dataset.")
        return  # Termina l'esecuzione se la colonna non esiste
    
    X = data.drop([stress_col], axis=1)
    y = data[stress_col]

    # Codifica della variabile target se è categoriale
    if y.dtype.name == 'category' or y.dtype.name == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)  # Trasformiamo in numeri per classificazione

    # Imposta K-Fold Cross-Validation
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    # Modelli con iperparametri da testare
    models_for_plot = {
        'Decision Tree': (DecisionTreeClassifier(), 'max_depth', range(1, 21)),
        'Logistic Regression': (LogisticRegression(max_iter=1000), 'C', [0.01, 0.1, 1, 10, 100]),
        'Random Forest': (RandomForestClassifier(random_state=42), 'max_depth', range(1, 21)),
        'Neural Network': (MLPClassifier(max_iter=1000), 'hidden_layer_sizes', [(10,), (50,), (100,), (100, 50)])
    }

    # Traccia il grafico degli errori su Training e Validation per ogni modello 
    plot_kfold_cross_validation(models_for_plot, X, y, kfold)

    # Suddivisione in training e test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    # Definire parametri per la ricerca in griglia (ottimizzazione dei parametri)
    param_grids = {
        'Decision Tree': {
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
        },
        'Logistic Regression': {
            'C': [0.1, 1, 10, 100],
            'solver': ['liblinear', 'lbfgs']
        },
        'Random Forest': {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        },
        'Neural Network': {
            'hidden_layer_sizes': [(50,), (100,), (50, 50)],
            'activation': ['relu', 'tanh'],
            'solver': ['adam', 'sgd'],
            'max_iter': [500, 1000]
        }
    }

    # Dizionario per i modelli GridSearchCV
    grid_models = {
        'Decision Tree': DecisionTreeClassifier(),
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Neural Network': MLPClassifier(max_iter=1000)
    }

    # Dizionario per memorizzare le metriche
    metrics = {}
    best_models = {}




    for model_name, model in grid_models.items():
        print(f"\nOttimizzazione del modello: {model_name}")
        
        # Ricerca in griglia con K-Fold Cross Validation
        grid_search = GridSearchCV(estimator=model, 
                                   param_grid=param_grids[model_name], 
                                   cv=kfold, 
                                   scoring='accuracy', 
                                   n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        best_models[model_name] = best_model
        
        print(f"Migliori parametri per {model_name}: {grid_search.best_params_}")


        # Suddivisione in training e test
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


        # Eseguire K-Fold Cross Validation con il modello ottimizzato
        cross_val_scores = cross_val_score(best_model, X_train, y_train, cv=kfold, scoring='accuracy')
        print(f"Accuratezza media in Cross Validation per {model_name}: {cross_val_scores.mean():.4f}")
        
        # Predizione sul test set usando il modello ottimizzato
        y_pred = best_model.predict(X_test)

        # Calcolo delle metriche
        metrics[model_name] = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, average='macro'),
            'Recall': recall_score(y_test, y_pred, average='macro'),
            'F1 Score': f1_score(y_test, y_pred, average='macro')
        }

        # Stampa delle metriche
        print(f"Metrics for {model_name}: {metrics[model_name]}")
        
        # Matrice di confusione
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
        disp.plot()
        plt.title(f'Matrice di Confusione - {model_name}')
        plt.show()

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
