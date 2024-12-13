import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# Caricamento del dataset
def load_data(file_path):
    # Carica il CSV in un DataFrame
    data = pd.read_csv(file_path)
    return data

# Preprocessing generale
def preprocess_data(data):
    # Identificazione dei dati categorici e numerici
    categorical_columns = data.select_dtypes(include=['object']).columns
    numerical_columns = data.select_dtypes(include=['int64', 'float64']).columns
    boolean_columns = data.select_dtypes(include=['bool']).columns

    # Gestione dei valori mancanti
    imputer_cat = SimpleImputer(strategy='most_frequent')
    imputer_num = SimpleImputer(strategy='mean')
    data[categorical_columns] = imputer_cat.fit_transform(data[categorical_columns])
    data[numerical_columns] = imputer_num.fit_transform(data[numerical_columns])

    # Restituisce feature, target e codificatori
    return data, categorical_columns, numerical_columns, boolean_columns

# Preprocessing per DecisionTreeClassifier e RandomForestClassifier
def preprocess_tree_models(X, categorica_columns):
    # LabelEncoding per modelli basati su alberi
    for col in categorica_columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
    return X

# Preprocessing per LogisticRegression e MLPClassifier
def preprocess_scalable_models(X, categorica_columns, numerica_columns, boolea_columns):
    # Gestione dei valori mancanti
    X[boolea_columns] = X[boolea_columns].astype(int)  # Converti i booleani in interi

    # OneHotEncoding per modelli che lo richiedono
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_encoded = ohe.fit_transform(X[categorica_columns])
    X_encoded = pd.DataFrame(X_encoded, columns=ohe.get_feature_names_out(categorica_columns))
    X = pd.concat([X.drop(columns=categorica_columns), X_encoded], axis=1)

    # Scalatura dei dati numerici (solo per modelli che lo richiedono)
    scaler = StandardScaler()
    X[numerica_columns] = scaler.fit_transform(X[numerica_columns])
    return X

# Suddivisione del dataset
def split_dataset(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# Pipeline completa
def main(file_path, target_column, model_type):
    # Carica il dataset
    data = load_data(file_path)

    #print("Colonne dopo il preprocessing:", data.dtypes)
    # Preprocessing generale
    data, cat_columns, num_columns, bool_columns = preprocess_data(data)

    # Preprocessing specifico per il modello
    if model_type in ['DecisionTreeClassifier', 'RandomForestClassifier']:
        X_processed = preprocess_tree_models(data, cat_columns)
    elif model_type in ['LogisticRegression', 'MLPClassifier']:
        X_processed = preprocess_scalable_models(data, cat_columns, num_columns, bool_columns)
    else:
        raise ValueError("Modello non supportato")

    # Suddivisione del dataset
    #X_train, X_test, y_train, y_test = split_dataset(X_processed, y)

    #print("Colonne dopo il preprocessing:", X.dtypes)

    return X_processed

# Esempio di utilizzo
if __name__ == "__main__":
    file_path = "employee_survey.csv"  # Sostituisci con il percorso del tuo file CSV
    target_column = "Stress"  # Sostituisci con il nome della colonna target
    

    # Esegui per ciascun modello
    for model in ['DecisionTreeClassifier', 'LogisticRegression', 'RandomForestClassifier', 'MLPClassifier']:
        print(f"Preprocessing per: {model}")
        X_train, X_test, y_train, y_test = main(file_path, target_column, model)
        print(f"Shape X_train: {X_train.shape}, X_test: {X_test.shape}")
    
