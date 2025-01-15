import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

import os

# Caricamento del dataset
def load_data():
    # Carica il CSV in un DataFrame
    #data = pd.read_csv("employee_survey.csv")
    #dataset_path = './dataset/employee_survey.csv'
    #data = pd.read_csv(dataset_path)

    dataset_path = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'employee_survey.csv')
    data = pd.read_csv(dataset_path)

    data.drop(['EmpID'], axis=1, inplace=True)
    data = data.dropna()
    data = data.drop_duplicates()

    ''' Index(['EmpID', 'Gender', 'Age', 'MaritalStatus', 'JobLevel', 'Experience',
    'Dept', 'EmpType', 'WLB', 'WorkEnv', 'PhysicalActivityHours',
    'Workload', 'Stress', 'SleepHours', 'CommuteMode', 'CommuteDistance',
    'NumCompanies', 'TeamSize', 'NumReports', 'EduLevel', 'haveOT',
    'TrainingHoursPerYear', 'JobSatisfaction'],
    dtype='object')


    ['ID', 'Genere', 'Eta', 'StatoCivile', 'LivelloLav', 'EsperienzaLav',
    'Dipartimento', 'TipoOccupazione', 'VEquVita-Lav', 'VAmbieteLav', 'OreAttivitaFisica',
    'VCaricoLav', 'Stress', 'OreSonno', 'TipoPendolare', 'DistanzaPercorsaPend',
    'NumCompagnie', 'GrandezzaTeam', 'NumSegnalazioni', 'LivelloIstruzione', 'HaStraordinari',
    'OreAnnualiFormative', 'ValLavorativa']
    '''

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
    
import pandas as pd
import matplotlib.pyplot as plt

# Import dataset



def convert_to_ranges(df):
    df_c = df.copy()

    # Convert 'Age' in intervals between 1 and 4
    df_c.loc[(df_c['Age'] >= 22) & (df_c['Age'] <= 31), ['Age']] = 1
    df_c.loc[(df_c['Age'] >= 32) & (df_c['Age'] <= 41), ['Age']] = 2
    df_c.loc[(df_c['Age'] >= 42) & (df_c['Age'] <= 51), ['Age']] = 3
    df_c.loc[(df_c['Age'] >= 52) & (df_c['Age'] <= 60), ['Age']] = 4

    # Convert 'Experience' in intervals between 1 and 4
    df_c.loc[(df_c['Experience'] >= 0) & (df_c['Experience'] <= 7), ['Experience']] = 1
    df_c.loc[(df_c['Experience'] >= 8) & (df_c['Experience'] <= 15), ['Experience']] = 2
    df_c.loc[(df_c['Experience'] >= 16) & (df_c['Experience'] <= 23), ['Experience']] = 3
    df_c.loc[(df_c['Experience'] >= 24) & (df_c['Experience'] <= 29), ['Experience']] = 4

    # Convert 'PhysicalActivityHours' in intervals between 1 and 5
    df_c.loc[(df_c['PhysicalActivityHours'] >= 0) & (df_c['PhysicalActivityHours'] <= 0.9), ['PhysicalActivityHours']] = 1
    df_c.loc[(df_c['PhysicalActivityHours'] >= 1) & (df_c['PhysicalActivityHours'] <= 1.9), ['PhysicalActivityHours']] = 2
    df_c.loc[(df_c['PhysicalActivityHours'] >= 2) & (df_c['PhysicalActivityHours'] <= 2.9), ['PhysicalActivityHours']] = 3
    df_c.loc[(df_c['PhysicalActivityHours'] >= 3) & (df_c['PhysicalActivityHours'] <= 3.9), ['PhysicalActivityHours']] = 4
    df_c.loc[(df_c['PhysicalActivityHours'] >= 4) & (df_c['PhysicalActivityHours'] <= 5), ['PhysicalActivityHours']] = 5

    # Convert 'SleepHours' in intervals between 1 and 6
    df_c.loc[(df_c['SleepHours'] >= 4) & (df_c['SleepHours'] <= 4.9), ['SleepHours']] = 1
    df_c.loc[(df_c['SleepHours'] >= 5) & (df_c['SleepHours'] <= 5.9), ['SleepHours']] = 2
    df_c.loc[(df_c['SleepHours'] >= 6) & (df_c['SleepHours'] <= 6.9), ['SleepHours']] = 3
    df_c.loc[(df_c['SleepHours'] >= 7) & (df_c['SleepHours'] <= 7.9), ['SleepHours']] = 4
    df_c.loc[(df_c['SleepHours'] >= 8) & (df_c['SleepHours'] <= 8.9), ['SleepHours']] = 5
    df_c.loc[(df_c['SleepHours'] >= 9) & (df_c['SleepHours'] <= 10), ['SleepHours']] = 6

    # Convert 'CommuteDistance' in intervals between 1 and 4
    df_c.loc[(df_c['CommuteDistance'] >= 1) & (df_c['CommuteDistance'] <= 7), ['CommuteDistance']] = 1
    df_c.loc[(df_c['CommuteDistance'] >= 8) & (df_c['CommuteDistance'] <= 14), ['CommuteDistance']] = 2
    df_c.loc[(df_c['CommuteDistance'] >= 15) & (df_c['CommuteDistance'] <= 21), ['CommuteDistance']] = 3
    df_c.loc[(df_c['CommuteDistance'] >= 22) & (df_c['CommuteDistance'] <= 29), ['CommuteDistance']] = 4

    # Convert 'NumCompanies' in intervals between 1 and 4
    df_c.loc[(df_c['NumCompanies'] >= 0) & (df_c['NumCompanies'] <= 2), ['NumCompanies']] = 1
    df_c.loc[(df_c['NumCompanies'] >= 3) & (df_c['NumCompanies'] <= 5), ['NumCompanies']] = 2
    df_c.loc[(df_c['NumCompanies'] >= 6) & (df_c['NumCompanies'] <= 8), ['NumCompanies']] = 3
    df_c.loc[(df_c['NumCompanies'] >= 9) & (df_c['NumCompanies'] <= 12), ['NumCompanies']] = 4

    # Convert 'TeamSize' in intervals between 1 and 5
    df_c.loc[(df_c['TeamSize'] >= 5) & (df_c['TeamSize'] <= 9), ['TeamSize']] = 1
    df_c.loc[(df_c['TeamSize'] >= 10) & (df_c['TeamSize'] <= 14), ['TeamSize']] = 2
    df_c.loc[(df_c['TeamSize'] >= 15) & (df_c['TeamSize'] <= 19), ['TeamSize']] = 3
    df_c.loc[(df_c['TeamSize'] >= 20) & (df_c['TeamSize'] <= 24), ['TeamSize']] = 4
    df_c.loc[(df_c['TeamSize'] >= 25) & (df_c['TeamSize'] <= 30), ['TeamSize']] = 5

    # Convert 'TrainingHoursPerYear' in intervals between 1 and 5
    df_c.loc[(df_c['TrainingHoursPerYear'] >= 10) & (df_c['TrainingHoursPerYear'] <= 20), ['TrainingHoursPerYear']] = 1
    df_c.loc[(df_c['TrainingHoursPerYear'] >= 21) & (df_c['TrainingHoursPerYear'] <= 31), ['TrainingHoursPerYear']] = 2
    df_c.loc[(df_c['TrainingHoursPerYear'] >= 32) & (df_c['TrainingHoursPerYear'] <= 42), ['TrainingHoursPerYear']] = 3
    df_c.loc[(df_c['TrainingHoursPerYear'] >= 43) & (df_c['TrainingHoursPerYear'] <= 53), ['TrainingHoursPerYear']] = 4
    df_c.loc[(df_c['TrainingHoursPerYear'] >= 54) & (df_c['TrainingHoursPerYear'] <= 65), ['TrainingHoursPerYear']] = 5

    return df_c