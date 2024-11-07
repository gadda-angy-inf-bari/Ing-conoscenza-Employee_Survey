import pandas as pd
import matplotlib.pyplot as plt

# Import dataset
def load_data():
    df = pd.read_csv("employee_survey.csv")
    #print(df)


    df = df.dropna()
    df.drop_duplicates()


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
    #print(df.columns)

    df.drop(columns=['EmpID'])


    #print(df)
    return df


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