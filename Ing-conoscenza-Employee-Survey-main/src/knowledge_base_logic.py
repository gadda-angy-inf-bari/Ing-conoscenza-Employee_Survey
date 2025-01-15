import pytholog
import pandas as pd
from preprocessing_data import load_data

def clausole():
    # Crea un database logico
    employee_survey = pytholog.KnowledgeBase("sondaggio_dipendenti")

    # Caricare il dataset
    df = load_data()

    # Fatti
    facts = []

    # 0
    data = df[['EduLevel', 'Dept']].drop_duplicates().to_dict(orient="records")
    for d in data:
        facts.append(f"settore_lav({d['Dept'].lower()},{d['EduLevel'].lower()})")

    # 1
    status = df.groupby(['Dept', 'MaritalStatus'])['MaritalStatus'].count().reset_index(name='counts')
    department = df.groupby(['Dept'])['Dept'].count().reset_index(name='counts')
    merged = pd.merge(status, department, on='Dept', how='inner')
    merged['percentage'] = merged['counts_x']/merged['counts_y'] * 100
    merged.drop(['counts_x', 'counts_y'], axis=1, inplace=True)
    data = merged.to_dict('records')
    for d in data:
        facts.append(f"dipartimento_stato_civile({d['Dept'].lower()},{d['MaritalStatus'].lower()},{d['percentage']})")

    # 2
    data = df.groupby(['Dept'])['WorkEnv'].mean().to_dict()
    for key, value in data.items():
        facts.append(f"media_ambiente_lav_sodd({key.lower()},{value})")

    # 3
    data = df.groupby(['Dept'])['JobSatisfaction'].mean().to_dict()
    for key, value in data.items():
        facts.append(f"media_sodd_lav({key.lower()},{value})")

    # 4
    res = df.groupby(['Dept', 'WLB'])['JobSatisfaction'].mean()
    data = res.to_frame().reset_index().to_dict('records')
    for d in data:
        facts.append(f"media_lav_equ_vitaLav({d['Dept'].lower()},{d['WLB']},{d['JobSatisfaction']})")

    # 5
    attrition = df.groupby(['Dept', 'WLB'])['WLB'].count().reset_index(name='counts')
    department = df.groupby(['Dept'])['Dept'].count().reset_index(name='counts')
    merged = pd.merge(attrition, department, on='Dept', how='inner')
    merged['percentage'] = merged['counts_x'] / merged['counts_y'] * 100
    merged.drop(['counts_x', 'counts_y'], axis=1, inplace=True)
    data = merged.to_dict('records')
    for d in data:
        facts.append(f"dipartimento_equ_vitaLav_percentuale({d['Dept'].lower()},{d['WLB']},{d['percentage']})")


    employee_survey(facts)

    # Regole
    rules = []

    employee_survey(rules)

    return employee_survey






def main():
    employee_survey = clausole()

    query = [
        'settore_lav',
        'dipartimento_stato_civile',
        'media_ambiente_lav_sodd',
        'media_sodd_lav',
        'media_lav_equ_vitaLav',
        'dipartimento_equ_vitaLav_percentuale',
    ]
    query_range = range(0, len(query))

    dept = [
        'it', 
        'hr', 
        'finance', 
        'marketing', 
        'sales', 
        'legal', 
        'operations', 
        'customer service',
        #'Q'
    ]
    dept_range = range(0, len(dept))

    eduLevel = [
        'High School', 
        'Bachelor', 
        'Master', 
        'PhD' 
        #'Q'
    ]
    eduLevel_range = range(0, len(eduLevel))

    marital_status = [
        'single',
        'married',
        'divorced', 
        'widowed'
    ]
    marital_status_range = range(0, len(marital_status))

    wlb = [
        1, 
        2, 
        3, 
        4,
        5
    ]
    wlb_range = range(1, 5)

    stress = range(1, 5)

    for index, value in enumerate(query):
        print(f'{index}: {value}')

    query_answ = None
    while query_answ not in query_range:
        query_answ = int(input('Scegli una domanda:'))

    if query_answ == 0:
        for index, value in enumerate(dept):
            print(f'{index}: {value}')
        dept_answ = None
        while dept_answ not in dept_range:
            dept_answ = int(input('Scegli un dipartimento:'))

        print(employee_survey.query(pytholog.Expr(f"{query[query_answ]}({dept[dept_answ]}, Q)")))#{eduLevel[eduLevel_answ]})")))

    elif query_answ == 1:
        for index, value in enumerate(dept):
            print(f'{index}: {value}')
        dept_answ = None
        while dept_answ not in dept_range:
            dept_answ = int(input('Scegli un dipartimento:'))

        for index, value in enumerate(marital_status):
            print(f'{index}: {value}')
        status_answ = None
        while status_answ not in marital_status_range:
            status_answ = int(input('Scegli uno stato civile:'))

        print(employee_survey.query(pytholog.Expr(f"{query[query_answ]}({dept[dept_answ]},{marital_status[status_answ]}, Q)")))

    elif query_answ == 2 or query_answ == 3:
        for index, value in enumerate(dept):
            print(f'{index}: {value}')
        dept_answ = None
        while dept_answ not in dept_range:
            dept_answ = int(input('Scegli un dipartimento:'))

        print(employee_survey.query(pytholog.Expr(f"{query[query_answ]}({dept[dept_answ]}, Q)")))

    elif query_answ == 4 or query_answ == 5:
        for index, value in enumerate(dept):
            print(f'{index}: {value}')
        dept_answ = None
        while dept_answ not in dept_range:
            dept_answ = int(input('Scegli un dipartimento:'))

        for index, value in enumerate(wlb):
            print(f'{index}: {value}')
        wlb_answ = None
        while wlb_answ not in wlb_range:
            wlb_answ = int(input('Scegli il livello di bilanciamento tra vita-lavoro:'))

        print(employee_survey.query(pytholog.Expr(f"{query[query_answ]}({dept[dept_answ]},{wlb[wlb_answ]}, Q)")))


if __name__ == "__main__":
    main()