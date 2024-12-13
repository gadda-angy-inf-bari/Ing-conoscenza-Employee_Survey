import pytholog
import pytholog as pl
import dataset as dt
import pandas as pd
from prova_preprocessing2 import preprocess_data, preprocess_scalable_models, preprocess_tree_models, split_dataset
#from pyswip import Prolog

# Crea un database logico
employee_survey = pytholog.KnowledgeBase("sondaggio_dipendenti")

df = pd.read_csv("employee_survey.csv")
df.drop(columns=['EmpID'])
# Check for empty elements
df = df.dropna()
# Check for duplicate rows
df.drop_duplicates(inplace=True)


facts = []

#data = df[['WLB', 'Stress']].drop_duplicates().to_dict(orient="records")
#for d in data:
#    facts.append(f"buon_equilibrio_vita_lavoro({d['WLB']}, {d['Stress']})")

# 0+1
data = df[['EduLevel', 'Dept']].drop_duplicates().to_dict(orient="records")
for d in data:
    #facts.append(f"suitable_department({d['EduLevel'].lower()},{d['Dept'].lower()})")
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







#print(facts)

employee_survey(facts)

rules = [
    #"buon_equilibrio_vita_lavoro(X) :- buon_equilibrio_stress(X, S), S < 3"
    #"buon_equilibrio_vita_lavoro(WLB, Stress) :- dipendente(WLB, Stress), (WLB >= 3), (Stress < 2)"
    #"tutti_i_risultati() :- dipendente(WLB, Stress), WLB >= 3, Stress < 2"
    #"stressato(ID, Causa) :- dipendente(_, _, _, _, _, _, _, _, _, _, _, Stress(1), _, CommuteDistance(Causa), _, _, _, _, _, _, _, _)",
    #"promozione(ID, Istruzione) :- dipendente(ID, _, _, _, _, esperienza_lav(E), _, _, _, _, _, _, _, _, _, _, _, _, num_segnalazioni(0), livello_istruzione(Istruzione), _, _, _), E > 5"
]

employee_survey(rules)










#for fact in employee_survey.db:
#    print(fact)

# Query per buon equilibrio vita-lavoro
#query_n = employee_survey.query(pytholog.Expr(f"buon_equilibrio_vita_lavoro(3, Q)"))
#print("Buon equilibrio vita-lavoro:", query_n)

# Query per buon equilibrio vita-lavoro
#query_1 = employee_survey.query(pytholog.Expr(f"buon_equilibrio_vita_lavoro(WLB, Stress)"))
#query_0 = employee_survey.query(pytholog.Expr(f"suitable_department(bachelor, Q)"))
#print("Educazione dipartimento:", query_0)

#query_1 = employee_survey.query(pytholog.Expr(f"working_field(bachelor, Q)"))
#print("Educazione dipartimento:", query_1)

#query_3 = employee_survey.query(pytholog.Expr(f"department_marital_status(bachelor, single, Q)"))
#print("Educazione dipartimento:", query_3)

#query_4 = employee_survey.query(pytholog.Expr(f"average_work_env_satisfaction(bachelor, Q)"))
#print("Educazione dipartimento:", query_4)

#query_5 = employee_survey.query(pytholog.Expr(f"average_job_satisfaction(bachelor, Q)"))
#print("Educazione dipartimento:", query_5)

#query_6 = employee_survey.query(pytholog.Expr(f"average_job_wlb(bachelor, 2, Q)"))
#print("Educazione dipartimento:", query_6)

#query_7 = employee_survey.query(pytholog.Expr(f"department_wlb_percentage(bachelor, 2)"))
#print("Educazione dipartimento:", query_7)





# 0-1
#data = df[[ 'Gender', 'Age', 'MaritalStatus', 'JobLevel', 
            #'Experience', 'Dept', 'EmpType', 'WLB', 
            #'WorkEnv', 'PhysicalActivityHours', 'Workload', 'Stress', 
            #'SleepHours', 'CommuteMode', 'CommuteDistance', 'NumCompanies', 
            #'TeamSize', 'NumReports', 'EduLevel', 'haveOT',
            #'TrainingHoursPerYear', 'JobSatisfaction', 
        #]].drop_duplicates().to_dict(orient="records")


            #kb.append(f"dipendente("
              #f"{d['Gender'].lower()},        {d['Age']},                     {d['MaritalStatus'].lower()},   {d['JobLevel']}, "
              #f"{d['Experience']},    {d['Dept'].lower()},                    {d['EmpType'].lower()},         {d['WLB']}, "
              #f"{d['WorkEnv']},       {d['PhysicalActivityHours']},   {d['Workload']},        {d['Stress']}, "
              #f"{d['SleepHours']},    {d['CommuteMode']},             {d['CommuteDistance']}, {d['NumCompanies']}, "
              #f"{d['TeamSize']},      {d['NumReports']},              {d['EduLevel'].lower()},        {d['haveOT']}, "
              #f"{d['TrainingHoursPerYear']}, {d['JobSatisfaction']})")


        #kb.append(f"suitable_department({d['EducationField'].lower()},{d['Department'].lower()})")

# Definizione dei fatti
#knowledge([
    #"dipendente()"
#])

# Definizione delle regole
#knowledge([

# Query per stress elevato
#query_2 = knowledge.query("stressato(ID, Causa)")
#print("Dipendenti stressati:", query_2)

# Query per promozioni
#query_3 = knowledge.query("promozione(ID, Istruzione)")
#print("Dipendenti idonei per promozione:", query_3)




def main():
    query = [
        #'buon_equilibrio_vita_lavoro',
        #'suitable_department',
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

    #if query_answ == 0:
        #for index, value in enumerate(eduLevel):
            #print(f'{index}: {value}')
        #eduLevel_answ = None
        #while eduLevel_answ not in eduLevel_range:
            #eduLevel_answ = int(input('Choose an education Level'))

        #for index, value in enumerate(dept):
            #print(f'{index}: {value}')
        #dept_answ = None
        #while dept_answ not in dept_range:
            #dept_answ = int(input('Choose a department:'))

        #print(employee_survey.query(pytholog.Expr(f"{query[query_answ]}({eduLevel[eduLevel_answ]},{dept[dept_answ]})")))

    if query_answ == 0:
        for index, value in enumerate(dept):
            print(f'{index}: {value}')
        dept_answ = None
        while dept_answ not in dept_range:
            dept_answ = int(input('Scegli un dipartimento:'))

        #for index, value in enumerate(eduLevel):
            #print(f'{index}: {value}')
        #eduLevel_answ = None
        #while eduLevel_answ not in eduLevel_range:
            #eduLevel_answ = int(input('Choose an education Level'))

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