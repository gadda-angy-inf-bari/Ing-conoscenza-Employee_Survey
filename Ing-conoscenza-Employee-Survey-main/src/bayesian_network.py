import bnlearn
import networkx as nx
import matplotlib.pyplot as plt
from dataset import load_data, convert_to_ranges
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


class IBMBayesianNetwork:
    def __init__(self):
        self.df = convert_to_ranges(load_data())
        self.edges = None
        self.model = self.create_edges()

        # Crea un modello Bayesian Network con gli edges trovati
        self.model = bnlearn.make_DAG(self.edges, verbose=1)
        self.plot_model_graph()

        # Score-based structure learning with hillclimb search
        print('[Structure learning with hillclimb search, BIC score...]')
        self.bic_model = bnlearn.structure_learning.fit(self.df, methodtype='hc', scoretype='bic', verbose=1)
        print('[Structure learning with hillclimb search, K2 score...]')
        self.k2_model = bnlearn.structure_learning.fit(self.df, methodtype='hc', scoretype='k2', verbose=1)
        print('[Structure learning with hillclimb search, BDeu score...]')
        self.bdeu_model = bnlearn.structure_learning.fit(self.df, methodtype='hc', scoretype='bdeu', verbose=1)
        
        # Comparison
        self.scores, self.adjmat_diff = bnlearn.compare_networks(self.bic_model, self.model)
        print('[Comparing your structure with BIC model...]')
        print('Accuracy ' + str(self.accuracy()))
        print('Precision ' + str(self.precision()))
        print('Recall ' + str(self.recall()))
        print('F-Score ' + str(self.f_score()))
        print('Error ' + str(self.error()))
        self.scores, self.adjmat_diff = bnlearn.compare_networks(self.k2_model, self.model)
        print('[Comparing your structure with K2 model...]')
        print('Accuracy ' + str(self.accuracy()))
        print('Precision ' + str(self.precision()))
        print('Recall ' + str(self.recall()))
        print('F-Score ' + str(self.f_score()))
        print('Error ' + str(self.error()))
        self.scores, self.adjmat_diff = bnlearn.compare_networks(self.bdeu_model, self.model)
        print('[Comparing your structure with BDeu model...]')
        print('Accuracy ' + str(self.accuracy()))
        print('Precision ' + str(self.precision()))
        print('Recall ' + str(self.recall()))
        print('F-Score ' + str(self.f_score()))
        print('Error ' + str(self.error()))

        self.bn = bnlearn.parameter_learning.fit(self.model, self.df, methodtype='maximumlikelihood', verbose=1)


    def create_edges(self):
        self.edges = [
                 ('Gender', 'WLB'),
                 ('Gender', 'WorkEnv'),
                      
                 ('JobLevel', 'Dept'),
                 ('Experience', 'Dept'),
                 ('EduLevel', 'Dept'),
                 
                 ('MaritalStatus', 'EmpType'),
                 ('MaritalStatus', 'haveOT'),

                 ('WLB', 'Stress'),
                 ('WorkEnv', 'Stress'),
                 ('Workload', 'Stress'),
                 
                 ('Age', 'JobLevel'),
                 ('Age', 'Experience'),
                 ('Age', 'WLB'),
                 ('Age', 'NumCompanies'),
                 ('Age', 'NumReports'),
                 ('Age', 'EduLevel'),

                 ('CommuteMode', 'PhysicalActivityHours'),
                 ('CommuteMode', 'Stress'),
                 ('CommuteMode', 'CommuteDistance'),
                 
                 ('Stress', 'SleepHours'),

                 ('JobLevel', 'TeamSize'),
                 ('Workload', 'TeamSize'),
                 ('Dept', 'TeamSize'),
                 
                 ('TrainingHoursPerYear', 'JobLevel'),
                 ('TrainingHoursPerYear', 'Dept'),

                 ('TeamSize', 'NumReports'),
                 ('WorkEnv', 'NumReports'),
                 
                 ('WLB', 'JobSatisfaction'),
                 ('WorkEnv', 'JobSatisfaction'),
                 ('Dept', 'JobSatisfaction'),
                 ('EmpType', 'JobSatisfaction'),
                 ('Workload', 'JobSatisfaction'),
                 ('TeamSize', 'JobSatisfaction'),
                 ('TrainingHoursPerYear', 'JobSatisfaction'),]
        return bnlearn.make_DAG(self.edges, verbose=1)

    def plot_model_graph(self):
        # Crea il grafo con NetworkX
        G = nx.DiGraph()  # Grafo diretto (DAG)
        G.add_edges_from(self.edges)  # Aggiungi gli edges

        # Disegna il grafo con matplotlib
        plt.figure(figsize=(12, 10))  # Imposta la dimensione della figura
        pos = nx.spring_layout(G, seed=42)  # Imposta la disposizione dei nodi
        nx.draw(G, pos, with_labels=True, node_size=3000, node_color='skyblue', font_size=10, font_weight='bold', edge_color='gray')
        plt.title("Bayesian Network Graph", fontsize=15)
        plt.show()

    def error(self):
        return (self.scores[0,1]+self.scores[1,0])/(self.scores[0,0]+self.scores[0,1]+self.scores[1,0]+self.scores[1,1])

    def accuracy(self):
        return 1-self.error()

    def precision(self):
        return self.scores[0,0] / (self.scores[0,0] + self.scores[0,1])

    def recall(self):
        return self.scores[0,0] / (self.scores[0,0] + self.scores[1,1])

    def f_score(self):
        return (2 * self.recall() * self.precision()) / (self.recall() + self.precision())

    def query(self, variable, evidence):
        query = bnlearn.inference.fit(self.bn, variables=variable, evidence=evidence, verbose=1)
        return query.df.to_dict(orient="records")


def main():
    bn = IBMBayesianNetwork()

    variable = [
        'Gender', 'Age', 'MaritalStatus', 'JobLevel', 'Experience',
       'Dept', 'EmpType', 'WLB', 'WorkEnv', 'PhysicalActivityHours',
       'Workload', 'Stress', 'SleepHours', 'CommuteMode', 'CommuteDistance',
       'NumCompanies', 'TeamSize', 'NumReports', 'EduLevel', 'haveOT',
       'TrainingHoursPerYear', 'JobSatisfaction'
    ]
    variable_range = range(0, len(variable))


    gender = [
        'Male', 
        'Female', 
        'Other'
    ] 
    gender_range = range(0, len(gender))
    
    age = [
        1,
        2,
        3,
        4
    ] 
    age_range = range(0, len(age))
    
    maritalStatus = [
        'Single', 
        'Married', 
        'Divorced', 
        'Widowed'
    ] 
    maritalStatus_range = range(0, len(maritalStatus))

    jobLevel = [
        'Senior',
        'Mid',
        'Junior',
        'Lead',
        'Intern/Fresher'        
    ] 
    jobLevel_range = range(0, len(jobLevel))

        
    experience = [
        1,
        2,
        3,
        4
    ] 
    experience_range = range(0, len(experience))

    dept = [
        'IT', 
        'Finance', 
        'Operations', 
        'Marketing', 
        'Sales', 
        'Customer Service', 
        'Legal', 
        'HR'
    ] 
    dept_range = range(0, len(dept))
    
    empType = [
        'Full-Time',
        'Part-Time',
        'Contract'  
    ] 
    empType_range = range(0, len(empType))
    
    wlb = [
        1,
        2,
        3,
        4,
        5
    ] 
    wlb_range = range(0, len(wlb))

    workEnv = [
        1,
        2,
        3,
        4,
        5
    ] 
    workEnv_range = range(0, len(workEnv))

        
    physicalActivityHours = [
        1,
        2,
        3,
        4,
        5
    ] 
    physicalActivityHours_range = range(0, len(physicalActivityHours))

    workload = [
        1,
        2,
        3,
        4,
        5
    ] 
    workload_range = range(0, len(workload))
    
    stress = [
        1,
        2,
        3,
        4,
        5
    ] 
    stress_range = range(0, len(stress))
    
    sleepHours = [
        1,
        2,
        3,
        4,
        5,
        6
    ] 
    sleepHours_range = range(0, len(sleepHours))

    commuteMode = [
        'Car', 
        'Public Transport', 
        'Bike', 
        'Motorbike',        
        'Walk'
    ] 
    commuteMode_range = range(0, len(commuteMode))

    commuteDistance = [
        1,
        2,
        3,
        4
    ] 
    commuteDistance_range = range(0, len(commuteDistance))

    numCompanies = [
        1,
        2,
        3,
        4
    ] 
    numCompanies_range = range(0, len(numCompanies))

    teamSize = [
        1,
        2,
        3,
        4,
        5
    ] 
    teamSize_range = range(0, len(teamSize))

    numReports = [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9
    ] 
    numReports_range = range(0, len(numReports))

    eduLevel = [
        'Car', 
        'Public', 
        'Transport', 
        'Bike', 
        'Walk', 
        'Motorbike'
    ] 
    eduLevel_range = range(0, len(eduLevel))

    haveOT = [
        'TRUE', 
        'FALSE'
    ] 
    haveOT_range = range(0, len(haveOT))

    trainingHoursPerYear = [
        1,
        2,
        3,
        4,
        5
    ] 
    trainingHoursPerYear_range = range(0, len(trainingHoursPerYear))

    jobSatisfaction = [
        1,
        2,
        3,
        4,
        5
    ] 
    jobSatisfaction_range = range(0, len(jobSatisfaction))

    print('[Prediction]')
    for index, value in enumerate(variable):
        print(f'{index}: {value}')

    already_chosen = []
    variable_answ = None
    while variable_answ not in variable_range:
        variable_answ = int(input('Choose a variable to predict:'))
    already_chosen.append(variable_answ)

    stop = False
    evidences = { }
    while not stop:
        print('------------------------------------------------------------------')
        for index, value in enumerate(variable):
            print(f'{index}: {value}')
        evidence_answ = None
        print('You already chose ' + str(already_chosen))
        while evidence_answ not in variable_range:
                evidence_answ = int(input('Choose an evidence:'))
        already_chosen.append(evidence_answ)

        # Gender
        if evidence_answ == 0:
            for index, value in enumerate(gender):
                print(f'{index}: {value}')
            answ = None
            while answ not in gender_range:
                answ = int(input('Choose a gender:'))
            evidences['Gender'] = gender[answ]

        # Age
        if evidence_answ == 1:
            print('1: [22, 31] \n2: [32, 41] \n3: [42, 51] \n4: [52, 60]')
            answ = None
            while answ not in age_range:
                answ = int(input('Choose a age range:'))
            evidences['Age'] = age[answ]

        # MaritalStatus
        if evidence_answ == 2:
            for index, value in enumerate(maritalStatus):
                print(f'{index}: {value}')
            answ = None
            while answ not in maritalStatus_range:
                answ = int(input('Choose a marital status:'))
            evidences['MaritalStatus'] = maritalStatus[answ]

        # JobLevel
        if evidence_answ == 3:
            for index, value in enumerate(jobLevel):
                print(f'{index}: {value}')
            answ = None
            while answ not in jobLevel_range:
                answ = int(input('Choose a job level:'))
            evidences['JobLevel'] = jobLevel[answ]
            
        # Experience
        if evidence_answ == 4:
            print('1: [0, 7] \n2: [8, 15] \n3: [16, 23] \n4: [24, 29]')
            answ = None
            while answ not in experience_range:
                answ = int(input('Choose a work experience range:'))
            evidences['Experience'] = experience[answ]

        # Dept
        if evidence_answ == 5:
            for index, value in enumerate(dept):
                print(f'{index}: {value}')
            answ = None
            while answ not in dept_range:
                answ = int(input('Choose a dept:'))
            evidences['Dept'] = dept[answ]

        # EmpType
        if evidence_answ == 6:
            for index, value in enumerate(empType):
                print(f'{index}: {value}')
            answ = None
            while answ not in empType_range:
                answ = int(input('Choose a type of employment:'))
            evidences['empType'] = empType[answ]

        # WLB
        if evidence_answ == 7:
            print('1: Very low \n2: Low \n3: Medium \n4: High \n5: Very high')
            answ = None
            while answ not in wlb_range:
                answ = int(input('Choose a work-life balance rating:'))
            evidences['WLB'] = wlb[answ]

        # WorkEnv
        if evidence_answ == 8:
            print('1: Very low \n2: Low \n3: Medium \n4: High \n5: Very high')
            answ = None
            while answ not in workEnv_range:
                answ = int(input('Choose a work environment rating:'))
            evidences['WorkEnv'] = workEnv[answ]

        # PhysicalActivityHours
        if evidence_answ == 9:
            print('1: [0, 0.9] \n2: [1, 1.9] \n3: [2, 2.9] \n4: [3, 3.9] \n5: [4, 5]')
            answ = None
            while answ not in physicalActivityHours_range:
                answ = int(input('Choose a number of hours of physical activity per week:'))
            evidences['PhysicalActivityHours'] = physicalActivityHours[answ]

        # Workload
        if evidence_answ == 10:
            print('1: Very low \n2: Low \n3: Medium \n4: High \n5: Very high')
            answ = None
            while answ not in workload_range:
                answ = int(input('Choose a workload rating:'))
            evidences['Workload'] = workload[answ]

        # Stress
        if evidence_answ == 11:
            print('1: Very low \n2: Low \n3: Medium \n4: High \n5: Very high')
            answ = None
            while answ not in stress_range:
                answ = int(input('Choose a stress level rating:'))
            evidences['Stress'] = stress[answ]

        # SleepHours
        if evidence_answ == 12:
            print('1: [4, 4.9] \n2: [5, 5.9] \n3: [6, 6.9] \n4: [7, 7.9] \n5: [8, 8.9] \n6: [9, 10]')
            answ = None
            while answ not in sleepHours_range:
                answ = int(input('Choose a number of hours of sleep per night:'))
            evidences['SleepHours'] = sleepHours[answ]

        # CommuteMode
        if evidence_answ == 13:
            for index, value in enumerate(commuteMode):
                print(f'{index}: {value}')
            answ = None
            while answ not in commuteMode_range:
                answ = int(input('Choose a mode of commute:'))
            evidences['commuteMode'] = commuteMode[answ]

        # CommuteDistance
        if evidence_answ == 14:
            print('1: [1, 7] \n2: [8, 14] \n3: [15, 21] \n4: [22, 29]')
            answ = None
            while answ not in commuteDistance_range:
                answ = int(input('Choose a distance traveled during the commute:'))
            evidences['CommuteDistance'] = commuteDistance[answ]

        # NumCompanies
        if evidence_answ == 15:
            print('1: [0, 2] \n2: [3, 5] \n3: [6, 8] \n4: [9, 12]')
            answ = None
            while answ not in numCompanies_range:
                answ = int(input('Choose a number of different companies he has worked for:'))
            evidences['NumCompanies'] = numCompanies[answ]

        # TeamSize
        if evidence_answ == 16:
            print('1: [5, 9] \n2: [10, 14] \n3: [15, 19] \n4: [20, 24] \n5: [25, 30]')
            answ = None
            while answ not in teamSize_range:
                answ = int(input('Choose a size of the team:'))
            evidences['TeamSize'] = teamSize[answ]

        # NumReports
        if evidence_answ == 17:
            for index, value in enumerate(numReports):
                print(f'{index}: {value}')
            answ = None
            while answ not in numReports_range:
                answ = int(input('Choose a number of people reported:'))
            evidences['NumReports'] = numReports[answ]

        # EduLevel
        if evidence_answ == 18:
            for index, value in enumerate(eduLevel):
                print(f'{index}: {value}')
            answ = None
            while answ not in eduLevel_range:
                answ = int(input('Choose a highest level of education:'))
            evidences['EduLevel'] = eduLevel[answ]

        # haveOT
        if evidence_answ == 19:
            for index, value in enumerate(haveOT):
                print(f'{index}: {value}')
            answ = None
            while answ not in haveOT_range:
                answ = int(input('Choose a indicator if the employee has overtime:'))
            evidences['haveOT'] = haveOT[answ]

        # TrainingHoursPerYear
        if evidence_answ == 20:
            print('1: [10, 20] \n2: [21, 31] \n3: [32, 42] \n4: [43, 53] \n5: [54, 65]')
            answ = None
            while answ not in trainingHoursPerYear_range:
                answ = int(input('Choose a hours of training received per year:'))
            evidences['TrainingHoursPerYear'] = trainingHoursPerYear[answ]

        # JobSatisfaction
        if evidence_answ == 21:
            print('1: Very low \n2: Low \n3: Medium \n4: High \n5: Very high')
            answ = None
            while answ not in jobSatisfaction_range:
                answ = int(input('Choose a rating of job satisfaction:'))
            evidences['JobSatisfaction'] = jobSatisfaction[answ]

        flag = input('Done choosing evidence(s)? y-n')
        if flag == 'y':
            stop = True
        else:
            stop = False

    print('Chosen evidence(s): \n' + str(evidences))
    query = bn.query(variable=[variable[variable_answ]], evidence=evidences)
    print(query)


if __name__ == "__main__":
    main()