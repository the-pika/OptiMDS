"""
OptiMDS: Optimizing Medical Discharge Summaries to Improve Patient Communication

Deepika Verma, "Optimizing Medical Discharge Summaries to Improve Patient Communication", (under review)
@author: Deepika Verma
"""



from jmetal.algorithm.multiobjective import NSGAII
from jmetal.operator import SBXCrossover, PolynomialMutation
from jmetal.util.termination_criterion import StoppingByEvaluations
from readability import Readability
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution

import pandas as pd
import gensim
import gensim.downloader
import jdk
import pygad
import language_tool_python
import numpy as np
import matplotlib.pyplot as plt

google_news_vectors = gensim.downloader.load('word2vec-google-news-300')


# Load the dataset of terms and definitions
csv_file = 'final_layman_datset_for_GA.csv'  # CSV file path
df = pd.read_csv(csv_file)

# Create a dictionary for easy lookup
term_to_definition = pd.Series(df.Definition.values, index=df.Term.str.lower()).to_dict()


text_array = []
index_array = []


text = '''

Mr. Abc was admitted to the hospital with a history of AAA repair complicated by MI, hypertension, and hyperlipidemia, who presented with nasal fractures and epistaxis after a mechanical fall. He was found to have a nasal bone and septal fracture on a CT scan and had Rhinorockets placed to control the bleeding. During his hospital course, he experienced an episode of nausea, coughed up blood, and became hypotensive and bradycardic, leading to a brief loss of consciousness. He was diagnosed with a focal consolidation at the left lung base, possibly representing aspiration or developing pneumonia, and was given antibiotics. He was also found to have mild regional left ventricular systolic dysfunction, mild aortic valve stenosis, and mild aortic regurgitation on an echocardiogram. His metoprolol was uptitrated, pravastatin was converted to atorvastatin, clopidogrel was maintained, and he was started on aspirin. He was discharged with a stable creatinine and a diagnosis of nasal fracture, epistaxis, and NSTEMI. He was advised to follow up with his PCP to discuss a stress test. He was also instructed to use oxymetolazone nasal spray and hold pressure if bleeding reoccurs. He was discharged home with service and is expected to be ambulatory and independent.

'''

r = Readability(text)
initial_score = r.ari().score

#Creates a dictionary in order to store all the synonyms in main memory
resource = text.split() 
Dict = {}
for i in resource:
    if ',' in i:
        i = i.replace(',', '')
    if '.' in i:
        i = i.replace('.', '') 

    if (not i[0].isupper() and len(i) > 3):
        if i in Dict.keys():
            print ("Processing...Please wait")
        else:
            try:
                synonyms = google_news_vectors.most_similar(i, topn=6)
            except KeyError as e:
                print (e)
                synonyms = None
            if synonyms is not None:
                Dict[i] = []
                Dict[i] = synonyms


def listToString(s):
    str1 = ""
    for ele in s:
        str1 += str(ele)
        str1 += " "
    
    str1 = str1.replace(' ,', ',')
    str1 = str1.replace('_', ' ')
    return str1


def correct_mistakes (text):
    my_tool = language_tool_python.LanguageTool('en-US')  
    my_text = text  
    my_matches = my_tool.check(my_text)  
  
    myMistakes = []  
    myCorrections = []  
    startPositions = []  
    endPositions = []  
  
    # using the for-loop  
    for rules in my_matches:  
        if len(rules.replacements) > 0:  
            startPositions.append(rules.offset)  
            endPositions.append(rules.errorLength + rules.offset)  
            myMistakes.append(my_text[rules.offset : rules.errorLength + rules.offset])  
            myCorrections.append(rules.replacements[0])  
  
    # creating new object  
    my_NewText = list(my_text)   
  
    # rewriting the correct passage  
    for n in range(len(startPositions)):  
        for i in range(len(my_text)):  
            my_NewText[startPositions[n]] = myCorrections[n]  
            if (i > startPositions[n] and i < endPositions[n]):  
                my_NewText[i] = ""  
  
    my_NewText = "".join(my_NewText)  
    
    return my_NewText 


# function to get synonyms using word2vec
def Synonym(word, number):
    synonyms = []
    
    if (Dict.get(word) is not None):
        synonyms = Dict.get(word)
             
    if (not synonyms):
        return -2, word
    elif number >= len(synonyms):
        return len(synonyms)-1, synonyms[len(synonyms)-1][0]
    else:
        return int(number), synonyms[int(number-1)][0]



# Function to replace terms with definitions from the dataset or word2vec
def replace_with_definition(word, number):
    word_lower = word.lower()
    
    if word_lower in term_to_definition:
        return number, term_to_definition[word_lower]  # Return definition from dataset
    
    # fallback to word2vec synonyms
    return Synonym(word, number)
    

# Process the text with the updated synonym replacement logic
def obtain_text(solution):
    res2 = text.split()
    text_converted = []
    index = 0

    # Ensure 'solution' is a list, not an object
    solution_values = solution.variables if hasattr(solution, 'variables') else solution

    for i in res2:
        if solution_values[index] < 1:
            text_converted.append(i)
        elif solution_values[index] >= 1:
            number, word = replace_with_definition(i, solution_values[index])
            text_converted.append(word.upper())
        else:
            print('Error')
        index += 1

    result = listToString(text_converted)
    return result

"""
Some parts of the code are derived from: 
ORUGA: Optimizing Readability Using Genetic Algorithms
[Martinez-Gil2023a] J. Martinez-Gil, "Optimizing Readability Using Genetic Algorithms", arXiv preprint arXiv:2301.00374, 2023
@author: Jorge Martinez-Gil
"""

# Update the fitness function to use the new synonym replacement logic
def fitness_func1(solution):
    print(f"Solution length: {len(solution)}, Index array length: {len(index_array)}")
    print(f"Solution: {solution}, Index Array: {index_array}")

    # Preprocessing
    a = 0
    for i in index_array:
        if index_array[a] <= 0:
            solution[a] = 0
        a += 1

    res2 = text.split()
    text_converted = []
    index = 0
    for i in res2:
        if solution[index] < 1:
            text_converted.append(i)
        elif solution[index] >= 1:
            number, word = replace_with_definition(i, solution[index])
            text_converted.append(word.upper())
        else:
            print('Error')
        index += 1

    result = listToString(text_converted)
    r = Readability(result)
    return r.ari().score

print(text)
res = text.split()


for i in res:
    flag = 0
    if ',' in i:
        i = i.replace(',', '')
        flag = 1
    if '.' in i:
        i = i.replace('.', '')
        flag = 2   
        
    if (not i[0].isupper() and len(i) > 3 and i[-2:] != 'ed'):
        number, word = replace_with_definition(i,6)
        text_array.append (word)
        index_array.append (number)
    else:
        text_array.append (i)
        index_array.append (0)
        
    if flag == 1:
        cad = str(text_array[-1])
        text_array.pop()
        cad = cad + str(',')
        text_array.append (cad)
        flag = 0
    if flag == 2:
        cad = str(text_array[-1])
        text_array.pop()
        cad = cad + str('.')
        text_array.append (cad)
        flag = 0


newText = listToString(text_array)
print(newText)
print(index_array)


class OptiMDS(FloatProblem):

    def __init__(self):
        super(OptiMDS, self).__init__()
        self.number_of_objectives = 2
        self.number_of_variables = len(index_array)
        self.number_of_constraints = 0

        self.obj_directions = [self.MINIMIZE, self.MAXIMIZE]
        self.obj_labels = ['f(x)', 'f(y)']

        self.lower_bound = self.number_of_variables * [-4]
        self.upper_bound = self.number_of_variables * [4]

        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        solution.objectives[1] = fitness_func1(solution.variables)
        solution.objectives[0] = len([1 for i in solution.variables if i >= 1])

        return solution


    def get_name(self):
        return 'OptiMDS'


from jmetal.util.solution import get_non_dominated_solutions, print_function_values_to_file, print_variables_to_file
from jmetal.lab.visualization import Plot

import os
import matplotlib.pyplot as plt

# Create a folder to store Pareto front images
output_folder = "pareto_fronts"
os.makedirs(output_folder, exist_ok=True)

# Create a list to store function values
data_records = []

# Run the optimization 30 times
num_runs = 30


problem = OptiMDS()

algorithm = NSGAII(
    problem=problem,
    population_size=20,
    offspring_population_size=30,
    mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
    crossover=SBXCrossover(probability=1.0, distribution_index=20),
    termination_criterion=StoppingByEvaluations(max_evaluations=800)
)


for run in range(1, num_runs + 1):
    print(f"Run {run}/{num_runs}")
    algorithm.run()
    front = get_non_dominated_solutions(algorithm.get_result())
    
    # Store function values in the list
    for solution in front:
        data_records.append([run] + solution.objectives)
    
    # Save Pareto front scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter([s.objectives[0] for s in front], [s.objectives[1] for s in front], color='blue', s=100)
    plt.xlabel('Words to be replaced', fontsize=16)
    plt.ylabel('Readability Score', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(False)
    plot_path = os.path.join(output_folder, f'pareto_front_run_{run}.png')
    plt.savefig(plot_path)
    plt.close()

# Save function values to CSV
csv_filename = "nsga2_word2vec_results.csv"
df_results = pd.DataFrame(data_records, columns=["Run", "Words Replaced", "Readability Score"])
df_results.to_csv(csv_filename, index=False)

print(f"Optimization completed for {num_runs} runs.")
print(f"Function values saved to {csv_filename}")
print(f"Pareto front images saved in {output_folder}/ folder.")

