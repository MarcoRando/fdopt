import pycutest
import numpy as np 

#probs = pycutest.find_problems(userN=None, m=None, n=[2, 1000])


#prob = pycutest.import_problem("BDEXP", sifParams={'N':100})


unique_names = []

with open("cutest_problem_names.log", 'r') as f:

    for line in f.readlines():
        unique_names.append(line.split(",")[0].split("_")[0])
        
print(len(set(unique_names)))
