import pycutest
import numpy as np 

probs = pycutest.find_problems(constraints='unconstrained', userN=True, m=None)


# probs = pycutest.find_problems(constraints='unconstrained', userN=True, m=None)
mask = np.ones(len(probs))
with open("cutest_problem_names.log", 'w') as f:
    f.write("(")
    for (i, problem) in enumerate(probs):
        try:
            prob_inst = pycutest.import_problem(problem)
            
            print("[--] Problem= {}\tn = {}".format(problem, prob_inst.n))
            if hasattr(prob_inst, 'close'):
                prob_inst.close()
        except Exception:
            mask[i] = 0
        if mask[i] and  prob_inst.n <= 5000:
            f.write("\"{}\" ".format(problem))
    f.write(")")
            

# print(mask)
# with open("cutest_problem_names.log", 'w') as f:
#     for i in range(len(probs)):
#         f.write("{},{}")
# print(problem.n)
#if __name__ == "__main__":
#    list_cutest_problems_without_constraints()
