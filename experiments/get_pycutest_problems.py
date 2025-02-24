import pycutest
import numpy as np 

#probs = pycutest.find_problems(userN=None, m=None, n=[2, 1000])


#prob = pycutest.import_problem("BDEXP", sifParams={'N':100})

dims = list(range(2, 6)) + list(range(10, 60, 10)) +  [100, 200, 500, 1000]

probs = pycutest.find_problems(userN=None, m=None, n=[2, 1000])
with open("cutest_problem_names.log", 'w') as f:
    for (i, problem) in enumerate(probs):
        for n in dims:
            try:
                prob_inst = pycutest.import_problem(problem, sifParams={'N' : n})
                
                pycutest.print_available_sif_params(problem)
                if prob_inst.n <= 1000:
                    print("[--] Added Problem {}\t[n = {}]".format(problem, prob_inst.n))
                    f.write("{},{},{}\n".format(problem, n, prob_inst.n))
                    f.flush()
                if hasattr(prob_inst, 'close'):
                    prob_inst.close()

            except Exception:
                print("[--] Problem {} with d = {} has been skipped".format(problem, n))
            

# print(mask)
# with open("cutest_problem_names.log", 'w') as f:
#     for i in range(len(probs)):
#         f.write("{},{}")
# print(problem.n)
#if __name__ == "__main__":
#    list_cutest_problems_without_constraints()
