import pycutest
import numpy as np 

#probs = pycutest.find_problems(userN=None, m=None, n=[2, 1000])


#prob = pycutest.import_problem("BDEXP", sifParams={'N':100})




dims = list(range(1, 10)) + list(range(10, 200, 10)) +  [97, 200, 300, 400, 500, 997, 1000]#, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]


num_points = 10
rnd_state = np.random.RandomState(1231415)

probs = pycutest.find_problems(userN=True, constraints='unconstrained')
print(probs, len(probs))
with open("cutest_problem_name_new.log", 'a') as f:

    for (i, problem) in enumerate(probs):
        pycutest.print_available_sif_params(problem)
        for n in dims:
            try:
                print(f"[++] Testing for problem = {problem} n = {n} ")
                prob_inst = pycutest.import_problem(problem, sifParams={'N' : n})
#                prob_inst = pycutest.import_problem("HADAMARD", sifParams={'N' : n})
                print(f"\t[++] Imported {problem} => n = {prob_inst.n} ||g|| = {np.linalg.norm(prob_inst.grad(prob_inst.x0)) }\tf(x0) = {prob_inst.obj(prob_inst.x0)}")

                x0 = prob_inst.x0.copy()
                if np.linalg.norm(prob_inst.grad(prob_inst.x0)) == 0.0:
                    for _ in range(10):
                        x0 = rnd_state.rand(prob_inst.n)
                        if np.linalg.norm(prob_inst.grad(prob_inst.x0)) > 0.0:
                            break
#                        print(np.linalg.norm(prob_inst.grad(prob_inst.x0)))

                if 6 <= prob_inst.n <= 1000 and np.linalg.norm(prob_inst.grad(prob_inst.x0)) > 0.0 and prob_inst.obj(x0) < 1e5:
#                    x0 = prob_inst.x0.copy()
                    x = x0
                    print("\t\t[--] Added Problem {}\t[n = {}]".format(problem, prob_inst.n))
                    f.write("{},{},{},".format(problem, n, prob_inst.n) + ','.join([str(elem) for elem in x]) + "\n")
                    f.flush()
#                     for j in range(num_points):
#                         g = prob_inst.grad(x)
#                         if not np.any(np.isnan(x)) and np.linalg.norm(g) > 0.0:
#                             print("\t\t[--] Added Problem {}\t[n = {}, j= {}]".format(problem, prob_inst.n, j))
#                             f.write("{}_{},{},{},".format(problem, j, n, prob_inst.n) + ','.join([str(elem) for elem in x]) + "\n")
#                             f.flush()
                        
# #                        x_near = x + 1e-5
# #                        L = np.linalg.norm(g - prob_inst.grad(x_near)) / np.linalg.norm(x - x_near)
#                         x = x0 + (1/prob_inst.n) * rnd_state.randn(prob_inst.n)
                        
                if hasattr(prob_inst, 'close'):
                    prob_inst.close()

            except Exception:                
                pass
#                print("[--] Problem {} with d = {} has been skipped".format(problem, n))
            

# print(mask)
# with open("cutest_problem_names.log", 'w') as f:
#     for i in range(len(probs)):
#         f.write("{},{}")
# print(problem.n)
#if __name__ == "__main__":
#    list_cutest_problems_without_constraints()
