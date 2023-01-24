from eszd.directions import HouseholderReflect

P = HouseholderReflect(d=5, l=3)

A = P()

print(A)

print(A.T @ A)