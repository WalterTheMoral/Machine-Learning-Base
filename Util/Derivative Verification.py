import numpy as np

def check_grad(f, x, grad, epsilon=1e-7):
    approximation = abs( (f(x + epsilon) - f(x - epsilon)) / (2 * epsilon) )
    derivative = abs( grad(x) )

    difference = abs(approximation - derivative) / (approximation + derivative)
    return difference < epsilon, difference

def f(x):
    return 2*(x**2)+4*x+3
def grad_f(x):
    return 4*x+4
def wrong_grad_f(x):
    return 4*x+3.999

check, diff = check_grad(f, 5.0, grad_f)
print("check:",str(check), ", diff:", str(diff))
check, diff = check_grad(f, 5.0, wrong_grad_f)
print("check:",str(check), ", diff:", str(diff))

print("\n" + "*"*80 + "\n")

def check_n_grad(f, params_vec: np.ndarray, grad_vec: np.ndarray, epsilon: float = 1e-7) -> (bool, float):
    n = params_vec.shape[0]
    approx = np.zeros(n)

    for i in range(n):
        plus_params = params_vec.copy()
        plus_params[i] += epsilon
        f_plus = f(plus_params)

        minus_params = params_vec.copy()
        minus_params[i] -= epsilon
        f_minus = f(minus_params)

        approx[i] = (f_plus - f_minus) / (2*epsilon)

    diff = np.linalg.norm(grad_vec - approx) / ((np.linalg.norm(grad_vec)) + np.linalg.norm(approx))
    return diff < epsilon, diff

def g(parms):
    a,b = parms[0], parms[1]
    return 2*a**2+4*a*b-3*b**2

def dg_da(a,b):
    return 4*a+4*b
def dg_db(a,b):
    return 4*a-6*b
def dg_db_wrong(a,b):
    return 4*a-6*b+0.001
a,b = 5.0,1.0
check, diff = check_n_grad(g, np.array([a,b]), np.array([dg_da(a,b),dg_db(a,b)]))
print("check:",str(check), ", diff:", str(diff))
check, diff = check_n_grad(g, np.array([a,b]), np.array([dg_da(a,b),dg_db_wrong(a,b)]))
print("check:",str(check), ", diff:", str(diff))
