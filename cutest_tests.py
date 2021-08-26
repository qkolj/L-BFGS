import numpy as np
import pycutest
from scipy import optimize
import time
from matplotlib import pyplot as plt

# Originalna implementacija L-BFGS metode predstavljena u prezentaciji

def wolfe_line_search(f, grad, x, p, max_iter=100, c1=10**-4, c2=0.9, alpha_1=1.0, alpha_max=1000):
        
    def phi(alpha):
        return f(x + alpha * p)

    def phi_grad(alpha):
        return np.dot(grad(x + alpha * p).T, p)

    alpha_i_1 = 0
    alpha_i = alpha_1

    for i in range(1, max_iter + 1):
        phi_alpha_i = phi(alpha_i)
        if (phi_alpha_i > phi(0) + c1 * alpha_i * phi_grad(0)) or (i > 1 and phi_alpha_i >= phi(alpha_i_1)):
            return zoom(phi, phi_grad, alpha_i_1, alpha_i, c1, c2)
        
        phi_grad_alpha_i = phi_grad(alpha_i)
        if np.abs(phi_grad_alpha_i) <= -c2 * phi_grad(0):
            return alpha_i
        if phi_grad_alpha_i >= 0:
            return zoom(phi, phi_grad, alpha_i, alpha_i_1, c1, c2)
        alpha_i_1 = alpha_i
        alpha_i = min(2 * alpha_i, alpha_max)

    if i == max_iter:
        return None
    
def zoom(phi, phi_grad, alpha_lo, alpha_hi, c1, c2, max_iter=100):
    i = 0
    while True:
        alpha_j = (alpha_lo + alpha_hi) / 2.0
        phi_alpha_j = phi(alpha_j)

        if (phi_alpha_j > phi(0) + c1 * alpha_j * phi_grad(0)) or (phi_alpha_j >= phi(alpha_lo)):
            alpha_hi = alpha_j
        else:
            phi_grad_alpha_j = phi_grad(alpha_j)
            if np.abs(phi_grad_alpha_j) <= -c2 * phi_grad(0):
                return alpha_j
            if phi_grad_alpha_j * (alpha_hi - alpha_lo) >= 0:
                alpha_hi = alpha_lo
            alpha_lo = alpha_j
        i += 1
        if i >= max_iter:
            return None

def lbfgs(f, grad, x0, eps=1e-4, max_iter=1000, history=10):
    
    n = len(x0)
    
    def two_loop_rec(x, m):
        q = grad(x)
        k = S.shape[0]
        rhos = np.zeros(k)
        alphas = np.zeros(k)
        beta = 0
        for i in range(k - 1, -1, -1):
            rhos[i] = np.dot(Y[i].T, S[i]) ** -1
            alphas[i] = np.dot(S[i].T, q) * rhos[i]
            q = q - alphas[i] * Y[i]
        if k > 0:
            gamma_k = np.dot(S[k - 1].T, Y[k - 1]) / np.dot(Y[k - 1], Y[k - 1])
            H_k0 = np.diag(gamma_k * np.ones(n))
        else:
            H_k0 = np.diag(np.ones(n))
        r = np.dot(H_k0, q)
        for i in range(k):
            beta = rhos[i] * np.dot(Y[i].T, r)
            r = r + S[i] * (alphas[i] - beta)
        
        return r
    
    S = np.empty([0, n])
    Y = np.empty([0, n])
    x_old = x0
    
    for k in range(1, max_iter + 1):
        p_k = -two_loop_rec(x_old, history)
        alpha_k = wolfe_line_search(f, grad, x_old, p_k)
        
        if alpha_k is None:
            print("Wolfe line search did not converge")
            return x_old, k
        
        x_new = x_old + alpha_k * p_k
        
        grad_diff = grad(x_new) - grad(x_old)
        if np.linalg.norm(grad_diff) < eps:
            break
        
        if k > history:
            S = S[1:]
            Y = Y[1:]
        S = np.append(S, [x_new - x_old], axis=0)
        Y = np.append(Y, [grad_diff], axis=0)
        x_old = x_new
        
    if k == max_iter:
        print("Optimization did not converge")
    else:
        print("Optimization converged in {} steps".format(k))
        
    return x_new, k

# Ucitavamo probleme sa odgovarajucim parametrima

problem_names = ['BOX', 'DIXMAANL', 'EIGENALS', 'FREUROTH', 'TRIDIA', 'VAREIGVL']
problem_params = [{'N': 10000}, {'M': 500}, {'N': 10}, {'N': 1000}, {'N': 1000}, {'N': 4999, 'M': 6}]
number_of_problems = len(problem_names)
problems = {}
results = {}

for i in range(number_of_problems):
    problems[problem_names[i]] = pycutest.import_problem(problem_names[i], sifParams=problem_params[i])
    
for name in problem_names:
    # Ispisujemo imena problema sa brojem promenljivih
    print(problems[name])
    results[name] = []

for name in problem_names:
    # Posto bibliotecke funkcije za optimizaciju zahtevaju da im se ciljna funkcija,
    # njen gradijent i hesijan nezavisno proslede, prilagodjavamo API pycutest-a
    # tim zahtevima.
    def prob_obj(x):
        return problems[name].obj(x, gradient=False)
    
    def prob_obj_grad(x):
        return problems[name].obj(x, gradient=True)[1]
    
    def prob_obj_hess(x):
        return problems[name].hess(x)
    
    # Svaku optimizaciju zapocinjemo od zadate pocetne tacke u okviru tog problema.
    # Koristimo promenljive start_time i end_time da izmerimo vreme izvrsavanja
    # svake od metoda optimizacije.
    start_time = time.time()
    result = lbfgs(prob_obj, prob_obj_grad, problems[name].x0, eps=1e-5, max_iter=500, history=100)
    end_time = time.time()
    results[name].append({'x': result[0], 'num_iters': result[1], 'time': end_time - start_time})
    
    start_time = time.time()
    result = optimize.minimize(prob_obj, problems[name].x0, method='L-BFGS-B', jac=prob_obj_grad, options={'maxcor': 100, 'gtol': 1e-5, 'maxiter': 500})
    end_time = time.time()
    results[name].append({'x': result['x'], 'num_iters': result['nit'], 'time': end_time - start_time})
    
    start_time = time.time()
    result = optimize.minimize(prob_obj, problems[name].x0, method='Newton-CG', jac=prob_obj_grad, hess=prob_obj_hess, options={'maxiter': 500})
    end_time = time.time()
    results[name].append({'x': result['x'], 'num_iters': result['nit'], 'time': end_time - start_time})

    
lbfgs_times = []
scipy_lbfgs_times = []
scipy_newton_times = []

for name in problem_names:
    lbfgs_times.append(results[name][0]['time'])
    scipy_lbfgs_times.append(results[name][1]['time'])
    scipy_newton_times.append(results[name][2]['time'])

x_iters = range(number_of_problems)

plt.bar(x_iters, lbfgs_times)
for i, t in enumerate(lbfgs_times):
    plt.text(x_iters[i], t + 0.1, "{:.2f}".format(t), ha='center')
plt.xticks(x_iters, problem_names)
plt.xlabel("Назив проблема")
plt.ylabel("Време извршавања у секундама")
plt.title("L-BFGS - време извршавања")
plt.savefig('figures/lbfgs.png')

plt.clf()
plt.bar(x_iters, scipy_lbfgs_times)
for i, t in enumerate(scipy_lbfgs_times):
    plt.text(x_iters[i], t + 0.01, "{:.2f}".format(t), ha='center')
plt.xticks(x_iters, problem_names)
plt.xlabel("Назив проблема")
plt.ylabel("Време извршавања у секундама")
plt.title("scipy L-BFGS-B - време извршавања")
plt.savefig('figures/scipy-lbfgs.png')

plt.clf()
plt.bar(x_iters, scipy_newton_times)
for i, t in enumerate(scipy_newton_times):
    plt.text(x_iters[i], t + 1.0, "{:.2f}".format(t), ha='center')
plt.xticks(x_iters, problem_names)
plt.xlabel("Назив проблема")
plt.ylabel("Време извршавања у секундама")
plt.title("scipy Newton-CG - време извршавања")
plt.savefig('figures/scipy-newton.png')
