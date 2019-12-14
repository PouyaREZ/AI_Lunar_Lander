import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

# define constant from environment
FPS    = 50
SCALE  = 30.0   # affects how fast-paced the game is, forces should be adjusted as well
LEG_DOWN = 18
VIEWPORT_W = 600
VIEWPORT_H = 400

h = 1 / FPS
g = 10.
m = 4.816666603088379 + 0.07111112028360367
Fmax = 650.
alpha = 0.0
gamma = 20.

def oracle(p0, v0, pk, vk):
    '''Oracle for LunarLander-v2 framing as convex optimization'''
    # define variable
    L = 0
    U = 1000
    eps = 1.5

    # quasi-convex optimization (bisection of feasibility problem)
    while U - L >= eps:
        # set value of K
        K = (L + U) // 2
        
        # solve feasibility problem
        p_k, f_k, prob = solve_cvx(p0, v0, pk, vk, K)

        # print iteration
        # print('accuracy', U - L)
        # print('status:',  prob.status)
        # print('')

        # recursivly find the solution
        if (prob.status == cp.OPTIMAL):
            U = K
        else:
            L = K

    # solve for solution with upper bound (always feasible)
    K = U
    p_k, f_k, prob = solve_cvx(p0, v0, pk, vk, K)

    # print result
    print('status:', prob.status)
    print('minimal time', U)

    return p_k, f_k

def solve_cvx(p0, v0, pk, vk, K):
    '''Solve convex optimization problem for trajectory and thruster direction'''
    # create problem variables
    f_k = cp.Variable((K, 2))
    v_k = cp.Variable((K + 1, 2))
    p_k = cp.Variable((K + 1, 2))

    # create problem constrains
    constraints = [p_k[0, :] == p0,
                   v_k[0, :] == v0,
                   p_k[-1, :] == pk,
                   v_k[-1, :] == vk]
    e2 = np.zeros(2)
    e2[1] = 1.0
    for i in range(K):
        constraints.append(v_k[i + 1, :] == v_k[i, :] 
                            + (h / m) * f_k[i, :] - h * g * e2)
        constraints.append(p_k[i + 1, :] == p_k[i, :] 
                            + (h / 2) * (v_k[i, :] + v_k[i + 1, :]))
        constraints.append(p_k[i, 1] >= alpha * cp.abs(p_k[i, 0]))
        constraints.append(cp.norm(f_k[i, :]) <= Fmax)

    # form objective
    # minimum fuel descent
    # obj = cp.Minimize(gamma * h * cp.sum(cp.norm(f_k, axis=1)))
    # minimum time descent
    obj = cp.Minimize(0)

    # form and solve the problem
    prob = cp.Problem(obj, constraints)
    prob.solve()

    # print result
    # print("status:", prob.status)
    # print("optimal value", prob.value)
    # print("")

    return p_k.T.value, f_k.T.value, prob

def plotting(p_k):
    '''Ploting of trajectory'''
    plt.plot(p_k[0, :], p_k[1, :], color='r')
    plt.xlim([0, 20])
    plt.ylim([0, 15])
    plt.grid()
    plt.xlabel('x')
    plt.ylabel('y')

if __name__ == "__main__":
    # find trajectory
    p0 = np.array([9.92769241, 13.32634347])
    v0 = np.array([-3.66211474, -0.65636823])
    pk = np.array([10.0, 10 / 3 + LEG_DOWN/SCALE])
    vk = np.array([0, 0])
    p_k, f_k = oracle(p0, v0, pk, vk)

    # plot result
    fig = plt.figure()
    plotting(p_k)
    plt.show()
