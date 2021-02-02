import numpy as np
from sklearn import linear_model


def bin_search(X, y, k, s):
    l_border = 0
    r_border = 1000
    curr_s = 0.0
    eps = np.finfo('float').eps
    coefficients = None
    epoch = 0
    max_epoch = 100
    zero_count = 0
    alpha = 0
    while zero_count != k or s > curr_s:
        alpha = l_border + (r_border - l_border) / 2
        regression = linear_model.Lasso(alpha=alpha, max_iter=3000).fit(X, y)
        curr_s = regression.score(X, y)
        coefficients = np.concatenate((regression.coef_, np.array([regression.intercept_])), axis=0)
        zero_count = np.count_nonzero(np.abs(coefficients) <= eps)
        if zero_count <= k:
            l_border = alpha
        elif zero_count > k:
            r_border = alpha
        if epoch == max_epoch:
            print('\tStop bin search — run out of epochs!')
            break
        else:
            epoch += 1
    print(f'\tCalculations takes {epoch} epochs')
    return alpha, curr_s, coefficients


with open('lasso_input.txt', 'r') as fin, open('lasso_output.txt', 'w') as fout:
    N = int(fin.readline().strip())
    for test in range(1, N + 1):
        n, m, k, s = list(map(float, fin.readline().strip().split()))
        n, m, k = list(map(int, [n, m, k]))
        data = np.array(list(map(lambda x: list(map(float, x.strip().split())), [fin.readline() for _ in range(n)])))
        X, y = np.delete(data, m, 1), data[:, m]
        print(f'Run test {test}, k = {k}, s = {s}')
        a, curr_s, coefficients = bin_search(X, y, k, s)
        print(f'\tDone with alpha = {a}, r^2 = {curr_s}\n')
        fout.write(' '.join(list(map(str, coefficients))) + '\n')

    fin.close()
    fout.close()