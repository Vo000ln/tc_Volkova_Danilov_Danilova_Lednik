import numpy as np
from itertools import product

def GOLAY_matrix_rmRIX():
    B = np.array([
        [1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1],
        [1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1],
        [0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1],
        [1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1],
        [1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1],
        [1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1],
        [0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1],
        [0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1],
        [0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1],
        [1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1],
        [0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
    ])
    G = np.hstack((np.eye(len(B), dtype=int), B))
    H = np.vstack((np.eye(len(B), dtype=int), B))
    return B, G, H

B, G, H = GOLAY_matrix_rmRIX()
print("G: ", G)
print("H: ", H)

def MAKE_ERRORS(code, n):
    code_err = code.copy()
    error_positions = np.random.choice(len(code), n, replace=False)
    for position in error_positions:
        if code_err[position]:
            code_err[position] = 0
        else:
            code_err[position] = 1
    return code_err


def DECODE(recv, H):
    syndrome = np.dot(recv, H) % 2
    return syndrome


def GOLAY_ERR(G, H):
    message = np.random.randint(2, size=12)
    code = np.dot(message, G) % 2
    print(f"message: {message}")
    print(f"code: {code}")
    words = [[0, 1] for _ in range(12)]
    words = np.array(list(product(*words)))

    words_dict = dict()
    for element in words:
        words_dict[np.array_str((element @ G) % 2)] = element

    for num_errors in range(1, 5):
        print(f"\nnumber errors: {num_errors}")
        recv_errors = MAKE_ERRORS(code, num_errors)
        print(f"recv errors: {recv_errors}")
        syndrome = np.array(DECODE(recv_errors, H))
        print(f"syndrome: {syndrome}")

        u = np.array([])
        if sum(syndrome) > 3:
            for i in range(B.shape[0]):
                if sum((syndrome + B[i]) % 2) <= 2:
                    e_i = np.zeros(12, dtype=int)
                    e_i[i] = 1
                    u = np.hstack(((syndrome + B[i]) % 2, e_i))
                    break
        else:
            u = np.hstack((syndrome, np.zeros(len(B), dtype=int)))

        if u.size == 0:
            syndrome2 = (syndrome @ B) % 2
            if sum(syndrome2) > 3:
                for i in range(B.shape[0]):
                    if sum((syndrome2 + B[i]) % 2) <= 2:
                        e_i = np.zeros(12, dtype=int)
                        e_i[i] = 1
                        u = np.hstack((e_i, (syndrome2 + B[i]) % 2))
                        break
            else:
                u = np.hstack((np.zeros(12, dtype=int), syndrome2))

        if u.size > 0:
            print(f"source: {words_dict[np.array_str((recv_errors + u) % 2)]}")
        else:
            print("error found, not fix")


GOLAY_ERR(G, H)


def RM_G(r, m):
    if not r:
        return np.ones((1, 2**m), dtype=int)
    if r == m:
        return np.vstack((RM_G(m - 1, m), np.array([0 for _ in range(2 ** m - 1)] + [1])))

    matrix_rm = RM_G(r, m - 1)
    matrix_rm2 = RM_G(r - 1, m - 1)
    return np.vstack((np.hstack((matrix_rm, matrix_rm)), np.hstack((np.zeros((matrix_rm2.shape[0], matrix_rm.shape[1]), dtype=int), matrix_rm2))))


def DECODE_WORD(w, Hs, k, G_key_words):
    pred = []
    d = np.inf
    for element in G_key_words:
        if sum((w + element) % 2) < d:
            pred.clear()
            pred.append(element)
            d = sum((w + element) % 2)
        elif sum((w + element) % 2) == d:
            pred.append(element)

    if (len(pred) == 1):
        w = pred[0]
        w_copy = w.copy()
        w_copy[w_copy == 0] = -1

        for i in range(len(Hs)):
            w_copy = w_copy @ Hs[i]

        w_copy_abs = np.abs(w_copy)
        j = np.argmax(w_copy_abs)
        bin_j = bin(j)[2:]

        tmp = "0"  * (m - len(bin_j))
        tmp = tmp + bin_j
        tmp = tmp[::-1]

        if(w_copy[j] > 0):
            fake_message = np.array(list("1" + tmp), dtype=int)
        else:
            fake_message = np.array(list("0" + tmp), dtype=int)
        print(f"source: {k}")
        print(f"decode word: {fake_message}")
    else:
        print("error found, not fix")


r, m = 1, 3
Hs = []
H = np.array([[1, 1], [1, -1]])

for i in range(1, m + 1):
    Hs.append(np.kron(np.kron(np.eye(int(2**(m - i)), dtype=int), H), np.eye(int(2**(i - 1)), dtype=int)))

bin_lists = [[0, 1], [0, 1], [0, 1], [0, 1]]
key_words = np.array(list(product(*bin_lists)))
G_key_words = np.array([(element @ RM_G(r, m)) % 2 for element in key_words])
k = np.array([1, 1, 0, 0])
w = (k @ RM_G(r, m)) % 2
w[0] = (w[0] + 1) % 2

DECODE_WORD(w, Hs, k, G_key_words)

k = np.array([1, 1, 0, 0])
w = (k @ RM_G(r, m)) % 2

w[0] = (w[0] + 1) % 2
w[1] = (w[1] + 1) % 2

DECODE_WORD(w, Hs, k, G_key_words)

r, m = 1, 4

Hs = []
H = np.array([[1, 1], [1, -1]])
for i in range(1, m + 1):
    Hs.append(np.kron(np.kron(np.eye(int(2**(m - i)), dtype=int), H), np.eye(int(2**(i - 1)), dtype=int)))

bin_lists = [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]
key_words = np.array(list(product(*bin_lists)))
G_key_words = np.array([(element @ RM_G(r, m)) % 2 for element in key_words])
k = np.array([1, 1, 1, 0, 0])
w = (k @ RM_G(r, m)) % 2
w[0] = (w[0] + 1) % 2

DECODE_WORD(w, Hs, k, G_key_words)

k = np.array([1, 1, 1, 0, 0])
w = (k @ RM_G(r, m)) % 2
w[0] = (w[0] + 1) % 2
w[1] = (w[1] + 1) % 2

DECODE_WORD(w, Hs, k, G_key_words)

k = np.array([1, 1, 1, 0, 0])
w = (k @ RM_G(r, m)) % 2
w[0] = (w[0] + 1) % 2
w[1] = (w[1] + 1) % 2
w[2] = (w[2] + 1) % 2

DECODE_WORD(w, Hs, k, G_key_words)

k = np.array([1, 1, 1, 0, 0])
w = (k @ RM_G(r, m)) % 2
w[0] = (w[0] + 1) % 2
w[1] = (w[1] + 1) % 2
w[2] = (w[2] + 1) % 2
w[3] = (w[3] + 1) % 2

DECODE_WORD(w, Hs, k, G_key_words)