import numpy as np
from itertools import combinations, product
from math import comb


def generateVector(vec, size):
    if not len(vec):
        return np.ones(2**size, dtype=int)

    indices = np.arange(2**size)
    # преобразуем индексы в двоичное представление и извлекаем нужные биты
    binary_vectors = (indices[:, None] >> np.arange(size)[::-1]) & 1
    result = np.prod((binary_vectors[:, vec] + 1) % 2, axis=1)

    return result

def generateMatrix(M, size):
    return [
        w
        for w in product([0, 1], repeat=size)
        if np.prod([(w[index] + 1) % 2 for index in M]) == 1
    ]

def compute_binary_products_with_t(M, m, t):
    if not M:
        return np.ones(2**m, dtype=int)
    return [
        np.prod([(w[j] + t[j] + 1) % 2 for j in M])
        for w in product([0, 1], repeat=m)
    ]

def RM(r, m):
    # размер порождающей матрицы
    matr_size = sum(comb(m, index) for index in range(r + 1))
    matr = np.zeros((matr_size, 2**m), dtype=int)
    index = 0

    for subset_size in range(r + 1):
        for subset in combinations(range(m), subset_size):
            matr[index] = generateVector(subset, m)
            index += 1

    return matr

G = RM(2, 4)
print(f"матрица кода Рида-Маллера:\n {G}")

def sortedWord(i, j):
    indices = range(i)
    # получаем все комбинации длиной k
    comb_list = list(combinations(indices, j))
    result_array = np.array(comb_list, dtype=int)

    return result_array

# генерация слова с указанным количеством ошибок
def mistakes(G, error, v):
    v = np.dot(v, G) % 2
    # генерация случайных позиций ошибок
    mistake_pos = np.random.choice(len(v), size=error, replace=False)
    # внесение ошибок
    v[mistake_pos] ^= 1
    return v


def majorDecoding(input_word, reference, matrix_size, total_length):
    counter = 0
    current_index = reference
    output_rows = np.zeros(total_length, dtype=int)
    max_weight = 2 ** (matrix_size - reference - 1) - 1
    current_word = input_word.copy()

    while True:
        for subset in sortedWord(matrix_size, current_index):
            max_count = 2 ** (matrix_size - current_index - 1)
            count_zeros = count_ones = 0
            for transformation in generateMatrix(subset, matrix_size):
                remaining_indices = [
                    index for index in range(matrix_size) if index not in subset
                ]
                binary_products = compute_binary_products_with_t(
                    remaining_indices, matrix_size, transformation
                )
                if not ((current_word @ binary_products) % 2):
                    count_zeros += 1
                else:
                    count_ones += 1
            if count_zeros > max_weight and count_ones > max_weight:
                return
            if count_zeros > max_count:
                output_rows[counter] = 0
                counter += 1
            if count_ones > max_count:
                vector = generateVector(subset, matrix_size)
                current_word = (current_word + vector) % 2
                output_rows[counter] = 1
                counter += 1
        if current_index > 0:
            if len(current_word) < max_weight:
                for subset in sortedWord(matrix_size, reference + 1):
                    output_rows[counter] = 0
                    counter += 1
                break
            current_index -= 1
        else:
            break
    return output_rows


print("однократная ошибка")
v = np.array([1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1])
print(f"сообщение изначальное: {v}")
error = mistakes(G, 1, v)
print(f"слово с ошибкой: {error}")
decoded = majorDecoding(error, 2, 4, len(G))
print("ошибка не исправна" if decoded is None else f"v*G: {(decoded @ G) % 2}\nисправленное слово: {decoded}")

print("двукратная ошибка")
v = np.array([1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1])
print(f"сообщение изначальное: {v}")
error = mistakes(G, 2, v)
print(f"слово с ошибкой: {error}")
decoded = majorDecoding(error, 2, 4, len(G))
print("ошибка не исправна" if decoded is None else f" v*G: {(decoded @ G) % 2}\nисправленное слово: {decoded}")
