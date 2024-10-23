import random


def analyze_hamming_code(r, extended=False):
    if extended:
        gen_matrix = create_extended_hamming_matrix(r)
        parity_matrix = create_extended_parity_matrix(r)
        max_errors = 4
        print("\nАнализ расширенного кода Хэмминга")
    else:
        gen_matrix = create_hamming_matrix(r)
        parity_matrix = create_parity_matrix(r)
        max_errors = 3
        print("\nАнализ стандартного кода Хэмминга")

    print("\nПорождающая матрица G:")
    for row in gen_matrix:
        print(row)

    print("\nПроверочная матрица H:")
    for row in parity_matrix:
        print(row)

    u_vecs = [list(col) for col in zip(*gen_matrix)]
    code_vec = u_vecs[random.randint(0, len(u_vecs) - 1)]
    print("\nСгенерированное кодовое слово:")
    print(code_vec)

    for num_errors in range(1, max_errors + 1):
        if num_errors > len(code_vec):
            break
        print(f"\nПроверка для {num_errors} ошибок:")

        err_vec = create_error_vector(len(code_vec), num_errors)
        print(f"Вектор ошибок: {err_vec}")

        received_vec = [(bit + err) % 2 for bit, err in zip(code_vec, err_vec)]
        print(f"Кодовое слово с ошибками: {received_vec}")

        corrected_vec, synd = correct_errors(parity_matrix, received_vec)
        print(f"Синдром: {synd}")
        print(f"Исправленное кодовое слово: {corrected_vec}")

        final_synd = vector_mult_matrix(corrected_vec, parity_matrix)
        print(f"Синдром после коррекции (должен быть [0,...,0]): {final_synd}")


def create_extended_hamming_matrix(r):
    gen_matrix = create_hamming_matrix(r)
    for row in gen_matrix:
        parity_bit = sum(row) % 2
        row.append(parity_bit)
    return gen_matrix


def create_extended_parity_matrix(r):
    parity_matrix = create_parity_matrix(r)
    extra_row = [0] * len(parity_matrix[0])
    parity_matrix.append(extra_row)
    for row in parity_matrix:
        row.append(1)
    return parity_matrix


def create_hamming_matrix(r):
    n = 2 ** r - 1
    k = 2 ** r - r - 1
    return horizontal_concat(create_id_matrix(k), generate_vectors_x(n, k))


def create_parity_matrix(r):
    n = 2 ** r - 1
    k = 2 ** r - r - 1
    return vertical_concat(generate_vectors_x(n, k), create_id_matrix(n - k))


def create_syndrome_table(h_mat):
    syndrome_tbl = {}
    for i in range(len(h_mat[0])):
        err_vector = [0] * len(h_mat[0])
        err_vector[i] = 1
        syndrome = vector_mult_matrix(err_vector, h_mat)
        syndrome_tbl[tuple(syndrome)] = err_vector
    return syndrome_tbl


def correct_errors(h_mat, received_vec):
    synd = vector_mult_matrix(received_vec, h_mat)
    synd_table = create_syndrome_table(h_mat)
    if tuple(synd) in synd_table:
        error_vec = synd_table[tuple(synd)]
        corrected_vec = [(bit + err) % 2 for bit, err in zip(received_vec, error_vec)]
        return corrected_vec, synd
    return received_vec, synd


def create_id_matrix(size):
    return [[1 if i == j else 0 for j in range(size)] for i in range(size)]


def horizontal_concat(matrix_a, matrix_b):
    return [row_a + row_b for row_a, row_b in zip(matrix_a, matrix_b)]


def vertical_concat(matrix_a, matrix_b):
    return matrix_a + matrix_b


def generate_vectors_x(n, k):
    vectors = []
    vec = [0] * (n - k)
    while len(vectors) < k:
        for i in reversed(range(len(vec))):
            if vec[i] == 0:
                vec[i] = 1
                vectors.append(vec[:])
                break
            else:
                vec[i] = 0
    return vectors


def create_error_vector(size, num_errors):
    vector = [0] * size
    pos_errors = random.sample(range(size), num_errors)
    for pos in pos_errors:
        vector[pos] = 1
    return vector


def vector_mult_matrix(vector, matrix):
    return [sum(v * m for v, m in zip(vector, col)) % 2 for col in zip(*matrix)]


# Примеры вызова функций
# Исследование стандартного кода Хэмминга для одно-, двух- и трехкратных ошибок
analyze_hamming_code(2)
analyze_hamming_code(3)
analyze_hamming_code(4)

# Исследование расширенного кода Хэмминга для одно-, двух-, трех- и четырехкратных ошибок
analyze_hamming_code(2, extended=True)
analyze_hamming_code(3, extended=True)
analyze_hamming_code(4, extended=True)
