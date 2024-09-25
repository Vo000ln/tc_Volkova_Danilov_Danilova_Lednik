import numpy as np

# Порождающая матрица G (7, 4, 3)
def generate_G():
    I = np.eye(4, dtype=int)  # Единичная матрица 4x4
    X = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]])  # Матрица дополнения X
    G = np.hstack((I, X))  # Объединение матриц G = [I | X]
    return G

G = generate_G()
print("Порождающая матрица G:")
print(G)

# Проверочная матрица H
def generate_H():
    X = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]])  # Матрица дополнения X (4x3)
    I = np.eye(3, dtype=int)  # Единичная матрица 3x3
    H = np.hstack((X.T, I))  # Проверочная матрица H = [X^T | I]
    return H

H = generate_H()
print("\nПроверочная матрица H:")
print(H)

# Синдромы для однократных ошибок
def generate_syndromes(H):
    syndromes = {}
    for i in range(H.shape[1]):
        error_vector = np.zeros(H.shape[1], dtype=int)
        error_vector[i] = 1
        syndrome = np.dot(H, error_vector) % 2
        syndromes[tuple(syndrome)] = error_vector
    return syndromes

syndromes = generate_syndromes(H)
print("\nТаблица синдромов для однократных ошибок:")
for syndrome, error in syndromes.items():
    print(f"Синдром {syndrome}: Ошибка {error}")

# Сформировать кодовое слово
def generate_codeword(data, G):
    return np.dot(data, G) % 2

# Внесение ошибки
def introduce_error(codeword, position):
    codeword[position] ^= 1  # Инвертируем бит на указанной позиции
    return codeword

# Вычисление синдрома
def calculate_syndrome(received_word, H):
    return np.dot(H, received_word) % 2

# Исправление ошибки с использованием синдрома
def correct_error(received_word, syndrome, syndromes):
    if tuple(syndrome) in syndromes:
        error_vector = syndromes[tuple(syndrome)]
        corrected_word = (received_word + error_vector) % 2
        return corrected_word
    return received_word

# Пример
data_word = np.array([1, 0, 1, 1])  # Данные длины k
codeword = generate_codeword(data_word, G)
print("\nКодовое слово:", codeword)

# Внесение ошибки в кодовое слово
error_position = 2
received_word = introduce_error(codeword.copy(), error_position)
print("Кодовое слово с ошибкой:", received_word)

# Вычисление синдрома
syndrome = calculate_syndrome(received_word, H)
print("Синдром:", syndrome)

# Исправление ошибки
corrected_word = correct_error(received_word, syndrome, syndromes)
print("Исправленное слово:", corrected_word)

# Внесение двукратной ошибки
def introduce_double_error(codeword, positions):
    for position in positions:
        codeword[position] ^= 1  # Инвертируем бит на каждой указанной позиции
    return codeword

# Пример для двукратной ошибки
data_word = np.array([1, 0, 1, 1])  # Данные длины k
codeword = generate_codeword(data_word, G)
print("\nКодовое слово:", codeword)

# Внесение двукратной ошибки
error_positions = [2, 5]  # Позиции для внесения ошибки
received_word = introduce_double_error(codeword.copy(), error_positions)
print("Кодовое слово с двукратной ошибкой:", received_word)

# Вычисление синдрома
syndrome = calculate_syndrome(received_word, H)
print("Синдром:", syndrome)

# Попытка исправления ошибки
corrected_word = correct_error(received_word, syndrome, syndromes)
print("Попытка исправления:", corrected_word)
print("Полученное слово отличается от исходного:", not np.array_equal(corrected_word, codeword))

# Пример для кода (n, k, 5) — n=10, k=5
def generate_G_5():
    I = np.eye(5, dtype=int)  # Единичная матрица
    X = np.array([[1, 1, 0, 1, 0], [1, 0, 1, 0, 1], [0, 1, 1, 1, 0], [1, 1, 0, 0, 1], [0, 0, 1, 1, 1]])  # Матрица дополнения X
    G = np.hstack((I, X))  # Объединение матриц G
    return G

G_5 = generate_G_5()
print("\nПорождающая матрица для кода (n=10, k=5, d=5):")
print(G_5)

def generate_H_5():
    X = G_5[:, 5:]  # Матрица дополнения X (взята из правой части порождающей матрицы G)
    I = np.eye(5, dtype=int)  # Единичная матрица
    H = np.hstack((X.T, I))  # Проверочная матрица H, размер
    return H

H_5 = generate_H_5()
print("\nПроверочная матрица H для кода (n=10, k=5):")
print(H_5)

# Генерация синдромов для однократных и двухкратных ошибок
def generate_error_syndromes(H):
    syndromes = {}
    n = H.shape[1]
    for i in range(n):
        for j in range(i+1, n):
            error_vector = np.zeros(n, dtype=int)
            error_vector[i] = 1
            error_vector[j] = 1
            syndrome = np.dot(H, error_vector) % 2
            syndromes[tuple(syndrome)] = error_vector
    return syndromes

double_error_syndromes = generate_error_syndromes(H_5)
print("\nТаблица синдромов для двукратных ошибок:")
for syndrome, error in double_error_syndromes.items():
    print(f"Синдром {syndrome}: Ошибка {error}")

# Пример для однократной ошибки
data_word_5 = np.array([1, 0, 1, 1, 0])  # Данные длины k
codeword_5 = generate_codeword(data_word_5, G_5)
print("\nКодовое слово (d=5):", codeword_5)

# Внесение однократной ошибки
error_position = 3
received_word_5 = introduce_error(codeword_5.copy(), error_position)
print("Кодовое слово с однократной ошибкой:", received_word_5)

# Вычисление синдрома
syndrome_5 = calculate_syndrome(received_word_5, H_5)
print("Синдром:", syndrome_5)

# Исправление ошибки
corrected_word_5 = correct_error(received_word_5, syndrome_5, syndromes)
print("Исправленное слово:", corrected_word_5)

# Внесение двукратной ошибки в кодовое слово (d=5)
error_positions = [2, 6]
received_word_5 = introduce_double_error(codeword_5.copy(), error_positions)
print("\nКодовое слово с двукратной ошибкой:", received_word_5)

# Вычисление синдрома
syndrome_5 = calculate_syndrome(received_word_5, H_5)
print("Синдром:", syndrome_5)

# Попытка исправления двукратной ошибки
corrected_word_5 = correct_error(received_word_5, syndrome_5, double_error_syndromes)
print("Попытка исправления:", corrected_word_5)
print("Полученное слово отличается от исходного:", not np.array_equal(corrected_word_5, codeword_5))

# Внесение трёхкратной ошибки
def introduce_triple_error(codeword, positions):
    for position in positions:
        codeword[position] ^= 1  # Инвертируем бит на каждой указанной позиции
    return codeword

# Пример для трёхкратной ошибки
data_word = np.array([1, 0, 1, 1])  # Данные длины k
codeword = generate_codeword(data_word, G)
print("\nКодовое слово:", codeword)

# Внесение трёхкратной ошибки
error_positions = [1, 3, 6]  # Позиции для внесения ошибки
received_word = introduce_triple_error(codeword.copy(), error_positions)
print("Кодовое слово с трёхкратной ошибкой:", received_word)

# Вычисление синдрома
syndrome = calculate_syndrome(received_word, H)
print("Синдром:", syndrome)

# Попытка исправления ошибки
corrected_word = correct_error(received_word, syndrome, syndromes)
print("Попытка исправления:", corrected_word)

# Проверка, отличается ли полученное слово от исходного
print("Полученное слово отличается от исходного:", not np.array_equal(corrected_word, codeword))




