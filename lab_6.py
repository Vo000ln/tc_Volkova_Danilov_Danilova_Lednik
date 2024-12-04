import numpy as np
import random


def find_ostat(delim, delit):
    ostat = list(delim)
    while len(ostat) >= delit.size:

        for i in range(delit.size):
            ostat[len(ostat) - delit.size + i] ^= delit[i]

        while ostat and not(ostat[-1]):
            ostat = ostat[:-1]

    return np.array(ostat)


def polynom_mul(matr_a, matr_b):
    res = np.zeros(matr_a.size + matr_b.size - 1, dtype=int)

    for i in range(matr_b.size):
        if matr_b[i]:
            # cдвигаем результат на 'i' позиций вправо
            res[i : i + matr_a.size] ^= matr_a.astype(int)

    return res


def find_error(a, g, err_rate):
    # умножаем исходное сообщение на порождающий полином = отправленное сообщение
    snd_msg = polynom_mul(a, g)
    print(f"отправленное сообщение:  {snd_msg}")
    w = snd_msg.copy()
    
    err = np.zeros(w.size, dtype=int)
    # генерируем случайные индексы для ошибок
    err_i = random.sample(range(w.size), err_rate)
    for i in err_i:
        err[i] = 1
    # вносим ошибки в сообщение
    w = (w + err) % 2
    print(f"сообщение с ошибкой:  {w}")

    ostat = find_ostat(w, g)
    # определяем шаблоны ошибок в зависимости от частоты
    err_example = [[1]] if err_rate == 1 else [[1, 1, 1], [1, 0, 1], [1, 1], [1]]

    idx = 0
    while not any(np.array_equal(ostat, ex) for ex in err_example):
        ostat = find_ostat(polynom_mul(ostat, np.array([0, 1])), g)
        idx += 1

    # Создаем массив для исправления ошибок
    tmp = np.zeros(w.size, dtype=int)
    tmp[0 if not(idx) else tmp.size - idx - 1] = 1  
    # остаток * временный массив = исправленный
    pmul = polynom_mul(ostat, tmp)[:w.size]
    msg = (w + pmul) % 2
    print(f"исправленное сообщение:  {msg}\n")


a = [np.array([1, 0, 0, 1]), np.array([1, 0, 0, 1, 0, 0, 0, 1, 1])] 
g = [np.array([1, 0, 1, 1]), np.array([1, 0, 0, 1, 1, 1, 1])]
for j in range(2):
    print(f'порождающий полином {'g = 1 + x^2 + x^3' if not(j) else 'g = 1 + x^3 + x^4 + x^5 + x^6'}')
    print(f"исходное сообщение:  {a[j]}")
    print(f"порождающий полином:  {g[j]}\n")
    for i in range(1, 4 if not(j) else 5):
        find_error(a[j], g[j], i)
