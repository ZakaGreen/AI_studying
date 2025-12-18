import random


def proceed(x):
    return x * k + ccd


# коэфф при х
k = random.uniform(-5, 5)
c = random.uniform(-5, 5)

# Вывод данных начальной прямой линии
print('Начальная прямая линия: ', k, '* X + ', c)
rate = 0.0001  # скорость обучения

# Набор X:Y
# зависимость роста мужчины от длины следа обуви
data = {22: 150, 23: 155, 24: 160, 25: 162, 26: 171, 27: 174, 28: 180, 29: 183, 30: 189, 31: 192}

for i in range(100000):
    x = random.choice(list(data.keys()))  # случайная координата Х
    true_result = data[x]  # У координаты
    out = proceed(x)  # ответ сети
    delta = true_result - out  # считаем ошибку сети
    k += delta * rate * x  # Меняем вес при постоянном входе с дельта-правилом
    c += delta * rate

print('Готовая прямая: Y = ', k, '* X + ', c)  # Вывод данных готовой прямой
