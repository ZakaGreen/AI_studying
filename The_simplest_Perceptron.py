import random


def perceptron(sensor):
    b = 7  # Порог функции активации
    s = 0  # Начальное значение суммы
    for i in range(n_sensor):
        s += int(sensor[i]) * weights[i]  # Цикл суммирования сигналов от сенсоров
    if s >= b:
        return True  # Сумма превысила порог
    else:
        return False


def decrease(number):  # Если 1 при цифре != tema
    for i in range(n_sensor):
        if int(number[i]) == 1:
            weights[i] -= 1


def increase(number):  # Если 1 при цифре == tema
    for i in range(n_sensor):
        if int(number[i]) == 1:
            weights[i] += 1


num0 = list('111101101101111')
num1 = list('001001001001001')
num2 = list('111001111100111')
num3 = list('111001111001111')
num4 = list('101101111001001')
num5 = list('111100111001111')
num6 = list('111100111101111')
num7 = list('111001001001001')
num8 = list('111101111101111')
num9 = list('111101111001111')
nums = [num0, num1, num2, num3, num4, num5, num6, num7, num8, num9]
tema = 5
n_sensor = 15
weights = [0 for i in range(n_sensor)]
n = 1

for i in range(n):
    j = random.randint(0, 9)
    print(j)
    r = perceptron(nums[j])  # 1 or 0

    if j != tema:
        if r:  # Ошибка: активировался на ≠5
            print("Ошибка: активировался на ≠5")
            decrease(nums[j])
    else:
        if not r:  # Ошибка: не активировался на 5
            print("Ошибка: не активировался на 5")
            increase(nums[tema])

print(weights)
