import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ssl

# Отключить проверку SSL
ssl._create_default_https_context = ssl._create_unverified_context

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
df = pd.read_csv(url, header=None)
#print('Данные об Ирисах')
#print(df.to_string())

# Выборка из DF 100 строк (столбец 4 - название цветков) и загрузка их в массив Y
y = df.iloc[0:100, 4].values
# Преобразование названий цветков (столбец 4 - название цветков) в массив чисел -1 и 1
y = np.where(y == 'Iris-setosa', -1, 1)
#print('Значение названий цветков в виде -1 и 1, Y - 100', y, sep='\n')

# Выборка из DF 100 строк (столбец 0 и столбец 2), загрузка их в массив Х
X = df.iloc[0:100, [0, 2]].values
#print('Значение X - 100', X, 'Конец X', sep='\n')

# Формирование параметров значений для вывода на график
# Первые 50 элементов обучающей выборки (строки 0-50, столбцы 0, 1)
plt.scatter(X[0:50, 0], X[0:50, 1], color='red', marker='o', label='щетинистый')
# Следующие 50 элементов обучающей выборки (строки 50-100, столбцы 0, 1)
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='разноцветный')

# Формирование название осей, вывод графика и легенды на экран
plt.xlabel('Длина чашелистика')
plt.ylabel('Длина лепестка')
plt.legend(loc='upper left')
plt.show()


# Описание класса Perceptron
class Perceptron(object):
    ''' Классификатор на основе персептрона.
            Параметры
            :param eta: float, темп обучения от 0.0 до 1.0 (ƞ - эта)
            :param n_iter: int -  Проходы по тренировочному набору данных(эпохи)
            Атрибуты
            :parameter w_: 1-мерный массив - Веса после подгонки.
            :parameter errors_ : список - Число случаев ошибочной классификации в каждой эпохе.
            '''
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta  # Темп обучения от 0 до 1 (ƞ - эта)
        self.n_iter = n_iter  # Количество итераций(уроков/эпох)
        # Эпоха — это один полный проход через весь тренировочный набор данных.
        ''' Выполнить подгонку под тренировочные данныею
            :param X: массив, форма = [n_samples, n_features] тренировочные вектора,
            где n_samples - число образцов,
            n_features - число признаков.
            :param y: массив, форма = [n_samples] - Целевые значения.
            :return: self : объект
            '''

    def fit(self, X, y):
        # w_: одномерный массив весов после обучения, инициализируется 0 + смещение w_[0]
        # Смещение нужно, так как мы ищем границу решения по уравнению - линейную плоскость.
        # Образцы выше линии → класс 1, Ниже линии → класс -1
        self.w_ = np.zeros(1 + X.shape[1])  # X.shape возвращает кортеж с размерами массива X

        self.errors_ = []  # errors_: список - ошибок
                           # классификация в каждой эпохе
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    # Рассчитать чистый вход
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    # Вернуть метку класса после единичного скачка(пороговая функция)
    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)

ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Эпохи')
plt.ylabel('Число случаев ошибочной классификации')
plt.show()

# Визуализация границы решений
from matplotlib.colors import ListedColormap
def plot_decision_regions(X, y, classfier, resolution=0.02):
    # настроить генератор маркеров и палитру
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'green', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # Вывести поверхность решения
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    Z = classfier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # показать образцы классов
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, color=colors[idx], marker=markers[idx], label=cl)

# Нарисовать картину
plot_decision_regions(X, y, classfier=ppn)
plt.xlabel('Длина чашелистика, см')
plt.ylabel('Длина лепестка, см')
plt.legend(loc='upper left')
plt.show()

#
class AdaptiveLinearNeuron(object):
    ''' Классификатор на основе ADALINE (ADAptive Linear Neuron).
        Параметры
        :param eta: float, темп обучения от 0.0 до 1.0 (ƞ - эта)
        :param n_iter: int -  Проходы по тренировочному набору данных
        Атрибуты
        :parameter w_: 1-мерный массив - Веса после подгонки.
        :parameter errors_ : список - Число случаев ошибочной классификации в каждой эпохе.
        '''
    def __init__(self, rate=0.01, niter=10):
        self.rate = rate
        self.niter = niter

    def fit(self, X, y):
        ''' Выполнить подгонку под тренировочные данныею
        :param X: (массив), форма = [n_samples, n_features],
        где n_samples - число образцов,
        n_features - число признаков.
        :param y: массив, форма = [n_samples] - Целевые значения.
        :return: self : объект
        '''
        self.weight = np.zeros(1 + X.shape[1])
        self.cost = []
        for i in range(self.niter):
            output = self.net_input(X)
            errors = y - output
            self.weight[1:] += self.rate * X.T.dot(errors)
            self.weight[0] += self.rate * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost.append(cost)
        return self

    def net_input(self, X):
        # Вычисление чистого входного сигнала
        return np.dot(X, self.weight[1:]) + self.weight[0]

    def activation(self, X):
        # Вычислительная линейная активация
        return self.net_input(X)

    def predict(self, X):
        # Вернуть метку класса после единичного скачка
        return np.where(self.activation(X) >= 0.0, 1, -1)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
# learning rate 0.01
aln1 = AdaptiveLinearNeuron(0.01, 10).fit(X, y)
ax[0].plot(range(1, len(aln1.cost) + 1), np.log10(aln1.cost), marker='o')
ax[0].set_xlabel('Эпохи')
ax[0].set_ylabel('Сумма квадратичных ошибок')
ax[0].set_title('Adaline. Темп обучения 0.01')

# learning rate 0.01
aln2 = AdaptiveLinearNeuron(0.0001, 10).fit(X, y)
ax[1].plot(range(1, len(aln2.cost) + 1), aln2.cost, marker='o')
ax[1].set_xlabel('Эпохи')
ax[1].set_ylabel('Сумма квадратичных ошибок')
ax[1].set_title('ADALINE. Темп обучения 0.0001')
plt.show()

X_std = np.copy(X)
X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

# learning rate 0.01
aln = AdaptiveLinearNeuron(0.01, 10)
aln.fit(X_std, y)

# нарисовать картинку
plot_decision_regions(X_std, y, classifier=aln)
plt.title('ADALINE (градиентный спуск)')
plt.xlabel('Длина чашелистика [стандартизированная]')
plt.ylabel('Длина лепестка [стандартизированная]')
plt.legend(loc='upper left')
plt.show()

plt.plot('')
plt.xlabel('Эпохи')
plt.ylabel('Сумма квадратичных ошибок')
plt.show()