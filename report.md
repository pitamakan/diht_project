# Отчет #

## Введение ##
(Мини-пакетный) стохастический градиентный спуск является популярным методом оптимизации, который был применен ко многим приложениям машинного обучения. Но довольно
высокая дисперсия, вносимая стохастическим градиентом на каждом шаге, может замедлить сходимость. В этой статье мы предлагаем антитетическую стратегию выборки для уменьшения дисперсии, используя преимущества внутренней структуры в наборе данных. В соответствии с этой новой стратегией стохастические градиенты в мини-партии больше не являются независимыми, но максимально отрицательно коррелируют, в то время как стохастический градиент мини-партии по-прежнему является беспристрастной оценкой полной gradient. Для задач бинарной классификации нам  просто нужно вычислить антитетические образцы заранее и повторно использовать результат в каждой итерации, что является практичным. Приведены эксперименты, подтверждающие эффективность предложенного метода.

## Алгоритм ##

```
def calculate_path_len(n): # какая-то функция определенеия длины шага
    rerutn 1.0 / n

# sample -  наши известные значения для задачи классификации,
# start_value - стартовый вектор
# functions_grads - градиенты функций потерь

def SGD_with_AS(sample, start_value, functions_grads):
    table = Calculate_antithetic_table(sample)
    w = start_value
    for i in range(100):#подставь нужное
        l = random.randint(0, len(sample) - 1)
        j = table[l]
        w = w - calculate_path_len(i+1)(functions_grads[l](w) - functions_grads[j](w))

    return w
```

Как же находить таблицу антитетики? Очень Просто:
Покажем на примере задачи бинарной классификации($y_i \in$ {1, -1})

Имеем
Samples = [{x0, y0}, {x1, y1}, ..., {x_(n-1), y_(n-1)}]

```
def count_dist(sample_1, sample_2, prev_sample):
    if prev_sample is None:
        return abs(sample_1[0] * sample_1[1] * sample_2[0][0] * sample_2[0][1])
    return abs(sample_1[0] * (sample_1[1] - prev_sample) * sample_2[0][0] * sample_2[0][1])

def NearestNeighbor(sample, DB, prev_sample):
    nearest_dist = count_dist(sample, DB[0], prev_sample)
    min_obj = DB[0]
    for obj in DB:
        new_dist = count_dist(sample, obj, prev_sample)
        if new_dist < nearest_dist:
            nearest_dist = new_dist
            min_obj = obj
    return min_obj

def Calculate_antithetic_table(samples):
    n = len(samples)
    table = np.zeros(n)
    DB = [(Samples[i], i) for i in range(n)]
    prev_sample = None
    for i in range(n):
        sample_id = NearestNeighbor(Samples[i], DB, Samples[i-1] if i > 0 else None)
        table[i] = sample_id[1]
        prev_sample = Samples[0]
        DB.erase(sample_id)
    return table
```

## О полигоне, тестах и сравниваемых алгоритмах ##
См. .ipynb

Тесты: - 

## Результаты ##

## Вывод ##
