import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import matplotlib.dates as mdates

# Загрузка данных из файлов
file_path_data = 'F:/ККАЛ/DataSet Газпром/opers.csv'
file_path_repairs = 'F:/ККАЛ/DataSet Газпром/repairs.csv'
data = pd.read_csv(file_path_data)
repairs = pd.read_csv(file_path_repairs)

# Преобразование столбца с датами в формат datetime
data['date'] = pd.to_datetime(data['date'])

# Разделение данных на группы по значению 'is_emergency'
false_data = data[data['is_emergency'] == False]
true_data = data[data['is_emergency'] == True]

# Функция для построения дерева
def build_tree(data):
    tree = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for _, row in data.iterrows():
        date = row['date']
        r_id = row['r_id']
        tree[date.year][date.month][date.day].append(r_id)
    return tree

# Построение деревьев
false_tree = build_tree(false_data)
true_tree = build_tree(true_data)

# Функция для нахождения даты с наибольшим и наименьшим количеством записей и соответствующих r_id
def find_extremes(tree):
    max_count = 0
    min_count = float('inf')
    peak_date = None
    low_date = None
    peak_r_id = []
    low_r_id = []
    for year in tree:
        for month in tree[year]:
            for day in tree[year][month]:
                count = len(tree[year][month][day])
                if count > max_count:
                    max_count = count
                    peak_date = (year, month, day)
                    peak_r_id = tree[year][month][day]
                if count < min_count:
                    min_count = count
                    low_date = (year, month, day)
                    low_r_id = tree[year][month][day]
    return (peak_date, max_count, peak_r_id), (low_date, min_count, low_r_id)

false_peak, false_low = find_extremes(false_tree)
true_peak, true_low = find_extremes(true_tree)

print(f"Пик False: {false_peak[0]} с {false_peak[1]} записями, r_id: {false_peak[2]}")
print(f"Минимум False: {false_low[0]} с {false_low[1]} записями, r_id: {false_low[2]}")
print(f"Пик True: {true_peak[0]} с {true_peak[1]} записями, r_id: {true_peak[2]}")
print(f"Минимум True: {true_low[0]} с {true_low[1]} записями, r_id: {true_low[2]}")

# Функция для визуализации общего количества записей по дням
def plot_total_counts(tree, title):
    dates = []
    counts = []
    for year in sorted(tree):
        for month in sorted(tree[year]):
            for day in sorted(tree[year][month]):
                dates.append(f"{year}-{month:02d}-{day:02d}")
                counts.append(len(tree[year][month][day]))
    dates = pd.to_datetime(dates)

    plt.figure(figsize=(14, 7))
    plt.plot(dates, counts, label=title)
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=90)
    plt.xlabel('Дата')
    plt.ylabel('Количество записей')
    plt.title(f'Общее количество записей {title} с течением времени')
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.show()

plot_total_counts(false_tree, "False")
plot_total_counts(true_tree, "True")

# Функция для визуализации наиболее активных и наименее активных периодов
def plot_extremes(extremes, title):
    dates = [f"{d[0]}-{d[1]:02d}-{d[2]:02d}" for d, _, _ in extremes]
    counts = [count for _, count, _ in extremes]
    dates = pd.to_datetime(dates)

    plt.figure(figsize=(12, 6))
    plt.bar(dates, counts, width=0.5, align='center', label=title)
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=90)
    plt.xlabel('Дата')
    plt.ylabel('Количество записей')
    plt.title(f'{title} периоды')
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.show()

plot_extremes([false_peak, false_low], "False Peak and Low")
plot_extremes([true_peak, true_low], "True Peak and Low")

# Функция для нахождения наиболее частого r_id
def most_common_r_id(r_ids):
    count_dict = defaultdict(int)
    for r_id in r_ids:
        count_dict[r_id] += 1
    return max(count_dict, key=count_dict.get)

most_common_false_low_r_id = most_common_r_id(false_low[2])
most_common_false_peak_r_id = most_common_r_id(false_peak[2])
most_common_true_peak_r_id = most_common_r_id(true_peak[2])
most_common_true_low_r_id = most_common_r_id(true_low[2])

most_common_false_low_b_id = repairs[repairs['r_id'] == most_common_false_low_r_id]['b_id'].iloc[0]
most_common_false_peak_b_id = repairs[repairs['r_id'] == most_common_false_peak_r_id]['b_id'].iloc[0]
most_common_true_peak_b_id = repairs[repairs['r_id'] == most_common_true_peak_r_id]['b_id'].iloc[0]
most_common_true_low_b_id = repairs[repairs['r_id'] == most_common_true_low_r_id]['b_id'].iloc[0]

print(f"Наиболее частый r_id для Минимума False: {most_common_false_low_r_id}, соответствующий b_id: {most_common_false_low_b_id}")
print(f"Наиболее частый r_id для Пика False: {most_common_false_peak_r_id}, соответствующий b_id: {most_common_false_peak_b_id}")
print(f"Наиболее частый r_id для Пика True: {most_common_true_peak_r_id}, соответствующий b_id: {most_common_true_peak_b_id}")
print(f"Наиболее частый r_id для Минимума True: {most_common_true_low_r_id}, соответствующий b_id: {most_common_true_low_b_id}")

# Функция для визуализации наиболее частых r_id
def plot_common_r_id(extremes, common_r_ids, title):
    dates = [f"{d[0]}-{d[1]:02d}-{d[2]:02d}" for d, _, _ in extremes]
    r_ids = [r for r in common_r_ids]
    dates = pd.to_datetime(dates)

    plt.figure(figsize=(12, 6))
    plt.bar(dates, r_ids, width=0.5, align='center', label=title)
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=90)
    plt.xlabel('Дата')
    plt.ylabel('Наиболее частый r_id')
    plt.title(f'Наиболее частые r_id в {title} периодах')
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.show()

plot_common_r_id([false_peak, false_low], [most_common_false_peak_r_id, most_common_false_low_r_id], "False")
plot_common_r_id([true_peak, true_low], [most_common_true_peak_r_id, most_common_true_low_r_id], "True")
