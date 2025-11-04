import os  # ИМпортируем модуль для взаимодействия с операционной системой папки/файлы
import pandas as pd  # Библиотека для работы с Excel
import matplotlib.pyplot as plt  # Библиотека построения графиков на основе Execl

# raw string позволяет игнорировать экранирование 
results_path = r'C:\Users\User\Desktop\classification_image_yoloV8\runs\detect\train3\results.csv'

# Чтение CSV файла по указанному пути 
results = pd.read_csv(results_path)

# Название колонок 
# Колонки возвращаются с индексом .column
# Колонки парсим в понятный формат для python .tolist() - список 
print("Доступные колонки:")
print(results.columns.tolist())

# Почистим названия колонок (убрать лишние пробелы)
results.columns = results.columns.str.strip()

# Посмотрим на очищенные названия
print("\nОчищенные колонки:")
print(results.columns.tolist())

# Создаем графики
# Создаем новую фигуру окно для графиков размером 15 дюймов ширина и 5 дюймов высота
plt.figure(figsize=(15, 5))

# График 1: Loss functions
plt.subplot(1, 2, 1)  # Построение сетки из 1 и 2 столбца 
# Проверка на существование колонки 
# Построение линейного графика x y label linewidth color='red'
if 'epoch' in results.columns and 'train/box_loss' in results.columns:
    plt.plot(results['epoch'], results['train/box_loss'], label='Train Box Loss', linewidth=2)
    plt.plot(results['epoch'], results['val/box_loss'], label='Val Box Loss', linewidth=2, color='red')
    plt.grid(True, alpha=0.3)
    plt.title('Box Loss vs Epochs')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend()
else:
    print("Колонки для графика loss не найдены")

# График 2: Metrics
plt.subplot(1, 2, 2)
if 'epoch' in results.columns and 'metrics/mAP50(B)' in results.columns:
    plt.plot(results['epoch'], results['metrics/mAP50(B)'] * 100, label='mAP50', linewidth=2, color='green')
    plt.plot(results['epoch'], results['metrics/mAP50-95(B)'] * 100, label='mAP50-95', linewidth=2, color='orange')
    plt.grid(True, alpha=0.3)  # Прозрачность сетки от 0 до 1
    plt.title('Validation Metrics vs Epochs') # Название таблицы
    plt.ylabel('Accuracy (%)') # Обозначение по оси х
    plt.xlabel('Epochs') # Обозначение по оси y
    plt.legend() # Отражение графа 
else:
    print("Колонки для графика метрик не найдены")

plt.tight_layout() # Перекрытие рамок 
plt.show() # Вывод графов после успешного построения на экран

# Дополнительные графики
plt.figure(figsize=(15, 10))

# График 3: Все метрики
plt.subplot(2, 2, 1)
if all(col in results.columns for col in ['metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)']):
    plt.plot(results['epoch'], results['metrics/precision(B)'] * 100, label='Precision', linewidth=2)
    plt.plot(results['epoch'], results['metrics/recall(B)'] * 100, label='Recall', linewidth=2)
    plt.plot(results['epoch'], results['metrics/mAP50(B)'] * 100, label='mAP50', linewidth=2)
    plt.grid(True, alpha=0.3)
    plt.title('Detection Metrics')
    plt.ylabel('Percentage (%)')
    plt.xlabel('Epochs')
    plt.legend()

# График 4: Learning rate
plt.subplot(2, 2, 2)
if 'lr/pg0' in results.columns:
    plt.plot(results['epoch'], results['lr/pg0'], label='Learning Rate', linewidth=2, color='purple')
    plt.grid(True, alpha=0.3)
    plt.title('Learning Rate Schedule')
    plt.ylabel('Learning Rate')
    plt.xlabel('Epochs')
    plt.legend()

# График 5: Все потери
plt.subplot(2, 2, 3)
if all(col in results.columns for col in ['train/box_loss', 'train/cls_loss', 'train/dfl_loss']):
    plt.plot(results['epoch'], results['train/box_loss'], label='Box Loss', linewidth=2)
    plt.plot(results['epoch'], results['train/cls_loss'], label='Cls Loss', linewidth=2)
    plt.plot(results['epoch'], results['train/dfl_loss'], label='DFL Loss', linewidth=2)
    plt.grid(True, alpha=0.3)
    plt.title('Training Loss Components')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend()

# График 6: Валидационные потери
plt.subplot(2, 2, 4)
if all(col in results.columns for col in ['val/box_loss', 'val/cls_loss', 'val/dfl_loss']):
    plt.plot(results['epoch'], results['val/box_loss'], label='Val Box Loss', linewidth=2)
    plt.plot(results['epoch'], results['val/cls_loss'], label='Val Cls Loss', linewidth=2)
    plt.plot(results['epoch'], results['val/dfl_loss'], label='Val DFL Loss', linewidth=2)
    plt.grid(True, alpha=0.3)
    plt.title('Validation Loss Components')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend()

plt.tight_layout()
plt.show()

# Вывод статистики
print("\n📊 Статистика обучения:")
print(f"Количество эпох: {len(results)}")
if 'metrics/mAP50(B)' in results.columns:
    print(f"Лучшая mAP50: {results['metrics/mAP50(B)'].max():.4f}")
if 'metrics/precision(B)' in results.columns:
    print(f"Лучшая Precision: {results['metrics/precision(B)'].max():.4f}")
