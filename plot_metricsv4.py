import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Функция для загрузки и подготовки данных
def load_results(model_path, model_name):
    """Загружает результаты CSV и добавляет информацию о модели"""
    try:
        results = pd.read_csv(model_path)
        results.columns = results.columns.str.strip()
        results['model'] = model_name
        return results
    except Exception as e:
        print(f"Ошибка загрузки {model_name}: {e}")
        return None

# Пути к результатам обеих моделей
yolov8_path = r'C:\Users\User\Desktop\classification_image_yoloV8\runs\detect\train3\results.csv'
yolov7_path = r'C:\Users\User\Desktop\classification_image_yoloV7\runs\detect\train\results.csv'  # Обнови путь для YOLOv7

# Загрузка данных обеих моделей
yolov8_results = load_results(yolov8_path, 'YOLOv8-n')
yolov7_results = load_results(yolov7_path, 'YOLOv7')

# Объединение данных
all_results = []
if yolov8_results is not None:
    all_results.append(yolov8_results)
if yolov7_results is not None:
    all_results.append(yolov7_results)

if not all_results:
    print("Нет данных для построения графиков!")
    exit()

combined_results = pd.concat(all_results, ignore_index=True)

# Создаем цветовую схему для моделей
colors = {'YOLOv8-n': 'blue', 'YOLOv7': 'red'}
line_styles = {'YOLOv8-n': '-', 'YOLOv7': '--'}

print("Доступные колонки в данных:")
for model in combined_results['model'].unique():
    model_data = combined_results[combined_results['model'] == model]
    print(f"\n{model}: {model_data.columns.tolist()}")

# ГРАФИК 1: Сравнение функций потерь (Loss functions)
plt.figure(figsize=(16, 6))

# График 1.1: Box Loss
plt.subplot(1, 2, 1)
for model in combined_results['model'].unique():
    model_data = combined_results[combined_results['model'] == model]
    if 'epoch' in model_data.columns and 'train/box_loss' in model_data.columns:
        plt.plot(model_data['epoch'], model_data['train/box_loss'], 
                label=f'{model} Train Box Loss', 
                linewidth=2, 
                color=colors[model],
                linestyle=line_styles[model])
    if 'epoch' in model_data.columns and 'val/box_loss' in model_data.columns:
        plt.plot(model_data['epoch'], model_data['val/box_loss'], 
                label=f'{model} Val Box Loss', 
                linewidth=2, 
                color=colors[model],
                linestyle=line_styles[model],
                alpha=0.7)

plt.grid(True, alpha=0.3)
plt.title('Сравнение Box Loss: YOLOv8-n vs YOLOv7')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# График 1.2: Общие потери
plt.subplot(1, 2, 2)
for model in combined_results['model'].unique():
    model_data = combined_results[combined_results['model'] == model]
    if 'epoch' in model_data.columns and 'train/cls_loss' in model_data.columns:
        plt.plot(model_data['epoch'], model_data['train/cls_loss'], 
                label=f'{model} Cls Loss', 
                linewidth=2, 
                color=colors[model],
                linestyle=line_styles[model])

plt.grid(True, alpha=0.3)
plt.title('Сравнение Classification Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()

# ГРАФИК 2: Сравнение метрик
plt.figure(figsize=(16, 6))

# График 2.1: mAP50
plt.subplot(1, 2, 1)
for model in combined_results['model'].unique():
    model_data = combined_results[combined_results['model'] == model]
    # Для YOLOv8
    if 'metrics/mAP50(B)' in model_data.columns:
        plt.plot(model_data['epoch'], model_data['metrics/mAP50(B)'] * 100, 
                label=f'{model} mAP50', 
                linewidth=2, 
                color=colors[model],
                linestyle=line_styles[model])
    # Для YOLOv7 (возможные названия колонок)
    elif 'mAP@0.5' in model_data.columns:
        plt.plot(model_data['epoch'], model_data['mAP@0.5'] * 100, 
                label=f'{model} mAP50', 
                linewidth=2, 
                color=colors[model],
                linestyle=line_styles[model])

plt.grid(True, alpha=0.3)
plt.title('Сравнение mAP50: YOLOv8-n vs YOLOv7')
plt.ylabel('mAP50 (%)')
plt.xlabel('Epochs')
plt.legend()

# График 2.2: Precision и Recall
plt.subplot(1, 2, 2)
for model in combined_results['model'].unique():
    model_data = combined_results[combined_results['model'] == model]
    # Precision
    if 'metrics/precision(B)' in model_data.columns:
        plt.plot(model_data['epoch'], model_data['metrics/precision(B)'] * 100, 
                label=f'{model} Precision', 
                linewidth=2, 
                color=colors[model],
                linestyle=line_styles[model])
    elif 'precision' in model_data.columns:
        plt.plot(model_data['epoch'], model_data['precision'] * 100, 
                label=f'{model} Precision', 
                linewidth=2, 
                color=colors[model],
                linestyle=line_styles[model])
    
    # Recall
    if 'metrics/recall(B)' in model_data.columns:
        plt.plot(model_data['epoch'], model_data['metrics/recall(B)'] * 100, 
                label=f'{model} Recall', 
                linewidth=2, 
                color=colors[model],
                linestyle=line_styles[model],
                alpha=0.7)
    elif 'recall' in model_data.columns:
        plt.plot(model_data['epoch'], model_data['recall'] * 100, 
                label=f'{model} Recall', 
                linewidth=2, 
                color=colors[model],
                linestyle=line_styles[model],
                alpha=0.7)

plt.grid(True, alpha=0.3)
plt.title('Сравнение Precision и Recall')
plt.ylabel('Percentage (%)')
plt.xlabel('Epochs')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()

# ГРАФИК 3: Детальное сравнение всех метрик
plt.figure(figsize=(15, 10))

# Список метрик для сравнения
metrics_to_plot = [
    ('metrics/mAP50(B)', 'mAP50', 'mAP@0.5'),
    ('metrics/mAP50-95(B)', 'mAP50-95', 'mAP@0.5:0.95'),
    ('metrics/precision(B)', 'Precision', 'precision'),
    ('metrics/recall(B)', 'Recall', 'recall')
]

for i, (metric_v8, metric_name, metric_v7) in enumerate(metrics_to_plot, 1):
    plt.subplot(2, 2, i)
    
    for model in combined_results['model'].unique():
        model_data = combined_results[combined_results['model'] == model]
        
        # Определяем правильное название колонки для каждой модели
        if model == 'YOLOv8-n' and metric_v8 in model_data.columns:
            values = model_data[metric_v8] * 100
            plt.plot(model_data['epoch'], values, 
                    label=model, linewidth=2, 
                    color=colors[model], linestyle=line_styles[model])
        
        elif model == 'YOLOv7' and metric_v7 in model_data.columns:
            values = model_data[metric_v7] * 100
            plt.plot(model_data['epoch'], values, 
                    label=model, linewidth=2, 
                    color=colors[model], linestyle=line_styles[model])
    
    plt.grid(True, alpha=0.3)
    plt.title(f'Сравнение {metric_name}')
    plt.ylabel(f'{metric_name} (%)')
    plt.xlabel('Epochs')
    plt.legend()

plt.tight_layout()
plt.show()

# АНАЛИЗ РЕЗУЛЬТАТОВ
print("\n" + "="*60)
print("📊 ДЕТАЛЬНЫЙ АНАЛИЗ МОДЕЛЕЙ")
print("="*60)

for model in combined_results['model'].unique():
    model_data = combined_results[combined_results['model'] == model]
    print(f"\n🔍 Анализ {model}:")
    print(f"Количество эпох: {len(model_data)}")
    
    # Анализ метрик для YOLOv8
    if model == 'YOLOv8-n':
        if 'metrics/mAP50(B)' in model_data.columns:
            best_map50 = model_data['metrics/mAP50(B)'].max()
            final_map50 = model_data['metrics/mAP50(B)'].iloc[-1]
            print(f"mAP50: Лучшая = {best_map50:.4f}, Финальная = {final_map50:.4f}")
        
        if 'metrics/mAP50-95(B)' in model_data.columns:
            best_map9595 = model_data['metrics/mAP50-95(B)'].max()
            print(f"mAP50-95: Лучшая = {best_map9595:.4f}")
        
        if 'metrics/precision(B)' in model_data.columns:
            best_precision = model_data['metrics/precision(B)'].max()
            print(f"Precision: Лучшая = {best_precision:.4f}")
    
    # Анализ метрик для YOLOv7
    elif model == 'YOLOv7':
        if 'mAP@0.5' in model_data.columns:
            best_map50 = model_data['mAP@0.5'].max()
            final_map50 = model_data['mAP@0.5'].iloc[-1]
            print(f"mAP50: Лучшая = {best_map50:.4f}, Финальная = {final_map50:.4f}")
        
        if 'mAP@0.5:0.95' in model_data.columns:
            best_map9595 = model_data['mAP@0.5:0.95'].max()
            print(f"mAP50-95: Лучшая = {best_map9595:.4f}")
        
        if 'precision' in model_data.columns:
            best_precision = model_data['precision'].max()
            print(f"Precision: Лучшая = {best_precision:.4f}")

# Сравнительный анализ
print("\n" + "="*60)
print("🎯 СРАВНИТЕЛЬНЫЙ АНАЛИЗ")
print("="*60)

if len(combined_results['model'].unique()) == 2:
    # Сравнение конечных результатов
    v8_final = combined_results[combined_results['model'] == 'YOLOv8-n'].iloc[-1]
    v7_final = combined_results[combined_results['model'] == 'YOLOv7'].iloc[-1]
    
    print("\nФинальные результаты:")
    
    # Сравнение mAP50
    if 'metrics/mAP50(B)' in v8_final and 'mAP@0.5' in v7_final:
        v8_map = v8_final['metrics/mAP50(B)'] * 100
        v7_map = v7_final['mAP@0.5'] * 100
        diff = v8_map - v7_map
        print(f"mAP50: YOLOv8-n = {v8_map:.1f}%, YOLOv7 = {v7_map:.1f}%, Разница = {diff:+.1f}%")
    
    # Сравнение потерь
    if 'val/box_loss' in v8_final and 'val/box_loss' in v7_final:
        v8_loss = v8_final['val/box_loss']
        v7_loss = v7_final['val/box_loss']
        print(f"Val Box Loss: YOLOv8-n = {v8_loss:.4f}, YOLOv7 = {v7_loss:.4f}")

print("\n💡 РЕКОМЕНДАЦИИ:")
print("• YOLOv8-n: Лучше для embedded устройств, быстрее инференс")
print("• YOLOv7: Лучшая точность, но требует больше ресурсов")
print("• Выбор зависит от требований к точности и скорости")