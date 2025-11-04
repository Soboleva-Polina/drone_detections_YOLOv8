import numpy as np
import matplotlib.pyplot as plt

# Number of epochs for the training
epochs = np.arange(0, 101)

# Generate synthetic data for Yolov7
yolov7_precision = np.random.uniform(0.5, 0.9, size=epochs.shape)
yolov7_recall = np.random.uniform(0.5, 0.9, size=epochs.shape)
yolov7_map50 = np.random.uniform(0.4, 0.8, size=epochs.shape)
yolov7_box_loss = np.exp(-0.03 * epochs) + np.random.normal(0, 0.05, size=epochs.shape)
yolov7_cls_loss = np.exp(-0.03 * epochs) + np.random.normal(0, 0.05, size=epochs.shape)
yolov7_dfl_loss = np.exp(-0.02 * epochs) + np.random.normal(0, 0.05, size=epochs.shape)

# Generate synthetic data for Yolov8
yolov8_precision = np.random.uniform(0.7, 1.0, size=epochs.shape)
yolov8_recall = np.random.uniform(0.7, 1.0, size=epochs.shape)
yolov8_map50 = np.random.uniform(0.6, 0.9, size=epochs.shape)
yolov8_box_loss = np.exp(-0.05 * epochs) + np.random.normal(0, 0.05, size=epochs.shape)
yolov8_cls_loss = np.exp(-0.05 * epochs) + np.random.normal(0, 0.05, size=epochs.shape)
yolov8_dfl_loss = np.exp(-0.04 * epochs) + np.random.normal(0, 0.05, size=epochs.shape)

# Learning rate schedule (same for both models)
lr_schedule = np.maximum(0.0001, np.exp(-0.01 * (epochs - 50) ** 2))

# Create subplots
fig, axs = plt.subplots(2, 2, figsize=(15, 10))

# Detection Metrics
axs[0, 0].plot(epochs, yolov7_precision, label="Yolov7 Precision", color='blue')
axs[0, 0].plot(epochs, yolov7_recall, label="Yolov7 Recall", color='orange')
axs[0, 0].plot(epochs, yolov7_map50, label="Yolov7 mAP50", color='green')
axs[0, 0].plot(epochs, yolov8_precision, label="Yolov8 Precision", color='blue', linestyle='--')
axs[0, 0].plot(epochs, yolov8_recall, label="Yolov8 Recall", color='orange', linestyle='--')
axs[0, 0].plot(epochs, yolov8_map50, label="Yolov8 mAP50", color='green', linestyle='--')
axs[0, 0].set_title("Detection Metrics")
axs[0, 0].set_xlabel("Epochs")
axs[0, 0].set_ylabel("Percentage (%)")
axs[0, 0].legend()

# Learning Rate Schedule
axs[0, 1].plot(epochs, lr_schedule, label="Learning Rate", color='purple')
axs[0, 1].set_title("Learning Rate Schedule")
axs[0, 1].set_xlabel("Epochs")
axs[0, 1].set_ylabel("Learning Rate")
axs[0, 1].legend()

# Training Loss Components
axs[1, 0].plot(epochs, yolov7_box_loss, label="Yolov7 Box Loss", color='blue')
axs[1, 0].plot(epochs, yolov7_cls_loss, label="Yolov7 Class Loss", color='orange')
axs[1, 0].plot(epochs, yolov7_dfl_loss, label="Yolov7 DFL Loss", color='green')
axs[1, 0].plot(epochs, yolov8_box_loss, label="Yolov8 Box Loss", color='blue', linestyle='--')
axs[1, 0].plot(epochs, yolov8_cls_loss, label="Yolov8 Class Loss", color='orange', linestyle='--')
axs[1, 0].plot(epochs, yolov8_dfl_loss, label="Yolov8 DFL Loss", color='green', linestyle='--')
axs[1, 0].set_title("Training Loss Components")
axs[1, 0].set_xlabel("Epochs")
axs[1, 0].set_ylabel("Loss")
axs[1, 0].legend()

# Validation Loss Components
axs[1, 1].plot(epochs, yolov7_box_loss * 1.5, label="Yolov7 Val Box Loss", color='blue')
axs[1, 1].plot(epochs, yolov7_cls_loss * 1.5, label="Yolov7 Val Class Loss", color='orange')
axs[1, 1].plot(epochs, yolov7_dfl_loss * 1.5, label="Yolov7 Val DFL Loss", color='green')
axs[1, 1].plot(epochs, yolov8_box_loss * 1.2, label="Yolov8 Val Box Loss", color='blue', linestyle='--')
axs[1, 1].plot(epochs, yolov8_cls_loss * 1.2, label="Yolov8 Val Class Loss", color='orange', linestyle='--')
axs[1, 1].plot(epochs, yolov8_dfl_loss * 1.2, label="Yolov8 Val DFL Loss", color='green', linestyle='--')
axs[1, 1].set_title("Validation Loss Components")
axs[1, 1].set_xlabel("Epochs")
axs[1, 1].set_ylabel("Loss")
axs[1, 1].legend()

# Show the plots
plt.tight_layout()
plt.show()