# ============================================================
#  MNIST Handwritten Digit Recognizer
#  Keras 3 + Torch Backend + Robust Visualization
# ============================================================

import os
import datetime
# Force Keras to use the Torch backend (MUST be done first)
os.environ["KERAS_BACKEND"] = "torch"

import keras
import numpy as np
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

# Make matplotlib optional
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# ─────────────────────────────────────────
# STEP 1 — Load dataset
# ─────────────────────────────────────────
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 784).astype("float32") / 255.0
x_test  = x_test.reshape(-1, 784).astype("float32")  / 255.0

# ─────────────────────────────────────────
# STEP 2 — Build the network
# ─────────────────────────────────────────
model = keras.Sequential([
    keras.layers.Input(shape=(784,)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ─────────────────────────────────────────
# STEP 3 — Smart TensorBoard Callback (Bypasses Blocks)
# ─────────────────────────────────────────
class SmartVisualizer(keras.callbacks.Callback):
    def __init__(self, log_dir):
        super().__init__()
        self.writer = SummaryWriter(log_dir)
        print(f"\n[INFO] Visual logs saving to: {log_dir}")

    def on_epoch_end(self, epoch, logs=None):
        if logs:
            for name, value in logs.items():
                self.writer.add_scalar(name, value, epoch)
        self.writer.flush()

    def on_train_end(self, logs=None):
        self.writer.close()

log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tb_callback = SmartVisualizer(log_dir)

# ─────────────────────────────────────────
# STEP 4 — Train
# ─────────────────────────────────────────
print("\nStarting Training with Smart Visualization...")
model.fit(
    x_train, y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    callbacks=[tb_callback],
    verbose=1
)

# ─────────────────────────────────────────
# STEP 5 — Final evaluation
# ─────────────────────────────────────────
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"\nFinal Test Accuracy: {test_accuracy * 100:.2f}%")

# ─────────────────────────────────────────
# STEP 6 — Predictions with Pillow
# ─────────────────────────────────────────
predictions = model.predict(x_test[:5])

def save_prediction_samples(x_data, y_true, y_pred, count=5):
    canvas_width = count * 100
    canvas_height = 120
    combined = Image.new('RGB', (canvas_width, canvas_height), (255, 255, 255))
    for i in range(count):
        img_array = (x_data[i].reshape(28, 28) * 255).astype(np.uint8)
        img = Image.fromarray(img_array).convert('RGB').resize((80, 80))
        combined.paste(img, (i * 100 + 10, 10))
        print(f"Sample {i+1}: Predicted={np.argmax(y_pred[i])}, Actual={y_true[i]}")
    combined.save('predictions_sample.png')
    print("\n[✔] Prediction images saved as predictions_sample.png")
    
    # Save the model for Netron visualization
    model.save('mnist_model.keras')
    print("[✔] Model saved as 'mnist_model.keras' for Netron.app visualization!")

save_prediction_samples(x_test, y_test, predictions)
