# Accessing Training History from Saved Keras Models

When working with saved Keras models, you may need to access training history (loss and accuracy curves) after the model has been saved and reloaded. Here's how to handle this:

## The Challenge

By default, Keras models don't automatically save the full training history when you call `model.save()`. The history object returned by `model.fit()` contains valuable epoch-by-epoch metrics that are lost if not explicitly saved.

## Solutions

### 1. Save History Separately (Recommended)

The most reliable approach is to save the history alongside your model:

```python
# During training:
history = model.fit(train_seq, validation_data=val_seq, epochs=10)
model.save('models/CNN_3_classes.keras')

# Save history separately
import pickle
with open('models/CNN_3_classes_history.pkl', 'wb') as f:
    pickle.dump(history.history, f)

# Later, load both:
model = keras.saving.load_model('models/CNN_3_classes.keras')
with open('models/CNN_3_classes_history.pkl', 'rb') as f:
    history = pickle.load(f)

# Now you can access all training metrics:
print(f"Training accuracy per epoch: {history['accuracy']}")
print(f"Validation loss per epoch: {history['val_loss']}")
```

### 2. Use Callbacks to Save History

Create a custom callback that automatically saves history when training ends:

```python
class HistorySaver(keras.callbacks.Callback):
    def on_train_end(self, logs=None):
        import pickle
        with open('models/CNN_3_classes_history.pkl', 'wb') as f:
            pickle.dump(self.model.history.history, f)

# During training:
history = model.fit(..., callbacks=[HistorySaver()])
```

### 3. Limited History from Saved Model

You can get some information from a saved model, but not the full history:

```python
model = keras.saving.load_model('models/CNN_3_classes.keras')

# This only gives you final metrics, not epoch-by-epoch history
final_metrics = model.evaluate(val_seq)
print(f"Final validation accuracy: {final_metrics[1]:.4f}")
```

### 4. TensorBoard for Persistent Logging

For comprehensive training tracking:

```python
# During training:
tensorboard_callback = keras.callbacks.TensorBoard(
    log_dir='./logs',
    histogram_freq=1
)

history = model.fit(..., callbacks=[tensorboard_callback])

# Later, you can:
# 1. Load TensorBoard to visualize training curves
# 2. Parse the event files to extract metrics programmatically
```

## Best Practices

### Complete Training Workflow

```python
# Training
history = model.fit(train_seq, validation_data=val_seq, epochs=10)

# Save model
model.save('models/CNN_3_classes.keras')

# Save history
import pickle
with open('models/CNN_3_classes_history.pkl', 'wb') as f:
    pickle.dump(history.history, f)

# Save metadata
import json
from datetime import datetime

metadata = {
    'training_date': str(datetime.now()),
    'epochs': 10,
    'final_train_acc': history.history['accuracy'][-1],
    'final_val_acc': history.history['val_accuracy'][-1],
    'model_architecture': model.get_config(),
    'optimizer': str(model.optimizer.get_config())
}

with open('models/CNN_3_classes_metadata.json', 'w') as f:
    json.dump(metadata, f)
```

### Loading and Analyzing Later

```python
# Load everything
model = keras.saving.load_model('models/CNN_3_classes.keras')

with open('models/CNN_3_classes_history.pkl', 'rb') as f:
    history = pickle.load(f)

with open('models/CNN_3_classes_metadata.json', 'r') as f:
    metadata = json.load(f)

# Plot training curves
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history['accuracy'], label='Train')
plt.plot(history['val_accuracy'], label='Validation')
plt.title('Accuracy over epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history['loss'], label='Train')
plt.plot(history['val_loss'], label='Validation')
plt.title('Loss over epochs')
plt.legend()
plt.show()
```

## Advanced Options

### Using MLflow for Experiment Tracking

```python
import mlflow
import mlflow.keras

# Start MLflow run
with mlflow.start_run():
    # Log parameters
    mlflow.log_param("epochs", 10)
    mlflow.log_param("batch_size", 32)
    
    # Train model
    history = model.fit(...)
    
    # Log metrics
    for epoch, (acc, val_acc, loss, val_loss) in enumerate(zip(
        history.history['accuracy'], 
        history.history['val_accuracy'],
        history.history['loss'],
        history.history['val_loss']
    )):
        mlflow.log_metric("accuracy", acc, step=epoch)
        mlflow.log_metric("val_accuracy", val_acc, step=epoch)
        mlflow.log_metric("loss", loss, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)
    
    # Log model
    mlflow.keras.log_model(model, "model")
```

### Using Weights & Biases

```python
import wandb
from wandb.keras import WandbCallback

# Initialize W&B
wandb.init(project="dog-breed-classification", config={
    "epochs": 10,
    "batch_size": 32,
    "optimizer": "adam"
})

# Train with W&B callback
history = model.fit(
    ...,
    callbacks=[WandbCallback()]
)

# W&B automatically logs:
# - Training and validation metrics
# - System metrics (CPU, GPU, memory)
# - Model architecture
# - Hyperparameters
```

## Important Notes

1. **History is ephemeral**: If you don't save it during training, you cannot recover it later
2. **Multiple formats**: You can save history as pickle, JSON, or use specialized tools
3. **Reproducibility**: Saving history alongside models ensures you can analyze training dynamics later
4. **Storage considerations**: History files are typically small but can grow with many epochs
5. **Versioning**: Consider versioning your models and their corresponding history files together

## Troubleshooting

### "History object not found"

If you get this error when trying to load history, it means the history wasn't saved. You'll need to:
1. Retrain the model with history saving enabled, or
2. Use the model as-is but without access to training curves

### Pickle loading errors

If you have issues loading pickled history:
- Ensure you're using the same Python version
- Try saving as JSON instead: `json.dump(history.history, f)`
- Consider using joblib for more robust serialization

### Memory issues with long training

For very long training runs:
- Save history periodically using callbacks
- Consider downsampling (save every N epochs)
- Use streaming solutions like TensorBoard or W&B

By following these approaches, you can ensure that your training history is preserved and accessible for analysis, even after the training process is complete.