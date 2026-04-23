# Accessing Training History from Saved Keras Models

Keras models don't save training history automatically. Here's how to preserve and access it:

## Quick Solutions

### 1. Save History with Joblib (Recommended)

```python
# After training
history = model.fit(train_seq, validation_data=val_seq, epochs=10)
model.save('models/model.keras')
import joblib
joblib.dump(history.history, 'models/history.joblib')

# Later
model = keras.saving.load_model('models/model.keras')
history = joblib.load('models/history.joblib')
```

### 2. Auto-Save with Callback

```python
class HistorySaver(keras.callbacks.Callback):
    def on_train_end(self, logs=None):
        import joblib
        joblib.dump(self.model.history.history, 'models/history.joblib')

model.fit(..., callbacks=[HistorySaver()])
```

### 3. TensorBoard for Advanced Tracking

```python
model.fit(..., callbacks=[
    keras.callbacks.TensorBoard(log_dir='./logs')
])
# View with: tensorboard --logdir=./logs
```

## Best Practices

```python
# Complete workflow
import joblib
import json
from datetime import datetime

# Train
history = model.fit(...)

# Save everything
model.save('models/model.keras')
joblib.dump(history.history, 'models/history.joblib')

# Save metadata
metadata = {
    'date': str(datetime.now()),
    'final_accuracy': history.history['accuracy'][-1]
}
with open('models/metadata.json', 'w') as f:
    json.dump(metadata, f)
```

## Key Points

- **History is ephemeral** - must be saved explicitly
- **Joblib is preferred** over JSON for complex objects
- **Callbacks automate** the saving process
- **TensorBoard provides** visualization and analysis tools
- **Always save metadata** for complete reproducibility

## Troubleshooting

**Joblib errors**: Use same Python version for saving/loading
**Missing history**: Retrain with proper saving or use model without history
**Large files**: Consider TensorBoard for long training runs