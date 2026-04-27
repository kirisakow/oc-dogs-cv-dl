from typing import Dict, Any, Optional
import keras
from keras import layers, optimizers
from sklearn.model_selection import ParameterGrid
import numpy as np
from build_model_from_scratch import MyKerasSequence


def create_model_from_params(params: Dict[str, Any], n_classes: int, data_augm: Optional[keras.Sequential] = None) -> keras.Model:
    """
    Create a model based on grid search parameters.
    
    Args:
        params: Dictionary of parameters including:
            - n_conv_layers: Number of convolutional layers
            - conv_filters: List of filters for each conv layer
            - conv_kernel_size: Kernel size for conv layers
            - dense_units: Number of units in dense layer
            - dropout_rate: Dropout rate (None to disable)
            - learning_rate: Learning rate for optimizer
        n_classes: Number of output classes
        data_augm: Optional data augmentation layer
    
    Returns:
        Compiled Keras model
    """
    inputs = keras.Input(shape=(params.get('img_size', 224), params.get('img_size', 224), 3))
    
    if data_augm:
        x = data_augm(inputs)
    
    x = layers.Rescaling(1./255)(inputs if not data_augm else x)
    
    # Add convolutional layers
    for i in range(params.get('n_conv_layers', 2)):
        filters = params.get('conv_filters', [32, 64])[i] if i < len(params.get('conv_filters', [32, 64])) else 64
        kernel_size = params.get('conv_kernel_size', 5)
        
        x = layers.Conv2D(filters, kernel_size, padding='same', activation='relu')(x)
        x = layers.MaxPooling2D(2)(x)
    
    # Flatten and add dense layers
    x = layers.Flatten()(x)
    x = layers.Dense(params.get('dense_units', 512), activation='relu')(x)
    
    # Add dropout if specified
    if params.get('dropout_rate', 0.0) > 0:
        x = layers.Dropout(params.get('dropout_rate'))(x)
    
    outputs = layers.Dense(n_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    
    # Compile with specified learning rate
    optimizer = optimizers.Adam(learning_rate=params.get('learning_rate', 0.001))
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def grid_search_models(
    param_grid: Dict[str, Any],
    train_data: MyKerasSequence,
    val_data: MyKerasSequence,
    n_classes: int,
    data_augm: Optional[keras.Sequential] = None,
    verbose: int = 1
) -> Dict[str, Any]:
    """
    Perform grid search over model hyperparameters.
    
    Args:
        param_grid: Dictionary of parameters to search
        train_data: Training data sequence
        val_data: Validation data sequence
        n_classes: Number of classes
        data_augm: Optional data augmentation
        verbose: Verbosity level
    
    Returns:
        Dictionary with best parameters and results
    """
    best_score = -1
    best_params = {}
    best_history = None
    results = []
    
    # Generate all parameter combinations
    param_combinations = list(ParameterGrid(param_grid))
    
    print(f"Testing {len(param_combinations)} parameter combinations...")
    
    for i, params in enumerate(param_combinations):
        if verbose:
            print(f"\nTesting combination {i+1}/{len(param_combinations)}: {params}")
        
        # Create and train model
        model = create_model_from_params(params, n_classes, data_augm)
        
        history = model.fit(
            train_data,
            validation_data=val_data,
            epochs=params.get('epochs', 10),
            batch_size=params.get('batch_size', 32),
            verbose=verbose-1 if verbose > 0 else 0
        )
        
        # Evaluate on validation set
        val_loss, val_acc = model.evaluate(val_data, verbose=0)
        
        result = {
            'params': params,
            'val_loss': val_loss,
            'val_accuracy': val_acc,
            'history': history.history
        }
        results.append(result)
        
        if verbose:
            print(f"Val accuracy: {val_acc:.4f}, Val loss: {val_loss:.4f}")
        
        # Track best model
        if val_acc > best_score:
            best_score = val_acc
            best_params = params
            best_history = history
    
    return {
        'best_params': best_params,
        'best_score': best_score,
        'best_history': best_history.history,
        'all_results': results
    }


# Example usage:
if __name__ == "__main__":
    # Define parameter grid
    param_grid = {
        'n_conv_layers': [2, 3],
        'conv_filters': [[32, 64], [32, 64, 128]],
        'conv_kernel_size': [3, 5],
        'dense_units': [256, 512],
        'dropout_rate': [0.0, 0.15, 0.3],
        'learning_rate': [0.001, 0.0001],
        'batch_size': [16, 32],
        'epochs': [5, 10]
    }
    
    # Assuming you have train_seq and val_seq from MyKerasSequence
    # results = grid_search_models(
    #     param_grid=param_grid,
    #     train_data=train_seq,
    #     val_data=val_seq,
    #     n_classes=3
    # )
    
    # print(f"Best parameters: {results['best_params']}")
    # print(f"Best validation accuracy: {results['best_score']:.4f}")