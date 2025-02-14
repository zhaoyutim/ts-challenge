import numpy as np
import h5py
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
import argparse
import backtrader as bt
import os


def train_models(X_np, y_np, target_dims, ctb_params):
    """
    Trains separate CatBoost models for each target specified in target_dims using provided hyperparameters.

    Parameters:
        X_np (np.ndarray): Features array with shape (samples, features).
        y_np (np.ndarray): Returns array with shape (samples, 4).
        target_dims (list): List of target indices to train.
        ctb_params (dict): Dictionary of training hyperparameters.

    Returns:
        list: A list of trained CatBoostRegressor models for the specified targets.
    """
    models = []
    for d in target_dims:
        print(f"Training model for target dimension {d}...")
        # Scale returns for the target
        y_target = y_np[:, d] * 100  
        model = CatBoostRegressor(**ctb_params)
        model.fit(X_np, y_target)
        model_save_path = f'models/catboost_model_{d}.cbm'
        model.save_model(model_save_path)
        print(f"Model for target dimension {d} saved to {model_save_path}")
        models.append(model)
    return models

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train CatBoost models for selected target dimension(s)."
    )
    parser.add_argument(
        "--dimension",
        type=str,
        default="all",
        help=("Target dimension to train/test. Use 'all' to train for all target dimensions, "
              "or specify a single dimension (e.g., '2') or multiple dimensions separated by commas (e.g., '0,2').")
    )
    args = parser.parse_args()

    # Determine which dimensions to use for training/testing.
    if args.dimension.lower() == "all":
        dims = list(range(4))
    else:
        # Allows specifying one or more dimensions as a comma-separated list.
        dims = [int(x.strip()) for x in args.dimension.split(",")]

    # Load training data.
    X_np = np.load("dataset/train/features.npy", allow_pickle=True)
    y_np = np.load("dataset/train/labels.npy", allow_pickle=True)

    print("Shape of X_np (features):", X_np.shape)
    print("Shape of y_np (returns):", y_np.shape)
    
    # Squeeze arrays if necessary (e.g., remove any singleton dimensions).
    X_train = np.squeeze(X_np)
    y_train = np.squeeze(y_np)
    
    # Define CatBoost hyperparameters (moved here so they can be logged later).
    ctb_params = {
        'iterations': 2000,
        'learning_rate': 0.1,
        'depth': 12,
        'l2_leaf_reg': 30,
        'bootstrap_type': 'Bernoulli',
        'subsample': 0.66,
        'loss_function': 'RMSE',
        'eval_metric': 'RMSE',
        'metric_period': 100,
        'od_type': 'Iter',
        'od_wait': 200,
        'task_type': 'GPU',
        'allow_writing_files': False,
        'random_strength': 4.428571428571429
    }
    
    # Train models for the selected target dimensions.
    models = train_models(X_train, y_train, dims, ctb_params)
    
    # Load test data.
    X_test = np.squeeze(np.load("dataset/test/features.npy", allow_pickle=True))
    y_test_full = np.squeeze(np.load("dataset/test/labels.npy", allow_pickle=True))
    
    # Select the appropriate target dimensions in the test set.
    # Ensure y_test is 2D for consistency.
    if len(dims) > 1:
        y_test = y_test_full[:, dims]
    else:
        y_test = y_test_full[:, dims[0]].reshape(-1, 1)

    # Generate predictions for each target using its respective model.
    predictions_list = []
    for idx, model in enumerate(models):
        print(f"Predicting for target dimension {dims[idx]}...")
        pred = model.predict(X_test)
        predictions_list.append(pred)
    
    # Combine predictions into a single array.
    predictions = np.column_stack(predictions_list)
    print("Combined predictions shape:", predictions.shape)
    
    # Calculate root mean squared error (RMSE)
    rmse_per_target = np.sqrt(np.mean((predictions - y_test * 100) ** 2, axis=0))
    overall_rmse = np.sqrt(mean_squared_error(predictions, y_test * 100))
    
    print("RMSE per target dimension:", rmse_per_target)
    print("Overall RMSE:", overall_rmse)
    
    # Save training parameters and testing results to a text file.
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    results_filepath = os.path.join(models_dir, "training_results.txt")
    
    with open(results_filepath, "w") as file:
        file.write("CatBoost Model Training Results\n")
        file.write("=================================\n\n")
        for idx, d in enumerate(dims):
            file.write(f"Target Dimension: {d}\n")
            file.write("Training Parameters:\n")
            for key, value in ctb_params.items():
                file.write(f"    {key}: {value}\n")
            file.write(f"Root Mean Squared Error (Test): {rmse_per_target[idx]}\n\n")
        file.write(f"Overall RMSE: {overall_rmse}\n")
    
    print(f"Training parameters and testing scores saved to {results_filepath}")