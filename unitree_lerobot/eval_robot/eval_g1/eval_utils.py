import torch
import numpy as np
import json


def save_eval_results(results, output_file):
    """
    Save evaluation results to JSON file.
    
    Args:
        results: Dictionary containing evaluation results
        output_file: Path to output JSON file
    """
    
    # Convert numpy arrays and numpy scalars to JSON serializable types
    def convert_numpy_types(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj
    
    serializable_results = {}
    for key, value in results.items():
        serializable_results[key] = convert_numpy_types(value)
    
    with open(output_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    print(f"Results saved to {output_file}")


def calculate_eval_metrics(ground_truth_actions, predicted_actions):
    """
    Calculate standardized evaluation metrics for fair comparison.
    Use consistent metrics regardless of model's training loss function.
    
    Args:
        ground_truth_actions: Ground truth action array
        predicted_actions: Predicted action array
        
    Returns:
        Dictionary containing evaluation metrics
    """
    # Convert to tensors for loss calculations
    gt_actions_tensor = torch.from_numpy(ground_truth_actions).float()
    pred_actions_tensor = torch.from_numpy(predicted_actions).float()
    
    # Standard evaluation metrics (same for all models)
    mse_loss = torch.nn.functional.mse_loss(pred_actions_tensor, gt_actions_tensor)
    l1_loss = torch.nn.functional.l1_loss(pred_actions_tensor, gt_actions_tensor)
    huber_loss = torch.nn.functional.huber_loss(pred_actions_tensor, gt_actions_tensor)
    
    # Return evaluation results with standardized metrics
    return {
        'ground_truth_actions': ground_truth_actions,
        'predicted_actions': predicted_actions,
        # Standardized evaluation losses (fair comparison)
        'eval_mse_loss': mse_loss.item(),
        'eval_l1_loss': l1_loss.item(), 
        'eval_huber_loss': huber_loss.item(),
        # Per-dimension metrics
        'mse_per_dim': np.mean((ground_truth_actions - predicted_actions) ** 2, axis=0),
        'mae_per_dim': np.mean(np.abs(ground_truth_actions - predicted_actions), axis=0),
        # Legacy metrics (for backward compatibility)
        'total_mse': np.mean((ground_truth_actions - predicted_actions) ** 2),
        'total_mae': np.mean(np.abs(ground_truth_actions - predicted_actions))
    }
