# utils/metrics.py - Adversarial Robustness Metrics

import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips
from typing import Dict, List, Tuple
import warnings

# Initialize LPIPS model (do this once globally to avoid reloading)
# At the top of the file
_lpips_model = None
_lpips_available = True  # Track if LPIPS is working

def get_lpips_model(device='cpu'):
    """Get or initialize LPIPS model"""
    global _lpips_model, _lpips_available
    
    if not _lpips_available:
        return None
    
    if _lpips_model is None:
        try:
            _lpips_model = lpips.LPIPS(net='alex').to(device)
            _lpips_model.eval()
        except Exception as e:
            print(f"Warning: LPIPS model failed to load: {e}")
            print("LPIPS metric will be skipped. This is usually due to SSL certificate issues.")
            _lpips_available = False
            return None
    
    return _lpips_model


def calculate_lpips(clean_img: torch.Tensor, adv_img: torch.Tensor, device='cpu') -> float:
    """
    Calculate Learned Perceptual Image Patch Similarity (LPIPS)
    Lower values indicate more perceptually similar images (range: 0 to 1)
    """
    lpips_model = get_lpips_model(device)
    
    # If LPIPS is not available, return NaN or a default value
    if lpips_model is None:
        return float('nan')
    
    # LPIPS expects images in range [-1, 1], we have [0, 1]
    clean_normalized = clean_img.unsqueeze(0).to(device) * 2 - 1
    adv_normalized = adv_img.unsqueeze(0).to(device) * 2 - 1
    
    with torch.no_grad():
        lpips_val = lpips_model(clean_normalized, adv_normalized)
    
    return lpips_val.item()




def calculate_l2_norm(clean_img: torch.Tensor, adv_img: torch.Tensor) -> float:
    """
    Calculate L2 (Euclidean) norm of perturbation
    
    Args:
        clean_img: Clean image tensor
        adv_img: Adversarial image tensor
    
    Returns:
        L2 norm value
    """
    diff = (adv_img - clean_img).flatten()
    return torch.norm(diff, p=2).item()


def calculate_linf_norm(clean_img: torch.Tensor, adv_img: torch.Tensor) -> float:
    """
    Calculate L∞ (maximum) norm of perturbation
    
    Args:
        clean_img: Clean image tensor
        adv_img: Adversarial image tensor
    
    Returns:
        L∞ norm value
    """
    diff = torch.abs(adv_img - clean_img)
    return torch.max(diff).item()


def calculate_l0_norm(clean_img: torch.Tensor, adv_img: torch.Tensor, threshold: float = 1e-6) -> int:
    """
    Calculate L0 norm (number of pixels changed)
    
    Args:
        clean_img: Clean image tensor
        adv_img: Adversarial image tensor
        threshold: Minimum change to count as modified pixel
    
    Returns:
        Number of changed pixels
    """
    diff = torch.abs(adv_img - clean_img)
    changed_pixels = (diff > threshold).float().sum().item()
    return int(changed_pixels)


def calculate_ssim(clean_img: torch.Tensor, adv_img: torch.Tensor) -> float:
    """
    Calculate Structural Similarity Index (SSIM)
    Higher values indicate more similar images (range: -1 to 1, typically 0 to 1)
    
    Args:
        clean_img: Clean image tensor (C, H, W)
        adv_img: Adversarial image tensor (C, H, W)
    
    Returns:
        SSIM score
    """
    # Convert to numpy and transpose to (H, W, C)
    clean_np = clean_img.cpu().numpy().transpose(1, 2, 0)
    adv_np = adv_img.cpu().numpy().transpose(1, 2, 0)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ssim_val = ssim(clean_np, adv_np, multichannel=True, channel_axis=2, 
                       data_range=1.0, win_size=11)
    
    return float(ssim_val)


def calculate_psnr(clean_img: torch.Tensor, adv_img: torch.Tensor) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR)
    Higher values indicate more similar images (typically 20-50 dB)
    
    Args:
        clean_img: Clean image tensor (C, H, W)
        adv_img: Adversarial image tensor (C, H, W)
    
    Returns:
        PSNR value in dB
    """
    # Convert to numpy and transpose to (H, W, C)
    clean_np = clean_img.cpu().numpy().transpose(1, 2, 0)
    adv_np = adv_img.cpu().numpy().transpose(1, 2, 0)
    
    psnr_val = psnr(clean_np, adv_np, data_range=1.0)
    
    return float(psnr_val)


# def calculate_lpips(clean_img: torch.Tensor, adv_img: torch.Tensor, device='cpu') -> float:
#     """
#     Calculate Learned Perceptual Image Patch Similarity (LPIPS)
#     Lower values indicate more perceptually similar images (range: 0 to 1)
    
#     Args:
#         clean_img: Clean image tensor (C, H, W)
#         adv_img: Adversarial image tensor (C, H, W)
#         device: Device to run computation on
    
#     Returns:
#         LPIPS distance
#     """
#     lpips_model = get_lpips_model(device)
    
#     # LPIPS expects images in range [-1, 1], we have [0, 1]
#     clean_normalized = clean_img.unsqueeze(0).to(device) * 2 - 1
#     adv_normalized = adv_img.unsqueeze(0).to(device) * 2 - 1
    
#     with torch.no_grad():
#         lpips_val = lpips_model(clean_normalized, adv_normalized)
    
#     return lpips_val.item()


def calculate_mse(clean_img: torch.Tensor, adv_img: torch.Tensor) -> float:
    """
    Calculate Mean Squared Error
    
    Args:
        clean_img: Clean image tensor
        adv_img: Adversarial image tensor
    
    Returns:
        MSE value
    """
    mse = torch.mean((clean_img - adv_img) ** 2)
    return mse.item()


def calculate_confidence_drop(clean_conf: torch.Tensor, adv_conf: torch.Tensor) -> float:
    """
    Calculate drop in prediction confidence for true class
    
    Args:
        clean_conf: Confidence on clean image
        adv_conf: Confidence on adversarial image
    
    Returns:
        Confidence drop
    """
    return (clean_conf - adv_conf).item()


def calculate_all_metrics(
    clean_imgs: torch.Tensor,
    adv_imgs: torch.Tensor,
    true_labels: torch.Tensor,
    clean_preds: torch.Tensor,
    adv_preds: torch.Tensor,
    clean_probs: torch.Tensor = None,
    adv_probs: torch.Tensor = None,
    total_samples: int = None,  # ADD THIS
    clean_correct: int = None,  # ADD THIS
    device: str = 'cpu'
) -> Dict:
    """
    Calculate all adversarial robustness metrics
    
    Args:
        clean_imgs: Batch of clean images (only fooled samples)
        adv_imgs: Batch of adversarial images (only fooled samples)
        true_labels: Ground truth labels (only fooled samples)
        clean_preds: Predictions on clean images (only fooled samples)
        adv_preds: Predictions on adversarial images (only fooled samples)
        clean_probs: Probability distributions on clean images
        adv_probs: Probability distributions on adversarial images
        total_samples: Total number of samples tested (not just fooled)
        clean_correct: Total number of samples correctly classified on clean images
        device: Device for computation
    
    Returns:
        Dictionary containing all metrics
    """
    # Use provided totals or fall back to fooled sample count
    if total_samples is None:
        total_samples = len(true_labels)
    if clean_correct is None:
        clean_correct = len(true_labels)  # Assume all fooled were correctly classified
    
    fooled = len(true_labels)  # Number of fooled samples
    
    # Attack Success Metrics - calculated on ALL samples tested
    attack_success_rate = (fooled / total_samples) * 100 if total_samples > 0 else 0
    # robust_accuracy = ((total_samples - fooled) / total_samples) * 100 if total_samples > 0 else 0
    clean_accuracy = (clean_correct / total_samples) * 100 if total_samples > 0 else 0
    fooling_rate = (fooled / clean_correct) * 100 if clean_correct > 0 else 0
    
    # Per-image metrics (calculated only on fooled samples)
    l2_norms = []
    linf_norms = []
    l0_norms = []
    ssim_scores = []
    psnr_scores = []
    lpips_scores = []
    mse_scores = []
    confidence_drops = []
    
    for i in range(len(clean_imgs)):
        clean = clean_imgs[i]
        adv = adv_imgs[i]
        
        # Perturbation norms
        l2_norms.append(calculate_l2_norm(clean, adv))
        linf_norms.append(calculate_linf_norm(clean, adv))
        l0_norms.append(calculate_l0_norm(clean, adv))
        
        # Perceptual metrics
        ssim_scores.append(calculate_ssim(clean, adv))
        psnr_scores.append(calculate_psnr(clean, adv))
        lpips_val = calculate_lpips(clean, adv, device)
        lpips_scores.append(lpips_val)
        mse_scores.append(calculate_mse(clean, adv))
        
        # Confidence drops (if probabilities provided)
        if clean_probs is not None and adv_probs is not None:
            true_label = true_labels[i]
            clean_conf = clean_probs[i, true_label]
            adv_conf = adv_probs[i, true_label]
            confidence_drops.append(calculate_confidence_drop(clean_conf, adv_conf))
    
    # Filter out NaN values for LPIPS
    valid_lpips = [x for x in lpips_scores if not np.isnan(x)]
    
    metrics = {
        # Count metrics
        "total_samples": total_samples,
        "correctly_classified": clean_correct,
        "fooled_samples": fooled,
        
        # Success metrics (on ALL samples)
        "attack_success_rate": attack_success_rate,
        # "robust_accuracy": robust_accuracy,
        "clean_accuracy": clean_accuracy,
        "fooling_rate": fooling_rate,
        
        # Perturbation norms (average of fooled samples)
        "avg_l2_norm": np.mean(l2_norms) if l2_norms else 0,
        "avg_linf_norm": np.mean(linf_norms) if linf_norms else 0,
        "avg_l0_norm": np.mean(l0_norms) if l0_norms else 0,
        "max_l2_norm": np.max(l2_norms) if l2_norms else 0,
        "max_linf_norm": np.max(linf_norms) if linf_norms else 0,
        
        # Perceptual metrics (average of fooled samples)
        "avg_ssim": np.mean(ssim_scores) if ssim_scores else 0,
        "avg_psnr": np.mean(psnr_scores) if psnr_scores else 0,
        "avg_lpips": np.mean(valid_lpips) if valid_lpips else float('nan'),
        "avg_mse": np.mean(mse_scores) if mse_scores else 0,
        
        # Confidence metrics
        "avg_confidence_drop": np.mean(confidence_drops) if confidence_drops else None,
        
        # Per-image metrics (for plotting)
        "l2_norms": l2_norms,
        "linf_norms": linf_norms,
        "l0_norms": l0_norms,
        "ssim_scores": ssim_scores,
        "psnr_scores": psnr_scores,
        "lpips_scores": lpips_scores,
        "mse_scores": mse_scores,
        "confidence_drops": confidence_drops if confidence_drops else None,
    }
    
    return metrics


def calculate_perturbation_statistics(metrics: Dict) -> Dict:
    """
    Calculate statistical properties of perturbations
    
    Args:
        metrics: Dictionary from calculate_all_metrics
    
    Returns:
        Dictionary with statistical measures
    """
    stats = {
        "l2_std": np.std(metrics["l2_norms"]),
        "l2_median": np.median(metrics["l2_norms"]),
        "linf_std": np.std(metrics["linf_norms"]),
        "linf_median": np.median(metrics["linf_norms"]),
        "ssim_std": np.std(metrics["ssim_scores"]),
        "ssim_median": np.median(metrics["ssim_scores"]),
        "psnr_std": np.std(metrics["psnr_scores"]),
        "psnr_median": np.median(metrics["psnr_scores"]),
        "lpips_std": np.std(metrics["lpips_scores"]),
        "lpips_median": np.median(metrics["lpips_scores"]),
    }
    
    return stats


def metrics_to_dataframe(metrics: Dict):
    """
    Convert metrics to pandas DataFrame for easy export
    
    Args:
        metrics: Dictionary from calculate_all_metrics
    
    Returns:
        pandas DataFrame
    """
    import pandas as pd
    
    # Per-image metrics
    data = {
        "Image_ID": range(len(metrics['l2_norms'])),
        "L2_Norm": metrics['l2_norms'],
        "Linf_Norm": metrics['linf_norms'],
        "L0_Norm": metrics['l0_norms'],
        "SSIM": metrics['ssim_scores'],
        "PSNR": metrics['psnr_scores'],
        "LPIPS": metrics['lpips_scores'],
        "MSE": metrics['mse_scores'],
    }
    
    if metrics['confidence_drops'] is not None:
        data["Confidence_Drop"] = metrics['confidence_drops']
    
    return pd.DataFrame(data)


def print_metrics_summary(metrics: Dict):
    """
    Print a formatted summary of metrics
    
    Args:
        metrics: Dictionary from calculate_all_metrics
    """
    print("=" * 60)
    print("ADVERSARIAL ROBUSTNESS METRICS SUMMARY")
    print("=" * 60)
    
    print("\n📊 Attack Success Metrics:")
    print(f"  Total Samples:           {metrics['total_samples']}")
    print(f"  Correctly Classified:    {metrics['correctly_classified']}")
    print(f"  Fooled Samples:          {metrics['fooled_samples']}")
    print(f"  Attack Success Rate:     {metrics['attack_success_rate']:.2f}%")
    print(f"  Robust Accuracy:         {metrics['robust_accuracy']:.2f}%")
    print(f"  Clean Accuracy:          {metrics['clean_accuracy']:.2f}%")
    print(f"  Fooling Rate:            {metrics['fooling_rate']:.2f}%")
    
    print("\n📏 Perturbation Metrics:")
    print(f"  Avg L2 Norm:             {metrics['avg_l2_norm']:.4f}")
    print(f"  Max L2 Norm:             {metrics['max_l2_norm']:.4f}")
    print(f"  Avg L∞ Norm:             {metrics['avg_linf_norm']:.4f}")
    print(f"  Max L∞ Norm:             {metrics['max_linf_norm']:.4f}")
    print(f"  Avg L0 Norm:             {metrics['avg_l0_norm']:.0f} pixels")
    
    print("\n👁️ Perceptual Similarity Metrics:")
    print(f"  Avg SSIM:                {metrics['avg_ssim']:.4f} (higher = more similar)")
    print(f"  Avg PSNR:                {metrics['avg_psnr']:.2f} dB (higher = more similar)")
    print(f"  Avg LPIPS:               {metrics['avg_lpips']:.4f} (lower = more similar)")
    print(f"  Avg MSE:                 {metrics['avg_mse']:.6f}")
    
    if metrics['avg_confidence_drop'] is not None:
        print("\n🎯 Confidence Metrics:")
        print(f"  Avg Confidence Drop:     {metrics['avg_confidence_drop']:.4f}")
    
    print("=" * 60)