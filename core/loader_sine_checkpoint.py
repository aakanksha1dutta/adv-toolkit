import torch
import os
from core.generator import SineAttack


def load_sine_attack(checkpoint_path='attacks/SINE/sine_attack_trained.pth', 
                     device='cpu', 
                     image_shape=None):
    """
    Load pre-trained Sine Attack parameters
    
    Args:
        checkpoint_path: Path to sine attack checkpoint
        device: Device to load on
        image_shape: Override image shape (C, H, W). If None, uses shape from checkpoint
    
    Returns:
        SineAttack instance with loaded parameters
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Sine attack checkpoint not found at {checkpoint_path}\n"
            f"Please ensure the sine_attack_trained.pth file is in attacks/SINE/"
        )
    
    print(f"Loading Sine Attack from: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract parameters
    amplitudes = checkpoint['amplitudes']
    freq_h = checkpoint['freq_h']
    freq_w = checkpoint['freq_w']
    phase_h = checkpoint['phase_h']
    phase_w = checkpoint['phase_w']
    epsilon = checkpoint['epsilon']
    num_freq = checkpoint['num_freq']
    
    # Use provided image_shape or fall back to checkpoint
    if image_shape is None:
        image_shape = checkpoint.get('image_shape', (3, 32, 32))
        print(f"  Using image shape from checkpoint: {image_shape}")
    else:
        print(f"  Using provided image shape: {image_shape}")
    
    # Move parameters to correct device
    amplitudes = amplitudes.to(device)
    freq_h = freq_h.to(device)
    freq_w = freq_w.to(device)
    phase_h = phase_h.to(device)
    phase_w = phase_w.to(device)
    
    # Create SineAttack instance with loaded parameters
    sine_attack = SineAttack(
        image_shape=image_shape,
        epsilon=epsilon,
        num_freq=num_freq,
        amps=amplitudes,
        freq_h=freq_h,
        freq_w=freq_w,
        phase_h=phase_h,
        phase_w=phase_w,
        device=device
    )
    
    print(f"Sine Attack loaded successfully")
    print(f"Epsilon: {epsilon}")
    print(f"Num frequencies: {num_freq}")
    
    return sine_attack


def check_sine_available():
    """
    Check if sine attack checkpoint is available
    
    Returns:
        bool: True if checkpoint exists
    """
    checkpoint_path = 'attacks/SINE/sine_attack_trained.pth'
    return os.path.exists(checkpoint_path)
