import torch 
import torch.nn as nn


def bia_attack(attacker_gan, target_model, test_loader, device, eps, user_transform=None, sine_attacker=None):
    """
    BIA attack with optional Sine attack ensemble
    
    Args:
        attacker_gan: BIA generator
        target_model: Target model to attack
        test_loader: DataLoader
        device: Device
        eps: Epsilon budget
        user_transform: Optional transform
        sine_attacker: Optional Sine attacker for ensemble (if None, BIA only)
    """
    clean_imgs_list = []
    adv_imgs_list = []
    true_labels_list = []
    adv_preds_list = []
    clean_preds_list = []
    clean_probs_list = []
    adv_probs_list = []
    
    clean_correct, adv_correct = 0, 0
    total = 0
    
    attacker_gan = nn.DataParallel(attacker_gan).to(device).eval()
    target_model.eval()
    
    # If sine_attacker provided, update its image shape from first batch
    if sine_attacker is not None:
        first_batch = next(iter(test_loader))
        image_shape = tuple(first_batch[0][0].shape)
        if sine_attacker.image_shape != image_shape:
            C, H, W = image_shape
            sine_attacker.image_shape = image_shape
            sine_attacker.grid_h = torch.linspace(0, 1, H, device=device).view(1, 1, H, 1)
            sine_attacker.grid_w = torch.linspace(0, 1, W, device=device).view(1, 1, 1, W)
    
    for img, label in test_loader:
        x_bia_adv = img.clone().float().to(device)
        label = label.long().to(device)
        
        # Generate BIA adversarial
        bia_adv = attacker_gan(x_bia_adv)
        bia_adv = torch.min(torch.max(bia_adv, img.to(device) - eps), img.to(device) + eps)
        bia_adv = torch.clamp(bia_adv, 0.0, 1.0)
        
        sa_adv = None
        # Generate Sine adversarial if provided
        if sine_attacker is not None:
            x_sa_adv = img.clone().float().to(device)
            batch_size = x_sa_adv.size(0)
            sa_delta = sine_attacker.generate_perturbation(batch_size)
            sa_adv = x_sa_adv + sa_delta
            sa_adv = torch.min(torch.max(sa_adv, img.to(device) - eps), img.to(device) + eps)
            sa_adv = torch.clamp(sa_adv, 0.0, 1.0)
        
        # Evaluate on user's model
        with torch.no_grad():
            x_clean = img.to(device)
            x_bia_eval = bia_adv
            
            if user_transform:
                x_clean = user_transform(img.to(device))
                x_bia_eval = user_transform(bia_adv)
            
            clean_out = target_model(x_clean)
            bia_out = target_model(x_bia_eval)
            
            sa_out = None
            # Evaluate Sine adversarial if provided
            if sine_attacker is not None:
                x_sa_eval = sa_adv
                if user_transform:
                    x_sa_eval = user_transform(sa_adv)
                sa_out = target_model(x_sa_eval)
            
            # Get predictions
            clean_probs = nn.functional.softmax(clean_out, dim=1)
            bia_probs = nn.functional.softmax(bia_out, dim=1)
            
            clean_pred = clean_out.argmax(dim=1)
            bia_pred = bia_out.argmax(dim=1)
            
            sa_probs, sa_pred = None, None
            if sine_attacker is not None:
                sa_probs = nn.functional.softmax(sa_out, dim=1)
                sa_pred = sa_out.argmax(dim=1)
            
            clean_correct += (clean_pred == label).sum().item()

            total += label.size(0)
            
            # Define masks FIRST
            bia_fooled_mask = (bia_pred != label) & (label == clean_pred)
            
            if sine_attacker is not None:
                sa_fooled_mask = (sa_pred != label) & (label == clean_pred)
                combined_mask = bia_fooled_mask | sa_fooled_mask
                
                # Track adversarial correct: use the final adversarial prediction for each sample
                # Use BIA prediction if BIA fooled, else use Sine prediction
                final_adv_pred = torch.where(bia_fooled_mask, bia_pred, 
                                            torch.where(sa_fooled_mask, sa_pred, bia_pred))
                adv_correct += (final_adv_pred == label).sum().item()
            else:
                combined_mask = bia_fooled_mask
                adv_correct += (bia_pred == label).sum().item()
            
            if combined_mask.any():
                # Vectorized: collect all fooled samples at once
                # For samples fooled by both, prefer BIA
                # Create a mask for which attack to use for each sample
                use_bia = bia_fooled_mask & combined_mask  # BIA fooled (includes samples fooled by both)
                use_sine = (~bia_fooled_mask) & combined_mask  # Only Sine fooled
                
                if use_bia.any():
                    clean_imgs_list.append(x_clean[use_bia].cpu())
                    adv_imgs_list.append(x_bia_eval[use_bia].cpu())
                    true_labels_list.append(label[use_bia].cpu())
                    adv_preds_list.append(bia_pred[use_bia].cpu())
                    clean_preds_list.append(clean_pred[use_bia].cpu())
                    clean_probs_list.append(clean_probs[use_bia].cpu())
                    adv_probs_list.append(bia_probs[use_bia].cpu())
                
                if sine_attacker is not None and use_sine.any():
                    clean_imgs_list.append(x_clean[use_sine].cpu())
                    adv_imgs_list.append(x_sa_eval[use_sine].cpu())
                    true_labels_list.append(label[use_sine].cpu())
                    adv_preds_list.append(sa_pred[use_sine].cpu())
                    clean_preds_list.append(clean_pred[use_sine].cpu())
                    clean_probs_list.append(clean_probs[use_sine].cpu())
                    adv_probs_list.append(sa_probs[use_sine].cpu())
    
    # Check if any samples were fooled
    if len(clean_imgs_list) == 0:
        return {
            "clean_imgs": torch.tensor([]),
            "adv_imgs": torch.tensor([]),
            "true_labels": torch.tensor([]),
            "adv_preds": torch.tensor([]),
            "clean_preds": torch.tensor([]),
            "clean_probs": None,
            "adv_probs": None,
            "num_fooled": 0,
            "total_samples": total,
            "clean_correct": clean_correct,
            "adv_correct": 0
        }
    
    # Concatenate results
    clean_imgs = torch.cat(clean_imgs_list).cpu()
    adv_imgs = torch.cat(adv_imgs_list).cpu()
    true_labels = torch.cat(true_labels_list).cpu()
    adv_preds = torch.cat(adv_preds_list).cpu()
    clean_preds = torch.cat(clean_preds_list).cpu()
    clean_probs = torch.cat(clean_probs_list).cpu()
    adv_probs = torch.cat(adv_probs_list).cpu()
    
    return {
        "clean_imgs": clean_imgs,
        "adv_imgs": adv_imgs,
        "true_labels": true_labels,
        "adv_preds": adv_preds,
        "clean_preds": clean_preds,
        "clean_probs": clean_probs,
        "adv_probs": adv_probs,
        "num_fooled": len(clean_imgs),
        "total_samples": total,
        "clean_correct": clean_correct,
        "adv_correct": adv_correct  # Now properly tracked
    }


# def bia_attack(attacker_gan, target_model, test_loader, device, eps, user_transform = None):

#     clean_imgs_list = []
#     adv_imgs_list = []
#     true_labels_list = []
#     adv_preds_list = []
#     clean_preds_list = []
#     clean_probs_list = []
#     adv_probs_list = []

#     clean_correct, adv_correct = 0, 0
#     total = 0
#     attacker_gan = nn.DataParallel(attacker_gan).to(device).eval()
#     target_model.eval()
    
#     for img, label in test_loader:
        
#         x_adv = (img.clone().float()).to(device)
#         label = label.long().to(device)

        
#         adv = attacker_gan(x_adv)
#         adv = torch.min(torch.max(adv, img - eps), img + eps)
#         adv = torch.clamp(adv, 0.0, 1.0)

#         #now put it through user's model

#         with torch.no_grad():
#             x_clean = img.to(device)

#             if user_transform:
#                 x_clean = user_transform(img)
#                 adv = user_transform(adv)
            
#             clean_out = target_model(x_clean)
#             adv_out = target_model(adv)

#             clean_probs = nn.functional.softmax(clean_out, dim=1)
#             adv_probs = nn.functional.softmax(adv_out, dim=1)

#             clean_pred = clean_out.argmax(dim=1)
#             adv_pred = adv_out.argmax(dim=1)

#             clean_correct += (clean_pred == label).sum().item()
#             adv_correct += (adv_pred == label).sum().item()
#             total += label.size(0)

#             # mask for fooled samples
#             mask = (adv_pred != label) & (label == clean_pred)
#             if mask.any():
#                 clean_imgs_list.append(x_clean[mask])
#                 adv_imgs_list.append(adv[mask])
#                 true_labels_list.append(label[mask])
#                 adv_preds_list.append(adv_pred[mask])
#                 clean_preds_list.append(clean_pred[mask])
#                 clean_probs_list.append(clean_probs[mask])
#                 adv_probs_list.append(adv_probs[mask])
    
#     # Check if any samples were fooled
#     if len(clean_imgs_list) == 0:
#         return {
#             "clean_imgs": torch.tensor([]),
#             "adv_imgs": torch.tensor([]),
#             "true_labels": torch.tensor([]),
#             "adv_preds": torch.tensor([]),
#             "clean_preds": torch.tensor([]),
#             "clean_probs": None,
#             "adv_probs": None,
#             "num_fooled": 0,
#             "total_samples": total,
#             "clean_correct": clean_correct,
#             "adv_correct": adv_correct
#         }
    
#     # Concatenate results
#     clean_imgs = torch.cat(clean_imgs_list).cpu()
#     adv_imgs = torch.cat(adv_imgs_list).cpu()
#     true_labels = torch.cat(true_labels_list).cpu()
#     adv_preds = torch.cat(adv_preds_list).cpu()
#     clean_preds = torch.cat(clean_preds_list).cpu()
#     clean_probs = torch.cat(clean_probs_list).cpu()
#     adv_probs = torch.cat(adv_probs_list).cpu()
    
#     return {
#         "clean_imgs": clean_imgs,
#         "adv_imgs": adv_imgs,
#         "true_labels": true_labels,
#         "adv_preds": adv_preds,
#         "clean_preds": clean_preds,
#         "clean_probs": clean_probs,
#         "adv_probs": adv_probs,
#         "num_fooled": len(clean_imgs),
#         "total_samples": total,  # ADD THIS
#         "clean_correct": clean_correct,  # ADD THIS
#         "adv_correct": adv_correct  # ADD THIS (samples correct on adversarial)
#     }

   
    