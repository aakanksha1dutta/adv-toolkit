import torch 
import torch.nn as nn


def bia_attack(attacker_gan, target_model, test_loader, device, eps, user_transform = None):

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
    
    for img, label in test_loader:
        
        x_adv = (img.clone().float()).to(device)
        label = label.long().to(device)

        
        adv = attacker_gan(x_adv)
        adv = torch.min(torch.max(adv, img - eps), img + eps)
        adv = torch.clamp(adv, 0.0, 1.0)

        #now put it through user's model

        with torch.no_grad():
            x_clean = img.to(device)

            if user_transform:
                x_clean = user_transform(img)
                adv = user_transform(adv)
            
            clean_out = target_model(x_clean)
            adv_out = target_model(adv)

            clean_probs = nn.functional.softmax(clean_out, dim=1)
            adv_probs = nn.functional.softmax(adv_out, dim=1)

            clean_pred = clean_out.argmax(dim=1)
            adv_pred = adv_out.argmax(dim=1)

            clean_correct += (clean_pred == label).sum().item()
            adv_correct += (adv_pred == label).sum().item()
            total += label.size(0)

            # mask for fooled samples
            mask = (adv_pred != label) & (label == clean_pred)
            if mask.any():
                clean_imgs_list.append(x_clean[mask])
                adv_imgs_list.append(adv[mask])
                true_labels_list.append(label[mask])
                adv_preds_list.append(adv_pred[mask])
                clean_preds_list.append(clean_pred[mask])
                clean_probs_list.append(clean_probs[mask])
                adv_probs_list.append(adv_probs[mask])
    
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
            "adv_correct": adv_correct
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
        "total_samples": total,  # ADD THIS
        "clean_correct": clean_correct,  # ADD THIS
        "adv_correct": adv_correct  # ADD THIS (samples correct on adversarial)
    }

    