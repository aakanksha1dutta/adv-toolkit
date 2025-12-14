from core.data_loader import get_dataloader
from core.attacker import bia_attack
from core.loader_checkpoint import load_gan
from core.model_loader import load_torchscript_model
from utils.transforms import load_transform_from_json
import matplotlib.pyplot as plt
import torch 
# import streamlit as st
import tempfile
import os
# def show_pairs(clean_imgs, adv_imgs, true_labels, adv_preds, clean_preds, n=6):
#     plt.figure(figsize=(12, 4))

#     for i in range(n):
#         clean = clean_imgs[i]
#         adv = adv_imgs[i]

#         # Clean image
#         plt.subplot(2, n, i+1)
#         plt.imshow(clean.permute(1,2,0))
#         plt.axis("off")
#         plt.title(f"True: {true_labels[i].item()}, Clean Pred: {clean_preds[i].item()}")

#         # Adv image
#         plt.subplot(2, n, n+i+1)
#         plt.imshow(adv.permute(1,2,0))
#         plt.axis("off")
#         plt.title(f"Adv Pred: {adv_preds[i].item()}")

#     plt.tight_layout()
#     plt.show()


def save_uploaded_file(uploaded_file):
    suffix = os.path.splitext(uploaded_file.name)[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        return tmp.name

class Args:
  def __init__(self, target_model = None, model_type = "vgg19", RN = True, DA=False, eps = 0.0314, user_transform_json = None, shuffle_data = False):
    self.model_type = model_type #model attacker is trained against - vgg19 vs resnet152
    self.target_model = target_model #user provided path
    self.RN = RN
    self.DA = DA
    self.eps = eps
    self.loaded_target_model = None
    self.user_transform_json = user_transform_json
    self.loaded_user_transform_json = None
    self.user_transform = None
    self.device =  "cuda:0" if torch.cuda.is_available() else "cpu"
    self.shuffle_data = shuffle_data


    if self.target_model:   
        target_model_path = save_uploaded_file(self.target_model)
        self.loaded_target_model, param_count = load_torchscript_model(target_model_path, device = self.device)
    else: 
       default_model_path = os.path.join("default_model", "cifar10.ts")
       self.loaded_target_model, param_count = load_torchscript_model(default_model_path, device=self.device)

    #if user uploads json, load it as a user transform, else default transform
    if self.user_transform_json:
        self.loaded_user_transform_json = save_uploaded_file(self.user_transform_json)
        self.user_transform = load_transform_from_json(self.loaded_user_transform_json)


    

if __name__ == '__main__':
    args = Args(eps=8/500, shuffle_data=True)
    loader = get_dataloader("/Users/aakankshadutta/MyProjects/adv-toolkit/data/test.zip", n_examples=64, shuffle=args.shuffle_data)
    

    #target_model
    # use streamlit to read and then load
    # args.target_model = st.file_uploader("Upload TorchScript model (.ts)", type=["ts"])
    # args.user_transform_json = st.file_uploader("Upload Transform JSON (.json) (optional)", type = ["json"])
    


    attacker_bia = load_gan(args)
    results_bia = bia_attack(attacker_bia, args.loaded_target_model, loader, args.device, args.eps, user_transform=args.user_transform)
    if results_bia["num_fooled"]>0:
       print(f"BIA fooled {results_bia["num_fooled"]} samples.")
        # show_pairs(
        #     results_bia["clean_imgs"], 
        #     results_bia["adv_imgs"],
        #     results_bia["true_labels"], 
        #     results_bia["adv_preds"], 
        #     results_bia["clean_preds"], 
        #     n=min(6, results_bia["num_fooled"]),
        # )



    

    

