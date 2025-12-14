---
title: Adv Toolkit
emoji: 🛡️
colorFrom: red
colorTo: red
sdk: docker
app_port: 8501
tags:
- streamlit
pinned: false
short_description: Adversarial Robustnes Toolkit for Black Box Attacks
license: apache-2.0
---

Your directory should look like this 
```
adv-toolkit/
├── attacks/
│   └── BIA/
│       └── vgg19/
│           ├── netG_BIA_0.pth
│           ├── netG_BIA+DA_0.pth
│           └── netG_BIA+RN_0.pth
├── Beyond-ImageNet-Attack/
├── core/
│   ├── __init__.py
│   ├── attacker.py              # Attack implementation
│   ├── data_loader.py            # Dataset loading utilities
│   ├── generator.py              # GAN generator models
│   ├── loader_checkpoint.py      # Model checkpoint loading
│   └── model_loader.py           # TorchScript model loader
├── data/
│   └── test.zip                  # Test dataset just for debugging
│   └── transform.json            # transform.json - a sample json that you could use for model
├── default_model/
│   └── cifar10.ts                # Default CIFAR-10 model
├── utils/
│   ├── __init__.py
│   ├── dashboard.py              # Streamlit interactive visualizations
│   ├── init.py                   # Initialization & SSL fixes
│   ├── metrics.py                # Robustness metrics calculation
│   ├── plotting.py               # Static plotting utilities
│   └── transforms.py             # Image preprocessing transforms
├── .gitignore
├── main.py                       # Streamlit app entry point
└── requirements.txt              # Python dependencies
```
