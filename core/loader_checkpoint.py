import torch
import torchvision
from .generator import GeneratorResnet
import pandas as pd


# Load a particular generator
def load_gan(args): 
    netG = GeneratorResnet()

    if args.RN:
        save_checkpoint_suffix = 'BIA+RN'
    elif args.DA:
        save_checkpoint_suffix = 'BIA+DA'
    else:
        save_checkpoint_suffix = 'BIA'  

    print('Substitute Model: {} \t RN: {} \t DA: {} \tSaving instance: {}'.format(args.model_type,
                                                                                  args.RN,
                                                                                  args.DA,
                                                                                  0))
                                                                                                           
    netG.load_state_dict(torch.load('attacks/BIA/{}/netG_{}_{}.pth'.format(args.model_type,
                                                                            save_checkpoint_suffix,
                                                                            0), map_location=args.device))

    return netG
