import glob

import numpy as np
import torch
from PIL import Image
from skimage.transform import resize
import matplotlib.pyplot as plt

from utils import *
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from datasets import SRDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data
image_list = glob.glob("../testing_lr_images/*.png") 

# Model checkpoints
srresnet_checkpoint = "./checkpoint_srresnet.pth.tar"

# Load model, either the SRResNet or the SRGAN
srresnet = torch.load(srresnet_checkpoint)['model'].to(device)
srresnet.eval()
model = srresnet

imagenet_mean_cuda = torch.FloatTensor([0.485, 0.456, 0.406]).to(device).unsqueeze(0).unsqueeze(2).unsqueeze(3)
imagenet_std_cuda = torch.FloatTensor([0.229, 0.224, 0.225]).to(device).unsqueeze(0).unsqueeze(2).unsqueeze(3)


# Prohibit gradient computation explicitly because I had some problems with memory
with torch.no_grad():
    # Batches
    for image_name in image_list:        # Move to default device
        print(image_name)
        lr_imgs = Image.open(image_name)
        lr_imgs = lr_imgs.convert('RGB')
        lr_imgs = convert_image(lr_imgs, source='pil', target='imagenet-norm')
        # print(lr_imgs.min(), lr_imgs.max(), lr_imgs.shape)
        # Forward prop.
        sr_imgs = model(lr_imgs.unsqueeze(0).to(device))  # (1, 3, w, h), in [-1, 1]
        # print(sr_imgs.max(), sr_imgs.min(), sr_imgs.shape)
        sr_imgs = convert_image(sr_imgs.squeeze().cpu(), source='[-1, 1]', target='[0, 1]').numpy().transpose([1,2,0])
        # Calculate PSNR and SSIM
        image_resized = resize(sr_imgs, (3 * sr_imgs.shape[0] // 4, 3 * sr_imgs.shape[1] // 4), anti_aliasing=True)
        # print(image_resized.shape)
        plt.imsave(os.path.join('..', 'testing_hr_images', os.path.basename(image_name)), image_resized)
