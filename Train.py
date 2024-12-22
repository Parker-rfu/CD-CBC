from Dataloader import Train_DataLoader
import torch
import torch.nn as nn
import numpy as np
import os
from tqdm import tqdm
from Model import MaskedAutoencoderViT
import random
import gc
import math
from torchvision.utils import save_image
import matplotlib.pyplot as plt
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
datasets_dir = './SAM'
model_exit = './model'
os.makedirs('./model', exist_ok=True)
ngpu = torch.cuda.device_count()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
seed = 1234

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

model = MaskedAutoencoderViT(noise_loss = True)
model_name = 'CD-CBC'
base_lr = 0.0001
epochs = 151

if ngpu > 1:
    model = nn.DataParallel(model)

model = model.to(device)

Dataset = Train_DataLoader(datasets_dir)
Datasets = Dataset.datasets_names
traindataloader = torch.utils.data.DataLoader(Dataset, batch_size=64, shuffle=True, num_workers=8)

max_iterations = len(traindataloader) * epochs
optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.05)
cri = nn.CrossEntropyLoss()
iterations = 0
Loss = []

def adjust_learning_rate(current_iteration, max_iteration, lr_min=0, lr_max=0.001, warmup_iteration=500):
    lr = 0.0
    if current_iteration <= warmup_iteration:
        lr = lr_max * current_iteration / warmup_iteration
    else:
        lr = lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos((current_iteration - warmup_iteration) / max_iteration * math.pi))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def train(epoch,save = False):
    global iterations
    model.train()
    torch.cuda.empty_cache()
    gc.collect()
    iterations += 1
    recon_losses = 0
    train_losses = 0
    ssim_losses = 0
    contrast_losses = 0
    print(f'epoch : {epoch + 1}/{epochs}')
    for _, scimg, _, ds in tqdm(traindataloader):
        adjust_learning_rate(current_iteration=iterations, max_iteration=max_iterations, lr_min=0, lr_max=0.001)
        scimg = scimg.float()
        scimg, ds = scimg.to(device), ds.to(device)
        optimizer.zero_grad()
        ssim_loss, contrast_loss, recon_loss, x1, x1_pred, x2, x2_pred = model(scimg)
        train_loss = 0.67* recon_loss + 0.03 * contrast_loss + 0.3 * ssim_loss
        train_loss.backward()
        optimizer.step()
        
        train_losses += train_loss.data.cpu()
        ssim_losses += ssim_loss.data.cpu()
        recon_losses += recon_loss.data.cpu()
        contrast_losses += contrast_loss.data.cpu()
        
    train_losses = train_losses / len(traindataloader)
    recon_losses = recon_losses / len(traindataloader)
    ssim_losses = ssim_losses / len(traindataloader)
    contrast_losses = contrast_losses / len(traindataloader)

    # Save the train images 
    # if (epoch+1) % 5 == 0 and save:
    #     save_image(x1, os.path.join('./img', f"{epoch}_real_1.jpg"), nrow=10, padding=2, pad_value=255)  # origin_images
    #     save_image(x1_pred, os.path.join('./img', f"{epoch}_model_1.jpg"), nrow=10, padding=2, pad_value=255) # model train images
    #     save_image(x2, os.path.join('./img', f"{epoch}_real_2.jpg"), nrow=10, padding=2, pad_value=255)  # origin_images
    #     save_image(x2_pred, os.path.join('./img', f"{epoch}_model_2.jpg"), nrow=10, padding=2, pad_value=255) # model train images

    # Save the model weights 
    if (epoch+1)%50 == 0:
        torch.save(model.state_dict(), os.path.join(model_exit, model_name + str(epoch + 1).zfill(3) + ".mdl"))

    Loss.append(train_losses.item())
    print('total_loss : {:.6f}, recon_loss : {:.6},contrastive_loss : {:.6},ssim_loss : {:.6}'.format(train_losses, recon_losses,contrast_losses,ssim_losses))
if __name__ == '__main__':
    for epoch in range(epochs):
        train(epoch)
    # save loss curve image
    plt.figure()
    plt.plot(range(1, epoch + 2), Loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.savefig(os.path.join('./img', 'train_loss_curve.png'))   
    