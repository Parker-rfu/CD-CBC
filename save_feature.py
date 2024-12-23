from Dataloader import Train_DataLoader
import torch
import torch.nn as nn
import numpy as np
import os
from tqdm import tqdm
from Model import MaskedAutoencoderViT
import random
import gc
from collections import OrderedDict

seed = 1234
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

pretrained_path = './model/CD-CBC.mdl'
datasets_dir = './SAM'

# load model weights
mae_vit = MaskedAutoencoderViT(noise_loss = False)
pretrained_weights = torch.load(pretrained_path)
new_state_dict = OrderedDict()

for k, v in pretrained_weights.items():
    # Removing the 'module.' prefix
    name = k.replace('module.', '')  
    new_state_dict[name] = v

mae_vit.load_state_dict(new_state_dict)
mae_vit = mae_vit.to(device)

Dataset = Train_DataLoader(datasets_dir)
traindataloader = torch.utils.data.DataLoader(Dataset, batch_size=64, shuffle=True, num_workers=32)

class MAE_ViT_with_CLS_and_Pool(nn.Module):
    def __init__(self, mae_vit):
        super(MAE_ViT_with_CLS_and_Pool, self).__init__()
        self.encoder = mae_vit.forward_encoder
        
    def forward(self, x):
        # use cls_token to classification
        features= self.encoder(x,0.75,False)
        feature = features[0]
        return feature

model = MAE_ViT_with_CLS_and_Pool(mae_vit).to(device)

def save_feature():
    model.eval()
    torch.cuda.empty_cache()
    gc.collect()
    print('saving...')
    x = []
    y = []
    dataset = []
    with torch.no_grad():
        for _, scimg, label, ds in tqdm(traindataloader):
            scimg = scimg.float()
            scimg = scimg.to(device)
            cls_feature = model(scimg)  
            cls_feature = cls_feature.cpu().detach().numpy()
            label = label.cpu().numpy()
            ds = ds.cpu().numpy()

            for i in range(cls_feature.shape[0]):
                x.append(cls_feature[i, :])  
                y.append(label[i])
                dataset.append(ds[i])

    x = np.array(x)
    y = np.array(y)
    dataset = np.array(dataset)
    os.makedirs('./feature', exist_ok=True)
    np.save(os.path.join('./feature', 'X.npy'), x)
    np.save(os.path.join('./feature', 'y.npy'), y)
    np.save(os.path.join('./feature', 'dataset.npy'), dataset)

if __name__ == '__main__':
    save_feature()