from functools import partial
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat, rearrange
from einops.layers.torch import Rearrange
from timm.models.vision_transformer import PatchEmbed, Block
from util.SSIM import SSIM
from losses import info_nce_loss, masked_mse_loss
from util.pos_embed import get_2d_sincos_pos_embed, get_1d_sincos_pos_embed
from torchvision.transforms import (Compose, RandomVerticalFlip,
                                    RandomHorizontalFlip,ToTensor)
    
class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=128, patch_size=8, in_chans=3,
                 embed_dim=768, depth=12, num_heads=12,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 projector_hidden_dim= 4096,projector_out_dim = 128,mask_ratio= 0.75,
                 noise_embed_in_dim= 768,noise_embed_hidden_dim= 768,noise_std_max= 0.05,
                 mlp_ratio=4., norm_layer=partial(nn.LayerNorm, eps=1e-6), norm_pix_loss=True,
                 noise_loss=False,pe_dims=128,channel_last=False,decoder_embed_unmasked_tokens = True,
                ):
        super().__init__()
        self.embed_dim = embed_dim
        self.temperature = 0.1
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.channel_last = False
        self.norm_pixel_loss = norm_pix_loss
        self.noise_std_max = noise_std_max
        self.transforms = Compose(
            [
                RandomHorizontalFlip(p=0.5),
                RandomVerticalFlip(p=0.5),
                ToTensor(),
            ]
        )
        self.projector_hidden_dim = projector_hidden_dim
        self.projector_out_dim = projector_out_dim
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, self.projector_hidden_dim),
            nn.BatchNorm1d(self.projector_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.projector_hidden_dim, self.projector_hidden_dim),
            nn.BatchNorm1d(self.projector_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.projector_hidden_dim, self.projector_out_dim),
        )
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # projection head changes
        feat_dim = 128
        self.projection_head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, feat_dim)
            )
        
        # noise loss
        self.noise_embed_in_dim = noise_embed_in_dim 
        self.noise_embed_hidden_dim = noise_embed_hidden_dim 
        
        self.decoder_embed_unmasked_tokens = decoder_embed_unmasked_tokens
        self.decoder_embed_dim = decoder_embed_dim
        self.noise_embed = nn.Sequential(
            nn.Linear(self.noise_embed_in_dim, self.noise_embed_hidden_dim),
            nn.ReLU(),
            nn.Linear(
                self.noise_embed_hidden_dim,
                self.embed_dim
                if self.decoder_embed_unmasked_tokens
                else self.decoder_embed_dim,
            ),
        )
        
        self.noise_loss = noise_loss
        self.pe_dims=pe_dims
        self.noise_pe_mlp = nn.Sequential(
                nn.Linear(pe_dims, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim)
            )

        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) 
        
        self.output = nn.Sequential(
            # 192 -> 128
            nn.ConvTranspose2d(192, 128, kernel_size=4, stride=2, padding=1),  # (64, 128, 32, 32)
            nn.ReLU(),
            # 128 -> 64
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),    # (64, 64, 64, 64)
            nn.ReLU(),
            # 64 -> 32
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),     # (64, 32, 128, 128)
            nn.ReLU(),
            # 32 -> 3
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=1, padding=1),      # (64, 3, 128, 128)
        )


        self.norm_pix_loss = norm_pix_loss
        if self.channel_last:
            self = self.to(memory_format=torch.channels_last)
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    def patchify(self, x: torch.Tensor):
        """Rearrange image into patches

        Args:
            x: Tensor of size (b, 3, h, w)

        Return:
            x: Tensor of size (b, h*w, patch_size^2 * 3)
        """
        assert x.shape[2] == x.shape[3] and x.shape[2] % self.patch_size == 0

        return rearrange(
            x,
            "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
            p1=self.patch_size,
            p2=self.patch_size,
        )

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio,shuffle_and_mask=True):
        # embed patches
        x = self.patch_embed(x)  
        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        if shuffle_and_mask:
            x, mask, ids_restore = self.random_masking(x, mask_ratio)
        else:
            mask = ids_restore = None

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, 0, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)


        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x) 
        
        if not shuffle_and_mask:
            x = rearrange(x, 'b t c -> t b c')
            return x

        return x, mask, ids_restore

    def forward_decoder(self, x, idx_unshuffle, p):
        if not self.decoder_embed_unmasked_tokens:
            # Project only masked tokens to decoder embed size
            x = self.embed(x)
    
        # Append mask tokens to input
        L = idx_unshuffle.shape[1]
        B, L_unmasked, D = x.shape
        mask_tokens = self.mask_token.repeat(B, L + 1 - L_unmasked, 1)
        temp = torch.concat([x[:, 1:, :], mask_tokens], dim=1)  # Skip cls token

        # Unshuffle tokens
        temp = torch.gather(
            temp, dim=1, index=repeat(idx_unshuffle, "b l -> b l d", d=D)
        )

        # Add noise level embedding
        if p is not None:
            temp = temp + p[:, None, :]

        # Prepend cls token
        x = torch.cat([x[:, :1, :], temp], dim=1)

        if self.decoder_embed_unmasked_tokens:
            # Project masked and unmasked tokens to decoder embed size
            x = self.decoder_embed(x)

        # Add pos embed
        x = x + self.decoder_pos_embed

        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        x =x[:,1:,:]
        recon_x = x  
        x = rearrange(x, 'b t c -> b c t')
        x = x.view(x.shape[0], x.shape[1], 16, 16)
        x = self.output(x)
        
        return x,recon_x# Don't return cls token

    
    @torch.no_grad()
    def add_noise(self, x: torch.Tensor):
        """Add noise to input image

        Args:
            x: Tensor of size (b, c, h, w)

        Return:
            x_noise: x tensor with added Gaussian noise of size (b, c, h, w)
            noise: Noise tensor of size (b, c, h, w)
            std: Noise standard deviation (noise level) tensor of size (b,)
        """
        # Sample std uniformly from [0, self.noise_std_max]
        std = torch.rand(x.size(0), device=x.device) * self.noise_std_max

        # Sample noise
        noise = torch.randn_like(x) * std[:, None, None, None]

        # Add noise to x
        x_noise = x + noise

        return x_noise, noise, std
    def apply_transforms_to_batch(self,images):
        N, C, H, W = images.shape
        transformed_images = []

        for i in range(N):
            img_tensor = images[i]
            img = transforms.ToPILImage()(img_tensor)
            transformed_img = self.transforms(img=img)
            transformed_images.append(transformed_img)
        transformed_images_tensor = torch.stack(transformed_images)
        return transformed_images_tensor
    
    def forward(self, imgs, mask_ratio=0.75):
        x1 = self.apply_transforms_to_batch(imgs)
        x2 = self.apply_transforms_to_batch(imgs)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x1, x2 = x1.to(device), x2.to(device)
        if self.channel_last:
            x1 = x1.to(memory_format=torch.channels_last)  
            x2 = x2.to(memory_format=torch.channels_last)  
        if self.noise_loss:
            # Add noise to views
            x1_noise, noise1, std1 = self.add_noise(x1)
            x2_noise, noise2, std2 = self.add_noise(x2)
            
        # Mask and extract features
        z1, mask1, idx_unshuffle1 = self.forward_encoder(x1, self.mask_ratio)
        z2, mask2, idx_unshuffle2 = self.forward_encoder(x2, self.mask_ratio)
        
        # Pass mean encoder features through projector
        u1 = self.projector(torch.mean(z1[:, 1:, :], dim=1))  # Skip cls token
        u2 = self.projector(torch.mean(z2[:, 1:, :], dim=1))
        
        # Generate noise level embedding
        p1 = self.noise_embed(
            get_1d_sincos_pos_embed(std1, dim=self.noise_embed_in_dim)
        )
        p2 = self.noise_embed(
            get_1d_sincos_pos_embed(std2, dim=self.noise_embed_in_dim)
        )
        
         # Predict masked patches and noise
        x1_pred,x1_recon = self.forward_decoder(z1, idx_unshuffle1, p1)
        x2_pred,x2_recon = self.forward_decoder(z2, idx_unshuffle2, p2)

        criterion = SSIM()
        
        # Contrastive loss
        loss_contrast = info_nce_loss(torch.cat([u1, u2]), temperature=self.temperature)
        
        #Patch reconstruction loss
        loss_recon = (
            masked_mse_loss(x1_recon, self.patchify(x1), mask1, self.norm_pixel_loss)
            + masked_mse_loss(x2_recon, self.patchify(x2), mask2, self.norm_pixel_loss)
        ) / 2
        
        loss_ssim = ((1 - criterion(x1, x1_pred)) + (1 - criterion(x2,x2_pred))) / 2
        return loss_ssim,loss_contrast,loss_recon,x1, x1_pred, x2, x2_pred


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks

if __name__ == '__main__':
    model=MaskedAutoencoderViT(noise_loss = True).cuda()
    loss_ssim, loss_contrast,loss_recon,x1, x1_pred, x2, x2_pred  = model(torch.randn((64, 3, 128, 128)))