from typing import Union

import numpy as np
import nibabel as nib
import torch
import matplotlib.pyplot as plt
from nibabel.processing import resample_from_to
from monai import transforms
from monai.data.meta_tensor import MetaTensor
from torch.utils.tensorboard.writer import SummaryWriter
import os
import torch.nn.functional as F

class AverageLoss:
    """
    Utility class to track losses
    and metrics during training.
    """

    def __init__(self):
        self.losses_accumulator = {}
    
    def put(self, loss_key:str, loss_value:Union[int,float]) -> None:
        """
        Store value

        Args:
            loss_key (str): Metric name
            loss_value (int | float): Metric value to store
        """
        if loss_key not in self.losses_accumulator:
            self.losses_accumulator[loss_key] = []
        self.losses_accumulator[loss_key].append(loss_value)
    
    def pop_avg(self, loss_key:str) -> float:
        """
        Average the stored values of a given metric

        Args:
            loss_key (str): Metric name

        Returns:
            float: average of the stored values
        """
        if loss_key not in self.losses_accumulator:
            return None
        losses = self.losses_accumulator[loss_key]
        self.losses_accumulator[loss_key] = []
        return sum(losses) / len(losses)
    
    def to_tensorboard(self, writer: SummaryWriter, step: int):
        """
        Logs the average value of all the metrics stored 
        into Tensorboard.

        Args:
            writer (SummaryWriter): Tensorboard writer
            step (int): Tensorboard logging global step 
        """
        for metric_key in self.losses_accumulator.keys():
            writer.add_scalar(metric_key, self.pop_avg(metric_key), step)
            
            
def to_vae_latent_trick(z: torch.Tensor, unpadded_z_shape: tuple = (3, 15, 18, 15)) -> torch.Tensor:
    """
    The latent for the VAE is not divisible by 4 (required to
    go through the UNet), therefore we apply padding before using 
    it with the UNet. This function removes the padding.

    Args:
        z (torch.Tensor): Padded latent
        unpadded_z_shape (tuple, optional): unpadded latent dimensions. Defaults to (3, 15, 18, 15).

    Returns:
        torch.Tensor: Latent without padding
    """
    padder = transforms.DivisiblePad(k=4)
    z = padder(MetaTensor(torch.zeros(unpadded_z_shape))) + z
    z = padder.inverse(z)
    return z


def to_mni_space_1p5mm_trick(x: torch.Tensor, mni1p5_dim: tuple = (122, 146, 122)) -> torch.Tensor:
    """
    The volume is resized to be divisible by 8 (required by 
    the autoencoder). This function restores the initial dimensions
    (i.e., the MNI152 space dimensions at 1.5 mm^3). 

    Args:
        x (torch.Tensor): Resized volume
        mni1p5_dim (tuple, optional): MNI152 space dims at 1.5 mm^3. Defaults to (122, 146, 122).

    Returns:
        torch.Tensor: input resized to original shape
    """
    resizer = transforms.ResizeWithPadOrCrop(spatial_size=mni1p5_dim, mode='minimum')
    return resizer(x)


def tb_display_reconstruction(writer, step, image, recon):
    """
    Display reconstruction in TensorBoard during AE training.
    """
    plt.style.use('dark_background')
    _, ax = plt.subplots(ncols=3, nrows=2, figsize=(7, 5))
    for _ax in ax.flatten(): _ax.set_axis_off()

    if len(image.shape) == 4: image = image.squeeze(0) 
    if len(recon.shape) == 4: recon = recon.squeeze(0)

    ax[0, 0].set_title('original image', color='cyan')
    ax[0, 0].imshow(image[image.shape[0] // 2, :, :], cmap='gray')
    ax[0, 1].imshow(image[:, image.shape[1] // 2, :], cmap='gray')
    ax[0, 2].imshow(image[:, :, image.shape[2] // 2], cmap='gray')

    ax[1, 0].set_title('reconstructed image', color='magenta')
    ax[1, 0].imshow(recon[recon.shape[0] // 2, :, :], cmap='gray')
    ax[1, 1].imshow(recon[:, recon.shape[1] // 2, :], cmap='gray')
    ax[1, 2].imshow(recon[:, :, recon.shape[2] // 2], cmap='gray')

    plt.tight_layout()
    writer.add_figure('Reconstruction', plt.gcf(), global_step=step)

    
def tb_display_generation(writer, step, tag, image):
    """
    Display generation result in TensorBoard during Diffusion Model training.
    """
    plt.style.use('dark_background')
    _, ax = plt.subplots(ncols=3, figsize=(7, 3))
    for _ax in ax.flatten(): _ax.set_axis_off()

    ax[0].imshow(image[image.shape[0] // 2, :, :], cmap='gray')
    ax[1].imshow(image[:, image.shape[1] // 2, :], cmap='gray')
    ax[2].imshow(image[:, :, image.shape[2] // 2], cmap='gray')

    plt.tight_layout()
    writer.add_figure(tag, plt.gcf(), global_step=step)


def tb_display_cond_generation(writer, step, tag, starting_image, followup_image, predicted_image):
    """
    Display conditional generation result in TensorBoard during ControlNet training.
    """
    plt.style.use('dark_background')
    _, ax = plt.subplots(ncols=3, nrows=3, figsize=(7, 7))
    for _ax in ax.flatten(): _ax.set_axis_off()

    ax[0, 0].set_title('starting image', color='cyan')
    ax[0, 0].imshow(starting_image[starting_image.shape[0] // 2, :, :], cmap='gray')
    ax[0, 1].imshow(starting_image[:, starting_image.shape[1] // 2, :], cmap='gray')
    ax[0, 2].imshow(starting_image[:, :, starting_image.shape[2] // 2], cmap='gray')

    ax[1, 0].set_title('follow-up image', color='magenta')
    ax[1, 0].imshow(followup_image[followup_image.shape[0] // 2, :, :], cmap='gray')
    ax[1, 1].imshow(followup_image[:, followup_image.shape[1] // 2, :], cmap='gray')
    ax[1, 2].imshow(followup_image[:, :, followup_image.shape[2] // 2], cmap='gray')

    ax[2, 0].set_title('predicted follow-up', color='yellow')
    ax[2, 0].imshow(predicted_image[predicted_image.shape[0] // 2, :, :], cmap='gray')
    ax[2, 1].imshow(predicted_image[:, predicted_image.shape[1] // 2, :], cmap='gray')
    ax[2, 2].imshow(predicted_image[:, :, predicted_image.shape[2] // 2], cmap='gray')
    
    plt.tight_layout()
    writer.add_figure(tag, plt.gcf(), global_step=step)


def percnorm_nifti(mri, lperc=1, uperc=99):
    '''
    Apply percnorm to NiFTI1Image class
    '''
    norm_arr = percnorm(mri.get_fdata(), lperc, uperc)
    return nib.Nifti1Image(norm_arr, mri.affine, mri.header)


def percnorm(arr, lperc=1, uperc=99):
    '''
    Remove outlier intensities from a brain component,
    similar to Tukey's fences method.
    '''
    upperbound = np.percentile(arr, uperc)
    lowerbound = np.percentile(arr, lperc)
    arr[arr > upperbound] = upperbound
    arr[arr < lowerbound] = lowerbound
    return arr


def apply_mask(mri, segm):
    """
    Performs brain extraction.
    """
    segm = resample_from_to(segm, mri, order=0)
    mask = segm.get_fdata() > 0
    mri_arr = mri.get_fdata()
    mri_arr[ mask == 0 ] = 0
    return nib.Nifti1Image(mri_arr, mri.affine, mri.header)

def save_reconstructed_volumes(output_dir, step, epoch, images, reconstructions, n_samples=5):
    """
    Save a specified number of original and reconstructed volumes to disk as NIfTI files.
    
    Args:
        output_dir (str): Directory to save the volumes
        step (int): Current training step
        epoch (int): Current epoch
        images (torch.Tensor): Original input images
        reconstructions (torch.Tensor): Reconstructed images from the model
        n_samples (int, optional): Number of samples to save. Defaults to 5.
    """
    # Create subdirectory for this evaluation
    subdir = os.path.join(output_dir, f'evaluation_ep{epoch}_step{step}')
    os.makedirs(subdir, exist_ok=True)
    
    # Ensure tensors are on CPU and in numpy format with float32 data type
    # (nibabel doesn't support float16 from mixed precision training)
    images = images.detach().cpu().to(torch.float32).numpy()
    reconstructions = reconstructions.detach().cpu().to(torch.float32).numpy()
    
    # Limit to the specified number of samples
    n_samples = min(n_samples, images.shape[0])
    
    for i in range(n_samples):
        # Save original image
        orig_img = images[i, 0]  # Assuming channel dimension is 1
        orig_nifti = nib.Nifti1Image(orig_img, np.eye(4))
        nib.save(orig_nifti, os.path.join(subdir, f'original_{i}.nii.gz'))
        
        # Save reconstruction
        recon_img = reconstructions[i, 0]  # Assuming channel dimension is 1
        recon_nifti = nib.Nifti1Image(recon_img, np.eye(4))
        nib.save(recon_nifti, os.path.join(subdir, f'reconstruction_{i}.nii.gz'))
    
    print(f"Saved {n_samples} original and reconstructed volumes to {subdir}")


def calculate_psnr(img1, img2, mask=None, max_val=None):
    """
    Calculate PSNR (Peak Signal-to-Noise Ratio) between two images.
    Higher values indicate better quality.
    
    Args:
        img1 (torch.Tensor): First image
        img2 (torch.Tensor): Second image
        mask (torch.Tensor, optional): Binary mask for region of interest. Defaults to None.
        max_val (float, optional): Maximum value of the signal. Defaults to None (will use max value in img1).
    
    Returns:
        float: PSNR value in dB
    """
    if max_val is None:
        max_val = img1.max()
    
    if mask is not None:
        # Apply mask
        img1 = img1[mask]
        img2 = img2[mask]
    
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return float('inf')
    
    return 20 * torch.log10(max_val / torch.sqrt(mse))


def calculate_ssim(img1, img2, mask=None, window_size=11, sigma=1.5, full=False):
    """
    Calculate SSIM (Structural Similarity Index) between two images.
    Implementation inspired by scikit-image and adapted for PyTorch.
    
    Args:
        img1 (torch.Tensor): First image
        img2 (torch.Tensor): Second image
        mask (torch.Tensor, optional): Binary mask for region of interest
        window_size (int, optional): Size of the gaussian window. Defaults to 11.
        sigma (float, optional): Standard deviation of the gaussian window. Defaults to 1.5.
        full (bool, optional): If True, return the full SSIM image. Defaults to False.
    
    Returns:
        float: SSIM value
    """
    if mask is not None:
        # Apply mask
        img1 = img1[mask]
        img2 = img2[mask]
    
    # Flatten to 1D if mask was applied
    if mask is not None:
        img1 = img1.view(-1)
        img2 = img2.view(-1)
    
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    
    mu1 = torch.mean(img1)
    mu2 = torch.mean(img2)
    
    sigma1_sq = torch.var(img1, unbiased=False)
    sigma2_sq = torch.var(img2, unbiased=False)
    sigma12 = torch.mean((img1 - mu1) * (img2 - mu2))
    
    ssim_num = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    ssim_den = (mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim = ssim_num / ssim_den
    
    return ssim


def calculate_metrics(images, reconstructions, mask=None):
    """
    Calculate PSNR and SSIM metrics between original and reconstructed images.
    
    Args:
        images (torch.Tensor): Original images
        reconstructions (torch.Tensor): Reconstructed images
        mask (torch.Tensor, optional): Binary mask to restrict calculation to brain region
    
    Returns:
        tuple: (psnr, ssim) values
    """
    # Ensure tensors are on the same device
    device = images.device
    
    if mask is not None:
        mask = mask.to(device)
    
    batch_psnr = []
    batch_ssim = []
    
    # Calculate metrics for each image in the batch
    for i in range(images.shape[0]):
        img = images[i].float()
        recon = reconstructions[i].float()
        
        # Use mask for current image if provided
        curr_mask = mask[i] if mask is not None else None
        
        psnr = calculate_psnr(img, recon, curr_mask)
        ssim = calculate_ssim(img, recon, curr_mask)
        
        batch_psnr.append(psnr.item())
        batch_ssim.append(ssim.item())
    
    return sum(batch_psnr) / len(batch_psnr), sum(batch_ssim) / len(batch_ssim)