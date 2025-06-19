import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips
import matplotlib.pyplot as plt

class ImageEvaluator:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.lpips_model = lpips.LPIPS(net='alex').to(self.device)
    
    def calculate_psnr(self, img1, img2):
        return psnr(img1, img2, data_range=255)
    
    def calculate_ssim(self, img1, img2):
        return ssim(img1, img2, multichannel=True, data_range=255)
    
    def calculate_lpips(self, img1, img2):
        img1_tensor = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        img2_tensor = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        
        img1_tensor = img1_tensor.to(self.device)
        img2_tensor = img2_tensor.to(self.device)
        
        # Normalize to [-1, 1]
        img1_tensor = img1_tensor * 2.0 - 1.0
        img2_tensor = img2_tensor * 2.0 - 1.0
        
        with torch.no_grad():
            distance = self.lpips_model(img1_tensor, img2_tensor)
        return distance.item()
    
    def evaluate_enhancement(self, original, enhanced):
        metrics = {
            'psnr': self.calculate_psnr(original, enhanced),
            'ssim': self.calculate_ssim(original, enhanced),
            'lpips': self.calculate_lpips(original, enhanced)
        }
        return metrics
    
    def plot_metrics(self, metrics_list, labels):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        metric_names = ['psnr', 'ssim', 'lpips']
        metric_titles = ['PSNR (Higher is Better)', 'SSIM (Higher is Better)', 'LPIPS (Lower is Better)']
        
        for i, metric in enumerate(metric_names):
            values = [m[metric] for m in metrics_list]
            axes[i].bar(labels, values)
            axes[i].set_title(metric_titles[i])
            axes[i].set_ylabel(metric.upper())
            
        plt.tight_layout()
        plt.show()