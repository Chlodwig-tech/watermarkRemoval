import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import torch.onnx
from ignite.metrics import PSNR, SSIM
import torchmetrics
from torch.utils.tensorboard import SummaryWriter
import optuna
from torch.utils.data import random_split
import numpy as np
import torchvision.transforms as T

class DenseBlock(nn.Module):
    def __init__(self, channels_init, growth_rate, layers):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(layers):
            channels = channels_init + i * growth_rate
            self.layers.append(nn.Sequential(
                nn.BatchNorm2d(channels),
                nn.ReLU(),
                nn.Conv2d(channels, 4 * growth_rate if channels > 4 * growth_rate else growth_rate, 3, padding=1)
            ))

    def forward(self, x):
        for layer in self.layers:
            new_x = layer(x)
            x = torch.cat([x, new_x], 1)
        return x

def selection_margin(masks, margin):
    kernel = torch.ones(1, 1, margin * 2 + 1, margin * 2 + 1, device=masks.device)
    selection = F.conv2d(masks, kernel, padding=margin)
    selection = torch.clamp(torch.abs(torch.ceil(selection)), 0, 1)
    return selection

class AtrousConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, rate, padding):
        super(AtrousConv2d, self).__init__()
        self.atrous_conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, dilation=rate)

    def forward(self, x):
        return self.atrous_conv(x)

class WatermarkRemovalCNN(nn.Module):
    def __init__(self, growth_rate, channels_init, bottleneck_channels):
        super(WatermarkRemovalCNN, self).__init__()

        self.initial_conv = nn.Conv2d(3, channels_init, 3, padding=1)
        self.dense_block = DenseBlock(channels_init, growth_rate, 4)
                                        # 16 + 7 * 8 = 72                    # 128
        self.bottleneck_conv = nn.Conv2d(channels_init + 7 * growth_rate, bottleneck_channels, 1, padding=0)
        self.atrous_block1 = AtrousConv2d(bottleneck_channels, bottleneck_channels, 3, rate=2, padding=2)
        self.atrous_block2 = AtrousConv2d(bottleneck_channels, bottleneck_channels, 3, rate=4, padding=4)
        self.atrous_block3 = AtrousConv2d(bottleneck_channels, bottleneck_channels, 3, rate=2, padding=2)

        self.final_conv = nn.Conv2d(bottleneck_channels, 3, 3, padding=1)

    def forward(self, x):
        #print("Input:", x.shape)
        x = self.initial_conv(x)
        #print("After initial_conv:", x.shape)
        x = self.dense_block(x)
        #print("After dense_block:", x.shape)
        x = self.bottleneck_conv(x)
        #print("After bottleneck_conv:", x.shape)
        x = self.atrous_block1(x)
        #print("After atrous_block1:", x.shape)
        x = self.atrous_block2(x)
        #print("After atrous_block2:", x.shape)
        x = self.atrous_block3(x)
        #print("After atrous_block3:", x.shape)
        x = self.final_conv(x)
        #print("After final_conv:", x.shape)
        return x

class WatermarkRemovalDataset(Dataset):
    def __init__(self, watermarked_dir, watermark_free_dir, transform=None):
        self.watermarked_dir = watermarked_dir
        self.watermark_free_dir = watermark_free_dir
        self.transform = transform
        self.images = os.listdir(watermarked_dir)
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        watermarked_img_path = os.path.join(self.watermarked_dir, self.images[idx])
        watermark_free_img_path = os.path.join(self.watermark_free_dir, self.images[idx])
        watermarked_image = Image.open(watermarked_img_path).convert("RGB")
        watermark_free_image = Image.open(watermark_free_img_path).convert("RGB")
        if self.transform:
            watermarked_image = self.transform(watermarked_image)
            watermark_free_image = self.transform(watermark_free_image)
        return watermarked_image, watermark_free_image

def calculate_loss(predictions, targets, masks):
    # Apply the mask to the targets
    image_mask = -(targets - masks)
    abs_loss = torch.mean(torch.abs(predictions - image_mask) ** 0.5)
    return abs_loss

def create_mask(height, width, min_opacity, max_opacity):
    # Random parameters for the mask
    mask_h = np.random.randint(int(height * .7), int(height * .9))
    mask_w = np.random.randint(int(width * .1), int(width * .3))
    opacity = np.random.uniform(min_opacity, max_opacity)
    max_angle = np.random.uniform(-1.5, 1.5)

    # Create mask
    mask = np.ones((mask_h, mask_w)) * opacity
    mask *= np.random.choice([-1, 1])
    y_pos = np.random.randint(0, height - mask_h)
    x_pos = np.random.randint(0, width - mask_w)

    # Apply padding
    mask = np.pad(mask, ((y_pos, height - mask_h - y_pos), (x_pos, width - mask_w - x_pos)))

    # Rotate mask
    mask = Image.fromarray(mask)
    mask = T.functional.rotate(mask, max_angle)

    return torch.tensor(np.array(mask), dtype=torch.float32).unsqueeze(0)  # Add channel dimension

def batch_masks(batch_size, height, width, min_opacity, max_opacity):
    return torch.stack([create_mask(height, width, min_opacity, max_opacity) for _ in range(batch_size)])

def save_model_onnx(model, epoch, file_path, input_size=(1, 3, 256, 256)):
    # Set the model to evaluation mode
    model.eval()
    # Create a dummy input tensor with the specified size
    dummy_input = torch.randn(input_size, device=next(model.parameters()).device)
    # Specify the path for the ONNX model
    onnx_file_path = f"{file_path}_epoch_{epoch}.onnx"
    # Export the model
    torch.onnx.export(model, dummy_input, onnx_file_path,
                      export_params=True,
                      opset_version=10,
                      do_constant_folding=True,
                      input_names=['input'],
                      output_names=['output'])

    print(f"Model saved to {onnx_file_path}")
    # Set the model back to training mode
    model.train()

# Transformation for the image
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Prepare the dataset and dataloaders
train_dataset = WatermarkRemovalDataset(
    watermarked_dir="../Data/CLWD/train/Watermarked_image",
    watermark_free_dir="../Data/CLWD/train/Watermark_free_image",
    transform=transform
)

# HYPERPARAMETERS ----------
num_epochs = 10
learning_rate = 0.001
batch_size = 32
growth_rate = 8
channels_init = growth_rate * 2  # 16
bottleneck_channels = 128
# --------------------------
save_step = 1
min_opacity = 0.3
max_opacity = 1.0

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Model, Loss Function, and Optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = WatermarkRemovalCNN(growth_rate, channels_init, bottleneck_channels).to(device)
#criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# PSNR Metric
# PSNR: the higher, the better
data_range = 255.0  # images in 0-255 range
psnr_metric = PSNR(data_range=data_range, device=device)
# SSIM: the higher (closer to 1), the better
ssim_metric = SSIM(data_range=255.0, kernel_size=(11, 11), sigma=(1.5, 1.5), k1=0.01, k2=0.03, gaussian=True, device=device)
# LPIPS: the lower, the better
lpips_metric = torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity(net_type='alex', reduction='mean').to(device)

# Dataset Splitting
total_size = len(train_dataset)
train_size = int(0.8 * total_size)
val_size = total_size - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

# DataLoader for Validation Set
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


writer = SummaryWriter()
hparams = {
    'learning_rate': learning_rate,
    'batch_size': batch_size,
    'num_epochs': num_epochs,
    'growth_rate': growth_rate,
    'channels_init': channels_init,
    'bottleneck_channels': bottleneck_channels
}
metrics = {
    'loss': 0.0,
    'PSNR': 0.0,
    'SSIM': 0.0,
    'LPIPS': 0.0,
}
writer.add_hparams(hparam_dict=hparams, metric_dict=metrics)

model = WatermarkRemovalCNN(growth_rate, channels_init, bottleneck_channels).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training Loop
for epoch in range(num_epochs):
    model.train()
    for i, (watermarked_images, watermark_free_images) in enumerate(train_loader):
        # Move data to GPU if available
        watermarked_images = watermarked_images.to(device)
        watermark_free_images = watermark_free_images.to(device)

        # Generate masks
        masks = batch_masks(watermarked_images.size(0), watermarked_images.size(2), watermarked_images.size(3), min_opacity, max_opacity).to(device)

        # Apply masks to images
        watermarked_images_with_mask = watermarked_images - masks

        # Forward pass
        outputs = model(watermarked_images_with_mask)
        loss = calculate_loss(outputs, watermark_free_images, masks)

        # Forward pass
        outputs = model(watermarked_images)
        #loss = criterion(outputs, watermark_free_images)
        loss = calculate_loss(outputs, watermark_free_images, masks)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Validation phase
        model.eval()
        with torch.no_grad():
            for watermarked_images, watermark_free_images in val_loader:
                # Move data to GPU if available
                watermarked_images = watermarked_images.to(device)
                watermark_free_images = watermark_free_images.to(device)

                # Forward pass
                outputs = model(watermarked_images)
                loss = calculate_loss(outputs, watermark_free_images, masks)

        # Calculate the metrics
        psnr_metric.update((outputs, watermark_free_images))
        psnr = psnr_metric.compute()
        ssim_metric.update((outputs, watermark_free_images))
        ssim = ssim_metric.compute()
        outputs_normalized = torch.clamp(2 * outputs - 1, min=-1, max=1)
        targets_normalized = torch.clamp(2 * watermark_free_images - 1, min=-1, max=1)
        lpips = lpips_metric(outputs_normalized, targets_normalized)

        #         if epoch == 0 and i == 0:
        #             print(f"Traing started on device: {device}")
        #         #print(f"i: {i}")

        if (i + 1) % 100 == 0 or (epoch == 0 and i == 0):
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], '
                  f'Loss: {loss.item():.4f}, PSNR: {psnr:.4f}, SSIM: {ssim:.4f}, LPIPS: {lpips:.4f}')
            writer.add_scalar("Loss/train", loss, epoch)
            writer.add_scalar("PSNR", psnr, epoch)
            writer.add_scalar("SSIM", ssim, epoch)
            writer.add_scalar("LPIPS", lpips, epoch)

        # Reset PSNR for next batch
        psnr_metric.reset()
        lpips_metric.reset()
        ssim_metric.reset()

    if epoch % save_step == 0:
        save_model_onnx(model, epoch + 1, "./results/model")

print("Training complete!")


# def objective(trial):
#     # Suggest values for the hyperparameters
#     learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
#     batch_size = trial.suggest_int('batch_size', 16, 128, log=True)
#     growth_rate = trial.suggest_int('growth_rate', 4, 16)
#     bottleneck_channels = trial.suggest_categorical('bottleneck_channels', [64, 128, 256])
#     num_epochs = 2
#     channels_init = growth_rate * 2
#
#     model = WatermarkRemovalCNN(growth_rate, channels_init, bottleneck_channels).to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#
#     # Training Loop with Validation and Mask Creation
#     for epoch in range(num_epochs):
#         model.train()
#         for i, (watermarked_images, watermark_free_images) in enumerate(train_loader):
#             watermarked_images = watermarked_images.to(device)
#             watermark_free_images = watermark_free_images.to(device)
#
#             # Move data to GPU if available
#             watermarked_images = watermarked_images.to(device)
#             watermark_free_images = watermark_free_images.to(device)
#
#             # Forward pass
#             outputs = model(watermarked_images)
#             #loss = criterion(outputs, watermark_free_images)
#             loss = calculate_loss(outputs, watermark_free_images)
#
#             # Backward and optimize
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#             # Validation phase
#             model.eval()
#             with torch.no_grad():
#                 for watermarked_images, watermark_free_images in val_loader:
#                     # Move data to GPU if available
#                     watermarked_images = watermarked_images.to(device)
#                     watermark_free_images = watermark_free_images.to(device)
#
#                     # Forward pass
#                     outputs = model(watermarked_images)
#                     loss = calculate_loss(outputs, watermark_free_images)
#
#             # Calculate the metrics
#             psnr_metric.update((outputs, watermark_free_images))
#             psnr = psnr_metric.compute()
#             ssim_metric.update((outputs, watermark_free_images))
#             ssim = ssim_metric.compute()
#             outputs_normalized = torch.clamp(2 * outputs - 1, min=-1, max=1)
#             targets_normalized = torch.clamp(2 * watermark_free_images - 1, min=-1, max=1)
#             lpips = lpips_metric(outputs_normalized, targets_normalized)
#
#             #         if epoch == 0 and i == 0:
#             #             print(f"Traing started on device: {device}")
#             #         #print(f"i: {i}")
#
#             if (i + 1) % 100 == 0 or (epoch == 0 and i == 0):
#                 print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], '
#                       f'Loss: {loss.item():.4f}, PSNR: {psnr:.4f}, SSIM: {ssim:.4f}, LPIPS: {lpips:.4f}')
#                 writer.add_scalar("Loss/train", loss, epoch)
#                 writer.add_scalar("PSNR", psnr, epoch)
#                 writer.add_scalar("SSIM", ssim, epoch)
#                 writer.add_scalar("LPIPS", lpips, epoch)
#
#             # Reset PSNR for next batch
#             psnr_metric.reset()
#             lpips_metric.reset()
#             ssim_metric.reset()
#
#         if epoch % save_step == 0:
#             save_model_onnx(model, epoch + 1, "./results/model")
#
#     print("Training complete!")
#     # Return the metric you want to optimize (e.g., -loss, psnr, etc.)
#     return lpips
#
# study = optuna.create_study(direction='maximize')  # or 'minimize' for loss
# study.optimize(objective, n_trials=3)
#
# print("Best trial:")
# trial = study.best_trial
#
# print(f"  Value: {trial.value}")
# print("  Params: ")
# for key, value in trial.params.items():
#     print(f"    {key}: {value}")


writer.close()

# See logs
# tensorboard --logdir="D:\Github\watermarkRemoval\WatermarkRemoval\runs"