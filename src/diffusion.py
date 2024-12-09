import torch
import torchvision.utils as vutils
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import v2
from torchvision.io import read_image
from diffusers import UNet2DModel, DDPMScheduler, DDPMPipeline
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
import os
torch.manual_seed(42)

def generate_images(
    pipeline,
    batch_size,
    num_inference_steps=1000,
    device="cuda" if torch.cuda.is_available() else "cpu",
    seed=0
):
    """Generate images using a fixed seed"""
    pipeline.to(device)
    with torch.no_grad():
        # Set the seed for reproducibility
        generator = torch.Generator(device=device).manual_seed(seed)
        images = pipeline(
            batch_size=batch_size,
            generator=generator,
            num_inference_steps=num_inference_steps,
            output_type="pt"
        ).images
    return images


class WildFishDataset(Dataset):
    """
    Dataset class for the WildFish++ dataset
    Written from scratch
    """
    def __init__(self, data_dir, transform=None):
        fish_species = os.listdir(data_dir)
        fish_species.sort()
        self.filepaths = [] 
        for species in fish_species:
            if species != "Carcharodon_carcharias":
                continue
            species_dir = os.path.join(data_dir, species)
            for img in sorted(os.listdir(species_dir)):
                self.filepaths.append((os.path.join(species_dir, img), species))
        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        img_path, species = self.filepaths[idx]
        image = read_image(img_path)
        if self.transform:
            image = self.transform(image)
        return {"image": image, "species": species}


num_epochs = 100
num_train_timesteps = 1000
num_warmup_steps = 500
num_training_steps = 10000
image_resolution = (64, 64)
batch_size = 16
learning_rate = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_dir = "checkpoints"
dataset_dir = "../../WildFish++_Release"
logging_dir = "logs"
num_images_to_log = 10
print(f"Using device: {device}")

transforms = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.uint8, scale=True),
    v2.Resize(size=image_resolution, antialias=True),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.5], std=[0.5]),
])
train_dataset = WildFishDataset(dataset_dir, transforms)

# the UNet2DModel architecture was taken from the diffusers library
model = UNet2DModel(
    sample_size=image_resolution,
    in_channels=3,
    out_channels=3,
    layers_per_block=2,
    block_out_channels=(128, 128, 256, 256, 512, 512),
    down_block_types=(
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",
        "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D",
        "AttnUpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    ),
)

model = model.to(device)
# Set up a DDPM (Denoising Diffusion Probabilistic Model) scheduler

noise_scheduler = DDPMScheduler(num_train_timesteps=num_train_timesteps, beta_schedule="linear")

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Optimizer and learning rate scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
version_num = len(os.listdir(checkpoint_dir))
logging_dir = os.path.join(logging_dir, f"version_{version_num}")
checkpoint_dir = os.path.join(checkpoint_dir, f"version_{version_num}")
os.makedirs(logging_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)
writer = SummaryWriter(logging_dir)

# Training loop

# The training loops was referenced from the diffusers library
for epoch in range(num_epochs):
    model.train()
    for step, batch in enumerate(tqdm(train_dataloader)):
        clean_images = batch["image"].to(device)
            
        # Sample noise and add to images
        noise = torch.randn(clean_images.shape).to(device)
        timesteps = torch.randint(
            0, noise_scheduler.num_train_timesteps, (clean_images.shape[0],),
            device=device
        ).long()
        noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
        
        # Predict noise and calculate loss
        noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
        loss = torch.nn.functional.mse_loss(noise_pred, noise)

        # Backpropagation and optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        writer.add_scalar("Loss", loss.item(), global_step=step)
        writer.add_scalar("Learning rate", lr_scheduler.get_last_lr()[0], global_step=step)

    # Save the model checkpoint
    pipeline = DDPMPipeline(
                    unet=model,
                    scheduler=noise_scheduler
                )
    pipeline.save_pretrained(os.path.join(checkpoint_dir, f"checkpoint-{epoch}.pt"))

    # Generate images for logging 
    model.eval()
    generated_images = generate_images(
        pipeline,
        batch_size=num_images_to_log,
        # num_inference_steps=1000,
        num_inference_steps=10,
        device=device,
        seed=42
    )

        # Stack and log images as a grid
        image_grid = vutils.make_grid(torch.cat(generated_images), nrow=5, normalize=True, scale_each=True)
        writer.add_image(f"Generated Images Epoch {epoch}", image_grid, global_step=epoch)

    print(f"Epoch {epoch} completed with loss: {loss.item()}")