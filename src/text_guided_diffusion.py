import torch
import torchvision.utils as vutils
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import v2
# from torchvision.io import read_image
from PIL import Image
from diffusers import StableDiffusionPipeline, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from transformers import get_cosine_schedule_with_warmup
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
import random
import torch.nn.functional as F
from tqdm import tqdm
import os
import shutil
from pygbif.species import name_backbone
import pickle as pkl
import json
import numpy as np

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

def generate_images(
    pipeline,
    prompts,
    device,
):
    """
    Generate images using a fixed seed
    This code is written from scratch 
    """
    pipeline.to(device)
    with torch.no_grad():
        images = pipeline(prompts).images
        images = np.array(images)
        return images

def get_taxon(fish_species_list):
    """
    Convert a list of fish species to a dictionary of taxon information
    Written from scratch
    """
    taxon_dict = {}
    unfound_species = 0
    for species in fish_species_list:
        taxon = name_backbone(name=species)
        taxon_dict[species] = taxon
        if taxon is None:
            print(f"Could not find taxon for {species}")
            unfound_species += 1
    print(f"Unfound species: {unfound_species} out of {len(fish_species_list)}")
    return taxon_dict

class WildFishDataset(Dataset):
    '''
    Dataset class for the WildFish++ dataset
    All written from scratch
    '''
    def __init__(self, data_dir, taxon_path, desired_species, transform=None):
        fish_species_list = os.listdir(data_dir)
        fish_species_list.sort()
        if os.path.exists(taxon_path):
            with open(taxon_path, "r") as json_file:
                taxon_dict = json.load(json_file)
        else:
            taxon_dict = get_taxon(fish_species_list)
            with open(taxon_path, "w") as json_file:
                json.dump(taxon_dict, json_file)
        
        self.taxon_dict = {}
        for fish_species in fish_species_list:
            if fish_species not in desired_species:
                continue
            taxon_info = taxon_dict.get(fish_species, None)
            if taxon_info is None: # there are no unknown species in the dataset so this should not happen
                taxon = "Unknown Unknown Unknown Unknown Unknown Unknown"
            else:
                kingdom = taxon_info.get("kingdom", "Unknown")
                phylum = taxon_info.get("phylum", "Unknown")
                order = taxon_info.get("order", "Unknown")
                family = taxon_info.get("family", "Unknown")
                genus = taxon_info.get("genus", "Unknown")
                species = taxon_info.get("species", "Unknown")
                taxon = f"{kingdom}  {phylum}  {order} {family}  {genus}  {species}"
            self.taxon_dict[fish_species] = taxon

        self.filepaths = [] 
        for fish_species in tqdm(fish_species_list) :
            if fish_species not in desired_species:
                continue
            species_dir = os.path.join(data_dir, fish_species)
            for img in os.listdir(species_dir):
                self.filepaths.append((os.path.join(species_dir, img), fish_species))
        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        img_path, species = self.filepaths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        taxon = self.taxon_dict[species]
        return {"image": image, "taxon": taxon}


num_epochs = 1
num_train_timesteps = 1000
image_resolution = (128, 128)
batch_size = 2
unet_learning_rate = 1e-5
vae_learning_rate = 1e-4
text_encoder_learning_rate = 1e-6
num_workers = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_dir = "checkpoints"
dataset_dir = "../../WildFish++_Release"
taxon_path = "taxon.json"
logging_dir = "logs"
num_species_train = 1000
num_images_to_log = 10
num_checkpoints_to_keep = 1
print(f"Using device: {device}")

transforms = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.uint8, scale=True),
    v2.Resize(size=image_resolution, antialias=True),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize([0.5], [0.5]),
])

# Load the dataset using only the first n species for training
species = os.listdir(dataset_dir)
random.shuffle(species)
# save as a pkl file
# with open("species.pkl", "wb") as f:
#     pkl.dump(species, f)
train_species = species[:num_species_train]
test_species = species[num_species_train:]
train_dataset = WildFishDataset(dataset_dir, taxon_path, train_species, transforms)
train_taxon = [train_dataset.taxon_dict[species] for species in train_species]
test_dataset = WildFishDataset(dataset_dir, taxon_path, test_species, transforms)

# Load the pretrained model
repo_id = "stabilityai/stable-diffusion-2-1-base"
pipeline = StableDiffusionPipeline.from_pretrained(repo_id)
pipeline.to(device)

unet = pipeline.unet
vae = pipeline.vae
text_encoder = pipeline.text_encoder
vae.requires_grad_(False)
text_encoder.requires_grad_(False)

# Set up noise scheduler
noise_scheduler = DDPMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

# Optimizer and learning rate scheduler
unet_optimizer = torch.optim.AdamW(unet.parameters(), lr=unet_learning_rate)
vae_optimizer = torch.optim.AdamW(vae.parameters(), lr=vae_learning_rate)
text_encoder_optimizer = torch.optim.AdamW(pipeline.text_encoder.parameters(), lr=text_encoder_learning_rate)
version_num = len(os.listdir(checkpoint_dir))
logging_dir = os.path.join(logging_dir, f"version_{version_num}")
checkpoint_dir = os.path.join(checkpoint_dir, f"version_{version_num}")
os.makedirs(logging_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)
writer = SummaryWriter(logging_dir)

# Training loop
# Training code is written by me but I did use the code from the stable diffusion repo as a reference to make sure I was adding noise correctly

training_pbar = tqdm(range(num_epochs), desc="Training", leave=True)
for epoch in range(num_epochs):
    unet.train()
    vae.train()
    epoch_loss = 0
    epoch_pbar = tqdm(train_dataloader, desc=f"Epoch {epoch}", leave=True)
    for step, batch in enumerate(epoch_pbar):
        images = batch["image"].to(device)

        # convert images to embeddings
        latents = pipeline.vae.encode(images).latent_dist.sample()
        latents *= 0.18125

        # convert text descriptions to embeddings
        text_input = pipeline.tokenizer(
            batch["taxon"],
            padding="max_length",
            max_length=pipeline.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )

        encoder_hidden_states = pipeline.text_encoder(text_input.input_ids.to(device))[0]
        
        # Add noise to the latents
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device)
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        
        # Predict the noise residual
        noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
        
        # Calculate loss
        loss = F.mse_loss(noise_pred, noise, reduction="mean")
        
        epoch_loss += loss.item()

        # Backpropagation and optimization step
        unet_optimizer.zero_grad()
        vae_optimizer.zero_grad()
        text_encoder_optimizer.zero_grad()
        loss.backward()
        unet_optimizer.step()
        vae_optimizer.step()
        writer.add_scalar("Loss Per Step", loss.item(), global_step=epoch*len(train_dataloader)+step)
        epoch_pbar.set_postfix({"Train Loss Per Step": f"{loss.item():.4f}"})

    epoch_loss /= len(train_dataloader)
    writer.add_scalar("Loss Per Epoch", epoch_loss, global_step=epoch)
    training_pbar.set_postfix({"Train Loss Per Epoch": f"{epoch_loss:.4f}"})

    # save checkpoint
    pipeline = StableDiffusionPipeline.from_pretrained(
        repo_id,
        vae=vae,
        unet=unet,
        text_encoder=pipeline.text_encoder,
    )

    # Generate images for logging
    vae.eval()
    unet.eval()
    text_encoder.eval()
    prompts = train_taxon[:num_images_to_log]

    generated_images = generate_images(
        pipeline,
        prompts,
        device,
    )

    generated_images = [torch.tensor(image).permute(2,0,1).to(device) for image in generated_images]

    image_grid = vutils.make_grid(generated_images, nrow=5)

    writer.add_image(f"Generated Images Per Epoch", image_grid, global_step=epoch, dataformats='CHW')

    # Remove old checkpoints and save new one
    checkpoints = os.listdir(checkpoint_dir)
    if len(checkpoints) >= num_checkpoints_to_keep:
        checkpoints.sort()
        for checkpoint in checkpoints[:num_checkpoints_to_keep+1]:
            shutil.rmtree(os.path.join(checkpoint_dir, checkpoint))

    pipeline.save_pretrained(os.path.join(checkpoint_dir, f"checkpoint-{epoch}"))

    # Log the loss per epoch
    print(f"Epoch {epoch} completed with loss: {epoch_loss:.3f}")
