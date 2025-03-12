import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
import math
import random

class MultimodalDataset(Dataset):
    def __init__(self, tsv_file, root_dir, transform=None):
        # Load text data
        self.data = pd.read_csv(tsv_file, sep="\t")
        self.root_dir = root_dir
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        # New: Print dataset size
        print(f"Total size of the dataset: {len(self.data)} samples")
 # New: Count sentence types
        self.sentence_type_counts = self.data['sentence_type'].value_counts()
        print("\nSentence type distribution:")
        print(f"Idiomatic sentences: {self.sentence_type_counts.get('idiomatic', 0)}")
        print(f"Literal sentences: {self.sentence_type_counts.get('literal', 0)}\n")

       
        self.image_folder_map = self._build_image_folder_map()
       
        self.image_folder_map = self._build_image_folder_map()

    def _build_image_folder_map(self):
        """Scans all folders and creates a mapping of image filenames to their actual folder path."""
        folder_map = {}
        for folder in os.listdir(self.root_dir):  # Iterate through idiom-named folders
            folder_path = os.path.join(self.root_dir, folder)
            if os.path.isdir(folder_path):
                for img_file in os.listdir(folder_path):
                    folder_map[img_file] = folder  # Store the mapping
        return folder_map

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        idiom = row['compound']
        sentence = row['sentence']
        
        # Load images and map their order
        images_with_names = []
        for i in range(1, 6):  # 5 images per sample
            img_name = row[f'image{i}_name']
            folder = self.image_folder_map.get(img_name, None)  # Find the correct folder
            
            if folder:
                img_path = os.path.join(self.root_dir, folder, img_name)
            else:
                img_path = None

            if img_path and os.path.exists(img_path):
                image = Image.open(img_path).convert("RGB")
                image = self.transform(image)
            else:
                print(f"Warning: {img_name} not found. Using a blank image.")
                image = torch.zeros((3, 224, 224))  # Placeholder for missing images
            
            images_with_names.append((img_name, image))
        
        # Shuffle images and create order mapping
        random.shuffle(images_with_names)
        image_order = [img_name for img_name, _ in images_with_names]
        image_tensors = [image for _, image in images_with_names]

        return {
            "idiom": idiom,
            "sentence": sentence,
            "image_order": image_order,  # Image order mapping
            "images": torch.stack(image_tensors)  # Shape: (5, 3, 224, 224)
        }

if __name__ == '__main__':
    # Paths to data
    tsv_path = "train.tsv"
    image_root = "train"

    # Create dataset
    dataset = MultimodalDataset(tsv_path, image_root)

    # Create DataLoader
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)  # Set num_workers=0 for Windows

    # Test the DataLoader with an iterator
    print("----------\n")
    dataiter = iter(train_loader)
    batch = next(dataiter)
    print(batch['idiom'], batch['sentence'], batch['image_order'], batch['images'].shape)
    print("----------\n")
    # Training loop
    num_epochs = 2
    total_samples = len(dataset)
    n_iterations = math.ceil(total_samples / 4)

    for epoch in range(num_epochs):
        for i, batch in enumerate(train_loader):
            if (i+1) % 5 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Step {i+1}/{n_iterations} | Images {batch['images'].shape}")
                print(f"Idiom {batch['idiom']}\n  Sentence {batch['sentence']}\n Image Order: {batch['image_order']}")
                print('---------\n')
