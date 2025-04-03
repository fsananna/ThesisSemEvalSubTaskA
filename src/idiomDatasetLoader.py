import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random

class IdiomDataset(Dataset):
    def __init__(self, dataset_path, dataset, transform=None):
        """
        Args:
            dataset_path (str): Path to the TSV file.
            transform (callable, optional): Optional transform to apply to images.
        """
        file_path = "../" + dataset_path +"/" + dataset + "/"+ f"subtask_a_{dataset}.tsv"
        
        self.data = pd.read_csv(file_path, sep='\t')
        self.dataset_dir = os.path.dirname(file_path)
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        idiom = row["compound"]
        sentence = row["sentence"]

        # Extract expected order of images
        expected_order = row["expected_order"].split(',')  # Convert string to list
        target = list(range(1, len(expected_order)+1))  # [ 1, 2, 3, 4, 5]  ## modify here if needed (suggested to start from zero)
        
        # Remove the square brackets from the first and last elements
        expected_order[0] = expected_order[0].lstrip('[')  # Remove '[' from the first element
        expected_order[-1] = expected_order[-1].rstrip(']')  # Remove ']' from the last element
        expected_order = [lambda x: x.strip().replace("'","") for x in  expected_order ]
        # Create mappings for image indexing
        image_to_idx = {img: i for i, img in enumerate(expected_order)}
        idx_to_image = {i: img for i, img in enumerate(expected_order)}

        # Shuffle order for training
        shuffled_order = expected_order.copy()
        random.shuffle(shuffled_order)

        # Convert shuffled image names back to indices
        shuffled_target = [image_to_idx[img] for img in shuffled_order]


        # Load images
        image_folder = os.path.join(self.dataset_dir, idiom)
        image_folder = self.dataset_dir + "/" +idiom
        images = []
        for img_name in shuffled_order:
            img_name_clean = img_name.replace("'", "")  # Remove single quotes
            img_name_clean = img_name_clean.strip()
            img_path = os.path.join(image_folder, img_name_clean)
            img_path = image_folder + "/" + img_name_clean 
            image = Image.open(img_path).convert("RGB")
            images.append(self.transform(image))

        images = torch.stack(images)  # Convert to tensor batch

        return {
            "idiom": idiom,
            "sentence": sentence,
            "actual_order": expected_order,  # Image (name) order
            "shuffled_order" : shuffled_order, #shuffled image (name) order
            "actual_target": target, # actual order in numeric form
            "shuffled_target": shuffled_target, 
            "images": images,  # Shape: (5, 3, 224, 224)
            "img_to_idx": image_to_idx,
            "idx_to_img": idx_to_image
        }

