import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import random

print("Script started")

class IdiomDataset(Dataset):
    def __init__(self, dataset_path, dataset, transform=None):
        file_path = "../" + dataset_path + "/" + dataset + "/" + f"subtask_a_{dataset}.tsv"
        
        self.data = pd.read_csv(file_path, sep='\t')
        print("Dataset loaded successfully.")
        print(self.data.head())  # First 5 rows

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

        # Process expected order list
        expected_order = row["expected_order"].split(',')
        expected_order[0] = expected_order[0].lstrip('[')
        expected_order[-1] = expected_order[-1].rstrip(']')
        expected_order = [x.strip().replace("'", "") for x in expected_order]

        target = list(range(1, len(expected_order)+1))
        image_to_idx = {img: i for i, img in enumerate(expected_order)}
        idx_to_image = {i: img for i, img in enumerate(expected_order)}

        shuffled_order = expected_order.copy()
        random.shuffle(shuffled_order)
        shuffled_target = [image_to_idx[img] for img in shuffled_order]

        image_folder = os.path.join(self.dataset_dir, idiom)
        images = []
        missing_images = []

        for img_name in shuffled_order:
            img_path = os.path.join(image_folder, img_name)
            if not os.path.exists(img_path):
                missing_images.append(img_name)
                continue
            try:
                image = Image.open(img_path).convert("RGB")
                images.append(self.transform(image))
            except Exception as e:
                print(f" Error loading image: {img_path}, {e}")
                missing_images.append(img_name)

        if len(images) != len(shuffled_order):
            print(f" Skipping idiom '{idiom}' due to missing images: {missing_images}")
            return None

        images = torch.stack(images)

        return {
            "idiom": idiom,
            "sentence": sentence,
            "actual_order": expected_order,
            "shuffled_order": shuffled_order,
            "actual_target": target,
            "shuffled_target": shuffled_target,
            "images": images,
            "img_to_idx": image_to_idx,
            "idx_to_img": idx_to_image
        }
