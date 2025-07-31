import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import models, transforms
from PIL import Image
import random

print("Script started")

class IdiomDataset(Dataset):
    def __init__(self, dataset_path, dataset, transform=None):
        if os.path.isabs(dataset_path):
            file_path = os.path.join(dataset_path, dataset, f"subtask_a_{dataset}.tsv")
        else:
            file_path = "../" + dataset_path + "/" + dataset + "/" + f"subtask_a_{dataset}.tsv"
        self.data = pd.read_csv(file_path, sep='\t')
        print("Dataset loaded successfully.")
        print(self.data.head())

        self.dataset_dir = os.path.dirname(file_path)

        # Load ResNet50 (without FC layer)
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        modules = list(resnet.children())[:-1]  # Remove final FC layer
        self.resnet = torch.nn.Sequential(*modules).eval()

        # Image Transform
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        idiom = row["compound"]
        sentence = row["sentence"]

        expected_order = [x.strip() for x in row["expected_order"].split(',')]
        target = list(range(1, len(expected_order) + 1))
        image_to_idx = {img: i for i, img in enumerate(expected_order)}
        idx_to_image = {i: img for i, img in enumerate(expected_order)}

        shuffled_order = expected_order.copy()
        random.shuffle(shuffled_order)
        shuffled_target = [image_to_idx[img] for img in shuffled_order if img in image_to_idx]

        image_folder = os.path.join(self.dataset_dir, idiom)
        image_vectors = []
        valid_images = []

        for img_name in shuffled_order:
            img_path = os.path.join(image_folder, img_name)
            if not os.path.exists(img_path):
                print(f"Warning: Image {img_path} not found, skipping.")
                continue
            try:
                image = Image.open(img_path).convert("RGB")
                image = self.transform(image).unsqueeze(0)  # (1, 3, 224, 224)
                with torch.no_grad():
                    resnet_output = self.resnet(image)  # (1, 2048, 1, 1)
                    resnet_output = resnet_output.view(1, -1)  # (1, 2048)
                image_vectors.append(resnet_output.squeeze())  # (2048,)
                valid_images.append(img_name)
            except Exception as e:
                print(f"Error loading image: {img_path}, {e}")
                continue

        if not image_vectors:
            print(f"Skipping idiom '{idiom}' due to no valid images.")
            return None

        actual_order = expected_order
        image_vectors = torch.stack(image_vectors)

        return {
            "idiom": idiom,
            "sentence": sentence,
            "actual_order": actual_order,
            "shuffled_order": valid_images,
            "actual_target": target[:len(valid_images)],
            "shuffled_target": shuffled_target,
            "image_vectors": image_vectors,  # Shape: (n, 2048)
            "img_to_idx": image_to_idx,
            "idx_to_img": idx_to_image
        }
