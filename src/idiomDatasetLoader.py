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
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.resnet = torch.nn.Sequential(
            *list(self.resnet.children())[:-1],
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
        ).eval()
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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
                image = self.transform(image).unsqueeze(0)
                with torch.no_grad():
                    vector = self.resnet(image)
                image_vectors.append(vector.squeeze())
                valid_images.append(img_name)
            except Exception as e:
                print(f"Error loading image: {img_path}, {e}")
                continue

        if not image_vectors:
            print(f"Skipping idiom '{idiom}' due to no valid images.")
            return None

        # actual_order = [img for img in expected_order if img in valid_images]
        actual_order = expected_order  # Replace the current line: actual_order = [img for img in expected_order if img in valid_images]
        image_vectors = torch.stack(image_vectors)

        return {
            "idiom": idiom,
            "sentence": sentence,
            "actual_order": actual_order,
            "shuffled_order": valid_images,
            "actual_target": target[:len(valid_images)],
            "shuffled_target": shuffled_target,
            "image_vectors": image_vectors,
            "img_to_idx": image_to_idx,
            "idx_to_img": idx_to_image
        }


# import os  # Import os for file and directory operations
# import pandas as pd  # Import pandas for reading TSV files
# import torch  # Import PyTorch for tensor operations
# from torch.utils.data import Dataset  # Import Dataset for custom dataset
# from torchvision import models, transforms  # Import models and transforms for ResNet50 and image preprocessing
# from PIL import Image  # Import PIL for image loading
# import random  # Import random for shuffling

# print("Script started")  # Indicate script has started

# class IdiomDataset(Dataset):  # Define custom dataset class
#     def __init__(self, dataset_path, dataset, transform=None):  # Initialize with dataset path and type
#         # Check if dataset_path is absolute; if so, use it directly, otherwise prepend ../
#         if os.path.isabs(dataset_path):
#             file_path = os.path.join(dataset_path, dataset, f"subtask_a_{dataset}.tsv")
#         else:
#             file_path = "../" + dataset_path + "/" + dataset + "/" + f"subtask_a_{dataset}.tsv"
#         self.data = pd.read_csv(file_path, sep='\t')  # Load TSV into DataFrame
#         print("Dataset loaded successfully.")
#         print(self.data.head())  # Display first 5 rows

#         self.dataset_dir = os.path.dirname(file_path)  # Get directory of TSV file
#         # Load pre-trained ResNet50 and add a projection layer to match BERT's 768 dimensions
#         self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
#         self.resnet = torch.nn.Sequential(
#             *list(self.resnet.children())[:-1],  # Remove final FC layer
#             torch.nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling to reduce spatial dimensions
#             torch.nn.Flatten(),  # Flatten to (batch_size, 2048)
#             torch.nn.Linear(2048, 768)  # Project to 768 dimensions
#         ).eval()
#         self.transform = transform if transform else transforms.Compose([
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         ])

#     def __len__(self):  # Return total number of samples
#         return len(self.data)

#     def __getitem__(self, idx):  # Get a sample by index
#         row = self.data.iloc[idx]  # Get row from DataFrame
#         idiom = row["compound"]  # Get idiom
#         sentence = row["sentence"]  # Get sentence

#         expected_order = row["expected_order"].split(',')  # Split expected order string
#         expected_order[0] = expected_order[0].lstrip('[')  # Remove opening bracket
#         expected_order[-1] = expected_order[-1].rstrip(']')  # Remove closing bracket
#         expected_order = [x.strip().replace("'", "") for x in expected_order]  # Clean the list

#         target = list(range(1, len(expected_order)+1))  # Create target indices
#         image_to_idx = {img: i for i, img in enumerate(expected_order)}  # Map images to indices
#         idx_to_image = {i: img for i, img in enumerate(expected_order)}  # Map indices to images

#         shuffled_order = expected_order.copy()  # Copy expected order
#         random.shuffle(shuffled_order)  # Shuffle the order
#         shuffled_target = [image_to_idx[img] for img in shuffled_order]  # Map shuffled images to indices

#         image_folder = os.path.join(self.dataset_dir, idiom)  # Get image folder path
#         image_vectors = []  # List to store image vectors
#         missing_images = []  # List to track missing images

#         for img_name in shuffled_order:  # Process each image
#             img_path = os.path.join(image_folder, img_name)  # Construct image path
#             if not os.path.exists(img_path):  # Check if image exists
#                 missing_images.append(img_name)  # Record missing image
#                 continue
#             try:
#                 image = Image.open(img_path).convert("RGB")  # Load and convert image to RGB
#                 image = self.transform(image).unsqueeze(0)  # Apply transform and add batch dimension
#                 with torch.no_grad():  # Disable gradient for efficiency
#                     vector = self.resnet(image)  # Get image vector from ResNet50 with projection
#                 image_vectors.append(vector.squeeze())  # Add vector to list (remove batch dimension)
#             except Exception as e:  # Handle image loading errors
#                 print(f"Error loading image: {img_path}, {e}")  # Print error
#                 missing_images.append(img_name)  # Record error

#         if len(image_vectors) != len(shuffled_order):  # Check for missing vectors
#             print(f"Skipping idiom '{idiom}' due to missing images: {missing_images}")  # Warn about skipping
#             return None

#         image_vectors = torch.stack(image_vectors)  # Stack vectors into a tensor

#         return {
#             "idiom": idiom,  # Idiom name
#             "sentence": sentence,  # Sentence
#             "actual_order": expected_order,  # Original order
#             "shuffled_order": shuffled_order,  # Shuffled order
#             "actual_target": target,  # Original targets
#             "shuffled_target": shuffled_target,  # Shuffled targets
#             "image_vectors": image_vectors,  # Tensor of image vectors
#             "img_to_idx": image_to_idx,  # Image to index mapping
#             "idx_to_img": idx_to_image  # Index to image mapping
#         }