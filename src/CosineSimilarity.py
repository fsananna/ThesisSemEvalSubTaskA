import os
import sys
import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
from torchvision import models, transforms
from transformers import BertTokenizer, BertModel
from PIL import Image

class ImageTextSimilarity:
    def __init__(self, dataset_path):
        """
        Initializes the model for extracting sentence embeddings (BERT) and image embeddings (ResNet50).
        """
        self.dataset_path = dataset_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load BERT model and tokenizer
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert_model = BertModel.from_pretrained("bert-base-uncased").to(self.device)
        self.bert_model.eval()  # Set to evaluation mode

        # Load Pretrained ResNet50 model
        self.resnet_model = models.resnet50(pretrained=True).to(self.device)
        self.resnet_model.eval()  # Set to evaluation mode
        self.resnet_model = torch.nn.Sequential(*list(self.resnet_model.children())[:-1])  # Remove final layer

        # Image transformation pipeline
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def get_sentence_embedding(self, sentence):
        """
        Generates a sentence embedding using BERT.
        """
        inputs = self.tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
        
        sentence_embedding = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
        return sentence_embedding

    def get_image_embedding(self, image_path):
        """
        Extracts an image feature vector using ResNet50.
        """
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.resnet_model(image)
        
        return features.squeeze().flatten()  # Flatten to 1D vector

    def cosine_similarity(self, vec1, vec2):
        """
        Computes cosine similarity between two vectors.
        """
        return F.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0)).item()

    def process_test_data(self):
        """
        Reads the test dataset TSV file, computes similarity scores, and ranks images.
        """
        # Load test dataset
        test_file = self.dataset_path
        data = pd.read_csv(test_file, sep="\t")

        results = []

        for _, row in data.iterrows():
            sentence = row["sentence"]
            compound = row["compound"]
            
            # Get sentence embedding
            sentence_embedding = self.get_sentence_embedding(sentence)

            # Get expected order (list of image filenames)
            image_names = row["expected_order"].strip("[]").replace("'", "").split(", ")
            
            # Compute similarity scores
            image_scores = []
            image_folder = os.path.join(self.dataset_path, compound)

            for img_name in image_names:
                img_name = img_name.strip()  # Clean image name
                img_path = os.path.join(image_folder, img_name)
                
                if os.path.exists(img_path):
                    image_vector = self.get_image_embedding(img_path)
                    similarity_score = self.cosine_similarity(sentence_embedding, image_vector)
                    image_scores.append((img_name, similarity_score))

            # Rank images by similarity score (descending order)
            image_scores.sort(key=lambda x: x[1], reverse=True)
            ranked_images = [img[0] for img in image_scores]

            # Store results
            results.append({
                "sentence": sentence,
                "compound": compound,
                "ranked_images": ranked_images
            })

        return results


