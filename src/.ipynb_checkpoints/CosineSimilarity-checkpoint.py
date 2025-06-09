import os
import torch
import pandas as pd
import torch.nn.functional as F
from torchvision import models, transforms
from transformers import BertTokenizer, BertModel
from PIL import Image
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.models")

# Unique identifier to verify file version
print("Using updated CosineSimilarity.py - May 31, 2025, 12:10 PM +06")

class ImageTextSimilarity:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")

        # Load BERT model and tokenizer
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert_model = BertModel.from_pretrained("bert-base-uncased").to(self.device)
        self.bert_model.eval()
        print("BERT model loaded")

        # Load pretrained ResNet50 model (remove final layer)
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        modules = list(resnet.children())[:-1]
        self.resnet_model = torch.nn.Sequential(*modules).to(self.device)
        self.resnet_model.eval()
        print("ResNet50 model loaded")

        # Add projection layer to match ResNet50 (2048) to BERT (768)
        self.projection = torch.nn.Linear(2048, 768).to(self.device)
        self.projection.eval()
        print("Projection layer initialized:", self.projection)

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def get_sentence_embedding(self, sentence):
        inputs = self.tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
        print(f"Sentence embedding shape: {embedding.shape}")
        return embedding

    def get_image_embedding(self, image_path):
        try:
            image = Image.open(image_path).convert("RGB")
            image = self.transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                embedding = self.resnet_model(image).squeeze()
                print(f"Raw ResNet50 embedding shape for {image_path}: {embedding.shape}")
                embedding = self.projection(embedding)  # Project to 768 dimensions
                print(f"Projected embedding shape for {image_path}: {embedding.shape}")
            return embedding
        except Exception as e:
            print(f"Error loading or processing image {image_path}: {e}")
            return None

    def cosine_similarity(self, vec1, vec2):
        if vec1 is None or vec2 is None:
            print(f"Cosine similarity failed: vec1={vec1}, vec2={vec2}")
            return -1.0
        print(f"Cosine similarity shapes: vec1={vec1.shape}, vec2={vec2.shape}")
        return F.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0)).item()

    def process_test_data(self):
        df = pd.read_csv(self.dataset_path, sep="\t")
        results = []

        root_dir = os.path.dirname(self.dataset_path)
        print(f"Root directory: {root_dir}")

        for _, row in df.iterrows():
            sentence = row["sentence"]
            compound = row["compound"]
            image_names = row["expected_order"].strip("[]").replace("'", "").split(", ")

            sentence_embedding = self.get_sentence_embedding(sentence)

            image_scores = []
            image_folder = os.path.join(root_dir, compound)
            print(f"Processing folder: {image_folder}")

            for img_name in image_names:
                img_path = os.path.join(image_folder, img_name.strip())
                if os.path.exists(img_path):
                    print(f"Found image: {img_path}")
                    image_embedding = self.get_image_embedding(img_path)
                    if image_embedding is not None:
                        score = self.cosine_similarity(sentence_embedding, image_embedding)
                        image_scores.append((img_name, score))
                else:
                    print(f"Missing image: {img_path}")

            image_scores.sort(key=lambda x: x[1], reverse=True)
            ranked_images = [img for img, _ in image_scores]
            print(f"Image scores for {compound}: {image_scores}")

            results.append({
                "sentence": sentence,
                "compound": compound,
                "ranked_images": ranked_images
            })

        return results