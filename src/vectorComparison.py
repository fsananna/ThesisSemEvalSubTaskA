import torch
from transformers import BertTokenizer, BertModel
from torch.nn.functional import cosine_similarity
from evaluationMetrics import topKAcc, NDCG

def get_text_vector(sentence, device="cpu"):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased").to(device)
    model.eval()
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze()

def rank_images(sentence, image_vectors, device="cpu"):
    text_vector = get_text_vector(sentence, device)
    similarities = []
    for i, img_vector in enumerate(image_vectors):
        sim = cosine_similarity(text_vector.unsqueeze(0), img_vector.unsqueeze(0)).item()
        similarities.append((i, sim))
    similarities.sort(key=lambda x: x[1], reverse=True)
    ranked_indices = [x[0] for x in similarities]
    return ranked_indices

def evaluate_ranking(actual_order, predicted_indices, img_to_idx):
    actual_indices = [img_to_idx[img] for img in actual_order]
    top1_acc = topKAcc(actual_indices, predicted_indices, K=1)
    ndcg = NDCG(actual_indices, [actual_indices[i] for i in predicted_indices])
    return top1_acc, ndcg