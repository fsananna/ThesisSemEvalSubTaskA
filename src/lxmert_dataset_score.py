# File: src/lxmert_dataset_score.py

import sys
sys.path.append(r"F:\Sem7\Thesis\CODE\src")
from idiomDatasetLoader import IdiomDataset
from transformers import LxmertTokenizer, LxmertModel
import torch
from torch.nn.functional import cosine_similarity

tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")
model = LxmertModel.from_pretrained("unc-nlp/lxmert-base-uncased")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

dataset_path = r"F:\Sem7\Thesis\CODE\dataset"
dataset = "test"
dataset = IdiomDataset(dataset_path=dataset_path, dataset=dataset)

results = []
for i in range(len(dataset)):
    sample = dataset[i]
    if sample is None:
        continue
    
    sentence = sample["sentence"]
    image_vectors = sample["image_vectors"]
    shuffled_order = sample["shuffled_order"]
    idiom = sample["idiom"]
    actual_order = sample["actual_order"]

    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=20)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    example_image_vector = image_vectors[0].unsqueeze(0).to(device)
    example_inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=20)
    example_input_ids = example_inputs["input_ids"].to(device)
    example_attention_mask = example_inputs["attention_mask"].to(device)

    visual_pos = torch.zeros((1, 4), device=device)

    scores = []
    for j in range(1, len(image_vectors)):  # Exclude the example (index 0)
        target_image_vector = image_vectors[j].unsqueeze(0).to(device)

        with torch.no_grad():
            example_outputs = model(
                input_ids=example_input_ids,
                attention_mask=example_attention_mask,
                visual_feats=example_image_vector.unsqueeze(1),
                visual_pos=visual_pos
            )
            example_pooled_output = example_outputs.pooled_output

            target_outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                visual_feats=target_image_vector.unsqueeze(1),
                visual_pos=visual_pos
            )
            target_pooled_output = target_outputs.pooled_output

            score = cosine_similarity(example_pooled_output, target_pooled_output).item()
            scores.append(score)

    for j, (image_name, score) in enumerate(zip(shuffled_order[1:], scores)):
        results.append({
            "idiom": idiom,
            "sentence": sentence,
            "image_name": image_name,
            "score": score
        })

    print(f"Sample {i}: Processed {len(shuffled_order) - 1} images for idiom '{idiom}'")

for i, result in enumerate(results):
    print(f"Idiom: {result['idiom']}")
    print(f"Sentence: {result['sentence']}")
    print(f"Image: {result['image_name']}")
    print(f"Score: {result['score']:.4f}\n")

total_scores = [result["score"] for result in results]
overall_average = sum(total_scores) / len(total_scores) if total_scores else 0
print(f"Overall Average Score Across All Images: {overall_average:.4f}")

idiom_averages = {}
for result in results:
    idiom = result["idiom"]
    if idiom not in idiom_averages:
        idiom_averages[idiom] = []
    idiom_averages[idiom].append(result["score"])

for idiom, scores in idiom_averages.items():
    average = sum(scores) / len(scores)
    print(f"Average Score for Idiom '{idiom}': {average:.4f}")

for i in range(0, len(results), len(actual_order) - 1):
    sample_results = results[i:i + len(actual_order) - 1]
    if not sample_results:
        continue
    best_match = max(sample_results, key=lambda x: x["score"])
    try:
        best_idx = actual_order.index(best_match["image_name"])
        print(f"Best match for idiom '{best_match['idiom']}': Image {best_match['image_name']} with score {best_match['score']:.4f} (Index in actual order: {best_idx})")
    except ValueError:
        print(f"Best match for idiom '{best_match['idiom']}': Image {best_match['image_name']} with score {best_match['score']:.4f} (Index not found in actual order)")