import sys
sys.path.append(r"F:\Sem7\Thesis\CODE\src")
from idiomDatasetLoader import IdiomDataset
from vectorComparison import rank_images, evaluate_ranking
import torch

dataset_path = r"F:\Sem7\Thesis\CODE\dataset"
dataset = "test"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = IdiomDataset(dataset_path=dataset_path, dataset=dataset)
print(f"Total samples: {len(dataset)}")

results = []
total_top1_acc = 0.0  # Track total Top-1 Accuracy
total_ndcg = 0.0      # Track total NDCG
valid_samples = 0     # Count valid samples

for i in range(len(dataset)):
    sample = dataset[i]
    if sample is None:
        continue
    valid_samples += 1
    ranked_indices = rank_images(sample["sentence"], sample["image_vectors"], device)
    top1_acc, ndcg = evaluate_ranking(sample["shuffled_order"], ranked_indices, sample["img_to_idx"])
    total_top1_acc += top1_acc  # Accumulate Top-1 Accuracy
    total_ndcg += ndcg          # Accumulate NDCG
    print(f"Sample {i}: Top-1 Acc = {top1_acc}, NDCG = {ndcg}")  # Debug print
    results.append({
        "idiom": sample["idiom"],
        "sentence": sample["sentence"],
        "actual_order": sample["actual_order"],
        "shuffled_order": sample["shuffled_order"],
        "ranked_order": [sample["idx_to_img"][i] for i in ranked_indices],
        "top1_acc": top1_acc,
        "ndcg": ndcg
    })

# Print per-sample results (first 5 samples)
for result in results[:5]:
    print(f"Idiom: {result['idiom']}")
    print(f"Sentence: {result['sentence']}")
    print(f"Actual Order: {result['actual_order']}")
    print(f"Shuffled Order: {result['shuffled_order']}")
    print(f"Ranked Order: {result['ranked_order']}")
    print(f"Top-1 Accuracy: {result['top1_acc']}")
    print(f"NDCG: {result['ndcg']}\n")

# Compute and print average metrics
if valid_samples > 0:
    avg_top1_acc = total_top1_acc / valid_samples
    avg_ndcg = total_ndcg / valid_samples
    print(f"Average Top-1 Accuracy across all samples: {avg_top1_acc:.2f}")
    print(f"Average NDCG across all samples: {avg_ndcg:.2f}")
else:
    print("No valid samples to compute average metrics.")