import json

# mapping from ground_truth letters to solver labels
mapping = {
    "A": "True",
    "B": "False",
    "C": "Unknown"
}

# Load input.json
with open("ProntoQA_gpt-4o_trans_decompose_no_negation.json", "r") as f:
    input_data = json.load(f)

# Load output.txt (assuming it's JSON too, otherwise adapt parsing)
with open("output_answer.json", "r") as f:
    output_data = json.load(f)

# Convert output_data into dict for fast lookup by id
solver_dict = {entry["id"]: entry["solver_label"] for entry in output_data}

# Compare
results = []
correct = 0
total = 0

for entry in input_data:
    _id = entry["id"]
    gt = entry["ground_truth"]
    expected_label = mapping[gt]

    if _id in solver_dict:
        solver_label = solver_dict[_id]
        match = (solver_label == expected_label)
        results.append({
            "id": _id,
            "ground_truth": gt,
            "expected_label": expected_label,
            "solver_label": solver_label,
            "match": match
        })
        total += 1
        if match:
            correct += 1

accuracy = correct / total if total > 0 else 0

print(f"Total compared: {total}")
print(f"Correct matches: {correct}")
print(f"Accuracy: {accuracy:.2%}")

# Optionally save results
with open("comparison_results.json", "w") as f:
    json.dump(results, f, indent=2)
