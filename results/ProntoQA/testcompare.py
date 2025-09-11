import json

# mapping ground_truth -> solver_label
mapping = {
    "A": "True",
    "B": "False",
    "C": "Unknown"
}

# Load input.json
with open("ProntoQA_gpt-4o_trans_decompose_no_negation.json", "r") as f:
    input_data = json.load(f)

# Load output.txt (assuming it's valid JSON)
with open("output_answer.json", "r") as f:
    output_data = json.load(f)

# Convert solver output into dictionary keyed by id
solver_dict = {entry["id"]: entry["solver_label"] for entry in output_data}

# Track missing ids
missing_ids = []
for entry in input_data:
    _id = entry["id"]
    if _id not in solver_dict:
        missing_ids.append(_id)

# Report missing
if missing_ids:
    print(f"❌ {len(missing_ids)} IDs from input are missing in solver output.")
    print("Example missing IDs:", missing_ids[:10])  # show first 10 for readability
else:
    print("✅ All input IDs exist in solver output.")

# (Optional) Still run mismatch check
for entry in input_data:
    _id = entry["id"]
    gt = entry["ground_truth"]
    expected_label = mapping.get(gt)

    if _id in solver_dict:
        solver_label = solver_dict[_id]
        if solver_label != expected_label:
            print("❌ First mismatch found:")
            print(json.dumps({
                "id": _id,
                "ground_truth": gt,
                "expected_label": expected_label,
                "solver_label": solver_label
            }, indent=2))
            break
else:
    if not missing_ids:
        print("✅ All entries match!")
