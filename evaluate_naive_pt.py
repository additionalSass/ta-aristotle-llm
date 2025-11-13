import json
import argparse


def load_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def map_final_answer_to_choice(final_answer_raw):
    if final_answer_raw is None:
        return 'C'

    s = str(final_answer_raw).strip().upper()

    if s in ['TRUE', 'True', 'T','A']:
        return 'A'

    if s in ['FALSE', 'False', 'F', 'No', 'B']:
        return 'B'

    if s in ['UNKNOWN', 'C', "Unknown", "Not enough information", "unknown"]:
        return 'C'

    if s in ['SELF-CONTRADICTORY', 'CONTRADICTORY', 'SELF_CONTRADICTORY', 'D']:
        return 'D'

    # Default , we can assume unknown
    return 'C'


def evaluate_instance(item):
    """
    Evaluate a single instance:
    - Read ground_truth ('A'/'B'/'C'/'D')
    - Map final_answer text to a choice
    - Return (is_correct, predicted_choice)
    """
    ground_truth = item.get('ground_truth', None)
    final_answer_raw = item.get('final_answer', None)

    if ground_truth is None:
        # No label means that we can't evaluate and assume it is False 
        return False, None

    pred_choice = map_final_answer_to_choice(final_answer_raw)
    is_correct = (pred_choice == ground_truth)
    return is_correct, pred_choice


def evaluate_file(dataset_name, model_name):

    file_path = f'./results/{dataset_name}/{dataset_name}_{model_name}_naive_prompt.json'
    data = load_json_file(file_path)

    total_instances = 0
    correct_instances = 0
    error_ids = []
    correct_ids = []

    for item in data:
        id_ = item.get('id', 'UNKNOWN_ID')
        ground_truth = item.get('ground_truth', None)

        if not ground_truth:
            # Skip items without ground truth
            continue

        total_instances += 1
        is_correct, pred_choice = evaluate_instance(item)

        if is_correct:
            correct_instances += 1
            correct_ids.append(id_)
        else:
            error_ids.append(id_)

    accuracy = correct_instances / total_instances if total_instances > 0 else 0.0

    print(f"Total instances: {total_instances}")
    print(f"Accuracy: {accuracy:.2%}")
    print("Error id:", error_ids)
    print("Correct id:", correct_ids)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    evaluate_file(args.dataset_name, args.model_name)