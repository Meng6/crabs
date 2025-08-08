import json, glob, re

ground_truth_path = "data/ProspectiveCorrectGivenSpecification.json"
records_path = "data/outputs/records/gpt-4o_log_crabs_"

with open(ground_truth_path, "r") as fin:
    ground_truth = json.load(fin)

instance_pred = []
instance_true = []
for file_path in sorted(glob.glob(f"{records_path}*.json")):
    notebook = file_path.replace(records_path, "").replace(".json", "")
    for item in ground_truth:
        if item[0]['id'] != "-".join(notebook.split("-")[1:]):
            continue

        with open(file_path, "r") as fin:
            cells = json.load(fin)
        for idx, cell in enumerate(cells):
            for qa in cell["log"]:
                if "prompt4in" in qa:
                    true_in = [x[0] if type(x) == list else x for x in (item[idx + 1]["inputs"] if "inputs" in item[idx + 1] else [])]
                    
                    pattern = r'In the following Python program block, is "([^"]+)" an input\?'
                    input_var = re.findall(pattern, qa["prompt4in"])[0]
                    
                    instance_pred.append(1 if qa["Answer"].startswith("Yes") else 0) # True if it is an input
                    instance_true.append(1 if input_var in true_in else 0)
                elif "prompt4out" in qa:
                    true_out = [x[0] if type(x) == list else x for x in (item[idx + 1]["outputs"] if "outputs" in item[idx + 1] else [])]
                    pattern = r'In the following Python program block, is "([^"]+)" an output candidate\?'
                    output_var = re.findall(pattern, qa["prompt4out"])[0]

                    instance_pred.append(1 if qa["Answer"].startswith("Yes") else 0) # True if it is an output
                    instance_true.append(1 if output_var in true_out else 0)
                else:
                    raise ValueError("Invalid key in log. Only 'prompt4in' or 'prompt4out' or 'Answer' are allowed.")
                
assert len(instance_pred) == len(instance_true)

# Calculate accuracy
accuracy = sum(p == t for p, t in zip(instance_pred, instance_true)) / len(instance_pred)
print(f"Accuracy: {accuracy:.4f}")
print(f"Number of instances: {len(instance_pred)}")
print(f"Number of correct instances: {sum(p == t for p, t in zip(instance_pred, instance_true))}")