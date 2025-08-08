from pydantic import BaseModel
from typing import List

class CellInfo(BaseModel):
    inputs: List[str]
    output_candidates: List[str]
    defines_code: List[str]
    refers_code: List[str]
    shared_references: str

with open("parsers/prompts/a1.txt", "r") as fin:
    base_prompt = fin.read()

def extract_records_using_a1(code_cells, model, temperature, client):

    llmonly_records = []
    context_shared_references, context_funcs_and_classes = "NA", set()
    for idx, code_cell in enumerate(code_cells):
        prompt = base_prompt.replace("[[CONTEXT_SHARED_REFERENCES]]", context_shared_references) \
                            .replace("[[CONTEXT_PYTHON]]", code_cell) \
                            .replace("[[CONTEXT_FUNCS_AND_CLASSES]]", ", ".join(context_funcs_and_classes))
        
        completion = client.beta.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful code assistant designed to output a JSON string."},
                {"role": "user", "content": prompt}
            ],
            response_format=CellInfo,
            temperature=temperature
        )
        cell_info = completion.choices[0].message.parsed.model_dump()
        context_shared_references = cell_info["shared_references"]
        context_funcs_and_classes.update(cell_info["defines_code"])

        cell_info["cell"] = idx + 1
        cell_info["log"] = prompt
        
        llmonly_records.append(cell_info)
    
    # Extract outputs by filtering out output candidates which are not used by any subsequent cells
    required_variables = set()
    for idx in range(len(llmonly_records)-1, -1, -1): # reverse order
        inputs, output_candidates = set(llmonly_records[idx]['inputs']), set(llmonly_records[idx]['output_candidates'])
        outputs = output_candidates & required_variables
        required_variables = (required_variables - outputs) | inputs
        llmonly_records[idx]['outputs'] = list(outputs)

    return llmonly_records