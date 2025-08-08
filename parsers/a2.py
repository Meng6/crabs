from parsers.yw_generator import extract_records
import re, json

def remove_inline_comments(code):
    return re.sub(r"#.*", "", code)

with open("parsers/prompts/a2.txt", "r") as fin:
    base_prompt = fin.read()

def extract_records_using_a2(code_cells, model, temperature, client):
    definite_records = extract_records(code_cells, is_definite=True)
    possible_records = extract_records(code_cells, is_definite=False)

    context_notebook = ""
    for idx, (code_cell, definite_record, possible_record) in enumerate(zip(code_cells, definite_records, possible_records)):
        context_notebook = context_notebook + """
Cell-{k}:
```python
# definite inputs: [{definite_inputs}]; outputs: [{definite_outputs}]
# possible inputs: [{possible_inputs}]; outputs: [{possible_outputs}]

{code}
```
""".format(k=idx+1, 
           definite_inputs=", ".join(definite_record["inputs"]), 
           definite_outputs=", ".join(definite_record["outputs"]), 
           possible_inputs=", ".join(possible_record["inputs"]), 
           possible_outputs=", ".join(possible_record["outputs"]),
           code=remove_inline_comments(code_cell))
    
    prompt = base_prompt.replace("[[CONTEXT_NOTEBOOK]]", context_notebook)

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful code assistant designed to output a JSON string."},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature
    )

    out = completion.choices[0].message.content

    try:
        match = re.search(r'```json\n(.*?)\n```', out, re.DOTALL)
        records = json.loads(match.group(1))
    except:
        records = None
    
    records = records if records and len(records) == len(possible_records) else None
    if records:
        for idx, possible_record in enumerate(possible_records):
            records[idx]["refers_code"] = list(possible_record["refers_code"])
            records[idx]["defines_code"] = list(possible_record["defines_code"])

    return records