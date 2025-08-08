import re, json

def remove_inline_comments(code):
    return re.sub(r"#.*", "", code)

with open("parsers/prompts/baseline.txt", "r") as fin:
    base_prompt = fin.read()

def extract_records_using_baseline_approach(code_cells, model, temperature, client):

    context_notebook = ""
    for idx, code_cell in enumerate(code_cells):
        context_notebook = context_notebook + """
Cell-{k}:
```python
{code}
```
""".format(k=idx+1,
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
    if not records:
        print("UNABLE TO PARSE")
        
    if records and len(records) != len(code_cells):
        print(len(records), len(code_cells))
    records = records if records and len(records) == len(code_cells) else None

    return records