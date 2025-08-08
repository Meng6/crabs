from parsers.yw_generator import extract_records

with open("parsers/prompts/crabs_subtask1.txt", "r") as fin:
    base_prompt4in = fin.read()
with open("parsers/prompts/crabs_subtask2.txt", "r") as fin:
    base_prompt4out = fin.read()

def extract_records_using_crabs(code_cells, model, temperature, client):
    definite_records = extract_records(code_cells, is_definite=True)
    possible_records = extract_records(code_cells, is_definite=False)

    crabs_records = []
    for idx, (code_cell, definite_record, possible_record) in enumerate(zip(code_cells, definite_records, possible_records)):
        
        ans_log = []
        cell_inputs = definite_record["inputs"].copy()
        cell_outputs = definite_record["outputs"].copy()

        # Context for alias analysis
        if possible_record["alias_stmt"]:
            context_python_alias = """
Preceding context (for shared reference):
```python
{alias_stmt}
```
""".format(alias_stmt=possible_record["alias_stmt"])
        else:
            context_python_alias = ""

        # Resolve conflicts for inputs
        context_alias_rule = "2. Shared References (Aliased Variables)\n" + \
                             "- If multiple variables reference the same object (e.g., through assignment or being stored inside a data structure), and one of them is an input, then all variables referring to that object are also inputs.\n" + \
                             "- If an container is an input, then all elements inside the container are also inputs.\n" + \
                             "- If an element inside a container is an input, then the container itself is also an input."
        if definite_record["inputs"] != possible_record["inputs"]:
            for var in possible_record["inputs"] - definite_record["inputs"]:
                context_python_alias = "" if var not in possible_record["alias_vars"] else context_python_alias
                context_alias_rule = "" if var not in possible_record["alias_vars"] else context_alias_rule
                prompt4in = base_prompt4in.replace("[[CONTEXT_PYTHON]]", code_cell).replace("[[CONTEXT_VAR]]", var).replace("[[CONTEXT_PYTHON_SHARED_REFERENCES]]", context_python_alias).replace("[[CONTEXT_SHARED_REFERENCES_RULE]]", context_alias_rule)
                completion = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a helpful code assistant designed to output Yes or No only."},
                        {"role": "user", "content": prompt4in}
                    ],
                    temperature=temperature
                )
                is_in = completion.choices[0].message.content
                ans_log.append({"prompt4in": prompt4in, "Answer": is_in})
                if is_in.startswith("Yes"):
                    cell_inputs.add(var)
        
        # Resolve conflicts for outputs
        context_alias_rule = "3. Shared References (Aliased Variables)\n" + \
                             "- If multiple variables reference the same mutable object (e.g., through assignment or being stored inside a data structure), modifying the object in place through one reference makes all references to that object output candidates.\n" + \
                             "- Modifying the container in place makes only the container an output candidate, but not its elements.\n" + \
                             "- Modifying an element inside a container in place makes both the container and the modified element output candidates.\n" + \
                             "- Operations like retrieving data, describing data, visualizing data, accessing properties, and creating a copy are not considered as modifying the object in place."
        
        if definite_record["outputs"] != possible_record["outputs"]:
            for var in possible_record["outputs"] - definite_record["outputs"]:
                context_python_alias = "" if var not in possible_record["alias_vars"] else context_python_alias
                context_alias_rule = "" if var not in possible_record["alias_vars"] else context_alias_rule
                prompt4out = base_prompt4out.replace("[[CONTEXT_PYTHON]]", code_cell).replace("[[CONTEXT_VAR]]", var).replace("[[CONTEXT_PYTHON_SHARED_REFERENCES]]", context_python_alias).replace("[[CONTEXT_SHARED_REFERENCES_RULE]]", context_alias_rule)
                completion = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a helpful code assistant designed to output Yes or No only."},
                        {"role": "user", "content": prompt4out}
                    ],
                    temperature=temperature
                )
                is_out = completion.choices[0].message.content
                ans_log.append({"prompt4out": prompt4out, "Answer": is_out})
                if is_out.startswith("Yes"):
                    cell_outputs.add(var)
        
        crabs_records.append({
            "cell": idx + 1, 
            "inputs": list(cell_inputs), 
            "outputs": list(cell_outputs), 
            "refers_code": list(definite_record["refers_code"]), 
            "defines_code": list(definite_record["defines_code"]),
            "log": ans_log
        })
    return crabs_records