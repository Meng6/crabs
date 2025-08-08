import nbformat, ast, copy, re

def clean_code(cell_code):
    # Remove syntaxes for line magics (%time, %timeit) along with any options (-r2, -n10, etc.)
    cleaned_code = re.sub(r"^\s*%time(it)?\s+(-\S+\s+)*", "", cell_code.strip(), flags=re.MULTILINE)
    # Remove syntaxes for cell magics (%%time, %%timeit)
    cleaned_code = re.sub(r"^\s*%%time(it)?\s+(-\S+\s+)*", "", cleaned_code, flags=re.MULTILINE)

    # Remove cells that start with %%
    if cleaned_code.startswith("%%"):
        cleaned_code = ""
    else: # Remove lines starts with % or !
        cleaned_code = re.sub(r"^\s*[%!].*\n?", "", cleaned_code, flags=re.MULTILINE)
    
    return cleaned_code

def extract_code_cells(notebook_path):
    """
    Extract code cells from a given Jupyter Notebook.
    :param notebook_path: String. Path of a Jupyter Notebook.
    :return code_cells: a list of code cells.
    """
    with open(notebook_path, 'r') as f:
        notebook = nbformat.read(f, as_version=4)
    code_cells = [clean_code(cell['source']) for cell in notebook.cells if cell.cell_type == 'code']
    return code_cells

class MinIOVisitor(ast.NodeVisitor):
    def __init__(self, alias_mapping={}, modules=set()):
        self.inputs = set()
        self.outputs = set()
        self.refers_code = set()
        self.defines_code = set()
        self.alias_mapping = alias_mapping
        self.modules = modules
        self.loopflag = False
    
    def visit_Import(self, node):
        for alias in node.names:
            if alias.asname:
                self.modules.add(alias.asname)
            else:
                self.modules.add(alias.name)
    
    def visit_ImportFrom(self, node):
        self.visit_Import(node)
    
    def _visit_list_of_nodes(self, nodes):
        for item in nodes:
            prev_inputs, prev_outputs = self.inputs.copy(), self.outputs.copy()
            self.visit(item)
            # Exclude previous outputs from current inputs
            curr_inputs = self.inputs - prev_inputs
            self.inputs = (curr_inputs - prev_outputs) | prev_inputs
    
    def visit_Try(self, node):
        # Union of the try-except-finally blocks
        body_inputs, body_outputs = set(), set()
        for item in node.handlers:
            self._visit_list_of_nodes(node.body + item.body + node.finalbody)
            body_inputs.update(self.inputs)
            body_outputs.update(self.outputs)
            self.inputs, self.outputs = set(), set()
        self._visit_list_of_nodes(node.body + node.orelse + node.finalbody)
        body_inputs.update(self.inputs)
        body_outputs.update(self.outputs)
    
    def visit_TryStar(self, node):
        self.visit_Try(node)

    def visit_If(self, node):
        prev_inputs, prev_outputs = self.inputs.copy(), self.outputs.copy()
        # In a loop
        if self.loopflag:
            # Order 1
            self._visit_list_of_nodes([node.test] + node.body + node.orelse)
            order1_inputs, order1_outputs = self.inputs.copy(), self.outputs.copy()
            self.inputs, self.outputs = prev_inputs.copy(), prev_outputs.copy()
            # Order 2
            self._visit_list_of_nodes([node.test] + node.orelse + node.body)
            # Keep the intersection of these two orders
            self.inputs = order1_inputs & self.inputs
            self.outputs = order1_outputs & self.outputs
        else:
            # Union of the if-else branches
            self._visit_list_of_nodes([node.test] + node.body)
            union_inputs, union_outputs = self.inputs.copy(), self.outputs.copy()
            self.inputs, self.outputs = prev_inputs.copy(), prev_outputs.copy()
            self._visit_list_of_nodes([node.test] + node.orelse)
            self.inputs.update(union_inputs)
            self.outputs.update(union_outputs)
    
    def visit_For(self, node):
        self.loopflag = True
        self._visit_list_of_nodes([node.iter, node.target] + node.body + node.orelse)
    
    def visit_AsyncFor(self, node):
        self.loopflag = True
        self._visit_list_of_nodes([node.iter, node.target] + node.body + node.orelse)
    
    def visit_While(self, node):
        self.loopflag = True
        self._visit_list_of_nodes([node.test] + node.body + node.orelse)
    
    def visit_comprehension(self, node):
        self._visit_list_of_nodes([node.iter, node.target] + node.ifs)
    
    def visit_ListComp(self, node):
        self._visit_list_of_nodes(node.generators + [node.elt])
    
    def visit_SetComp(self, node):
        self._visit_list_of_nodes(node.generators + [node.elt])
    
    def visit_DictComp(self, node):
        self._visit_list_of_nodes(node.generators + [node.key, node.value])
    
    def visit_GeneratorExp(self, node):
        self._visit_list_of_nodes(node.generators + [node.elt])

    def visit_With(self, node):
        self._visit_list_of_nodes(node.items + node.body)
    
    def visit_AugAssign(self, node):
        if isinstance(node.target, ast.Name):
            """
            Add the variable as an input as well
            Example: print(ast.dump(ast.parse('a += 1'), indent=4))
            # a is both an input and an output
            Module(
                body=[
                    AugAssign(
                        target=Name(id='a', ctx=Store()),
                        op=Add(),
                        value=Constant(value=1))],
                type_ignores=[])
            """
            self.inputs.add(node.target.id)
        self.generic_visit(node)

    def _visit_Name_add2inorout(self, node):
        if isinstance(node.ctx, ast.Load):
            self.inputs.add(node.id)
        elif isinstance(node.ctx, (ast.Store, ast.Del)):
            self.outputs.add(node.id)
    
    def _visit_Name_add2in(self, node):
        self.inputs.add(node.id)
        self._visit_Name_add2inorout(node)
        
    def _visit_Name_add2out(self, node):
        self.outputs.add(node.id)
        self._visit_Name_add2inorout(node)
    
    def _visit_Name_add2inandout(self, node):
        self.inputs.add(node.id)
        self.outputs.add(node.id)

    def visit_Name(self, node):
        """
        Reference: https://docs.python.org/3/library/ast.html#ast.Name
        Example: print(ast.dump(ast.parse('b = a + 1'), indent=4))
        Module(
            body=[
                Assign(
                    targets=[
                        Name(id='b', ctx=Store())],
                    value=BinOp(
                        left=Name(id='a', ctx=Load()),
                        op=Add(),
                        right=Constant(value=1)))],
            type_ignores=[])
        """
        self._visit_Name_add2inorout(node)
    
    def visit_AsyncFunctionDef(self, node):
        self.defines_code.add(node.name)

    def visit_FunctionDef(self, node):
        self.defines_code.add(node.name)
    
    def visit_ClassDef(self, node):
        self.defines_code.add(node.name)
    
    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            self.refers_code.add(node.func.id)
        self.generic_visit(node)
    
    def visit_Lambda(self, node):
        self.generic_visit(node.body)
        for arg in node.args.args:
            if arg.arg in self.inputs:
                self.inputs.remove(arg.arg)
            if arg.arg in self.outputs:
                self.outputs.remove(arg.arg)

    def _reset_name_for_store_and_del(self, prev_visit_name_func):
        if prev_visit_name_func != self._visit_Name_add2inandout:
            if self.visit_Name == self._visit_Name_add2in:
                self.visit_Name = self._visit_Name_add2inandout
            else:
                self.visit_Name = self._visit_Name_add2out

    def _reset_name_for_load(self, prev_visit_name_func):
        if prev_visit_name_func != self._visit_Name_add2inandout:
            if self.visit_Name == self._visit_Name_add2out:
                self.visit_Name = self._visit_Name_add2inandout
            else:
                self.visit_Name = self._visit_Name_add2in

    def visit_Attribute(self, node):
        prev_visit_name_func = self.visit_Name
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self._reset_name_for_store_and_del(prev_visit_name_func)
            self.generic_visit(node)
            self.visit_Name = prev_visit_name_func
        if isinstance(node.ctx, ast.Load):
            self._reset_name_for_load(prev_visit_name_func)
            self.generic_visit(node)
            self.visit_Name = prev_visit_name_func

    def visit_Subscript(self, node):
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self.visit(node.slice)
            prev_visit_name_func = self.visit_Name
            self._reset_name_for_store_and_del(prev_visit_name_func)
            self.visit(node.value)
            self.visit_Name = prev_visit_name_func
        if isinstance(node.ctx, ast.Load):
            prev_visit_name_func = self.visit_Name
            self._reset_name_for_load(prev_visit_name_func)
            self.generic_visit(node)
            self.visit_Name = prev_visit_name_func

    def visit(self, node):
        super().visit(node)
        return self.inputs - self.refers_code, self.outputs - self.refers_code, self.refers_code, self.defines_code, self.alias_mapping, self.modules

class MaxIOVisitor(MinIOVisitor):

    def _reset_name_before(self, prev_visit_name_func):
        self.visit_Attribute = self.generic_visit
        self.visit_Subscript = self.generic_visit
        super()._reset_name_for_store_and_del(prev_visit_name_func)
    
    def _reset_name_after(self, prev_visit_name_func):
        self.visit_Name = prev_visit_name_func
        self.visit_Attribute = super().visit_Attribute
        self.visit_Subscript = super().visit_Subscript

    def visit_Assign(self, node):
        for target in node.targets:
            if isinstance(target, ast.Name):
                items = set()
                # Direct assignment
                if isinstance(node.value, ast.Name):
                    items.add(node.value.id)
                # List or Tuple
                elif isinstance(node.value, (ast.List, ast.Tuple)):
                    for item in node.value.elts:
                        if isinstance(item, ast.Name):
                            items.add(item.id)
                # Dict
                elif isinstance(node.value, ast.Dict):
                    for item in node.value.values:
                        if isinstance(item, ast.Name):
                            items.add(item.id)
                if target.id in self.alias_mapping and self.alias_mapping[target.id]:
                    # Remove previous references: items -> target.id
                    for item in self.alias_mapping[target.id]:
                        self.alias_mapping[item].remove(target.id)
                # Overwrite the target.id mapping: target.id -> items
                self.alias_mapping[target.id] = items.copy()
                # Update the mapping for the items: item -> target.id
                for item in items:
                    self.alias_mapping[item] = self.alias_mapping.get(item, set()) | {target.id}
                # Remove empty mappings in the alias_mapping
                self.alias_mapping = {k: v for k, v in self.alias_mapping.items() if v}
        self.generic_visit(node)

    # Rule: if a variable is used as an iterator in a for, a while loop, or a comprehension, 
    # it is regarded as both input and output
    def _visit_list_of_nodes(self, nodes):
        for idx, item in enumerate(nodes):
            prev_inputs, prev_outputs = self.inputs.copy(), self.outputs.copy()
            if idx == 0:
                prev_visit_name_func = self.visit_Name
                # Handle the iter node (for loop) or the test node (while loop) or the iter node (comprehension)
                self._reset_name_before(prev_visit_name_func)
                self.visit(item)
                self._reset_name_after(prev_visit_name_func)
            else:
                self.visit(item)
            # Exclude outputs from previous iterations
            curr_inputs = self.inputs - prev_inputs
            self.inputs = (curr_inputs - prev_outputs) | prev_inputs
    
    def _visit_list_of_nodes_in_if(self, nodes):
        # visit the super calss method
        super()._visit_list_of_nodes(nodes)
    
    def visit_If(self, node):
        prev_inputs, prev_outputs = self.inputs.copy(), self.outputs.copy()
        # In a loop
        if self.loopflag:
            # Order 1
            self._visit_list_of_nodes_in_if([node.test] + node.body + node.orelse)
            order1_inputs, order1_outputs = self.inputs.copy(), self.outputs.copy()
            self.inputs, self.outputs = prev_inputs.copy(), prev_outputs.copy()
            # Order 2
            self._visit_list_of_nodes_in_if([node.test] + node.orelse + node.body)
            # Keep the union of these two orders
            self.inputs = order1_inputs | self.inputs
            self.outputs = order1_outputs | self.outputs
        else:
            # Union of the if-else branches
            self._visit_list_of_nodes_in_if([node.test] + node.body)
            union_inputs, union_outputs = self.inputs.copy(), self.outputs.copy()
            self.inputs, self.outputs = prev_inputs.copy(), prev_outputs.copy()
            self._visit_list_of_nodes_in_if([node.test] + node.orelse)
            self.inputs.update(union_inputs)
            self.outputs.update(union_outputs)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            self.refers_code.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            while isinstance(node.func.value, ast.Attribute):
                node.func = node.func.value
        prev_visit_name_func = self.visit_Name
        self._reset_name_before(prev_visit_name_func)
        self.generic_visit(node)
        self._reset_name_after(prev_visit_name_func)

def _add_refers(data, alias_mapping):
    for node in copy.deepcopy(data):
        if node in alias_mapping:
            data.update(alias_mapping[node])

def parse_per_statement(node, alias_mapping, modules, is_definite):
    Visitor = MinIOVisitor(alias_mapping, modules) if is_definite else MaxIOVisitor(alias_mapping, modules)
    return Visitor.visit(node)

def capture_variables(code_cell, alias_mapping={}, modules=set(), is_definite=True, parse_per_statement=parse_per_statement):
    cell_inputs, cell_outputs, cell_refers_code, cell_defines_code = set(), set(), set(), set()
    # Parse the code cell
    tree = ast.parse(code_cell, mode='exec', type_comments=False)
    excluded_inputs = set()
    prev_alias_mapping = copy.deepcopy(alias_mapping)
    alias_stmt = None
    for node in tree.body:
        # For each statement in a code cell
        inputs, outputs, refers_code, defines_code, alias_mapping, modules = parse_per_statement(node, alias_mapping, modules, is_definite)
        curr_inputs = inputs - excluded_inputs
        if prev_alias_mapping == alias_mapping:
            _add_refers(curr_inputs, alias_mapping)
            _add_refers(outputs, alias_mapping)
        else:
            alias_stmt = alias_stmt + '\n' + ast.unparse(node) if alias_stmt else ast.unparse(node)
        prev_alias_mapping = copy.deepcopy(alias_mapping)
        cell_inputs.update(curr_inputs)
        cell_outputs.update(outputs)
        cell_refers_code.update(refers_code)
        cell_defines_code.update(defines_code)
        excluded_inputs.update(outputs)
    return cell_inputs, cell_outputs, cell_refers_code, cell_defines_code, alias_mapping, alias_stmt, modules

def extract_records(code_cells, is_definite=True, parse_per_statement=parse_per_statement):
    """
    Extract inputs, outputs, and output_candidates from a list of code cells.
    :param code_cells: a list of code cells.
    :return records: a list of records (dict). Records are in reversed order for each code cell, containing inputs, outputs, and output_candidates.
    """
    # Consider alias_mapping in order
    # and exclude used funcs and classes (refers_code) that are not defined in the previous code cells
    alias_mapping, modules, records = {}, set(), []
    defined_code = set()
    prev_alias_stmt = None
    for idx, code_cell in enumerate(code_cells):
        inputs, output_candidates, refers_code, defines_code, alias_mapping, alias_stmt, modules = capture_variables(code_cell, alias_mapping=alias_mapping, modules=modules, is_definite=is_definite, parse_per_statement=parse_per_statement)
        defined_code = defined_code | defines_code
        prev_alias_stmt = alias_stmt if alias_stmt else prev_alias_stmt
        alias_vars = set(alias_mapping.keys()).union(*alias_mapping.values())
        records.append({'inputs': inputs - defined_code, 'output_candidates': output_candidates - defined_code, 'refers_code': (inputs | refers_code) & defined_code, 'defines_code': defines_code, 'alias_stmt': prev_alias_stmt, 'alias_vars': alias_vars})

    # Only keep variables that are used in the subsequent code cells as outputs
    required_variables = set()
    for idx in range(len(records)-1, -1, -1): # reverse order
        inputs, output_candidates = records[idx]['inputs'], records[idx]['output_candidates']
        outputs = output_candidates & required_variables
        required_variables = (required_variables - outputs) | inputs
        records[idx]['outputs'] = outputs

    # Exclude imported modules (e.g., pd) and built-in functions/variables (e.g.,len and __name__) from inputs, outputs, and output_candidates
    with open("data/builtin_funcs_and_vars.txt", "r") as fin:
        builtin_funcs_and_vars = set(fin.read().splitlines()[1:])
    modules_and_builtins = modules | builtin_funcs_and_vars
    existing_outputs = set()
    for idx, record in enumerate(records):
        # If a variable is not defined in the previous code cells
        # and is one of the imported modules or built-in functions/variables
        excluded_vars = (record['inputs'] - existing_outputs) & modules_and_builtins
        inputs = record['inputs'] - excluded_vars
        outputs = record['outputs'] - excluded_vars
        output_candidates = record['output_candidates'] - excluded_vars
        records[idx]['inputs'], records[idx]['outputs'], records[idx]['output_candidates'] = inputs, outputs, output_candidates
        existing_outputs.update(outputs)
    return records

def map_records(records):
    """
    Map records to avoid duplicate names in inputs and outputs.
    :param records: a list of records (dict). Records are in the same order as code cells, containing inputs, outputs, and output_candidates.
    :return records: a list of records (dict). Records are in the same order as code cells, containing inputs and outputs.
    """
    # Rename generated output with -{v} suffix, where v is 1 OR 2 OR 3, ...
    name_mappings = {}
    for record in records:
        mapped_inputs, mapped_outputs = set(), set()
        for input in record['inputs']:
            if (input in name_mappings and name_mappings[input] == 0) or (input not in name_mappings):
                name_mappings[input] = 0
                mapped_inputs.add(input)
            else:
                mapped_inputs.add(input + '-' + str(name_mappings[input]))
        for output in record['outputs']:
            if output in name_mappings:
                name_mappings[output] = name_mappings[output] + 1
                mapped_outputs.add(output + '-' + str(name_mappings[output]))
            else:
                name_mappings[output] = 0
                mapped_outputs.add(output)
        record['inputs'], record['outputs'] = mapped_inputs, mapped_outputs
    return records
