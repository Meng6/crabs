from parsers.yw_generator import extract_code_cells, extract_records
from parsers.crabs import extract_records_using_crabs
from parsers.a1 import extract_records_using_a1
from parsers.a2 import extract_records_using_a2
from parsers.baseline import extract_records_using_baseline_approach
import networkx as nx
import pandas as pd
from openai import OpenAI
import json, glob

#############################################################
# Parameters                                                #
#############################################################

# LLM settings
model="gpt-4o" # => gpt-4o-2024-08-06
temperatrue=0

# Inputs
openai_key_file = "key"
target_notebook_folder = "data/inputs"
groundtruth_file = "data/ProspectiveCorrectGivenSpecification.json"
# Outputs
target_informationflowgraph_folder = "data/outputs/informationflowgraph"
target_cellexedependencygraph_folder = "data/outputs/cellexecutiondependencygraph"

# Run selected methods
run_baseline = False # baseline approach
run_crabs = False # CRABS [our main approach]
run_a1 = False # Ablation Study 1
run_a2 = False # Ablation Study 2

#############################################################
# Helper functions:                                         #
#############################################################

def extract_edges(records, is_mapped=False):
    # (c_s, c_t, i): information i flows from the source cell c_s to the target cell c_t
    edges = set()
    data2definedProgramBlock, code2definedProgramBlock = {}, {}
    for idx, record in enumerate(records):
        programBlock = "cell{}".format(idx + 1)
        # data edges
        if "inputs" in record:
            for data in record["inputs"]:
                if is_mapped and type(data) == list:
                    data = data[1]
                if data in data2definedProgramBlock:
                    edges.add((data2definedProgramBlock[data], programBlock, f"data:{data}".split("-")[0]))
        if "outputs" in record:
            for data in record["outputs"]:
                if is_mapped and type(data) == list:
                    data = data[1]
                data2definedProgramBlock[data] = programBlock
        # code edges
        if "defines_code" in record:
            for code in record["defines_code"]:
                code2definedProgramBlock[code] = programBlock
        if "refers_code" in record:
            for code in record["refers_code"]:
                if code in code2definedProgramBlock and code2definedProgramBlock[code] != programBlock:
                    edges.add((code2definedProgramBlock[code], programBlock, f"code:{code}"))
    return edges

# Ignore labels to get the cell execution dependency graph, then extract transitive dependencies
def edges2transitivedependencies(edges):
    # cell execution dependency graph
    G = nx.DiGraph()
    for edge in edges:
        # (c_t, c_s): cell c_t depends on cell c_s
        G.add_edge(edge[1], edge[0])
    cell_exe_dependency_graph = nx.transitive_closure_dag(G)
    transitive_dependencies = set(cell_exe_dependency_graph.edges())
    return transitive_dependencies

######################################################
# Construct DAGs from the human annotated JSON file  #
######################################################

def extract_graphs_from_human_annotations(notebook, groundtruth):
    """
    Extract the human annotated DAGs from a JSON file.
    :param notebook: a string. The path to the notebook.
    :return information_flows: a set of information flows.
    :return transitive_dependencies: a set of transitive dependencies.
    """
    for item in groundtruth:
        if item[0]['id'] != "-".join(notebook.split('-')[1:])[:-6]:
            continue
        information_flows = extract_edges(records=item[1:], is_mapped=True) 
    transitive_dependencies = edges2transitivedependencies(information_flows)
    return (information_flows, transitive_dependencies)

###########################################################################################
# Extract syntactic parsers generated DAGs from a notebook:                               #
# 1. is_definite=True: minimal estimate of information flows and transitive dependencies  #
# 2. is_definite=False: maximal estimate of information flows and transitive dependencies #
###########################################################################################

def extract_graphs_using_syntactic_parsers(notebook, is_definite):
    """
    Extract syntactic parsers generated DAGs from a notebook.
    :param notebook: a string. The path to the notebook.
    :return information_flows: a set of information flows.
    :return transitive_dependencies: a set of transitive dependencies.
    """
    code_cells = extract_code_cells(notebook)
    records = extract_records(code_cells, is_definite)
    information_flows =  extract_edges(records, is_mapped=False)
    transitive_dependencies = edges2transitivedependencies(information_flows)
    return (information_flows,transitive_dependencies)

########################################################################################################
# Extract LLM-based methods generated DAGs from a notebook                                             #
# 1. ask an LLM using a single prompt for the complete notebook directly => baseline                   #
# 1. syntactic parsing + semantic parsing + cell-by-cell prompting => crabs  [Main Approach]           #
# 2.                     semantic parsing + cell-by-cell prompting => a1  [Ablation Study 1]           #
# 3. syntactic parsing + semantic parsing                          => a2  [Ablation Study 2]           #
########################################################################################################

def extract_graph_from_llm_based_output(notebook, notebook_name, model, temperatrue, client, flag='crabs'):
    """
    Extract LLM-based approaches generated DAGs from a notebook.
    :param notebook: a string. The path to the notebook.
    :return information_flows: a set of information flows.
    :return transitive_dependencies: a set of transitive dependencies.
    """
    code_cells = extract_code_cells(notebook)
    if flag == 'baseline': # Baseline
        records = extract_records_using_baseline_approach(code_cells, model, temperatrue, client)
    elif flag == 'crabs': # CRABS
        records = extract_records_using_crabs(code_cells, model, temperatrue, client)
    elif flag == 'a1': # Ablation Study 1
        records = extract_records_using_a1(code_cells, model, temperatrue, client)
    elif flag == 'a2': # Ablation Study 2
        records = extract_records_using_a2(code_cells, model, temperatrue, client)
    else:
        raise ValueError("Invalid flag. Choose from 'baseline', 'crabs', 'a1', and 'a2'.")
    with open(f"data/outputs/records/{model}_log_{flag}_{notebook_name}.json", "w") as fout:
        json.dump(records, fout, indent=4)
    try:
        information_flows =  extract_edges(records, is_mapped=False)
        transitive_dependencies = edges2transitivedependencies(information_flows)
    except:
        information_flows, transitive_dependencies = set(), set()
    return (information_flows, transitive_dependencies)

#########################################################################################
# Evaluate information flows and transitive dependencies                                #
# Metrics: precision, recall, F1, and accuracy (Jaccard Index)                          #
#########################################################################################

def compute_set_metrics(set_true, set_pred):
    """
    Calculate precision, recall, F1 score, and accuracy.
    :param set_true: a set. The true set.
    :param set_pred: a set. The predicted set.
    :return: precision, recall, F1 score, and accuracy.
    """
    tp = len(set_true & set_pred)
    fp = len(set_pred - set_true)
    fn = len(set_true - set_pred)

    if len(set_true) == 0:
        return None, None, None, None
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = tp / (len(set_true | set_pred)) if len(set_true | set_pred) > 0 else 0
    return precision, recall, f1_score, accuracy

def compute_metrics(edges, run_crabs=False, run_a1=False, run_a2=False):
    metrics_def = compute_set_metrics(edges["true"], edges["min"])
    metrics_pos = compute_set_metrics(edges["true"], edges["max"])
    metrics = pd.DataFrame({"metrics": ["precision", "recall", "f1", "accuracy"], "MinIOParser": metrics_def, "MaxIOParser": metrics_pos})
    if run_baseline:
        metrics["Baseline"] = compute_set_metrics(edges["true"], edges["baseline"])
    if run_crabs:
        metrics["CRABS"] = compute_set_metrics(edges["true"], edges["crabs"])
    if run_a1:
        metrics["A1"] = compute_set_metrics(edges["true"], edges["a1"])
    if run_a2:
        metrics["A2"] = compute_set_metrics(edges["true"], edges["a2"])
    return metrics.round(4)

# Load your OpenAI key
with open(openai_key_file, "r") as fin:
    api_key = fin.read()
client = OpenAI(api_key=api_key)

# Load the ground truth
with open(groundtruth_file) as fin:
    groundtruth = json.loads(fin.read())

for notebook in sorted(glob.glob(f'{target_notebook_folder}/*.ipynb')):
    print(notebook)
    notebook_name = notebook.split("/")[-1].replace(".ipynb", "")
    
    # Extract information flows and transitive dependencies
    (informationflows_true, transitivedependencies_true) = extract_graphs_from_human_annotations(notebook, groundtruth)
    (informationflows_min, transitivedependencies_def) = extract_graphs_using_syntactic_parsers(notebook, is_definite=True)
    (informationflows_max, transitivedependencies_pos) = extract_graphs_using_syntactic_parsers(notebook, is_definite=False)
    informationflows = {"true": informationflows_true, "min": informationflows_min, "max": informationflows_max}
    transitivedependencies = {"true": transitivedependencies_true, "min": transitivedependencies_def, "max": transitivedependencies_pos}
    
    if run_crabs:
        (informationflows["crabs"], transitivedependencies["crabs"]) = extract_graph_from_llm_based_output(notebook, notebook_name, model, temperatrue, client, flag='crabs')
    if run_a1:
        (informationflows["a1"], transitivedependencies["a1"]) = extract_graph_from_llm_based_output(notebook, notebook_name, model, temperatrue, client, flag='a1')
    if run_a2:
        (informationflows["a2"], transitivedependencies["a2"]) = extract_graph_from_llm_based_output(notebook, notebook_name, model, temperatrue, client, flag='a2')
    if run_baseline:
        (informationflows["baseline"], transitivedependencies["baseline"]) = extract_graph_from_llm_based_output(notebook, notebook_name, model, temperatrue, client, flag='baseline')

    # Compute metrics
    informationflowgraph_metrics = compute_metrics(informationflows, run_crabs=run_crabs, run_a1=run_a1, run_a2=run_a2)
    cellexedependencygraph_metrics = compute_metrics(transitivedependencies, run_crabs=run_crabs, run_a1=run_a1, run_a2=run_a2)

    # Save metrics
    informationflowgraph_metrics.to_csv(f"{target_informationflowgraph_folder}/{model}_{notebook_name}.csv", index=False)
    cellexedependencygraph_metrics.to_csv(f"{target_cellexedependencygraph_folder}/{model}_{notebook_name}.csv", index=False)

    # For the cell execution dependency graph: ensure precision of MinIOParser and recall of MaxIOParser are 100%
    assert cellexedependencygraph_metrics.loc[0, "MinIOParser"] in [1, None] and cellexedependencygraph_metrics.loc[1, "MaxIOParser"] in [1, None]
