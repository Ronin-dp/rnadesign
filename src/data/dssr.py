import subprocess
import json
import numpy as np
import os


def run_dssr(pdb_file):

    output_json = pdb_file + ".dssr.json"

    cmd = [
        "x3dna-dssr",
        f"-i={pdb_file}",
        "--json",
        f"-o={output_json}"
    ]

    subprocess.run(
        cmd,
        check=True
    )
    
    print("DSSR output:", output_json)
    
    print(
        "JSON exists:",
        os.path.exists(output_json),
        "size:",
        os.path.getsize(output_json)
    )
    
    with open(output_json) as f:
        data = json.load(f)
    return data

def extract_dbn(dssr_result):
    if "dbn" in dssr_result and isinstance(dssr_result["dbn"], dict):
        dbn_dict = dssr_result["dbn"]
        chain_keys = sorted([k for k in dbn_dict.keys() if k.startswith("chain_")])
        if chain_keys:
            sstr_parts = [dbn_dict[k]["sstr"] for k in chain_keys]
            return "".join(sstr_parts)
        if "all_chains" in dbn_dict:
            import re
            sstr = dbn_dict["all_chains"]["sstr"]
            sstr = re.sub(r'[^\.\(\)]', '', sstr)
            return sstr
        raise ValueError("No chain-specific or all_chains sstr found in dbn dictionary")
    
    if "dbn" in dssr_result and isinstance(dssr_result["dbn"], str):
        return dssr_result["dbn"]
    
    for key, value in dssr_result.items():
        if "dbn" in key.lower() and isinstance(value, str):
            return value
    
    raise ValueError("No dot bracket string found in DSSR output.")

def dbn_to_matrix(dbn):

    L=len(dbn)

    ss=np.zeros(
        (L,L),
        dtype=np.float32
    )

    stack=[]


    for i,c in enumerate(dbn):

        if c=="(":
            stack.append(i)

        elif c==")":

            j=stack.pop()

            ss[i,j]=1
            ss[j,i]=1


    return ss