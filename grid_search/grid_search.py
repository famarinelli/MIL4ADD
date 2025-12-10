import json
import itertools
import os
import subprocess
import time
from copy import deepcopy

def get_combinations(grid_dict):
    """Genera tutte le combinazioni (prodotto cartesiano) da un dizionario di liste."""
    keys = grid_dict.keys()
    values = grid_dict.values()
    for combination in itertools.product(*values):
        yield dict(zip(keys, combination))

def format_arg_list(key, value):
    """Restituisce una lista [key, val] invece di una stringa."""
    if isinstance(value, list):
        # Per liste (es. embedding files), appiattiamo: --arg val1 val2
        # Esempio: ['--embedding_files', 'file1', 'file2']
        return [f"--{key}"] + [str(v) for v in value]
    else:
        return [f"--{key}", str(value)]

def main():
    CONFIG_FILE = "grid_search/grid_config.json"
    
    with open(CONFIG_FILE, 'r') as f:
        config = json.load(f)

    # 1. Genera combinazioni base
    base_combinations = list(get_combinations(config["base_grid"]))
    
    job_counter = 0
    
    # 2. Itera sui gruppi condizionali (Scheduler logic)
    for group in config["conditional_groups"]:
        group_name = group["name"]
        group_combinations = list(get_combinations(group["grid"]))
        
        # 3. Prodotto Cartesiano: Base x Gruppo Corrente
        for base_params in base_combinations:
            for group_params in group_combinations:
                job_counter += 1

                if job_counter > 3: continue
                
                # Unione dei parametri: Fixed + Base + Group
                final_params = deepcopy(config["fixed_params"])
                final_params.update(base_params)
                final_params.update(group_params)
                
                # Generazione Nome Esperimento Univoco
                # Es: MultiMTRB_A0.8_B5_LR0.001_Plateau
                short_sched = final_params['scheduler'][:4] # Plat, Cosi...
                exp_name = (f"GS_{job_counter}_"
                            f"A{final_params['alpha']}_"
                            f"B{final_params['beta']}_"
                            f"H{final_params['hidden_dim']}_"
                            f"{short_sched}")
                
                final_params["exp_name"] = exp_name
                
                # Costruzione LISTA argomenti (non stringa)
                args_list = []
                for k, v in final_params.items():
                    args_list.extend(format_arg_list(k, v))
                
                print(f"[{job_counter}] Submitting: {exp_name}")
                
                # Comando sbatch: passiamo gli argomenti DOPO il template
                # Sintassi: sbatch [flags] template.sh [args_per_python...]
                cmd = [
                    "sbatch",
                    "--job-name", exp_name,
                    "--output", f"logs/{exp_name}.out",
                    "--error", f"logs/{exp_name}.err",
                    # Rimuoviamo --export complesso, lasciamo solo ALL per l'ambiente base
                    "--export", "ALL", 
                    config["slurm_template"]
                ] + args_list # <--- Appendiamo la lista degli argomenti qui
                
                subprocess.run(cmd)
                time.sleep(0.5)

    print(f"\n--- Grid Search Launched: {job_counter} jobs submitted ---")

if __name__ == "__main__":
    main()