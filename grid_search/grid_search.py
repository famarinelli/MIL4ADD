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

def format_arg(key, value):
    """Formatta chiave-valore per la riga di comando."""
    if isinstance(value, list):
        # Per argomenti come --embedding_files che prendono liste
        return f"--{key} {' '.join(map(str, value))}"
    else:
        return f"--{key} {value}"

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
                
                # Costruzione stringa argomenti Python
                args_list = []
                for k, v in final_params.items():
                    args_list.append(format_arg(k, v))
                
                py_args = " ".join(args_list)
                
                # 4. Sottomissione Job Slurm
                # Passiamo gli argomenti al template tramite variabile d'ambiente o direttamente
                print(f"[{job_counter}] Submitting: {exp_name}")
                
                # Comando sbatch
                # Usiamo --export per passare la stringa degli argomenti allo script bash
                cmd = [
                    "sbatch",
                    "--job-name", exp_name,
                    "--output", f"logs/{exp_name}.out",
                    "--error", f"logs/{exp_name}.err",
                    "--export", f"ALL,PY_ARGS=\"{py_args}\"",
                    config["slurm_template"]
                ]
                
                subprocess.run(cmd)
                
                # Piccolo sleep per non intasare lo scheduler Slurm
                time.sleep(0.5)

    print(f"\n--- Grid Search Launched: {job_counter} jobs submitted ---")

if __name__ == "__main__":
    main()