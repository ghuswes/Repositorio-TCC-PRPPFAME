import optuna
import os
import json
import importlib
import multiprocessing
import time

# ==============================================================================
# CONFIGURAÇÕES DE PESO E TIMEOUT
# ==============================================================================
TIMEOUT = 2460  # 41 minutos
FAILURE_SCORE = -100_000_000 # Penalidade para "Sem Solução" ou "Timeout"

# ISOLAMENTO DE RESULTADOS DO OPTUNA
os.environ["MMKP_OUTPUT_ROOT"] = "optuna_results_json"
# ==============================================================================

def run_mmkp_worker(module_name, *args):
    """
    Função auxiliar para rodar o teste em um processo separado.
    """
    try:
        module = importlib.import_module(module_name)
        # Chama run_test com a lista de argumentos fornecida
        module.run_test(*args)
    except Exception as e:
        print(f"Erro no processo worker ({module_name}): {e}")

def objective(trial):
    """
    Função objetivo que o Optuna tentará maximizar.
    """
    print(f"\n[DEBUG] Iniciando Trial #{trial.number}")
    
    database = "base120"
    
    variant_map = {
        "penalidade-aplicada": "MMKP_ordenado-peso_com-penalidade-aplicada",
        "penalidade-final-dp": "MMKP_ordenado-peso_com-penalidade-final-dp",
        "hibrido_ordenado": "MMKP_ordenado-peso_hibrido",
        "chaveamento_ordenado": "MMKP_ordenado-peso_chaveamento",
        "hibrido_nao-ordenado": "MMKP_nao-ordenado_hibrido",
        "chaveamento_nao-ordenado": "MMKP_nao-ordenado_chaveamento"
    }
    
    # Espaço de busca sincronizado com o banco optuna_mmkp.db original
    search_space = [
        "penalidade-aplicada", 
        "penalidade-final-dp", 
        "hibrido_ordenado", 
        "chaveamento_ordenado", 
        "hibrido_nao-ordenado", 
        "chaveamento_nao-ordenado"
    ]
    variant_key = trial.suggest_categorical("variant", search_space)

    module_name = variant_map[variant_key]
    
    heuristica = trial.suggest_categorical("heuristica", ["N", "A", "MG", "MGA", "MGR", "MM", "DV"])
    scale_vpl = trial.suggest_int("scale_vpl", 1, 100000)
    limit = trial.suggest_int("limit", 1, 300)
    alpha = 1.0 # Cap_buffer padrão
    
    extra_val = None
    if heuristica == "A":
        extra_val = trial.suggest_float("extra_alpha", 0.01, 1.0, step=0.01)
        extra_val = round(extra_val, 2)
    elif heuristica in ["MG", "MGA", "MGR", "MM"]:
        extra_val = trial.suggest_categorical("extra_order", ["C", "D"])
    
    # Preparação dos argumentos para o worker
    if "chaveamento" in variant_key:
        cut_val = trial.suggest_int("cut_val", 1, 10)
        worker_args = (database, heuristica, scale_vpl, limit, alpha, cut_val, extra_val)
    else:
        worker_args = (database, heuristica, scale_vpl, limit, alpha, extra_val)

    # EXECUÇÃO COM TIMEOUT
    p = multiprocessing.Process(
        target=run_mmkp_worker, 
        args=(module_name,) + worker_args
    )
    
    p.start()
    p.join(timeout=TIMEOUT) 
    
    if p.is_alive():
        print(f"\n[TIMEOUT] Trial {trial.number} excedeu {TIMEOUT}s.")
        p.terminate() 
        p.join()
        trial.set_user_attr("constraints", (1.0,))
        return FAILURE_SCORE 

    # LEITURA DO RESULTADO
    root_dir = os.environ.get("MMKP_OUTPUT_ROOT", ".")
    folder_suffix = module_name.replace("MMKP_", "")
    folder = f"resultados_{folder_suffix}"
    
    prefix = ""
    if "hibrido" in variant_key:
        prefix = "H"
    elif "chaveamento" in variant_key:
        prefix = "C"
    
    if "chaveamento" in variant_key:
        if extra_val is None:
            filename = f"{database}_{prefix}{heuristica}_{int(scale_vpl)}_{limit}_{alpha}_{cut_val}.json"
        else:
            extra_str = f"{extra_val:g}" if isinstance(extra_val, float) else extra_val
            filename = f"{database}_{prefix}{heuristica}_{int(scale_vpl)}_{limit}_{alpha}_{cut_val}_{extra_str}.json"
    else:
        if extra_val is None:
            filename = f"{database}_{prefix}{heuristica}_{int(scale_vpl)}_{limit}_{alpha}.json"
        else:
            extra_str = f"{extra_val:g}" if isinstance(extra_val, float) else extra_val
            filename = f"{database}_{prefix}{heuristica}_{int(scale_vpl)}_{limit}_{alpha}_{extra_str}.json"
    
    filename = filename.replace("__", "_")
    filepath = os.path.join(root_dir, folder, filename)
    print(f"[DEBUG] Procurando resultado em: {filepath}")
    
    if not os.path.exists(filepath):
        print(f"[AVISO] Arquivo {filepath} não encontrado.")
        trial.set_user_attr("constraints", (1.0,))
        return FAILURE_SCORE
        
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
            if data["solution"]:
                lucro = data["solution"][0]["liquid_profit"]
                tempo = data.get("tempo", 0)
                trial.set_user_attr("tempo", tempo)
                trial.set_user_attr("constraints", (0.0,))
                return lucro
            else:
                trial.set_user_attr("constraints", (1.0,))
                return FAILURE_SCORE
    except Exception as e:
        print(f"Erro ao ler resultado: {e}")
        trial.set_user_attr("constraints", (1.0,))
        return FAILURE_SCORE

def constraints(trial):
    return trial.user_attrs.get("constraints", (0.0,))

def run_optimization():
    sampler = optuna.samplers.TPESampler(
        constraints_func=constraints, 
        n_startup_trials=0,
        multivariate=True,
        group=True
    )

    study = optuna.create_study(
        study_name="MMKP_Multi_Variant_Optimization",
        direction="maximize",
        storage="sqlite:///optuna_mmkp.db", 
        load_if_exists=True,
        sampler=sampler
    )

    print("--- Iniciando Otimização Autônoma com Optuna (Versão Penalidade) ---")
    print(f"Estudo: {study.study_name}")
    print(f"Timeout por teste: {TIMEOUT}s | Foco: base120")
    print("O estudo continuará a partir dos dados existentes.")
    print("Pressione Ctrl+C para interromper.\n")

    try:
        # Mantendo 100 trials para a versão completa/penalidade
        study.optimize(objective, n_trials=100) 
    except KeyboardInterrupt:
        print("\nOtimização interrompida pelo usuário.")

    if len(study.trials) > 0:
        print("\n" + "="*40)
        print("MELHORES PARÂMETROS ENCONTRADOS:")
        try:
            print(f"  Melhor Lucro: {study.best_value:.2f}")
            for key, value in study.best_params.items():
                print(f"  {key}: {value}")
        except:
            print("  Nenhum trial viável encontrado ainda.")
        print("="*40)

if __name__ == "__main__":
    run_optimization()
