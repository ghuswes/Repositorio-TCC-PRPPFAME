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
os.environ["MMKP_OUTPUT_ROOT"] = "."
# ==============================================================================

def run_mmkp_worker(module_name, *args):
    """
    Função auxiliar para rodar o teste em um processo separado.
    """
    try:
        module = importlib.import_module(module_name)
        module.run_test(*args)
    except Exception as e:
        print(f"Erro no processo worker ({module_name}): {e}")

def objective(trial):
    """
    Função objetivo que o Optuna tentará maximizar.
    """
    print(f"\n[DEBUG] Iniciando Trial #{trial.number}")
    
    database = "Base 120"
    module_name = "MMKP_Recozimento_Simulado"
    
    # Parâmetros solicitados pelo usuário
    temp_ini = trial.suggest_float("temp_ini", 10000, 10000000, log=True)
    temp_ini = round(temp_ini, 2)
    
    taxa_resf = trial.suggest_float("taxa_resf", 0.001, 0.999)
    taxa_resf = round(taxa_resf, 3)
    
    max_iter_temp = trial.suggest_int("max_iter_temp", 1, 1000)
    total_iteracoes = trial.suggest_int("total_iteracoes", 5000, 1000000)
    
    worker_args = (database, temp_ini, taxa_resf, max_iter_temp, total_iteracoes)

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
    folder = "resultados_recozimento-simulado"
    
    # Formato: {nome_base}_SA_{temp_ini}_{taxa_resf}_{max_iter_temp}_{total_iteracoes}.json
    # Importante: os floats no nome do arquivo podem vir com .0 ou não dependendo do json.dump
    # O script SA usa f-strings diretas.
    filename = f"{database.replace(' ', '_')}_SA_{temp_ini}_{taxa_resf}_{max_iter_temp}_{total_iteracoes}.json"
    
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
                
                # Consideramos viável se retornou lucro positivo (ou conforme a lógica da base 120)
                # Na base 120, liquid_profit pode ser o fitness se não houver viável
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
        n_startup_trials=5, # Pequeno aquecimento
        multivariate=True
    )

    study = optuna.create_study(
        study_name="MMKP_SA_Optimization",
        direction="maximize",
        storage="sqlite:///optuna_mmkp_sa.db", 
        load_if_exists=True,
        sampler=sampler
    )

    print("--- Iniciando Otimização de Recozimento Simulado com Optuna ---")
    print(f"Estudo: {study.study_name}")
    print(f"Timeout por teste: {TIMEOUT}s | Foco: base120")
    print("Pressione Ctrl+C para interromper.\n")

    try:
        study.optimize(objective, n_trials=125) 
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
