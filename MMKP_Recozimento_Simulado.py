import pandas as pd
import numpy as np
import random
import math
import time
import json
import os
import sys

# ==============================================================================
# 1. PREPARAÇÃO DOS DADOS
# ==============================================================================
def preparar_problema(df, col_conjunto, col_item, col_valor, colunas_peso):
    problema = {}
    conjuntos = sorted(df[col_conjunto].unique())
    for c in conjuntos:
        dados_conj = df[df[col_conjunto] == c]
        itens = dados_conj[col_item].values.tolist()
        problema[c] = {
            'itens': itens,
            'valores': dados_conj[col_valor].values.tolist(),
            'pesos': dados_conj[colunas_peso].values.tolist(),
            'mapa_idx': {item_id: i for i, item_id in enumerate(itens)}
        }
    return problema, conjuntos

# ==============================================================================
# 2. ALGORITMO SIMULATED ANNEALING
# ==============================================================================
def simulated_annealing(problema, conjuntos, cap_sup, cap_inf, temp_ini, temp_fin, 
                        taxa_resf, max_iter_temp, total_iteracoes, nome_base):
    
    start_time_base = time.time()
    pasta_saida = "resultados_recozimento-simulado"
    os.makedirs(pasta_saida, exist_ok=True)
    
    def avaliar(solucao_vetor):
        v_total = 0
        num_dim = len(problema[conjuntos[0]]['pesos'][0])
        p_total = np.zeros(num_dim)
        
        for i, item_id in enumerate(solucao_vetor):
            c_id = conjuntos[i]
            idx = problema[c_id]['mapa_idx'][item_id]
            v_total += problema[c_id]['valores'][idx]
            p_total += np.array(problema[c_id]['pesos'][idx])
            
        estouro_sup = np.maximum(0, p_total - cap_sup)
        estouro_inf = np.maximum(0, cap_inf - p_total)
        # Penalidade por unidade de estouro (500 como no código original)
        penalidade_valor = (np.sum(estouro_sup) + np.sum(estouro_inf)) * 500
        
        fitness = v_total - penalidade_valor
        is_viavel = (np.sum(estouro_sup) == 0 and np.sum(estouro_inf) == 0)
        return fitness, is_viavel, v_total, p_total, penalidade_valor

    # --- Inicialização ---
    sol_atual = [random.choice(problema[c]['itens']) for c in conjuntos]
    fit_atual, viavel_atual, vpl_atual, pesos_atual, pen_atual = avaliar(sol_atual)
    
    melhor_sol_global = list(sol_atual)
    melhor_val_global = vpl_atual if viavel_atual else 0
    melhor_fit_global = fit_atual 
    melhor_pesos_global = np.array(pesos_atual)
    melhor_pen_global = pen_atual
    
    cont_melhorias = 0
    primeiro_resf_passou = False
    temp = temp_ini

    print(f"Processando {nome_base} ({total_iteracoes} iterações)...")

    for cont_iter in range(1, total_iteracoes + 1):
        vizinho = list(sol_atual)
        idx_c = random.randint(0, len(conjuntos) - 1)
        vizinho[idx_c] = random.choice(problema[conjuntos[idx_c]]['itens'])
        
        fit_viz, viavel_viz, vpl_viz, pesos_viz, pen_viz = avaliar(vizinho)
        
        delta = fit_viz - fit_atual
        if delta > 0:
            sol_atual, fit_atual, viavel_atual, vpl_atual, pesos_atual, pen_atual = vizinho, fit_viz, viavel_viz, vpl_viz, pesos_viz, pen_viz
        elif temp > 1e-10:
            if math.exp(delta / (temp * 500)) > random.random():
                sol_atual, fit_atual, viavel_atual, vpl_atual, pesos_atual, pen_atual = vizinho, fit_viz, viavel_viz, vpl_viz, pesos_viz, pen_viz
            
        if viavel_atual and vpl_atual > melhor_val_global:
            melhor_val_global = vpl_atual
            melhor_sol_global = list(sol_atual)
            melhor_pesos_global = np.array(pesos_viz)
            melhor_pen_global = pen_viz
            cont_melhorias += 1
            
        if fit_atual > melhor_fit_global:
            melhor_fit_global = fit_atual
            if melhor_val_global == 0:
                melhor_sol_global = list(sol_atual)
                melhor_pesos_global = np.array(pesos_atual)
                melhor_pen_global = pen_atual

        if cont_iter % max_iter_temp == 0:
            if temp > temp_fin: temp *= taxa_resf
            primeiro_resf_passou = True

    exec_time = round(time.time() - start_time_base, 4)
    
    # Estruturação da saída conforme base120_DV_...json
    formatted_weights = []
    for d in range(len(melhor_pesos_global)):
        w = melhor_pesos_global[d]
        min_c = cap_inf[d] if isinstance(cap_inf, (list, np.ndarray)) else cap_inf
        max_c = cap_sup[d] if isinstance(cap_sup, (list, np.ndarray)) else cap_sup
        
        # Penalidade individual da dimensão (estimada com taxa de 500)
        p_dim = (max(0, min_c - w) + max(0, w - max_c)) * 500
        line = f"['dimension: {d+1}', 'weight: {w}', 'penalty: {p_dim}']:('cap_min: {min_c}', 'cap_max: {max_c}')"
        formatted_weights.append(line)

    formatted_items = []
    for i, item_id in enumerate(melhor_sol_global):
        c_id = conjuntos[i]
        idx = problema[c_id]['mapa_idx'][item_id]
        profit = problema[c_id]['valores'][idx]
        weights = problema[c_id]['pesos'][idx]
        w_str = ", ".join(map(str, weights))
        line = f"['set: {c_id}', 'item: {item_id}']:('weights: {w_str}'):('profit: {profit}')"
        formatted_items.append(line)

    resultado = {
        "database": f"benchmarks/{nome_base.lower().replace(' ', '')}.csv",
        "status": "sucess",
        "method": "SA",
        "temp_ini": temp_ini,
        "taxa_resf": taxa_resf,
        "max_iter_temp": max_iter_temp,
        "total_iteracoes": total_iteracoes,
        "tempo": exec_time,
        "solution": [
            {
                "total_profit": float(sum(problema[conjuntos[i]]['valores'][problema[conjuntos[i]]['mapa_idx'][item_id]] 
                                     for i, item_id in enumerate(melhor_sol_global))),
                "penalty": float(melhor_pen_global),
                "liquid_profit": float(melhor_val_global if melhor_val_global > 0 else melhor_fit_global),
                "weights": formatted_weights,
                "items": formatted_items
            }
        ]
    }
    
    nome_arq = f"{nome_base.replace(' ', '_')}_SA_{temp_ini}_{taxa_resf}_{max_iter_temp}_{total_iteracoes}.json"
    caminho_completo = os.path.join(pasta_saida, nome_arq)
    
    with open(caminho_completo, 'w') as f:
        json.dump(resultado, f, indent=4)
        
    print(f"Concluído {nome_base}. Melhor Valor: {melhor_val_global}")
    return melhor_val_global

def find_benchmark_file(filename):
    for candidate in [
        os.path.join("benchmarks", filename),
        filename
    ]:
        if os.path.exists(candidate):
            return candidate
    return os.path.join("benchmarks", filename)

def run_test(nome_base, temp_ini, taxa_resf, max_iter_temp, total_iteracoes):
    try:
        csv_path = find_benchmark_file('base120.csv')
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Erro ao carregar base120.csv em benchmarks: {e}")
        return

    # Configuração fixa para Base 120 conforme solicitado
    configs = [
        (df, 'Conjunto', 'Item', 'Lucro', [str(i) for i in range(1, 17)], [160000]*16, [140000]*16, "Base 120"),
    ]
    
    for df_p, c_conj, c_item, c_val, c_pesos, sup, inf, nome in configs:
        if nome_base.lower() in nome.lower():
            p, c = preparar_problema(df_p, c_conj, c_item, c_val, c_pesos)
            simulated_annealing(p, c, np.array(sup), np.array(inf), temp_ini, 1, taxa_resf, max_iter_temp, total_iteracoes, nome)
            return

def executar():
    # Tenta carregar os arquivos buscando na pasta benchmarks
    arquivos = {
        'base120': find_benchmark_file('base120.csv'),
        'mochila_5x10': find_benchmark_file('5x10_cap100.csv'),
        'mochila_10x25': find_benchmark_file('10x25_cap300.csv'),
        'mochila_25x50': find_benchmark_file('20x50_cap500.csv')
    }

    try:
        b120 = pd.read_csv(arquivos['base120'])
        # Outras bases podem não existir, então carregamos sob demanda
    except Exception as e: 
        print(f"Erro ao carregar base120.csv em benchmarks: {e}")
        return

    configs = [
        (b120, 'Conjunto', 'Item', 'Lucro', [str(i) for i in range(1, 17)], [160000]*16, [140000]*16, "Base 120"),
    ]

    # Se houver argumentos suficientes, usa-os (formato: base temp_ini taxa_resf max_iter_temp total_iter)
    if len(sys.argv) >= 6:
        alvo = sys.argv[1].lower().replace('_', ' ')
        temp_ini = float(sys.argv[2])
        taxa_resf = float(sys.argv[3])
        max_iter_temp = int(sys.argv[4])
        total_iteracoes = int(sys.argv[5])
        
        for df, c_conj, c_item, c_val, c_pesos, sup, inf, nome in configs:
            if alvo in nome.lower():
                p, c = preparar_problema(df, c_conj, c_item, c_val, c_pesos)
                simulated_annealing(p, c, np.array(sup), np.array(inf), temp_ini, 1, taxa_resf, max_iter_temp, total_iteracoes, nome)
                return

    # Comportamento padrão se não houver argumentos específicos
    if len(sys.argv) > 1:
        alvo = sys.argv[1].lower()
        configs = [c for c in configs if alvo in c[-1].lower()]

    for df, c_conj, c_item, c_val, c_pesos, sup, inf, nome in configs:
        p, c = preparar_problema(df, c_conj, c_item, c_val, c_pesos)
        simulated_annealing(p, c, np.array(sup), np.array(inf), 100000, 1, 0.990, 40, 50000, nome)


if __name__ == "__main__": executar()
