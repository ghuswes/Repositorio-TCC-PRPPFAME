import pandas as pd
import numpy as np
import time
import json
import os
import sys


# Heurística de Déficit de Viabilidade
def viability_deficit(solutions, min_capacities, limit):
    for sol in solutions:
        deficits = np.maximum(0, min_capacities - sol["weights"]) / min_capacities
        sol["v_deficit"] = np.sum(deficits)

    half = limit // 2
    
    # 1. Metade "Pura" (Igual ao sem-demanda): Prioriza apenas a métrica da heurística
    # Para v_deficit, o melhor é o menor valor (ASC)
    solutions.sort(key=lambda x: x["v_deficit"])
    pure_selected = solutions[:half]
    
    # 2. Metade "Líquida" (Igual ao com-penalidade-aplicada): Prioriza Lucro Líquido
    pure_ids = {id(s) for s in pure_selected}
    remaining = [s for s in solutions if id(s) not in pure_ids]
    # Prioriza Lucro Líquido (DESC), desempata por v_deficit (ASC)
    remaining.sort(key=lambda x: (-x["total_profit"], x["v_deficit"]))
    liquid_selected = remaining[:limit - half]

    return pure_selected + liquid_selected


# Heurística de Média Geométrica
def geometric_mean(solutions, capacities, ord, limit):
    is_reverse = True if (ord == "C" or ord == None) else False
    for sol in solutions:
        fill_ratios = np.maximum(1e-12, sol["weights"]) / np.maximum(1e-12, capacities)
        sol["gm_score"] = np.power(np.prod(fill_ratios), 1 / len(capacities))

    half = limit // 2
    
    # 1. Metade "Pura": Apenas gm_score
    solutions.sort(
        key=lambda x: x["gm_score"] if is_reverse else -x["gm_score"],
        reverse=True,
    )
    pure_selected = solutions[:half]
    
    # 2. Metade "Líquida": Lucro Líquido > gm_score
    pure_ids = {id(s) for s in pure_selected}
    remaining = [s for s in solutions if id(s) not in pure_ids]
    remaining.sort(
        key=lambda x: (-x["total_profit"], x["gm_score"] if is_reverse else -x["gm_score"]),
        reverse=True,
    )
    liquid_selected = remaining[:limit - half]

    return pure_selected + liquid_selected


# Heurística de Média Geométrica Alternativa
def geometric_mean_alt(solutions, capacities, ord, limit):
    is_reverse = True if (ord == "C" or ord == None) else False
    for sol in solutions:
        fill_ratios = np.maximum(1e-12, sol["weights"]) / np.maximum(1e-12, capacities)
        gm = np.power(np.prod(fill_ratios), 1 / len(capacities))
        sol["gm_score"] = abs(1.0 - gm)

    half = limit // 2
    
    # 1. Metade "Pura": Apenas gm_score
    solutions.sort(
        key=lambda x: x["gm_score"] if is_reverse else -x["gm_score"],
        reverse=True,
    )
    pure_selected = solutions[:half]
    
    # 2. Metade "Líquida": Lucro Líquido > gm_score
    pure_ids = {id(s) for s in pure_selected}
    remaining = [s for s in solutions if id(s) not in pure_ids]
    remaining.sort(
        key=lambda x: (-x["total_profit"], x["gm_score"] if is_reverse else -x["gm_score"]),
        reverse=True,
    )
    liquid_selected = remaining[:limit - half]

    return pure_selected + liquid_selected


# Heurística de Média Geométrica Inversa
def geometric_mean_reverse(solutions, capacities, ord, limit):
    is_reverse = True if (ord == "C" or ord == None) else False
    for sol in solutions:
        fill_ratios = np.maximum(1e-12, sol["weights"]) / np.maximum(1e-12, capacities)
        fill_ratios = 1.0 - fill_ratios
        gm = np.power(np.prod(fill_ratios), 1 / len(capacities))
        sol["gm_score"] = abs(1.0 - gm)

    half = limit // 2
    
    # 1. Metade "Pura": Apenas gm_score
    solutions.sort(
        key=lambda x: x["gm_score"] if is_reverse else -x["gm_score"],
        reverse=True,
    )
    pure_selected = solutions[:half]
    
    # 2. Metade "Líquida": Lucro Líquido > gm_score
    pure_ids = {id(s) for s in pure_selected}
    remaining = [s for s in solutions if id(s) not in pure_ids]
    remaining.sort(
        key=lambda x: (-x["total_profit"], x["gm_score"] if is_reverse else -x["gm_score"]),
        reverse=True,
    )
    liquid_selected = remaining[:limit - half]

    return pure_selected + liquid_selected


# Heurística Mínimo Máximo
def min_max(solutions, capacities, ord, limit):
    is_reverse = True if (ord == "C" or ord == None) else False
    for sol in solutions:
        fill_ratios = np.maximum(1e-12, sol["weights"]) / np.maximum(1e-12, capacities)
        sol["max_ratio"] = np.max(fill_ratios)

    half = limit // 2
    
    # 1. Metade "Pura": Apenas max_ratio
    solutions.sort(
        key=lambda x: x["max_ratio"] if is_reverse else -x["max_ratio"],
        reverse=True,
    )
    pure_selected = solutions[:half]
    
    # 2. Metade "Líquida": Lucro Líquido > max_ratio
    pure_ids = {id(s) for s in pure_selected}
    remaining = [s for s in solutions if id(s) not in pure_ids]
    remaining.sort(
        key=lambda x: (-x["total_profit"], x["max_ratio"] if is_reverse else -x["max_ratio"]),
        reverse=True,
    )
    liquid_selected = remaining[:limit - half]

    return pure_selected + liquid_selected


# Filtro de Pareto Normal
def pareto_normal(solutions, limit):
    for sol in solutions:
        sol["w_sum"] = np.sum(sol["weights"])

    half = limit // 2
    
    # 1. Metade "Pura": Apenas w_sum (ASC)
    solutions.sort(key=lambda x: x["w_sum"])
    pure_selected = solutions[:half]
    
    # 2. Metade "Líquida": Lucro Líquido (DESC) > w_sum (ASC)
    pure_ids = {id(s) for s in pure_selected}
    remaining = [s for s in solutions if id(s) not in pure_ids]
    remaining.sort(key=lambda x: (-x["total_profit"], x["w_sum"]))
    liquid_selected = remaining[:limit - half]

    return pure_selected + liquid_selected


# Filtro de Pareto Avançado
def pareto_advanced(solutions, limit, frac_limit_sec):
    # 1. Base Global Híbrida (Metade Pura, Metade Líquida)
    base_selection = pareto_normal(solutions, limit)
    
    # 2. Diversidade Dimensão (Adiciona as soluções mais leves por dimensão)
    num_dims = len(solutions[0]["weights"])
    try:
        f_limit_frac = float(frac_limit_sec)
    except (ValueError, TypeError):
        f_limit_frac = 0.1
        
    add_limit = max(1, int(limit * f_limit_frac))
    
    dim_selected_ids = set()
    for d in range(num_dims):
        solutions.sort(key=lambda x: (x["weights"][d], x["w_sum"]))
        for sol in solutions[:add_limit]:
            dim_selected_ids.add(id(sol))
            
    # 3. União da Base com a Diversidade
    final_selection = {id(s): s for s in base_selection}
    for sol in solutions:
        if id(sol) in dim_selected_ids:
            final_selection[id(sol)] = sol
            
    return list(final_selection.values())


# Função para seleção da heurística de truncamento (agora com lógica híbrida 50/50 corrigida)
def select_heuristic(
    solutions, capacities, limit, trunc_type, extra, min_capacities=None
):
    if not solutions:
        return []

    # 1. Remover duplicatas
    unique_map = {}
    for sol in solutions:
        w_tuple = tuple(sol["weights"])
        # Mantemos a que tiver o melhor lucro líquido para representar o peso
        if (
            w_tuple not in unique_map
            or sol["total_profit"] > unique_map[w_tuple]["total_profit"]
        ):
            unique_map[w_tuple] = sol

    trunc_solutions = list(unique_map.values())
    if len(trunc_solutions) <= limit:
        return trunc_solutions

    trunc_type = trunc_type.upper()
    match trunc_type:
        case "MG":
            return geometric_mean(trunc_solutions, capacities, extra, limit)
        case "MGA":
            return geometric_mean_alt(trunc_solutions, capacities, extra, limit)
        case "MGR":
            return geometric_mean_reverse(trunc_solutions, capacities, extra, limit)
        case "MM":
            return min_max(trunc_solutions, capacities, extra, limit)
        case "N":
            return pareto_normal(trunc_solutions, limit)
        case "A":
            return pareto_advanced(trunc_solutions, limit, extra)
        case "DV":
            return viability_deficit(trunc_solutions, min_capacities, limit)
        case _:
            return []


# Solver MMKP Híbrido
def solve_mmkp(
    start_time,
    groups_with_id,
    capacities,
    trunc_type,
    scale_vpl,
    limit,
    cap_buffer,
    extra,
    min_capacities=None,
    penalty_rate=0,
):
    m_dims = len(capacities)
    processed_groups = []

    for g_id, group in groups_with_id:
        min_p = min(item[0] for item in group)
        new_items = []
        for idx_in_group, (p, w, item_id) in enumerate(group):
            scaled_p = (
                int((p - min_p) // scale_vpl) if scale_vpl > 0 else int(p - min_p)
            )
            new_items.append(
                {
                    "scaled_p": scaled_p,
                    "weights": w,
                    "real_p": p,
                    "item_id": item_id,
                    "g_id": g_id,
                }
            )
        processed_groups.append(new_items)

    dp = {
        0: [
            {
                "weights": np.zeros(m_dims, dtype=float),
                "total_profit": 0.0,
                "gross_profit": 0.0,
                "items": [],
            }
        ]
    }
    max_caps = capacities * cap_buffer

    for group_idx, group in enumerate(processed_groups):
        if time.time() - start_time > 3660:
            print("Timeout reached (3660s). Terminating with empty solution.")
            return {}

        new_dp = {}
        for item in group:
            item_id_tuple = (int(item["g_id"]), int(item["item_id"]))
            item_weights = item["weights"]
            item_real_p = item["real_p"]
            item_scaled_p = item["scaled_p"]

            for prev_p, solutions in dp.items():
                curr_p = prev_p + item_scaled_p
                for sol in solutions:
                    new_w = sol["weights"] + item_weights
                    if np.all(new_w <= max_caps):
                        if curr_p not in new_dp:
                            new_dp[curr_p] = []

                        new_gross = sol["gross_profit"] + item_real_p
                        penalty = calculate_penalty(new_w, capacities, min_capacities, penalty_rate)
                        new_total = new_gross - penalty

                        new_dp[curr_p].append(
                            {
                                "weights": new_w,
                                "total_profit": new_total,
                                "gross_profit": new_gross,
                                "items": sol["items"] + [item_id_tuple],
                            }
                        )

        for p in list(new_dp.keys()):
            new_dp[p] = select_heuristic(
                new_dp[p], capacities, limit, trunc_type, extra, min_capacities
            )
            if not new_dp[p]:
                del new_dp[p]

        dp = new_dp
        if not dp:
            print(f"Group {group_idx}: No solutions left.")
            break

    return dp


def csv_ordenation(groups_with_id):
    scored_groups = []
    for g_id, group in groups_with_id:
        total_weight_sum = 0
        for item in group:
            total_weight_sum += np.sum(item[1])
        average_weight = total_weight_sum / len(group)
        scored_groups.append((average_weight, g_id, group))
    scored_groups.sort(key=lambda x: x[0], reverse=True)
    return [(g_id, group) for _, g_id, group in scored_groups]


def calculate_penalty(weights, capacities, min_capacities, penalty_rate):
    if penalty_rate == 0:
        return 0
    penalty = 0
    if min_capacities is not None:
        penalty += np.sum(np.maximum(0, min_capacities - weights))
    penalty += np.sum(np.maximum(0, weights - capacities))
    return penalty * penalty_rate


def best_result_in_final_dp(final_dp):
    best_overall_liquid = -float("inf")
    best_sol_overall = None
    for _, solutions in final_dp.items():
        for sol_cand in solutions:
            liquid_cand = sol_cand["total_profit"]
            if liquid_cand > best_overall_liquid:
                best_overall_liquid = liquid_cand
                best_sol_overall = sol_cand
    return best_overall_liquid, best_sol_overall


def load_csv(file_path):
    if not os.path.exists(file_path):
        return None, None
    df = pd.read_csv(file_path)
    all_cols = list(df.columns)
    idx_item = all_cols.index("Item")
    idx_lucro = all_cols.index("Lucro")
    weight_cols = all_cols[idx_item + 1 : idx_lucro]
    groups_data = []
    lookup_dict = {}
    for g_id in sorted(df["Conjunto"].unique()):
        g_df = df[df["Conjunto"] == g_id]
        items = []
        for _, row in g_df.iterrows():
            profit = float(row["Lucro"])
            weights = row[weight_cols].values.astype(float)
            item_id = row["Item"]
            items.append((profit, weights, item_id))
            lookup_dict[(int(g_id), int(item_id))] = (profit, weights)
        groups_data.append((g_id, items))
    return groups_data, lookup_dict


def get_gabarito(dataset_tag):
    gabarito_path = "benchmarks/RESUMO_GABARITO.txt"
    if not os.path.exists(gabarito_path):
        return "null", "null"
    try:
        with open(gabarito_path, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith(f"{dataset_tag}:"):
                    best_str = line.split("Best=")[1].split(",")[0].strip()
                    best = float(best_str) if best_str.lower() != "unknown" else "null"
                    ub_str = line.split("Upper_Bound=")[1].split(",")[0].strip()
                    ub = float(ub_str) if ub_str.lower() != "unknown" else "null"
                    return best, ub
    except Exception as e:
        print(f"Erro ao ler gabarito para {dataset_tag}: {e}")
    return "null", "null"


def create_json(
    dataset_tag,
    file_path,
    final_dp,
    trunc_type,
    scale_vpl,
    limit,
    cap_buffer,
    exec_time,
    capacities,
    min_capacities,
    p_rate,
    extra,
    lookup_dict,
):
    best, ub = get_gabarito(dataset_tag)
    order = extra if (extra == "C" or extra == "D") else "N/A"
    try:
        frac_limit = float(extra)
    except (ValueError, TypeError):
        frac_limit = "N/A"

    results = {
        "database": file_path,
        "status": "sucess" if final_dp else "no solution",
        "trunc": "H" + trunc_type,
        "scale_vpl": scale_vpl,
        "limit": limit,
        "buffer": cap_buffer,
        "order": order,
        "frac_limit": frac_limit,
        "tempo": exec_time,
        "known_best": best if best != "null" else "N/A",
        "upper_bound": ub,
        "solution": [],
    }

    if final_dp:
        best_overall_liquid, best_sol_overall = best_result_in_final_dp(final_dp)
        sol = best_sol_overall
        liquid_profit = best_overall_liquid
        penalty = calculate_penalty(sol["weights"], capacities, min_capacities, p_rate)

        formatted_weights = []
        for d in range(len(capacities)):
            w = sol["weights"][d]
            min_c = min_capacities[d] if min_capacities is not None else 0
            max_c = capacities[d]
            p_dim = (max(0, min_c - w) + max(0, w - max_c)) * p_rate
            line = f"['dimension: {d+1}', 'weight: {w}', 'penalty: {p_dim}']:('cap_min: {min_c}', 'cap_max: {max_c}')"
            formatted_weights.append(line)

        selected_items_ids = sorted(sol["items"], key=lambda x: x[0])
        formatted_items = []
        for g_id, item_id in selected_items_ids:
            profit, weights = lookup_dict[(g_id, item_id)]
            w_str = ", ".join(map(str, weights))
            line = f"['set: {g_id}', 'item: {item_id}']:('weights: {w_str}'):('profit: {profit}')"
            formatted_items.append(line)

        results["solution"].append(
            {
                "total_profit": sol["gross_profit"],
                "penalty": penalty if p_rate > 0 else "N/A",
                "liquid_profit": liquid_profit,
                "gap_best": (
                    round(abs(best - liquid_profit) / best * 100, 4)
                    if best != "null"
                    else "N/A"
                ),
                "gap_upper_bound": (
                    round(abs(ub - liquid_profit) / ub * 100, 4)
                    if ub != "null"
                    else "N/A"
                ),
                "weights": formatted_weights,
                "items": formatted_items,
            }
        )

    root_dir = os.environ.get("MMKP_OUTPUT_ROOT", ".")
    out_dir = os.path.join(root_dir, "resultados_ordenado-peso_hibrido")
    os.makedirs(out_dir, exist_ok=True)
    if extra == None:
        out_name = f"{out_dir}/{dataset_tag}_H{trunc_type}_{int(scale_vpl)}_{limit}_{cap_buffer}.json"
    else:
        out_name = f"{out_dir}/{dataset_tag}_H{trunc_type}_{int(scale_vpl)}_{limit}_{cap_buffer}_{extra}.json"

    with open(out_name, "w") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"Concluído em {exec_time}s | VPL Líquido: {liquid_profit if final_dp else 'N/A'}")


def run_test(dataset_tag, trunc_type, scale_vpl, limit, cap_buffer, extra):
    bench_dir = "benchmarks"
    configs = {
        "5x10": (f"{bench_dir}/5x10_cap100.csv", [100] * 5, None, 0),
        "10x25": (f"{bench_dir}/10x25_cap300.csv", [300] * 5, None, 0),
        "20x50": (f"{bench_dir}/20x50_cap500.csv", [500] * 5, None, 0),
        "base120": (f"{bench_dir}/base120.csv", [160000] * 16, [140000] * 16, 500),
        "I01": (f"{bench_dir}/I01.csv", [25] * 5, None, 0),
        "I02": (f"{bench_dir}/I02.csv", [50] * 5, None, 0),
        "I03": (f"{bench_dir}/I03.csv", [75] * 10, None, 0),
        "I04": (f"{bench_dir}/I04.csv", [100] * 10, None, 0),
        "I05": (f"{bench_dir}/I05.csv", [125] * 10, None, 0),
        "I06": (f"{bench_dir}/I06.csv", [150] * 10, None, 0),
        "I07": (f"{bench_dir}/I07.csv", [500] * 10, None, 0),
        "I08": (f"{bench_dir}/I08.csv", [750] * 10, None, 0),
        "I09": (f"{bench_dir}/I09.csv", [1000] * 10, None, 0),
        "I10": (f"{bench_dir}/I10.csv", [1250] * 10, None, 0),
        "I11": (f"{bench_dir}/I11.csv", [1500] * 10, None, 0),
        "I12": (f"{bench_dir}/I12.csv", [1750] * 10, None, 0),
        "I13": (f"{bench_dir}/I13.csv", [2000] * 10, None, 0),
    }
    if dataset_tag not in configs:
        return
    file_path, default_caps, min_caps, p_rate = configs[dataset_tag]
    capacities = np.array(default_caps, dtype=float)
    min_capacities = np.array(min_caps, dtype=float) if min_caps else None
    groups, lookup_dict = load_csv(file_path)
    if not groups:
        return
    groups = csv_ordenation(groups)
    print(f"Exec {dataset_tag} | Híbrido | Trunc: {trunc_type} | Scale: {scale_vpl} | Limit: {limit} | Buffer: {cap_buffer} | Extra: {extra}")
    start_t = time.time()
    final_dp = solve_mmkp(start_t, groups, capacities, trunc_type, scale_vpl, limit, cap_buffer, extra, min_capacities, p_rate)
    exec_time = round(time.time() - start_t, 4)
    create_json(dataset_tag, file_path, final_dp, trunc_type, scale_vpl, limit, cap_buffer, exec_time, capacities, min_capacities, p_rate, extra, lookup_dict)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(1)
    with open(sys.argv[1], "r") as f:
        found_ini = False
        for line in f:
            line = line.strip()
            if not found_ini:
                if line.lower() == "ini":
                    found_ini = True
                continue
            parts = line.split()
            if len(parts) >= 3:
                tag, trunc = parts[0], parts[1]
                scale = float(parts[2])
                limit = int(parts[3]) if len(parts) > 3 else 100
                buf = float(parts[4]) if len(parts) > 4 else 1.0
                extra = parts[5] if len(parts) > 5 else None
                run_test(tag, trunc, scale, limit, buf, extra)
