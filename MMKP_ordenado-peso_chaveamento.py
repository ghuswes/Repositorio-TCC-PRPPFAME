import pandas as pd
import numpy as np
import time
import json
import os
import sys


# Heurística de Déficit de Viabilidade, onde damos prioridade às soluções mais próximas de atingirem a demanda mínima
def viability_deficit(solutions, min_capacities, use_profit=True):
    for sol in solutions:
        deficits = np.maximum(0, min_capacities - sol["weights"]) / min_capacities
        sol["v_deficit"] = np.sum(deficits)

    if use_profit:
        solutions.sort(key=lambda x: (-x["total_profit"], x["v_deficit"]))
    else:
        solutions.sort(key=lambda x: x["v_deficit"])
    return solutions


# Heurística de Média Geométrica: (prod(wi/Ci))^(1/n)
def geometric_mean(solutions, capacities, ord, use_profit=True):
    is_reverse = True if (ord == "C" or ord == None) else False
    for sol in solutions:
        fill_ratios = np.maximum(1e-12, sol["weights"]) / np.maximum(1e-12, capacities)
        sol["gm_score"] = np.power(np.prod(fill_ratios), 1 / len(capacities))

    if use_profit:
        solutions.sort(
            key=lambda x: (-x["total_profit"], x["gm_score"] if is_reverse else -x["gm_score"]),
            reverse=True,
        )
    else:
        solutions.sort(
            key=lambda x: x["gm_score"] if is_reverse else -x["gm_score"],
            reverse=True,
        )
    return solutions


# Heurística de Média Geométrica Alternativa: 1 - (prod(wi/Ci))^(1/n)
def geometric_mean_alt(solutions, capacities, ord, use_profit=True):
    is_reverse = True if (ord == "C" or ord == None) else False
    for sol in solutions:
        fill_ratios = np.maximum(1e-12, sol["weights"]) / np.maximum(1e-12, capacities)
        gm = np.power(np.prod(fill_ratios), 1 / len(capacities))
        sol["gm_score"] = abs(1.0 - gm)

    if use_profit:
        solutions.sort(
            key=lambda x: (-x["total_profit"], x["gm_score"] if is_reverse else -x["gm_score"]),
            reverse=True,
        )
    else:
        solutions.sort(
            key=lambda x: x["gm_score"] if is_reverse else -x["gm_score"],
            reverse=True,
        )
    return solutions


# Heurística de Média Geométrica Inversa: 1 - (prod(1 - wi/Ci))^(1/n)
def geometric_mean_reverse(solutions, capacities, ord, use_profit=True):
    is_reverse = True if (ord == "C" or ord == None) else False
    for sol in solutions:
        fill_ratios = np.maximum(1e-12, sol["weights"]) / np.maximum(1e-12, capacities)
        fill_ratios = 1.0 - fill_ratios
        gm = np.power(np.prod(fill_ratios), 1 / len(capacities))
        sol["gm_score"] = abs(1.0 - gm)

    if use_profit:
        solutions.sort(
            key=lambda x: (-x["total_profit"], x["gm_score"] if is_reverse else -x["gm_score"]),
            reverse=True,
        )
    else:
        solutions.sort(
            key=lambda x: x["gm_score"] if is_reverse else -x["gm_score"],
            reverse=True,
        )
    return solutions


# Heurística Mínimo Máximo: max(wi/Ci)
def min_max(solutions, capacities, ord, use_profit=True):
    is_reverse = True if (ord == "C" or ord == None) else False
    for sol in solutions:
        fill_ratios = np.maximum(1e-12, sol["weights"]) / np.maximum(1e-12, capacities)
        sol["max_ratio"] = np.max(fill_ratios)

    if use_profit:
        solutions.sort(
            key=lambda x: (-x["total_profit"], x["max_ratio"] if is_reverse else -x["max_ratio"]),
            reverse=True,
        )
    else:
        solutions.sort(
            key=lambda x: x["max_ratio"] if is_reverse else -x["max_ratio"],
            reverse=True,
        )
    return solutions


# Filtro de Pareto Normal (Lucro Líquido e Soma dos Pesos)
def pareto_normal(solutions, use_profit=True):
    for sol in solutions:
        sol["w_sum"] = np.sum(sol["weights"])

    if use_profit:
        # Prioriza o maior lucro líquido (total_profit)
        # Se empatar no lucro, pega a que tiver a menor soma de pesos
        solutions.sort(key=lambda x: (-x["total_profit"], x["w_sum"]))
    else:
        # Apenas soma dos pesos (como no sem-demanda)
        solutions.sort(key=lambda x: x["w_sum"])
    return solutions


# Filtro de Pareto Avançado, permite diversidade
def pareto_advanced(solutions, limit, frac_limit_sec, use_profit=True):
    for sol in solutions:
        sol["w_sum"] = np.sum(sol["weights"])

    selected_ids = set()
    # 1. Melhores globais (Peso Total ou Lucro + Peso)
    if use_profit:
        solutions.sort(key=lambda x: (-x["total_profit"], x["w_sum"]))
    else:
        solutions.sort(key=lambda x: x["w_sum"])
        
    for sol in solutions[:limit]:
        selected_ids.add(id(sol))

    num_dims = len(solutions[0]["weights"])
    try:
        frac_limit_sec = float(frac_limit_sec)
    except (ValueError, TypeError):
        frac_limit_sec = 0.1
    add_limit = max(1, int(limit * frac_limit_sec))

    # 2. Melhores por dimensão (Peso Dimensão)
    for d in range(num_dims):
        solutions.sort(key=lambda x: (x["weights"][d], x["w_sum"]))
        for sol in solutions[:add_limit]:
            selected_ids.add(id(sol))

    return [sol for sol in solutions if id(sol) in selected_ids]


# Função para seleção da heurística de truncamento
def select_heuristic(
    solutions, capacities, limit, trunc_type, extra, min_capacities=None, use_profit=True
):
    if not solutions:
        return []

    # 1. Remover duplicatas
    unique_map = {}
    for sol in solutions:
        w_tuple = tuple(sol["weights"])
        if (
            w_tuple not in unique_map
            or sol["total_profit"] > unique_map[w_tuple]["total_profit"]
        ):
            unique_map[w_tuple] = sol

    trunc_solutions = list(unique_map.values())

    trunc_type = trunc_type.upper()
    match trunc_type:
        case "MG":
            return geometric_mean(trunc_solutions, capacities, extra, use_profit)[:limit]
        case "MGA":
            return geometric_mean_alt(trunc_solutions, capacities, extra, use_profit)[:limit]
        case "MGR":
            return geometric_mean_reverse(trunc_solutions, capacities, extra, use_profit)[:limit]
        case "MM":
            return min_max(trunc_solutions, capacities, extra, use_profit)[:limit]
        case "N":
            return pareto_normal(trunc_solutions, use_profit)[:limit]
        case "A":
            return pareto_advanced(trunc_solutions, limit, extra, use_profit)
        case "DV":
            return viability_deficit(trunc_solutions, min_capacities, use_profit)[:limit]
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
    cut_val,
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

    num_groups = len(processed_groups)
    # cut_val: 1 = sem cortes (100%), 2 = 50%, 3 = 33%...
    switch_point = num_groups // cut_val if cut_val > 0 else num_groups

    for group_idx, group in enumerate(processed_groups):
        if time.time() - start_time > 3660:
            print("Timeout reached (3660s). Terminating with empty solution.")
            return {}

        is_penalty_active = group_idx >= switch_point
        
        # Transição: No ponto de troca, atualizamos o total_profit de todas as soluções na DP
        # de bruto para líquido para que o truncamento use a penalidade como critério.
        if group_idx == switch_point:
            for p_val in dp:
                for sol in dp[p_val]:
                    penalty = calculate_penalty(sol["weights"], capacities, min_capacities, penalty_rate)
                    sol["total_profit"] = sol["gross_profit"] - penalty

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
                        
                        if is_penalty_active:
                            penalty = calculate_penalty(new_w, capacities, min_capacities, penalty_rate)
                            new_total = new_gross - penalty
                        else:
                            new_total = new_gross

                        new_dp[curr_p].append(
                            {
                                "weights": new_w,
                                "total_profit": new_total,
                                "gross_profit": new_gross,
                                "items": sol["items"] + [item_id_tuple],
                            }
                        )

        for p in list(new_dp.keys()):
            # Passamos is_penalty_active como use_profit para controlar o critério de desempate
            new_dp[p] = select_heuristic(
                new_dp[p], capacities, limit, trunc_type, extra, min_capacities, is_penalty_active
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
    cut_val,
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
        "trunc": "C" + trunc_type,
        "scale_vpl": scale_vpl,
        "limit": limit,
        "buffer": cap_buffer,
        "cut_val": cut_val,
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
    out_dir = os.path.join(root_dir, "resultados_ordenado-peso_chaveamento")
    os.makedirs(out_dir, exist_ok=True)
    if extra == None:
        out_name = f"{out_dir}/{dataset_tag}_C{trunc_type}_{int(scale_vpl)}_{limit}_{cap_buffer}_{cut_val}.json"
    else:
        out_name = f"{out_dir}/{dataset_tag}_C{trunc_type}_{int(scale_vpl)}_{limit}_{cap_buffer}_{cut_val}_{extra}.json"

    with open(out_name, "w") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"Concluído em {exec_time}s | VPL Líquido: {liquid_profit if final_dp else 'N/A'}")


def run_test(dataset_tag, trunc_type, scale_vpl, limit, cap_buffer, cut_val, extra):
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
    print(f"Exec {dataset_tag} | Híbrido | Trunc: {trunc_type} | Scale: {scale_vpl} | Limit: {limit} | Buffer: {cap_buffer} | Cut: {cut_val} | Extra: {extra}")
    start_t = time.time()
    final_dp = solve_mmkp(start_t, groups, capacities, trunc_type, scale_vpl, limit, cap_buffer, cut_val, extra, min_capacities, p_rate)
    exec_time = round(time.time() - start_t, 4)
    create_json(dataset_tag, file_path, final_dp, trunc_type, scale_vpl, limit, cap_buffer, cut_val, exec_time, capacities, min_capacities, p_rate, extra, lookup_dict)


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
                cut_val = int(parts[5]) if len(parts) > 5 else 2
                extra = parts[6] if len(parts) > 6 else None
                run_test(tag, trunc, scale, limit, buf, cut_val, extra)
