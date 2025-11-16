# compare_ts_ucb_s.py
# TS / UCB / Random を S ごとに比較
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count

from pba_ts_multi   import Config as TSConfig,   run_sim as run_sim_ts
from pba_ucb_multi  import Config as UCBConfig,  run_sim as run_sim_ucb
from pba_rand_multi import Config as RandConfig, run_sim as run_sim_rand

def run_single_trial(params):
    """
    1本の試行を実行するワーカー
    戻り値: (algo, S, normalized_avg_rate)
    """
    algo, S, trial_idx, R, Q, base_seed = params

    seed = base_seed + trial_idx * 999 + S

    if algo == "ts":
        cfg = TSConfig(R=R, Q=Q, S=S, seed=seed, plot=False)
        avg, oracle, regret = run_sim_ts(cfg, verbose=False)
    elif algo == "ucb":
        cfg = UCBConfig(R=R, Q=Q, S=S, seed=seed, plot=False)
        avg, oracle, regret = run_sim_ucb(cfg, verbose=False)
    else:  # "rand"
        cfg = RandConfig(R=R, Q=Q, S=S, seed=seed, plot=False)
        avg, oracle, regret = run_sim_rand(cfg, verbose=False)

    return algo, S, avg / cfg.W_Hz  # normalized [bps/Hz]

def compare_vs_S():
    S_list = [5, 10, 15, 20, 25, 30, 35, 40]
    Q_fixed = 10     # ETCゲート数は固定
    R = 20000
    num_trials = 20
    base_seed = 1234

    algos = ["ts", "ucb", "rand"]

    tasks = []
    for algo in algos:
        for S in S_list:
            for k in range(num_trials):
                tasks.append((algo, S, k, R, Q_fixed, base_seed))

    print(f"[INFO] Total tasks = {len(tasks)}")

    if __name__ == "__main__":
        n_procs = cpu_count()
    else:
        n_procs = 1

    results = []
    with Pool(processes=n_procs) as pool:
        for idx, res in enumerate(pool.imap_unordered(run_single_trial, tasks), 1):
            results.append(res)
            if idx % 10 == 0 or idx == len(tasks):
                algo_cur, S_cur, rate_cur = res
                print(f"[INFO] finished {idx}/{len(tasks)} "
                      f"(last: algo={algo_cur}, S={S_cur}, rate={rate_cur:.3f})")

    # --- 集計 ---
    S_arr = np.array(S_list, dtype=int)

    def agg_for_algo(target_algo: str):
        mean = np.zeros_like(S_arr, dtype=float)
        sem  = np.zeros_like(S_arr, dtype=float)
        for i, S in enumerate(S_list):
            vals = [rate for (algo, Ss, rate) in results
                    if (algo == target_algo and Ss == S)]
            vals = np.array(vals, dtype=float)
            mean[i] = vals.mean()
            sem[i]  = vals.std(ddof=1) / np.sqrt(len(vals))
        return mean, sem

    mean_ts,   sem_ts   = agg_for_algo("ts")
    mean_ucb,  sem_ucb  = agg_for_algo("ucb")
    mean_rand, sem_rand = agg_for_algo("rand")

    # --- プロット ---
    plt.figure()
    plt.errorbar(S_arr, mean_ts,   yerr=sem_ts,
                 marker="o", capsize=4, label="PBA-TS")
    plt.errorbar(S_arr, mean_ucb,  yerr=sem_ucb,
                 marker="s", capsize=4, label="PBA-UCB")
    plt.errorbar(S_arr, mean_rand, yerr=sem_rand,
                 marker="^", capsize=4, label="Random")

    plt.xlabel("Number of UAVs (S)")
    plt.ylabel("Normalized average sum rate [bps/Hz]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    compare_vs_S()
