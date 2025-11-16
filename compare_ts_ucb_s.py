# ==========================================================
# compare_ts_ucb_vs_S.py
#   TS vs UCB の S 依存比較（横軸：S、縦軸：normalized rate）
# ==========================================================
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count

from pba_ts_multi import Config as TSConfig, run_sim as run_sim_ts
from pba_ucb_multi import Config as UCBConfig, run_sim as run_sim_ucb

# --------------------------------------------------------
# 単一試行: TS or UCB を実行して normalized rate を返す
# --------------------------------------------------------
def run_single_trial(params):
    algo, S, trial_idx, R, Q, base_seed = params

    # 試行ごとに seed を少しずつずらす
    seed = base_seed + trial_idx * 999 + S

    if algo == "ts":
        cfg = TSConfig(R=R, Q=Q, S=S, seed=seed, plot=False)
        avg, oracle, regret = run_sim_ts(cfg, verbose=False)
    else:
        cfg = UCBConfig(R=R, Q=Q, S=S, seed=seed, plot=False)
        avg, oracle, regret = run_sim_ucb(cfg, verbose=False)

    # normalized average sum rate [bps/Hz]
    return algo, S, avg / cfg.W_Hz



# --------------------------------------------------------
# TS/UCB の S ごとの比較
# --------------------------------------------------------
def compare_vs_S(S_list=[5,10,15,20,25,30,35,40], num_trials=5, R=10000, Q=10):

    tasks = []
    base_seed = 1234

    # TS と UCB 両方の試行タスクを作成
    for S in S_list:
        for trial in range(num_trials):
            tasks.append(("ts",  S, trial, R, Q, base_seed))
            tasks.append(("ucb", S, trial, R, Q, base_seed))

    print(f"[INFO] Total tasks = {len(tasks)}")

    # 並列処理
    with Pool(processes=cpu_count()) as pool:
        results = list(pool.imap_unordered(run_single_trial, tasks))

    # 結果整理
    S_arr = np.array(S_list)

    ts_mean, ts_sem = [], []
    ucb_mean, ucb_sem = [], []

    for S in S_list:
        ts_vals  = [rate for (algo, Ss, rate) in results if algo=="ts"  and Ss==S]
        ucb_vals = [rate for (algo, Ss, rate) in results if algo=="ucb" and Ss==S]

        ts_mean.append(np.mean(ts_vals))
        ucb_mean.append(np.mean(ucb_vals))

        ts_sem.append(np.std(ts_vals,  ddof=1) / np.sqrt(num_trials))
        ucb_sem.append(np.std(ucb_vals, ddof=1) / np.sqrt(num_trials))

    # → numpy array
    ts_mean = np.array(ts_mean)
    ucb_mean = np.array(ucb_mean)
    ts_sem = np.array(ts_sem)
    ucb_sem = np.array(ucb_sem)

    # プロット
    plt.figure(figsize=(10,5))
    plt.errorbar(S_arr, ts_mean, yerr=ts_sem, marker='o', capsize=4, label="PBA-TS")
    plt.errorbar(S_arr, ucb_mean, yerr=ucb_sem, marker='s', capsize=4, label="PBA-UCB")

    plt.xlabel("Number of UAVs (S)")
    plt.ylabel("Normalized average sum rate [bps/Hz]")
    plt.title("TS vs UCB: Average normalized rate vs S")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return ts_mean, ucb_mean


if __name__ == "__main__":
    compare_vs_S()
