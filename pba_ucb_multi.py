# ==========================================================
# pba_ucb_multi.py  — PBA-UCB (path-loss/SINR/QoS/最悪干渉) 多エージェント完全版
# 依存: numpy, matplotlib
# 実行例:
#   python pba_ts_multi.py
#   python pba_ts_multi.py --R 20000 --seed 42 --no-plot
# ==========================================================
from __future__ import annotations
import math, argparse
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count

# ---------------------- 物理ユーティリティ（自前定義） ----------------------
def dbm_to_w(dbm: float) -> float:
    return 10.0 ** ((dbm - 30.0) / 10.0)

def db_to_linear(db: float) -> float:
    return 10.0 ** (db / 10.0)

def linear_to_db(x: float) -> float:
    return 10.0 * math.log10(max(x, 1e-30))

# ----------------------------- 設定（CLI対応） -----------------------------
@dataclass
class Config:
    # シミュレーション基本
    R: int = 20000              # 総スロット
    Q: int = 10                 # サブバンド数（= PUリンク数）
    S: int = 10                 # SU数
    seed: int = 121
    eta: float = 0.5            # TS探索係数 (Table 3.1 相当、ただしここでは φ のバイアスには未使用)
    print_every: int = 1000     # 進捗表示間隔
    show_init_lines: int = 20   # （未使用）初期の詳細表示行数
    plot: bool = True           # Regret 図を表示

    # 物理パラメータ（Table 3.1 と式(3.1)〜(3.4)）
    area_km: float = 5.0        # 5km × 5km
    W_Hz: float = 10e6          # サブバンド帯域 W
    fq_GHz: float = 5.8         # 搬送周波数
    c: float = 3.0e8            # 光速
    xi: float = 3.0             # 減衰定数 ξ（path-loss exponent）
    Gt: float = 1.0             # 送信アンテナ利得（無指向=1）
    Gr: float = 1.0             # 受信アンテナ利得（無指向=1）

    P_pu_tx_dbm: float = 24.0   # PU送信電力 [dBm]
    P_su_max_dbm: float = 30.0  # SU最大送信電力 [dBm]
    N0_dbm: float = -100.0      # 熱雑音パワー [dBm]（1 Hz基準扱い）
    gTH_pu_db: float = 30.0     # PU閾値SINR [dB]（論文と同等）
    gTH_su_db: float = 5.0      # SU閾値SINR [dB]

    # 電力離散化（L段階 → L本のパワー・アーム）
    power_levels: int = 10
    use_power_penalty: bool = True  # ※多エージェント版TSでは明示的には使っていない

def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Multi-agent PBA-TS (path-loss/SINR/QoS) simulator")
    p.add_argument("--R", type=int, default=20000)
    p.add_argument("--Q", type=int, default=10)
    p.add_argument("--S", type=int, default=10)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--eta", type=float, default=0.5)
    p.add_argument("--print-every", type=int, default=1000)
    p.add_argument("--show-init-lines", type=int, default=20)
    p.add_argument("--no-plot", action="store_true")
    a = p.parse_args()
    cfg = Config(R=a.R, Q=a.Q, S=a.S, seed=a.seed, eta=a.eta,
                 print_every=a.print_every, show_init_lines=a.show_init_lines,
                 plot=(not a.no_plot))
    return cfg

# ----------------------------- 幾何/経路損失 ------------------------------
# 30 dB 閾値を満たす最大距離 delta の計算　干渉は考慮しない
def compute_delta_for_30dB(cfg: Config) -> float:
    fq = cfg.fq_GHz * 1e9
    Ppu = dbm_to_w(cfg.P_pu_tx_dbm)
    N0  = dbm_to_w(cfg.N0_dbm)        # 論文は境界SNR基準。干渉は無視してOK
    gamma = db_to_linear(cfg.gTH_pu_db)  # 30 dB → 1000
    lam_term = (cfg.c / (4.0 * math.pi * fq)) ** 2
    num = Ppu * cfg.Gt * cfg.Gr * lam_term
    delta = (num / (N0 * gamma)) ** (1.0 / cfg.xi)
    return float(delta)

# (x,y)座標系で Tx/Rx ペアを配置
def place_pairs(cfg: Config,
                rng: np.random.Generator,
                n_pairs: int,
                rx_offset_mean_m: float = 100.0,
                rx_offset_std_m: float = 20.0):
    """送信機Txを一様に配置し、Rxは Tx 近傍に配置（ガウスオフセット）"""
    L = cfg.area_km * 1000.0
    tx = rng.random((n_pairs, 2)) * L
    off = rng.normal(loc=rx_offset_mean_m, scale=rx_offset_std_m, size=(n_pairs, 2))
    ang = rng.random(n_pairs) * 2.0 * math.pi
    rx = tx + np.stack([off[:, 0] * np.cos(ang), off[:, 1] * np.sin(ang)], axis=1)
    rx = np.clip(rx, 0.0, L)  # rxを0〜Lに制限
    return tx, rx  # (n,2),(n,2)

def distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a: (N,2), b: (M,2) → 距離行列 (N,M)"""
    diff = a[:, None, :] - b[None, :, :]
    return np.linalg.norm(diff, axis=2)

def path_gain(cfg: Config, d_m: np.ndarray, fq_Hz: float) -> np.ndarray:
    """式(3.1) L = (Gt*Gr / d^xi) * (c/(4π f))^2  （チャネル利得）"""
    lam_term = (cfg.c / (4.0 * math.pi * fq_Hz)) ** 2
    with np.errstate(divide="ignore"):
        return cfg.Gt * cfg.Gr * lam_term / np.maximum(d_m, 1.0) ** cfg.xi

# ----------------------------- SINR / レート ------------------------------
def achievable_rate(cfg: Config, gamma_su_lin: float) -> float:
    """式(3.3) R = W log2(1+γ)（両閾値を満たすときのみ呼ぶ）"""
    return cfg.W_Hz * math.log2(1.0 + gamma_su_lin)

# （おまけ）レイアウト図
def plot_layout(su_tx, su_rx, pu_tx, pu_rx, area_m=5000, save=None, show=True):
    fig, ax = plt.subplots(figsize=(9, 7))

    # 先に線を引く（下地）
    for i in range(su_tx.shape[0]):
        ax.plot([su_tx[i, 0], su_rx[i, 0]], [su_tx[i, 1], su_rx[i, 1]],
                linestyle='--', color='red', alpha=0.5, zorder=1)
    for i in range(pu_tx.shape[0]):
        ax.plot([pu_tx[i, 0], pu_rx[i, 0]], [pu_tx[i, 1], pu_rx[i, 1]],
                linestyle='--', color='tab:blue', alpha=0.5, zorder=1)

    # BS（四角）は中段
    ax.scatter(su_tx[:, 0], su_tx[:, 1], marker='s', s=70,
               facecolors='none', edgecolors='red', linewidths=1.5,
               label='SU BS', zorder=2)
    ax.scatter(pu_tx[:, 0], pu_tx[:, 1], marker='s', s=70,
               facecolors='none', edgecolors='tab:blue', linewidths=1.5,
               label='PU BS', zorder=2)

    # UE（星）はいちばん上
    ax.scatter(su_rx[:, 0], su_rx[:, 1], marker='*', s=120,
               color='red', label='SU UE', zorder=3)
    ax.scatter(pu_rx[:, 0], pu_rx[:, 1], marker='*', s=120,
               color='tab:blue', label='PU UE', zorder=3)

    ax.set_xlim(0, area_m)
    ax.set_ylim(0, area_m)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("SU/PU Tx–Rx layout")

    # 凡例（重複除去）
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    hh = []
    ll = []
    for h, l in zip(handles, labels):
        if l not in seen:
            hh.append(h)
            ll.append(l)
            seen.add(l)
    ax.legend(hh, ll, loc='upper right', frameon=True)

    fig.tight_layout()
    if save:
        plt.savefig(save, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


# ----------------------------- メイン処理（多エージェント PBA-TS） ------------------------------
def run_sim(cfg: Config, verbose: bool = True):
    """
    多エージェント版 PBA-UCB
      - SU ω = 0..S-1 が全員 UCB を実行
      - 腕 = (ω, q, ℓ)
      - TS との差分は「φ の代わりに UCBスコア」を使うだけ
    """
    rng = np.random.default_rng(cfg.seed)

    fq_Hz   = cfg.fq_GHz * 1e9
    Pmax_W  = dbm_to_w(cfg.P_su_max_dbm)
    N0_W    = dbm_to_w(cfg.N0_dbm)
    gTH_pu  = db_to_linear(cfg.gTH_pu_db)
    gTH_su  = db_to_linear(cfg.gTH_su_db)

    L = cfg.power_levels
    P_set = (np.arange(1, L+1) / L) * Pmax_W

    S = cfg.S
    Q = cfg.Q
    R = cfg.R

    # 配置
    delta_m = compute_delta_for_30dB(cfg)
    if verbose:
        print(f"[INFO] delta (30 dB) ≈ {delta_m:.2f} m")

    pu_tx, pu_rx = place_pairs(cfg, rng, n_pairs=Q, rx_offset_mean_m=delta_m/2.0)
    su_tx, su_rx = place_pairs(cfg, rng, n_pairs=S)

    # 経路利得
    d_su_su  = distances(su_tx, su_rx)
    g_su_su  = path_gain(cfg, np.diag(d_su_su), fq_Hz)
    g_su_su_x = path_gain(cfg, d_su_su, fq_Hz)

    d_su_purx = distances(su_tx, pu_rx)
    g_su_purx = path_gain(cfg, d_su_purx, fq_Hz)

    d_pu_self = np.diag(distances(pu_tx, pu_rx))
    g_pu_self = path_gain(cfg, d_pu_self, fq_Hz)

    d_pu_surx = distances(pu_tx, su_rx)
    g_pu_surx = path_gain(cfg, d_pu_surx, fq_Hz)

    # UCB 統計（TS の sigma2 は不要）
    T  = np.zeros((S, Q, L), dtype=int)
    mu = np.zeros((S, Q, L), dtype=float)

    regret = np.zeros(R)
    total_sum_rate = 0.0

    Ppu_W = dbm_to_w(cfg.P_pu_tx_dbm)

    # Oracle
    oracle_best = 0.0
    for w in range(S):
        for q in range(Q):
            best = 0.0
            for p in P_set:
                I_su = Ppu_W * g_pu_surx[q, w]
                gamma_su = (p * g_su_su[w]) / (N0_W + I_su)

                I_pu = p * g_su_purx[w, q]
                gamma_pu = (Ppu_W * g_pu_self[q]) / (N0_W + I_pu)

                if gamma_su >= gTH_su and gamma_pu >= gTH_pu:
                    best = max(best, achievable_rate(cfg, gamma_su))
            oracle_best += best

    if verbose:
        print(f"[INFO] Approx. oracle best sum rate: {oracle_best/1e6:.3f} Mbit/slot")

    if verbose:
        print("[INFO] Multi-agent PBA-UCB simulation start...")

    # ===== メインループ =====
    for t in range(R):

        p_residual = np.full(S, Pmax_W)
        p_su = np.zeros((S, Q))
        chosen = -np.ones((S, Q), dtype=int)

        slot_rate = 0.0

        for q in range(Q):
            for w in range(S):

                if p_residual[w] <= 0:
                    continue

                # feasible arm
                feas = np.where(P_set <= p_residual[w])[0]
                if feas.size == 0:
                    continue

                # UCB スコア μ + √(η log t / T) - p/p_residual
                ucb_scores = []
                for l in feas:
                    if T[w,q,l] == 0:
                        bonus = float("inf")
                    else:
                        bonus = math.sqrt(cfg.eta * math.log(t+2) / T[w,q,l])
                    score = mu[w,q,l] + bonus - (P_set[l] / p_residual[w])
                    ucb_scores.append(score)

                best_pos = int(np.argmax(ucb_scores))
                l_star = int(feas[best_pos])

                p = float(P_set[l_star])
                p_su[w,q] = p
                chosen[w,q] = l_star

            # --- SINR 一括計算 ---
            I_pu_q = float(np.sum(p_su[:,q] * g_su_purx[:,q]))
            gamma_pu_q = (Ppu_W * g_pu_self[q]) / (N0_W + I_pu_q)

            for w in range(S):
                l = chosen[w,q]
                if l < 0:
                    continue

                p = p_su[w,q]

                I_su_wq = float(np.sum(p_su[:,q] * g_su_su_x[:,w]) - p*g_su_su_x[w,w])
                I_pu_to_su = Ppu_W * g_pu_surx[q,w]

                gamma_su_wq = (p*g_su_su[w]) / (N0_W + I_su_wq + I_pu_to_su)

                success = (gamma_pu_q >= gTH_pu) and (gamma_su_wq >= gTH_su)

                if success:
                    r = achievable_rate(cfg, gamma_su_wq)
                    slot_rate += r
                    total_sum_rate += r
                else:
                    r = 0.0

                T[w,q,l] += 1
                cnt = T[w,q,l]
                mu[w,q,l] += (r - mu[w,q,l]) / cnt

                # 残余電力更新
                p_residual[w] = max(p_residual[w] - p, 0)

        if t == 0:
            regret[t] = oracle_best - slot_rate
        else:
            regret[t] = regret[t-1] + (oracle_best - slot_rate)

        if verbose and (t % cfg.print_every == 0):
            print(f"[UCB] t={t:5d}  sum={slot_rate/1e6:.3f} Mbit  norm={slot_rate/cfg.W_Hz:.3f}  regret={regret[t]/1e6:.3f}")

    avg_sum_rate = total_sum_rate / R

    if verbose:
        print("[INFO] UCB simulation done.")
        print(f"平均サムレート = {avg_sum_rate/1e6:.3f} Mbit/slot")
        print(f"正規化 = {avg_sum_rate/cfg.W_Hz:.3f} bps/Hz")

    return avg_sum_rate, oracle_best, regret


# ===================== 並列実行用ワーカー（S掃引） =====================
def _trial_vs_S(params):
    S, trial_idx, R, Q_fixed, base_seed = params
    # 試行ごとに seed をずらす
    seed = base_seed + 1000 * trial_idx + S
    cfg = Config(R=R, Q=Q_fixed, S=S, seed=seed, plot=False)
    avg_rate, oracle, _ = run_sim(cfg, verbose=False)
    # S, 何回目のトライアルか, 正規化平均レート を返す
    return S, trial_idx, avg_rate / cfg.W_Hz  # [bps/Hz]


# ===================== 並列実行用ワーカー（Q掃引） =====================
def _trial_vs_Q(params):
    Q, trial_idx, R, S_fixed, base_seed = params
    seed = base_seed + 1000 * trial_idx + Q
    cfg = Config(R=R, Q=Q, S=S_fixed, seed=seed, plot=False)
    avg_rate, oracle, _ = run_sim(cfg, verbose=False)
    # Q, 何回目のトライアルか, 正規化平均レート
    return Q, trial_idx, avg_rate / cfg.W_Hz


# ----------------------------- パラメータ掃引（S を変える：並列MC版） ------------------------------
def experiment_vs_S_mc(num_trials: int = 20, base_seed: int = 1234, n_procs: int | None = None):
    S_list = [5, 10, 15, 20, 25, 30, 35, 40]
    Q_fixed = 10
    R = 20000

    # 並列実行用タスク列を作る
    tasks = []
    for S in S_list:
        for k in range(num_trials):
            tasks.append((S, k, R, Q_fixed, base_seed))

    if n_procs is None:
        n_procs = cpu_count()

    total_tasks = len(tasks)
    print(f"[INFO] experiment_vs_S_mc: trials={num_trials}, procs={n_procs}, total_tasks={total_tasks}")

    results = []
    # 並列実行 + 進捗表示
    with Pool(processes=n_procs) as pool:
        for idx, res in enumerate(pool.imap_unordered(_trial_vs_S, tasks), 1):
            results.append(res)
            if idx % 10 == 0 or idx == total_tasks:
                S_cur, trial_idx_cur, _ = res
                print(f"[INFO] finished {idx}/{total_tasks} (last: S={S_cur}, trial={trial_idx_cur+1})")

    # results: [(S, trial_idx, rate), ...] を S ごとにまとめて平均・標準誤差を計算
    S_arr = np.array(S_list, dtype=int)
    mean_rates = np.zeros_like(S_arr, dtype=float)
    sem_rates  = np.zeros_like(S_arr, dtype=float)   # 標準誤差

    for i, S in enumerate(S_list):
        vals = [rate for (Ss, trial_idx, rate) in results if Ss == S]
        vals = np.array(vals, dtype=float)
        mean_rates[i] = vals.mean()
        # 標準誤差 = 標準偏差 / sqrt(N)
        sem_rates[i]  = vals.std(ddof=1) / np.sqrt(len(vals))

    # 図を描画（平均＋標準誤差）
    plt.figure()
    plt.errorbar(S_arr, mean_rates, yerr=sem_rates,
                 marker="o", capsize=4,
                 label=f"PBA-TS (mean ± 1 SE, {num_trials} trials)")
    plt.xlabel("Number of UAVs (S)")
    plt.ylabel("Normalized average sum rate [bps/Hz]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# ----------------------------- パラメータ掃引（Q を変える：並列MC版） ------------------------------
def experiment_vs_Q_mc(num_trials: int = 50, base_seed: int = 1234, n_procs: int | None = None):
    Q_list = [5, 10, 15, 20, 25, 30, 35, 40]
    S_fixed = 10
    R = 20000

    tasks = []
    for Q in Q_list:
        for k in range(num_trials):
            tasks.append((Q, k, R, S_fixed, base_seed))

    if n_procs is None:
        n_procs = cpu_count()

    total_tasks = len(tasks)
    print(f"[INFO] experiment_vs_Q_mc: trials={num_trials}, procs={n_procs}, total_tasks={total_tasks}")

    results = []
    # 並列実行 + 進捗表示
    with Pool(processes=n_procs) as pool:
        for idx, res in enumerate(pool.imap_unordered(_trial_vs_Q, tasks), 1):
            results.append(res)
            if idx % 10 == 0 or idx == total_tasks:
                Q_cur, trial_idx_cur, _ = res
                print(f"[INFO] finished {idx}/{total_tasks} (last: Q={Q_cur}, trial={trial_idx_cur+1})")

    Q_arr = np.array(Q_list, dtype=int)
    mean_rates = np.zeros_like(Q_arr, dtype=float)
    sem_rates  = np.zeros_like(Q_arr, dtype=float)

    for i, Q in enumerate(Q_list):
        vals = [rate for (Qq, trial_idx, rate) in results if Qq == Q]
        vals = np.array(vals, dtype=float)
        mean_rates[i] = vals.mean()
        sem_rates[i]  = vals.std(ddof=1) / np.sqrt(len(vals))

    plt.figure()
    plt.errorbar(Q_arr, mean_rates, yerr=sem_rates,
                 marker="o", capsize=4,
                 label=f"PBA-TS (mean ± 1 SE, {num_trials} trials)")
    plt.xlabel("Number of ETC gates (Q)")
    plt.ylabel("Normalized average sum rate [bps/Hz]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    # 単発実験
    cfg = parse_args()
    run_sim(cfg, verbose=True)

    # 並列MC版：S掃引
    #experiment_vs_S_mc(num_trials=20)

    # 並列MC版：Q掃引もやりたいならこちらも
    # experiment_vs_Q_mc(num_trials=20)

if __name__ == "__main__":
    main()