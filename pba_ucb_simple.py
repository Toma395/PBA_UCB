# ==========================================================
# pba_ucb_sim_console.py  — PBA-UCB (path-loss/SINR/QoS/最悪干渉) コンソール完全版（修正版）
# 依存: numpy, matplotlib
# 実行例:
#   python pba_ucb_sim_console.py
#   python pba_ucb_sim_console.py --R 20000 --seed 42 --no-plot
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
    S: int = 10                  # 想定SU数（学習対象は1台。他(S-1)は最悪干渉源）
    seed: int = 121
    eta: float = 0.5            # UCB探索係数 (Table 3.1)
    print_every: int = 1000     # 進捗表示間隔
    show_init_lines: int = 20   # 初期の詳細表示行数
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
    gTH_pu_db: float = 30.0     # PU閾値SINR [dB]（論文。安全側にするなら35.0）
    gTH_su_db: float = 5.0      # SU閾値SINR [dB]

    # 電力離散化（10段階 → 10本のパワー・アーム）
    power_levels: int = 10
    use_power_penalty: bool = True

def parse_args() -> Config:
    p = argparse.ArgumentParser(description="PBA-UCB (path-loss/SINR/QoS) console simulator")
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

#(x,y)座標系で Tx/Rx ペアを配置
def place_pairs(cfg: Config, 
                rng: np.random.Generator, 
                n_pairs: int,
                rx_offset_mean_m: float = 80.0, 
                rx_offset_std_m: float = 20.0):
    """送信機Txを一様に配置し、Rxは Tx 近傍に配置（ガウスオフセット）"""
    #n_pairs: TxとRxのペア数
    L = cfg.area_km * 1000.0
    tx = rng.random((n_pairs, 2)) * L
    off = rng.normal(loc=rx_offset_mean_m, scale=rx_offset_std_m, size=(n_pairs, 2))
    ang = rng.random(n_pairs) * 2.0 * math.pi
    rx = tx + np.stack([off[:,0]*np.cos(ang), off[:,1]*np.sin(ang)], axis=1)
    rx = np.clip(rx, 0.0, L)#rxを0〜Lに制限
    return tx, rx  #(n,2),(n,2)

def distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a: (N,2), b: (M,2) → 距離行列 (N,M)"""
    diff = a[:,None,:] - b[None,:,:]
    return np.linalg.norm(diff, axis=2)

def path_gain(cfg: Config, d_m: np.ndarray, fq_Hz: float) -> np.ndarray:
    """式(3.1) L = (Gt*Gr / d^xi) * (c/(4π f))^2  （チャネル利得）"""
    lam_term = (cfg.c / (4.0 * math.pi * fq_Hz)) ** 2
    with np.errstate(divide="ignore"):
        return cfg.Gt * cfg.Gr * lam_term / np.maximum(d_m, 1.0)**cfg.xi

# ----------------------------- SINR / レート ------------------------------
def sinr_su(cfg: Config,
            p_su_w: float,
            gain_su_su: float,
            gains_other_su_to_su: np.ndarray,
            gain_pu_to_su: float) -> float:
    """SU側の式(3.2)。他SU + PU を干渉に入れる."""
    N0 = dbm_to_w(cfg.N0_dbm)  # ノイズパワー

    # 他(S-1) SU からの最悪干渉
    I_su = 0.0
    if cfg.S > 1:
        Pmax = dbm_to_w(cfg.P_su_max_dbm)
        I_su = Pmax * float(np.sum(gains_other_su_to_su[:cfg.S-1]))

    # ★PU からの干渉（サブバンド q 上の PU_q を1本とみなす）
    Ppu = dbm_to_w(cfg.P_pu_tx_dbm)
    I_pu = Ppu * gain_pu_to_su

    return (p_su_w * gain_su_su) / (N0 + I_su + I_pu)


def sinr_pu_q(cfg: Config, 
              g_pu_self_q: float,
              g_su_to_purx_q: np.ndarray,  # shape (S,)
              p_self_w: float) -> float:
    """PUリンクqのSINR。最悪: 他(S-1)はPmax、自分は選択p。"""
    N0   = dbm_to_w(cfg.N0_dbm)
    Ppu  = dbm_to_w(cfg.P_pu_tx_dbm)
    Pmax = dbm_to_w(cfg.P_su_max_dbm)

    # 自分(インデックス0)のSU→PU_q干渉
    I_self   = p_self_w * float(g_su_to_purx_q[0])
    # 他(S-1)台の最悪干渉はPmax固定
    I_others = 0.0
    if cfg.S > 1:
        I_others = Pmax * float(np.sum(g_su_to_purx_q[1:cfg.S]))
    I = I_self + I_others
    return (Ppu * g_pu_self_q) / (N0 + I)


def achievable_rate(cfg: Config, gamma_su_lin: float) -> float:
    """式(3.3) R = W log2(1+γ)（両閾値を満たすときのみ呼ぶ）"""
    return cfg.W_Hz * math.log2(1.0 + gamma_su_lin)

# （おまけ）レイアウト図
def plot_layout(su_tx, su_rx, pu_tx, pu_rx, area_m=5000, save=None, show=True):
    fig, ax = plt.subplots(figsize=(9,7))

    # 先に線を引く（下地）
    for i in range(su_tx.shape[0]):
        ax.plot([su_tx[i,0], su_rx[i,0]], [su_tx[i,1], su_rx[i,1]],
                linestyle='--', color='red', alpha=0.5, zorder=1)
    for i in range(pu_tx.shape[0]):
        ax.plot([pu_tx[i,0], pu_rx[i,0]], [pu_tx[i,1], pu_rx[i,1]],
                linestyle='--', color='tab:blue', alpha=0.5, zorder=1)

    # BS（四角）は中段
    ax.scatter(su_tx[:,0], su_tx[:,1], marker='s', s=70,
               facecolors='none', edgecolors='red', linewidths=1.5,
               label='SU BS', zorder=2)
    ax.scatter(pu_tx[:,0], pu_tx[:,1], marker='s', s=70,
               facecolors='none', edgecolors='tab:blue', linewidths=1.5,
               label='PU BS', zorder=2)

    # UE（星）はいちばん上
    ax.scatter(su_rx[:,0], su_rx[:,1], marker='*', s=120,
               color='red', label='SU UE', zorder=3)
    ax.scatter(pu_rx[:,0], pu_rx[:,1], marker='*', s=120,
               color='tab:blue', label='PU UE', zorder=3)

    ax.set_xlim(0, area_m); ax.set_ylim(0, area_m)
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.set_title("SU/PU Tx–Rx layout")

    # 凡例（重複除去）
    handles, labels = ax.get_legend_handles_labels()
    seen=set(); hh=[]; ll=[]
    for h,l in zip(handles,labels):
        if l not in seen: hh.append(h); ll.append(l); seen.add(l)
    ax.legend(hh, ll, loc='upper right', frameon=True)

    fig.tight_layout()
    if save: plt.savefig(save, dpi=150, bbox_inches="tight")
    if show: plt.show()
    plt.close(fig)


# ----------------------------- メイン処理 ------------------------------
def run_sim(cfg: Config, algo: str = "ucb", verbose: bool = True):
    rng = np.random.default_rng(cfg.seed)

    fq_Hz = cfg.fq_GHz * 1e9
    Pmax_W = dbm_to_w(cfg.P_su_max_dbm)
    gTH_pu = db_to_linear(cfg.gTH_pu_db)
    gTH_su = db_to_linear(cfg.gTH_su_db)

    # ★電力レベル：厳密に 1/L, 2/L, ..., 1 × Pmax
    P_set = (np.arange(1, cfg.power_levels+1) / cfg.power_levels) * Pmax_W

    # 腕 = (q, i)
    arms = [(q, i) for q in range(cfg.Q) for i in range(cfg.power_levels)]
    A = len(arms)
    
    #PUのΔ計算と配置
    delta_m = compute_delta_for_30dB(cfg)
    if verbose:
        print(delta_m)

    # ★PUは Q 本配置（サブバンドと1対1）
    pu_tx, pu_rx = place_pairs(cfg, rng, n_pairs=cfg.Q, rx_offset_mean_m=delta_m/2.0,) #(Q,2),(Q,2)
    su_tx, su_rx = place_pairs(cfg, rng, n_pairs=cfg.S) #(S,2),(S,2)

    # 経路利得
    d_su_su = distances(su_tx, su_rx)                       # (S,S) 対角: 自己リンク距離
    g_su_su = path_gain(cfg, np.diag(d_su_su), fq_Hz)       # (S,)　
    g_other_to_su = path_gain(cfg, d_su_su, fq_Hz)          # (S,S) 他SU→各SU-Rx

    d_su_purx = distances(su_tx, pu_rx)                     # (S,Q) SU→各PU_q-Rx
    g_su_purx = path_gain(cfg, d_su_purx, fq_Hz)            # (S,Q)

    d_pu_self = np.diag(distances(pu_tx, pu_rx))            # (Q,) PU_q Tx→Rx　対角行列化
    g_pu_self = path_gain(cfg, d_pu_self, fq_Hz)            # (Q,)

    # PU Tx → SU Rx（PU_q → SU_ω の干渉用）
    d_pu_surx = distances(pu_tx, su_rx)                     # (Q,S)
    g_pu_surx = path_gain(cfg, d_pu_surx, fq_Hz)            # (Q,S)

    # UCB統計
    T = np.zeros(A, dtype=int)
    mu = np.zeros(A, dtype=float)
    regret = np.zeros(cfg.R, dtype=float)

    # ★Oracle（サブバンドqごとに PU_q を評価）
    oracle_mean = np.zeros(A, dtype=float)
    for a,(q,i) in enumerate(arms):
        p = float(P_set[i])
        
        # PU_q → 「学習対象 SU (index 0)」への利得
        gain_pu_to_su = float(g_pu_surx[q, 0])
        
        gamma_su = sinr_su(
            cfg,
            p,
            float(g_su_su[0]),
            gains_other_su_to_su=g_other_to_su[1:,0],
            gain_pu_to_su=gain_pu_to_su
        )
        
        gamma_pu = sinr_pu_q(cfg, float(g_pu_self[q]), g_su_to_purx_q=g_su_purx[:,q],p_self_w=p)
        oracle_mean[a] = achievable_rate(cfg, gamma_su) if (gamma_pu >= gTH_pu) and (gamma_su >= gTH_su) else 0.0
    oracle_best = float(np.max(oracle_mean))
    if verbose:
        print(f"[INFO] Oracle best achievable rate: {oracle_best/1e6:.3f} Mbit/s  (under worst-case design)")
        
    # UCBスコア
    def ucb_score(idx: int, t: int) -> float:
        if T[idx] == 0:
            return float("inf")
        bonus = math.sqrt(cfg.eta * math.log(max(2, t)) / T[idx])
        q, i = arms[idx]
        p = float(P_set[i])
        pen = (p / Pmax_W) if cfg.use_power_penalty else 0.0
        return mu[idx] + bonus - pen

    # 初期化フェーズ（各腕を1回ずつ）
    t = 0
    if verbose:
        print("[INFO] Initialization phase...")
    for a in range(A):
        if t >= cfg.R: break
        q, i = arms[a]
        p = float(P_set[i])
        
        gain_pu_to_su = float(g_pu_surx[q, 0])

        gamma_su = sinr_su(
            cfg,
            p,
            float(g_su_su[0]),
            gains_other_su_to_su=g_other_to_su[1:,0],
            gain_pu_to_su=gain_pu_to_su
        )
        gamma_pu = sinr_pu_q(cfg, float(g_pu_self[q]), g_su_to_purx_q=g_su_purx[:,q],p_self_w=p)

        r = achievable_rate(cfg, gamma_su) if (gamma_pu >= gTH_pu) and (gamma_su >= gTH_su) else 0.0
        T[a] += 1
        mu[a] += (r - mu[a]) / T[a]
        regret[t] = (regret[t-1] if t>0 else 0.0) + (oracle_best - r)

        if (verbose and t < cfg.show_init_lines) or (t % cfg.print_every == 0):
            su_snr_db = linear_to_db(gamma_su)
            pu_snr_db = linear_to_db(gamma_pu)
            print(f"[INIT] t={t:5d} arm={a:3d} q={q} P={p*1e3:6.1f} mW  "
                  f"SINR_su={su_snr_db:6.2f} dB  SINR_pu={pu_snr_db:6.2f} dB  "
                  f"R={r/1e6:7.3f} Mb/s  regret={regret[t]:.3f}")
        t += 1

    # レート最大化フェーズ（UCB）
    if verbose:
        print("[INFO] Rate maximization phase (UCB)...")
    for tt in range(t, cfg.R):
        scores = np.array([ucb_score(a, tt+1) for a in range(A)], dtype=float)
        a_star = int(np.argmax(scores))
        q, i = arms[a_star]
        p = float(P_set[i])
        gain_pu_to_su = float(g_pu_surx[q, 0])
        
        gamma_su = sinr_su(
            cfg,
            p,
            float(g_su_su[0]),
            gains_other_su_to_su=g_other_to_su[1:,0],
           gain_pu_to_su=gain_pu_to_su
        )
        gamma_pu = sinr_pu_q(cfg, float(g_pu_self[q]), g_su_to_purx_q=g_su_purx[:,q], p_self_w=p)

        r = achievable_rate(cfg, gamma_su) if (gamma_pu >= gTH_pu) and (gamma_su >= gTH_su) else 0.0
        T[a_star] += 1
        mu[a_star] += (r - mu[a_star]) / T[a_star]
        regret[tt] = regret[tt-1] + (oracle_best - r)

        if verbose and (tt % cfg.print_every == 0):
            su_snr_db = linear_to_db(gamma_su)
            pu_snr_db = linear_to_db(gamma_pu)
            print(f"[UCB ] t={tt:5d} arm={a_star:3d} q={q} P={p*1e3:6.1f} mW  "
                  f"SINR_su={su_snr_db:6.2f} dB  SINR_pu={pu_snr_db:6.2f} dB  "
                  f"R={r/1e6:7.3f} Mb/s  regret={regret[tt]:.3f}")

    print("[INFO] Simulation done.\n")

    # 結果まとめ
    pulls = int(np.sum(T))
    avg_rate = float(np.sum(mu * T) / max(pulls,1))
    best_idx = int(np.argmax(mu))
    bq, bi = arms[best_idx]
    
    if verbose:
        print("[INFO] Simulation done.\n")
        print("========== 結果まとめ ==========")
        print(f"平均SUスループット: {avg_rate/1e6:.3f} Mbit/s/slot (QoS満足時のみカウント)")
        print(f"累積Regret(最終):  {regret[-1]/1e6:.3f} Mbit")
        print(f"学習で選ばれがちな腕: subband={bq}, power={P_set[bi]*1e3:.1f} mW")
        print(f"Oracle最良腕の平均: {oracle_best/1e6:.3f} Mbit/s")
        print("================================")

    # 図（任意表示）
    if cfg.plot:
        plt.figure()
        plt.plot(regret/1e6, label="cumulative regret [Mbit]")
        plt.xlabel("time slot")
        plt.ylabel("regret [Mbit]")
        plt.title("PBA-UCB regret (path-loss/QoS/worst-case interference)")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # レイアウト図（欲しければ）
    plot_layout(su_tx, su_rx, pu_tx, pu_rx, area_m=int(cfg.area_km*1000),
                save="layout.png", show=cfg.plot)
    
    # ★ 掃引用に結果を返す ★
    return avg_rate, oracle_best, regret

# ===================== 並列実行用ワーカー（S掃引） =====================
def _trial_vs_S(params):
    S, trial_idx, R, Q_fixed, base_seed = params
    # 試行ごとに seed をずらす
    seed = base_seed + 1000 * trial_idx + S
    cfg = Config(R=R, Q=Q_fixed, S=S, seed=seed, plot=False)
    avg_rate, oracle, _ = run_sim(cfg, algo="ucb", verbose=False)
    # S, 何回目のトライアルか, 正規化平均レート を返す
    return S, trial_idx, avg_rate / cfg.W_Hz  # [bps/Hz]


# ===================== 並列実行用ワーカー（Q掃引） =====================
def _trial_vs_Q(params):
    Q, trial_idx, R, S_fixed, base_seed = params
    seed = base_seed + 1000 * trial_idx + Q
    cfg = Config(R=R, Q=Q, S=S_fixed, seed=seed, plot=False)
    avg_rate, oracle, _ = run_sim(cfg, algo="ucb", verbose=False)
    # Q, 何回目のトライアルか, 正規化平均レート
    return Q, trial_idx, avg_rate / cfg.W_Hz


    
# ----------------------------- パラメータ掃引（S を変える：並列MC版） ------------------------------
def experiment_vs_S_mc(num_trials: int = 50, base_seed: int = 1234, n_procs: int | None = None):
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
                 label=f"PBA-UCB (mean ± 1 SE, {num_trials} trials)")
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
                 label=f"PBA-UCB (mean ± 1 SE, {num_trials} trials)")
    plt.xlabel("Number of ETC gates (Q)")
    plt.ylabel("Normalized average sum rate [bps/Hz]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    # 単発実験
    cfg = parse_args()
    run_sim(cfg, algo="ucb", verbose=True)

    # 並列MC版：S掃引
    #experiment_vs_S_mc(num_trials=50)

    # 並列MC版：Q掃引もやりたいならこちらも
    # experiment_vs_Q_mc(num_trials=50)

if __name__ == "__main__":
    main()
