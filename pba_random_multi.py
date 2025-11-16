# ==========================================================
# pba_rand_multi.py  — Random (path-loss/SINR/QoS/最悪干渉) 多エージェント版
# TS/UCB と同じ物理モデルで、腕選択だけランダムにしたベースライン
# 戻り値: avg_sum_rate, oracle_best, regret
# ==========================================================
from __future__ import annotations
import math
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt

# ---------------------- 物理ユーティリティ（自前定義） ----------------------
def dbm_to_w(dbm: float) -> float:
    return 10.0 ** ((dbm - 30.0) / 10.0)

def db_to_linear(db: float) -> float:
    return 10.0 ** (db / 10.0)

# ----------------------------- 設定 -----------------------------
@dataclass
class Config:
    # シミュレーション基本
    R: int = 20000              # 総スロット
    Q: int = 10                 # サブバンド数（= PUリンク数）
    S: int = 10                 # SU数
    seed: int = 121
    eta: float = 0.5            # 未使用（TS/UCBとの互換用）
    print_every: int = 1000
    show_init_lines: int = 20   # 未使用
    plot: bool = False          # 通常は比較スクリプト側でまとめて描画

    # 物理パラメータ
    area_km: float = 5.0
    W_Hz: float = 10e6
    fq_GHz: float = 5.8
    c: float = 3.0e8
    xi: float = 3.0
    Gt: float = 1.0
    Gr: float = 1.0

    P_pu_tx_dbm: float = 24.0
    P_su_max_dbm: float = 30.0
    N0_dbm: float = -100.0
    gTH_pu_db: float = 30.0
    gTH_su_db: float = 5.0

    power_levels: int = 10
    use_power_penalty: bool = False   # ランダムなので未使用

# ----------------------------- 幾何/経路損失 ------------------------------
def compute_delta_for_30dB(cfg: Config) -> float:
    fq = cfg.fq_GHz * 1e9
    Ppu = dbm_to_w(cfg.P_pu_tx_dbm)
    N0  = dbm_to_w(cfg.N0_dbm)
    gamma = db_to_linear(cfg.gTH_pu_db)
    lam_term = (cfg.c / (4.0 * math.pi * fq)) ** 2
    num = Ppu * cfg.Gt * cfg.Gr * lam_term
    delta = (num / (N0 * gamma)) ** (1.0 / cfg.xi)
    return float(delta)

def place_pairs(cfg: Config,
                rng: np.random.Generator,
                n_pairs: int,
                rx_offset_mean_m: float = 100.0,
                rx_offset_std_m: float = 20.0):
    L = cfg.area_km * 1000.0
    tx = rng.random((n_pairs, 2)) * L
    off = rng.normal(loc=rx_offset_mean_m, scale=rx_offset_std_m, size=(n_pairs, 2))
    ang = rng.random(n_pairs) * 2.0 * math.pi
    rx = tx + np.stack([off[:, 0] * np.cos(ang), off[:, 1] * np.sin(ang)], axis=1)
    rx = np.clip(rx, 0.0, L)
    return tx, rx

def distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    diff = a[:, None, :] - b[None, :, :]
    return np.linalg.norm(diff, axis=2)

def path_gain(cfg: Config, d_m: np.ndarray, fq_Hz: float) -> np.ndarray:
    lam_term = (cfg.c / (4.0 * math.pi * fq_Hz)) ** 2
    with np.errstate(divide="ignore"):
        return cfg.Gt * cfg.Gr * lam_term / np.maximum(d_m, 1.0) ** cfg.xi

# ----------------------------- SINR / レート ------------------------------
def achievable_rate(cfg: Config, gamma_su_lin: float) -> float:
    return cfg.W_Hz * math.log2(1.0 + gamma_su_lin)

# ----------------------------- メイン処理（多エージェント Random） ------------------------------
def run_sim(cfg: Config, verbose: bool = True):
    """
    多エージェント版 Random:
      - SU ω = 0..S-1 が全員「残余電力以下のレベルから一様ランダムに選ぶ」
      - 学習・更新なし（T, μ は使わない）
      - 物理モデルと QoS 条件、oracle の定義は TS/UCB と同じ
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

    # --- 配置 ---
    delta_m = compute_delta_for_30dB(cfg)
    if verbose:
        print(f"[RAND] delta (30 dB) ≈ {delta_m:.2f} m")

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

    regret = np.zeros(R)
    total_sum_rate = 0.0
    Ppu_W = dbm_to_w(cfg.P_pu_tx_dbm)

    # --- Oracle（TS/UCBと同じ）---
    oracle_best = 0.0
    for w in range(S):
        for q in range(Q):
            best = 0.0
            for p in P_set:
                I_su = Ppu_W * g_pu_surx[q, w]
                gamma_su = (p * g_su_su[w]) / (N0_W + I_su)

                I_pu = p * g_su_purx[w, q]
                gamma_pu = (Ppu_W * g_pu_self[q]) / (N0_W + I_pu)

                if (gamma_su >= gTH_su) and (gamma_pu >= gTH_pu):
                    best = max(best, achievable_rate(cfg, gamma_su))
            oracle_best += best

    if verbose:
        print(f"[RAND] Approx. oracle best sum rate: {oracle_best/1e6:.3f} Mbit/slot")

    # --- メインループ ---
    if verbose:
        print("[RAND] Multi-agent Random simulation start...")

    for t in range(R):
        p_residual = np.full(S, Pmax_W)
        p_su = np.zeros((S, Q))
        chosen = -np.ones((S, Q), dtype=int)
        slot_rate = 0.0

        for q in range(Q):
            # 各 SU が「残余電力以下のレベルからランダムに選択」
            for w in range(S):
                if p_residual[w] <= 0:
                    continue

                feas = np.where(P_set <= p_residual[w])[0]
                if feas.size == 0:
                    continue

                l_star = int(rng.choice(feas))
                p = float(P_set[l_star])
                p_su[w, q] = p
                chosen[w, q] = l_star

            # PU_q の SINR
            I_pu_q = float(np.sum(p_su[:, q] * g_su_purx[:, q]))
            gamma_pu_q = (Ppu_W * g_pu_self[q]) / (N0_W + I_pu_q)

            # 各 SU の SINR & レート
            for w in range(S):
                l = chosen[w, q]
                if l < 0:
                    continue

                p = p_su[w, q]
                I_su_wq = float(np.sum(p_su[:, q] * g_su_su_x[:, w]) - p * g_su_su_x[w, w])
                I_pu_to_su = Ppu_W * g_pu_surx[q, w]

                gamma_su_wq = (p * g_su_su[w]) / (N0_W + I_su_wq + I_pu_to_su)
                success = (gamma_pu_q >= gTH_pu) and (gamma_su_wq >= gTH_su)

                if success:
                    r = achievable_rate(cfg, gamma_su_wq)
                    slot_rate += r
                    total_sum_rate += r

                # ランダム方策では T, μ の更新は無し
                p_residual[w] = max(p_residual[w] - p, 0.0)

        # regret 更新
        if t == 0:
            regret[t] = oracle_best - slot_rate
        else:
            regret[t] = regret[t-1] + (oracle_best - slot_rate)

        if verbose and (t % cfg.print_every == 0):
            print(f"[RAND] t={t:5d} sum={slot_rate/1e6:.3f} Mbit  "
                  f"norm={slot_rate/cfg.W_Hz:.3f}  regret={regret[t]/1e6:.3f}")

    avg_sum_rate = total_sum_rate / R

    if verbose:
        print("[RAND] Simulation done.")
        print(f"平均サムレート = {avg_sum_rate/1e6:.3f} Mbit/slot")
        print(f"正規化 = {avg_sum_rate/cfg.W_Hz:.3f} bps/Hz")

    # 通常は比較スクリプト側でまとめて描画するのでここでは plot はしない
    if cfg.plot:
        plt.figure()
        plt.plot(regret/1e6)
        plt.xlabel("time slot")
        plt.ylabel("regret [Mbit]")
        plt.title("Random policy regret")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return avg_sum_rate, oracle_best, regret
