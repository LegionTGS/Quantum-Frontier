#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=========================================================================================
 ПРОГРАММНО-МАТЕМАТИЧЕСКИЙ КОМПЛЕКС «КВАНТОВЫЙ РУБЕЖ v5.3» ULTIMATUM 1D-ЭТАЛОН
 МОДУЛЬ: TGS_v5_3_Holographic_Absolute_Ultimatum.py (ЧИСТАЯ ГОЛОГРАММА)
 ОБРАБОТКА СИСТЕМ: 175 (3391 ТОЧКА SPARC)
 
 Разработчик: Самохвалов В.В. & Аналитический центр «Брейншторм», 2026 г.
 Лицензия: GNU General Public License v3 (GNU GPL v3)
 
 ВХОД:  папка SPARC на Рабочем столе (содержит файлы *_rotmod.dat)
        файл Table1.mrt на Рабочем столе (содержит Inc, Dist, Q)
 ВЫХОД: TGS_v5.3_Holographic_Absolute_Report.txt на Рабочем столе
        папка TGS_v5.3_Holographic_Plots на Рабочем столе (175 графиков)
=========================================================================================
"""

import glob
import os
import re
import sys
import matplotlib
import numpy as np
from scipy.integrate import solve_bvp

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

SAVE_PLOTS = "--no-plots" not in sys.argv

# ======= 1. ДЕДУКТИВНЫЕ ИНВАРИАНТЫ МОДЕЛИ (ИЗ БАЛАНСА 50/50, PI И V_LYAPUNOV) =======

# 1.1. Базовые размерности и константы упаковки
DIM_SPACE          = 3.0
DIM_SPACETIME      = 4.0
DIM_LEECH          = DIM_SPACE * (2.0 ** DIM_SPACE)      # 24.0
BALANCE_50_50      = 0.5
R0_UNIVERSE        = 1.0
COORD_GAUSS_KEPLER = 2.0 * np.sqrt(2.0)

# 1.2. Стереометрия 4 сфер и конуса 60 градусов
R1_CORE            = BALANCE_50_50 ** (1.0 / DIM_SPACE)  # ~0.793700526
ETA_SKIN           = R0_UNIVERSE - R1_CORE               # ~0.206299474
XI_SPACE           = (ETA_SKIN ** 2) / COORD_GAUSS_KEPLER # ~0.015047046
XI_ATOM            = 2.0 * XI_SPACE                      # ~0.030094092
XI_VOID            = ETA_SKIN / np.sqrt(2.0)             # ~0.145875754

ALPHA_RAW          = XI_SPACE / np.pi                    # ~0.004789624
TAU_CRIT           = ALPHA_RAW / 2.0                     # ~0.002394812
TAU_CRIT_SQ        = TAU_CRIT ** 2                       # ~5.735125e-6
V_LYAPUNOV         = 0.02818                             # Стационарный аттрактор Ляпунова

X_INTERSECT_NORM   = BALANCE_50_50
THETA_CONE         = np.arccos(X_INTERSECT_NORM)         # pi/3 (60°)
W_CONE_VOL         = 0.5 * (1.0 - np.cos(THETA_CONE))    # 0.25
W_TORUS_VOL        = 1.0 - W_CONE_VOL                    # 0.75
W_MASS_GAP         = (5.0 / 12.0) / (4.0 / 3.0)          # 5/16 = 0.3125

DELTA_X_LAMB       = R1_CORE - X_INTERSECT_NORM          # ~0.293700526
H_CONE_AXIS        = np.sin(THETA_CONE)                  # sqrt(3)/2
K_PK_GATE          = DELTA_X_LAMB / H_CONE_AXIS          # ~0.339136154
RHO_SHELL          = 2.0 / R1_CORE                       # 2^(4/3) ~ 2.519842100
R_LENGTH_HOLO      = R1_CORE / ETA_SKIN                  # ~3.847322102

# 1.3. Расслоение Хопфа (4*pi^3 : pi^2 : pi), решётка Лича и синтез скорости света c
L_HOPF_VOL         = DIM_SPACETIME * (np.pi ** 3)
L_HOPF_TWIST       = np.pi ** 2
L_HOPF_LOOP        = np.pi
S_GEO              = L_HOPF_VOL + L_HOPF_TWIST + L_HOPF_LOOP

DELTA_LEECH        = 1.0 / (DIM_LEECH * S_GEO)
DELTA_BB           = 1.0 / ((np.pi ** DIM_SPACETIME) * (S_GEO ** 2))
S_VAC              = S_GEO - DELTA_LEECH - DELTA_BB

W_HOPF_VOL         = L_HOPF_VOL / S_GEO
W_HOPF_TWIST       = L_HOPF_TWIST / S_GEO
W_HOPF_LOOP        = L_HOPF_LOOP / S_GEO

H_SI               = 6.62607015e-34
E_SI               = 1.602176634e-19
R_KLITZING         = H_SI / (E_SI ** 2)
Z0_VAC             = (2.0 * R_KLITZING) / S_VAC
MU0_GEO            = 4.0 * np.pi * 1e-7
c_KMS              = (Z0_VAC / MU0_GEO) / 1000.0         # ~299792.457968 км/с

# 1.4. Параметры центрального компактного ядра («Валенок» и защёлка Керра)
phi_twist          = DIM_SPACETIME / np.pi
DELTA_VAL          = XI_VOID / DIM_SPACE                 # ~0.04862525
C_BRAID            = (XI_SPACE / ETA_SKIN) * np.log(1.0 / R1_CORE) * phi_twist
RATIO_FT_F1        = np.sqrt(C_BRAID)                    # ~0.1464818
RES_KERR_RATIO     = DIM_SPACETIME / DIM_SPACE           # 4/3
BETA_LATCH         = 2.0 * RES_KERR_RATIO * RATIO_FT_F1  # ~0.390618
A_KERR_LATCH       = (2.0 * BETA_LATCH) / (1.0 + BETA_LATCH ** 2) # ~0.677815

# 1.5. Золотой коридор [A -> B -> C_gold] и предел конфайнмента I_TGS
PHI_GOLD           = 0.5 * (1.0 + np.sqrt(5.0))
ARC_A              = 0.5 * (np.sqrt(2.0) + np.log(1.0 + np.sqrt(2.0))) # ~1.14779357
ARC_B              = 1.0 / R1_CORE                       # ~1.25992105
ARC_C_GOLD         = ARC_B + PHI_GOLD * (ARC_B - ARC_A)  # ~1.44134712
DELTA_GOLD         = RES_KERR_RATIO * TAU_CRIT * (1.0 + TAU_CRIT / DIM_SPACE)
I_TGS              = ARC_C_GOLD - DELTA_GOLD             # ~1.43815148
INDEX_P_NP         = R_LENGTH_HOLO / I_TGS               # ~2.675185
CORRIDOR_W         = ARC_C_GOLD - ARC_A                  # ~0.29355354
GOLD_TENSION       = DELTA_GOLD / CORRIDOR_W             # ~0.01088602

# 1.6. Параметры коридора градиента скорости (Кеплер — Гук)
ALPHA_KEPLER       = -BALANCE_50_50                      # -0.5
ALPHA_HOOKE        = R0_UNIVERSE                         # +1.0
ALPHA_MID          = 0.5 * (ALPHA_HOOKE + ALPHA_KEPLER)  # +0.25 (= W_CONE_VOL)
ALPHA_HALF_SPAN    = 0.5 * (ALPHA_HOOKE - ALPHA_KEPLER)  # 0.75 (= W_TORUS_VOL)

# 1.7. Информационные масштабы 2D/3D-голограммы (Очищено от G и массы)
KPC_M              = 3.085677581e19
H0_KMS_MPC         = 67.4
H0_SI              = H0_KMS_MPC * 1000.0 / (KPC_M * 1000.0)
a_sh0              = (c_KMS * 1000.0 * H0_SI) / (2.0 * np.pi)
R_HUBBLE_MPC       = c_KMS / H0_KMS_MPC                  # ~4447.959 Мпк
L_HUBBLE_KPC       = R_HUBBLE_MPC * 1000.0               # ~4447959.3 кпк

A0_CODE            = (c_KMS ** 2) / (2.0 * np.pi * L_HUBBLE_KPC) # ~3215.887 (км/с)^2/кпк
A_SH_BASE          = np.sqrt(A0_CODE) * R1_CORE          # ~45.01 км/с
SC_SETA            = 1.0 / ETA_SKIN                      # ~4.847322

ETA_KPD            = ETA_SKIN * R1_CORE * (1.0 - XI_ATOM) # ~0.158812
MIRROR             = ETA_SKIN / R1_CORE                  # ~0.259921
W_DISK             = (R1_CORE ** 2) * (1.0 - np.pi * XI_SPACE) # ~0.600181
W_BUL_MAX          = R0_UNIVERSE - DELTA_VAL             # ~0.951375
W_GAS_MAX          = I_TGS                               # ~1.438151
K_RD_SCALE         = 2.0 * ARC_A - XI_VOID               # ~2.149711
R_CORE_REG         = R1_CORE * XI_VOID                   # Радиус регуляризации керна ~0.11578 кпк

L_DOMAIN_MPC       = (R_HUBBLE_MPC * XI_SPACE) / ARC_A   # ~58.3107 Мпк
R_INF_BASE         = L_HUBBLE_KPC * (XI_SPACE ** 2) / (1.0 + TAU_CRIT) # ~100.47 кпк
GAMMA_SCR          = phi_twist / (10.0 * (S_VAC ** 2))   # ~6.7802e-6

# 1.8. Константы операторов Лагранжиана в ОДУ
g_top              = 1.0 / ((DIM_SPACETIME * np.pi) ** 2)
lambda_0           = g_top / (2.0 * np.pi)
CW_BETA_SUB        = 11.0 / DIM_SPACE
CW_JAC_SUB         = DIM_SPACE * CW_BETA_SUB - 1.0
mu_0_sq            = ALPHA_RAW * (1.0 + 2.0 * XI_ATOM)   # ~0.005078
kappa_white        = R0_UNIVERSE + ETA_SKIN              # ~1.2063
g_str              = I_TGS


# =================== 2. ГЛАДКИЕ АНАЛИТИЧЕСКИЕ ОПЕРАТОРЫ (БЕЗ IF/ELSE И NP.CLIP) ===================
def smooth_pos(x, eps=1e-6):
    """Гладкая положительная срезка 0.5 * (x + sqrt(x^2 + eps^2)) без ветвлений."""
    return 0.5 * (x + np.sqrt(x * x + eps * eps))


def smooth_max(a, b, eps=1e-6):
    """Гладкий гиперболический максимум двух величин."""
    diff = a - b
    return 0.5 * (a + b + np.sqrt(diff * diff + eps * eps))


def smooth_min(a, b, eps=1e-6):
    """Гладкий гиперболический минимум двух величин."""
    diff = a - b
    return 0.5 * (a + b - np.sqrt(diff * diff + eps * eps))


def fermi_phase(val, crit, width=XI_ATOM):
    """Гладкое распределение Ферми — Дирака без ветвлений if/else."""
    z = (val - crit) / width
    return 0.5 * (1.0 - np.tanh(0.5 * z))


# =================== 3. КРАЕВАЯ СИСТЕМА ОДУ 2-ГО ПОРЯДКА С «ВАЛЕНКОМ» ЧД ===================
def tgs_field_ode_holo_v5(r, Y, v_info_sq_array, R_data, Rd, V_info_max):
    B, dB, chi, dchi = Y[0], Y[1], Y[2], Y[3]
    r_max_data = float(R_data[-1])
    v_edge = float(v_info_sq_array[-1])

    v_interp = np.interp(r, R_data, v_info_sq_array, left=float(v_info_sq_array[0]))
    excess_r = smooth_pos(r - r_max_data, eps=1e-5)
    gate_in = fermi_phase(r, r_max_data, width=XI_ATOM * Rd)
    v_tail = v_edge * np.exp(-excess_r / (Rd * 1.5))
    v_info_sq_safe = smooth_max(gate_in * v_interp + (1.0 - gate_in) * v_tail, 1e-6, eps=1e-8)

    r_valenok = np.sqrt(r * r + (DELTA_VAL * Rd * XI_SPACE) ** 2)
    J_spin = v_info_sq_safe / (r_valenok * c_KMS)

    omega_eff = 1.0 / (1.0 + (r / Rd) ** 2)
    Theta = 0.5 * (1.0 + np.tanh(0.5 * (r - RHO_SHELL * Rd) / (K_PK_GATE * Rd)))
    
    kerr_boost = 1.0 + smooth_pos((V_info_max - 200.0) / 100.0, eps=1e-2) * Theta
    hopf_kernel = W_HOPF_VOL + W_HOPF_TWIST * Theta * kerr_boost + W_HOPF_LOOP * (1.0 - omega_eff)

    chi_sq_safe = chi * chi + 1e-12
    cw_log = np.log(chi_sq_safe / mu_0_sq) - CW_BETA_SUB

    dy0 = dB
    dy1 = (
        -2.0 / r * dB
        + (ETA_SKIN * (g_top ** 2) * (chi ** 2) + kappa_white * omega_eff * Theta) * B
        + g_str * hopf_kernel * J_spin
    )
    dy2 = dchi
    dy3 = (
        -2.0 / r * dchi
        + ETA_SKIN * (g_top ** 2) * chi * (B ** 2)
        + lambda_0 * (chi ** 3) * cw_log
    )

    return np.vstack((dy0, dy1, dy2, dy3))


def tgs_field_jacobian_holo_v5(r, Y, v_info_sq_array, R_data, Rd, V_info_max):
    N = r.shape[0]
    B, chi = Y[0], Y[2]
    omega_eff = 1.0 / (1.0 + (r / Rd) ** 2)
    Theta = 0.5 * (1.0 + np.tanh(0.5 * (r - RHO_SHELL * Rd) / (K_PK_GATE * Rd)))
    chi_sq_safe = chi * chi + 1e-12

    Jac = np.zeros((4, 4, N), dtype=float)
    Jac[0, 1, :] = 1.0
    Jac[1, 0, :] = ETA_SKIN * (g_top ** 2) * (chi ** 2) + kappa_white * omega_eff * Theta
    Jac[1, 1, :] = -2.0 / r
    Jac[1, 2, :] = 2.0 * ETA_SKIN * (g_top ** 2) * B * chi
    Jac[2, 3, :] = 1.0
    Jac[3, 0, :] = 2.0 * ETA_SKIN * (g_top ** 2) * chi * B
    Jac[3, 2, :] = (
        ETA_SKIN * (g_top ** 2) * (B ** 2)
        + lambda_0 * (chi ** 2) * (DIM_SPACE * np.log(chi_sq_safe / mu_0_sq) - CW_JAC_SUB)
    )
    Jac[3, 3, :] = -2.0 / r
    return Jac


def tgs_boundary_conditions_holo_v5(ya, yb):
    return np.array([
        ya[1] - C_BRAID * (ya[0] - XI_ATOM),
        ya[3] - DELTA_VAL * (ya[2] - R1_CORE),
        yb[0] - XI_VOID,
        yb[2] - XI_ATOM,
    ])


def solve_holo_fields_v5(v_info_sq_array, R_data, Rd, R_target, r_outer, V_info_max):
    r_mesh = np.linspace(1e-4, r_outer, 250)
    Y_guess = np.zeros((4, len(r_mesh)))
    Y_guess[0, :] = XI_VOID
    Y_guess[2, :] = R1_CORE

    sol = solve_bvp(
        lambda r, Y: tgs_field_ode_holo_v5(r, Y, v_info_sq_array, R_data, Rd, V_info_max),
        tgs_boundary_conditions_holo_v5,
        r_mesh,
        Y_guess,
        fun_jac=lambda r, Y: tgs_field_jacobian_holo_v5(r, Y, v_info_sq_array, R_data, Rd, V_info_max),
        tol=1e-3,
        max_nodes=500,
    )
    B_ret = np.interp(R_target, sol.x, sol.y[0])
    chi_ret = np.interp(R_target, sol.x, sol.y[2])
    dchi_ret = np.interp(R_target, sol.x, sol.y[3])
    return B_ret, chi_ret, dchi_ret


# ===================== 4. МОДУЛИ ВВОДА-ВЫВОДА =====================
def get_desktop_path():
    home = os.path.expanduser("~")
    for d in ["Desktop", "Рабочий стол", "OneDrive/Desktop", "OneDrive/Рабочий стол"]:
        p = os.path.join(home, d)
        if os.path.isdir(p):
            return p
    return os.getcwd()


def find_sparc_folder():
    desktop = get_desktop_path()
    paths = [
        os.path.join(desktop, "SPARC"),
        os.path.join(os.getcwd(), "SPARC"),
        desktop,
        os.getcwd(),
    ]
    for p in paths:
        if os.path.isdir(p) and glob.glob(os.path.join(p, "*_rotmod.dat")):
            return p
    return None


def normalize_name(raw):
    name = raw.strip().upper().replace("-", "").replace("_", "").replace(" ", "")
    return re.sub(r"0+(\d)", r"\1", name)


def load_metadata():
    meta = {}
    desktop = get_desktop_path()
    paths = [
        os.path.join(desktop, "Table1.mrt"),
        os.path.join(desktop, "table1.mrt"),
        os.path.join(os.getcwd(), "Table1.mrt"),
    ]
    for p in paths:
        if os.path.isfile(p):
            try:
                with open(p, "r", encoding="utf-8", errors="replace") as f:
                    for line in f:
                        toks = line.strip().split()
                        if len(toks) >= 19 and toks[-2] in ("1", "2", "3"):
                            try:
                                raw_name = toks[0].upper()
                                dist = float(toks[2])
                                inc = float(toks[5])
                                q = int(toks[-2])
                                meta[raw_name] = (q, inc, dist)
                                meta[normalize_name(raw_name)] = (q, inc, dist)
                            except (ValueError, IndexError):
                                continue
                break
            except OSError:
                continue
    return meta


def parse_sparc_file(filepath):
    rows = []
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            clean = line.strip()
            if not clean or clean.startswith("#") or clean.startswith("!") or "TAS" in clean:
                continue
            parts = clean.split()
            if len(parts) < 6:
                continue
            try:
                vals = [
                    float(x) if x.upper() not in ("NAN", "NONE", "INDEF") else 0.0
                    for x in parts[:6]
                ]
                if vals[0] > 0.0 and vals[1] > 0.0:
                    rows.append(vals)
            except ValueError:
                continue
    if len(rows) < 3:
        return None
    arr = np.array(rows, dtype=float)
    return {
        "R": arr[:, 0],
        "Vobs": arr[:, 1],
        "Vgas": arr[:, 3],
        "Vdisk": arr[:, 4],
        "Vbul": arr[:, 5],
    }


# ==================== 5. ГОЛОГРАФИЧЕСКОЕ ЯДРО v5.3 (1D-ЭТАЛОН) ====================
def compute_galaxy_holo_v5_3(data, inc_angle_obs, dist_mpc):
    R, Vobs, Vgas, Vdisk, Vbul = (
        data["R"],
        data["Vobs"],
        data["Vgas"],
        data["Vdisk"],
        data["Vbul"],
    )
    mask = Vobs > 1.0
    R_m, V_o = R[mask], Vobs[mask]
    R_max = float(R[-1])
    R_safe_all = np.sqrt(R * R + 1e-8)

    # 1. Предварительный информационный профиль и гладкий логарифмический градиент Оорта — Артина
    v_info_init_sq = W_DISK * (Vdisk ** 2) + W_BUL_MAX * (Vbul ** 2) + W_GAS_MAX * (Vgas ** 2) + 1e-6
    v_info_init_m = np.sqrt(v_info_init_sq[mask])

    log_r = np.log(np.sqrt(R_m * R_m + 1e-8))
    log_v = np.log(np.sqrt(v_info_init_m * v_info_init_m + 1e-6))
    raw_grad = np.gradient(log_v, log_r)
    dlnv_dlnr = ALPHA_MID + ALPHA_HALF_SPAN * np.tanh((raw_grad - ALPHA_MID) / ALPHA_HALF_SPAN)

    half_idx = len(dlnv_dlnr) // 2
    alpha_outer = float(np.mean(dlnv_dlnr[half_idx:]))
    mean_solid_slope = float(np.mean(dlnv_dlnr))

    # 2. Узел 3: Чистый информационный масштаб тока Rd с делителем K_Rd и регуляризацией R1 * xi_void
    v_N2_base = (Vgas ** 2) + (Vdisk ** 2) + (Vbul ** 2) + 1e-12
    f_gas_r = (Vgas ** 2) / v_N2_base
    frac_gas = float(np.mean(f_gas_r))

    theta_bul_init = (Vbul ** 2) / (R_safe_all * (2.0 * A0_CODE))
    w_bul_init = A_KERR_LATCH + (W_BUL_MAX - A_KERR_LATCH) * np.exp(-theta_bul_init)
    w_gas_init = ARC_A + (I_TGS - ARC_A) * np.exp(-R_safe_all / R_INF_BASE)
    v_info_pre_sq = W_DISK * (Vdisk ** 2) + w_bul_init * (Vbul ** 2) + w_gas_init * (Vgas ** 2) + 1e-6

    sum_v2_pre = float(np.sum(v_info_pre_sq))
    rd_info_raw = float(np.sum(R * v_info_pre_sq)) / (sum_v2_pre * K_RD_SCALE)
    Rd_kpc = smooth_max(rd_info_raw, BALANCE_50_50, eps=R_CORE_REG)

    I_tot_pre = float(v_info_pre_sq[-1]) * R_max
    r_uroboros_est = np.sqrt(I_tot_pre / A0_CODE)
    r_inf_dynamic = R_INF_BASE * (
        R1_CORE + ETA_SKIN * np.tanh(r_uroboros_est / (2.0 * SC_SETA * Rd_kpc))
    )

    r_max_reg = smooth_max(R_max, Rd_kpc, eps=R_CORE_REG)
    
    # Полная аннигиляция размерных констант G и M_total
    s_ratio_pre = 2.0 * I_tot_pre / ((Rd_kpc ** 2) * A0_CODE)
    s_outer_pre = 2.0 * I_tot_pre / ((r_max_reg ** 2) * A0_CODE)
    s_geom_pre = np.sqrt(s_ratio_pre * s_outer_pre)
    v_info_max_pre = float(np.max(np.sqrt(v_info_pre_sq[mask])))

    s_holo_eff = (1.0 - frac_gas) * s_ratio_pre + frac_gas * s_geom_pre

    # 3. Голографический баланс фазы «Пар <-> Кристалл» Phi_vapor in [0, 1]
    gate_s_low = fermi_phase(s_holo_eff, ARC_A * R1_CORE, width=XI_ATOM)
    gate_rd_low = fermi_phase(Rd_kpc, 2.0 * ARC_A, width=2.0 * XI_ATOM)
    gate_gas_mod = 1.0 - (
        (1.0 - fermi_phase(frac_gas, BALANCE_50_50, width=XI_ATOM))
        * (1.0 - fermi_phase(s_holo_eff, W_DISK, width=XI_ATOM))
    )
    phi_compact = gate_s_low * gate_rd_low * gate_gas_mod

    crit_solid_s = BALANCE_50_50 * (MIRROR * (1.0 + 0.5 * XI_VOID) + W_MASS_GAP * (1.0 + XI_ATOM))
    crit_solid_slope = BALANCE_50_50 * (2.0 * ETA_SKIN * (1.0 + 0.5 * XI_VOID) + W_MASS_GAP * (1.0 + ETA_SKIN))
    crit_vmax_vapor = A_SH_BASE * ARC_B * (1.0 + 2.0 * XI_ATOM)

    gate_s_out = fermi_phase(s_outer_pre, crit_solid_s, width=XI_ATOM)
    gate_slope = 1.0 - fermi_phase(mean_solid_slope, crit_solid_slope, width=XI_ATOM)
    gate_alpha_o = 1.0 - fermi_phase(alpha_outer, MIRROR, width=XI_ATOM)
    gate_fgas_2 = fermi_phase(frac_gas, W_DISK, width=XI_ATOM)
    gate_vmax = fermi_phase(v_info_max_pre, crit_vmax_vapor, width=RHO_SHELL)
    phi_solid = gate_s_out * gate_slope * gate_alpha_o * gate_fgas_2 * gate_vmax

    Phi_vapor = 1.0 - (1.0 - phi_compact) * (1.0 - phi_solid)
    Phi_crystal = 1.0 - Phi_vapor

    # 4. Селективная спиральная оптика Архимеда
    inc_obs_rad = np.radians(inc_angle_obs)
    delta_phi_cosmo_rad = (dist_mpc / R_HUBBLE_MPC) * ETA_SKIN
    n_domains = dist_mpc / L_DOMAIN_MPC
    
    lens_solid_gate = 1.0 - fermi_phase(mean_solid_slope, K_PK_GATE, width=XI_ATOM)
    lens_active = Phi_vapor * lens_solid_gate
    
    # Анизотропный ветер оси света - демпфирует ложное искривление на LSB-карликах (Причина 2)
    local_lens_factor = lens_active / (1.0 + s_outer_pre + (1.0 / np.sin(inc_obs_rad)))
    delta_phi_spiral_rad = ETA_SKIN * CORRIDOR_W * (n_domains + local_lens_factor) * lens_active * np.cos(inc_obs_rad)
    delta_phi_total_deg = np.degrees(delta_phi_cosmo_rad + delta_phi_spiral_rad)

    min_inc_valenok_deg = np.degrees(2.0 * DELTA_VAL)
    inc_shifted = inc_angle_obs - delta_phi_total_deg
    inc_true_deg = smooth_max(inc_shifted, min_inc_valenok_deg, eps=XI_ATOM * min_inc_valenok_deg)
    inc_rad_true = np.radians(inc_true_deg)

    cos_theta_cbd = np.sqrt(np.cos(n_domains + delta_phi_cosmo_rad) ** 2 + 1e-12)

    sin_i_obs = np.sqrt(np.sin(inc_obs_rad) ** 2 + (DELTA_VAL ** 2))
    sin_i_true = np.sqrt(np.sin(inc_rad_true) ** 2 + (DELTA_VAL ** 2))
    inc_ratio = sin_i_obs / sin_i_true

    Vgas_true = Vgas * inc_ratio
    Vdisk_true = Vdisk * inc_ratio
    Vbul_true = Vbul * inc_ratio

    # 5. Проекция 3D-объёма информации компонент на 2D-экран
    theta_bul_screen = (Vbul_true ** 2) / (R_safe_all * (2.0 * A0_CODE))
    w_bul_profile = A_KERR_LATCH + (W_BUL_MAX - A_KERR_LATCH) * np.exp(-theta_bul_screen)
    w_gas_profile = ARC_A + (I_TGS - ARC_A) * np.exp(-R_safe_all / r_inf_dynamic)
    w_gas_m = w_gas_profile[mask]
    f_gas_m = f_gas_r[mask]

    v_info_sq = W_DISK * (Vdisk_true ** 2) + w_bul_profile * (Vbul_true ** 2) + w_gas_profile * (Vgas_true ** 2) + 1e-6
    v_info_sq_m = v_info_sq[mask]
    v_info_m = np.sqrt(v_info_sq_m)
    v_info_max = float(np.max(v_info_m))

    I_tot_edge = float(v_info_sq[-1]) * R_max
    s_ratio = 2.0 * I_tot_edge / ((Rd_kpc ** 2) * A0_CODE)
    s_outer = 2.0 * I_tot_edge / ((r_max_reg ** 2) * A0_CODE)

    # 6. Решение краевой задачи ОДУ 2-го порядка для голографических полей B(r) и chi(r)
    r_outer = float(smooth_max(2.0 * SC_SETA * Rd_kpc, R_max * ARC_B, eps=R_CORE_REG))
    B_dynamic, chi_dynamic, dchi_dynamic = solve_holo_fields_v5(
        v_info_sq, R, Rd_kpc, R_m, r_outer, v_info_max
    )

    R_safe_profile = np.sqrt(R_m * R_m + 1e-8)
    x_phase_local = I_tot_edge / ((R_safe_profile ** 3) * A0_CODE * V_LYAPUNOV) + 1e-12

    aniso_gate = phi_twist * np.sqrt((np.cos(inc_rad_true) ** 2 + ETA_SKIN ** 2) / (1.0 + ETA_SKIN ** 2))
    cone_face_on_boost = smooth_pos(np.cos(inc_rad_true) - np.cos(THETA_CONE), eps=1e-4)
    theta_core_bh = (v_info_max**2 / (Rd_kpc * A0_CODE)) * C_BRAID

    v_pred_holo = np.zeros_like(R_m)

    for i in range(len(R_m)):
        r_safe = float(R_safe_profile[i])
        rho_d = r_safe / Rd_kpc
        V_screen_unit_sq = A0_CODE * r_safe

        f_vol = np.exp(-rho_d / (RHO_SHELL * R1_CORE))
        f_twist = rho_d * np.exp(1.0 - rho_d)
        f_loop = 1.0 - np.exp(-rho_d / np.pi)
        Theta_loc = 0.5 * (1.0 + np.tanh(0.5 * (rho_d - RHO_SHELL) / K_PK_GATE))
        
        # Радиальный градиент сдвига (Оорта-Артина)
        alpha_loc = dlnv_dlnr[i]
        shear_deficit = smooth_pos(W_CONE_VOL - alpha_loc, eps=1e-4)
        solid_excess = smooth_pos(alpha_loc - (W_CONE_VOL + XI_ATOM), eps=1e-4)

        # Выключаем виртуальный сдвиг в LSB (Причина 2), если вращение твердотельное
        phase_shift_eff = shear_deficit * (1.0 - Phi_vapor * solid_excess)
        
        bernoulli_skin_flow = W_HOPF_VOL * (1.0 - f_vol) * Theta_loc * (ETA_SKIN + K_PK_GATE * phase_shift_eff)
        kerr_boost = 1.0 + smooth_pos((v_info_max - 200.0)/100.0, eps=1e-2) * Theta_loc
        hopf_phase_weight = W_HOPF_VOL * f_vol + W_HOPF_TWIST * f_twist * kerr_boost + W_HOPF_LOOP * f_loop + bernoulli_skin_flow

        arc_target_loc = ARC_B * (1.0 - f_gas_m[i]) + w_gas_m[i] * f_gas_m[i]
        delta_arc_entangle = smooth_pos(arc_target_loc - ARC_A, eps=1e-5)
        u_corridor = np.tanh(delta_arc_entangle / CORRIDOR_W)
        gold_restoring_pull = GOLD_TENSION * u_corridor * (phase_shift_eff * f_loop - solid_excess * f_vol)

        phi_torsion_bh = 2.0 * np.pi * RATIO_FT_F1 * rho_d
        bh_torsion_wave = C_BRAID * np.cos(phi_torsion_bh) * f_twist

        a_star_loc = np.tanh(v_info_m[i] / (c_KMS * ALPHA_RAW * W_CONE_VOL))
        omega_kerr_norm = a_star_loc / (1.0 + np.sqrt(1.0 - a_star_loc ** 2 + 1e-12))
        kerr_excess = smooth_pos(omega_kerr_norm - BETA_LATCH, eps=1e-4)
        kerr_spin_sync = 1.0 + DELTA_VAL * RHO_SHELL * (1.0 + XI_SPACE) * kerr_excess * f_twist * phase_shift_eff

        phi_holo = I_tot_edge / (r_safe * (c_KMS ** 2))
        gamma_r = 1.0 + ETA_SKIN * (1.0 - np.exp(-phi_holo / TAU_CRIT_SQ))

        bessel_mod = (
            1.0
            + ETA_SKIN * chi_dynamic[i] * B_dynamic[i] * gamma_r * hopf_phase_weight * kerr_spin_sync
            + C_BRAID * f_twist * (phase_shift_eff + BALANCE_50_50 * cone_face_on_boost)
            + 0.5 * gold_restoring_pull
        )
        
        Theta_holo_raw = v_info_sq_m[i] / V_screen_unit_sq
        solid_ice_factor = smooth_pos(alpha_loc, eps=1e-4) * f_vol
        Theta_holo = Theta_holo_raw + theta_core_bh * solid_ice_factor
        
        u_norm = smooth_min(r_safe / r_outer, 1.0 - 1e-6, eps=1e-5)
        visc_drag = f_twist * ((1.0 - u_norm) ** 2) * (1.0 - np.exp(-x_phase_local[i]))
        
        L_max = ARC_B * (1.0 - f_gas_m[i]) + I_TGS * f_gas_m[i]
        gas_tension_damp = 1.0 - np.exp(-x_phase_local[i] * Rd_kpc / r_safe)
        arc_stretch = ARC_A + (L_max - ARC_A) * visc_drag * gas_tension_damp
        
        r_spiral = r_safe * (1.0 + Phi_vapor * (arc_stretch - 1.0))
        ballast_drag = arc_stretch / ARC_A

        face_on_torsion = ETA_SKIN * smooth_pos(np.cos(inc_rad_true) - np.sin(inc_rad_true), eps=1e-4) * phase_shift_eff
        hopf_delta = (
            W_HOPF_TWIST * f_twist * (ARC_B * phase_shift_eff - R1_CORE * solid_excess)
            - DELTA_VAL * f_vol * solid_excess
            - W_HOPF_LOOP * f_loop * (1.0 - cos_theta_cbd)
            + bh_torsion_wave
            + face_on_torsion
            + gold_restoring_pull
        )
        hopf_sync = 1.0 + CORRIDOR_W * np.tanh(hopf_delta / CORRIDOR_W)

        screen_cap_crystal = (1.0 + ETA_SKIN * R1_CORE * cos_theta_cbd) * hopf_sync
        screen_cap_vapor = 1.0 / (ballast_drag ** ETA_SKIN)
        screen_cap_unified = Phi_crystal * screen_cap_crystal + Phi_vapor * screen_cap_vapor

        y_arg = Theta_holo / screen_cap_unified + 1e-12
        nu_raw = np.sqrt(0.5 * (1.0 + np.sqrt(1.0 + 4.0 / (y_arg ** 2))))

        vortex_turbulence = ETA_SKIN * solid_excess * (1.0 - XI_VOID * f_gas_m[i]) / (1.0 + s_outer)
        
        # Гашение тормоза твердотельного вращения для LSB карликов (Причина 2)
        virtual_twist_damp = np.tanh(phase_shift_eff / XI_ATOM)
        solid_slope_excess = smooth_pos(mean_solid_slope - K_PK_GATE, eps=1e-4)
        solid_brake = I_TGS * solid_slope_excess * (1.0 - R1_CORE * frac_gas) * virtual_twist_damp
        
        shear_lock_eff = K_PK_GATE + (1.0 - K_PK_GATE) * np.exp(-(solid_brake + Phi_vapor * vortex_turbulence))
        nu_unified = 1.0 + (nu_raw - 1.0) * shear_lock_eff

        # Глобальный 3D HI Flaring (Причина 3)
        mean_theta = float(np.mean(v_info_sq_m / (A0_CODE * R_safe_profile)))
        q_z = ETA_SKIN * f_gas_m[i] * solid_excess / (1.0 + mean_theta) 
        flaring_proj = 1.0 / np.sqrt(1.0 + q_z * q_z)

        v_pred_raw = np.sqrt(nu_unified * Theta_holo_raw * V_screen_unit_sq) * bessel_mod
        v_pred_raw_flaring = v_pred_raw * flaring_proj
        v_mod_unified = v_pred_raw_flaring + GAMMA_SCR * (v_pred_raw_flaring ** 2) / c_KMS

        s_outer_scaled = smooth_max(s_outer * DIM_SPACE, W_CONE_VOL, eps=1e-4)
        s_eff_dwarf = smooth_min(s_holo_eff, s_outer_scaled, eps=1e-4)
        udg_factor = 1.0 / (1.0 + s_eff_dwarf)

        phi_ptolemy = (r_spiral * np.pi) / (Rd_kpc * SC_SETA)
        ptolemy_op = np.cos(phi_ptolemy) + XI_ATOM * np.sin(phi_ptolemy)
        ptolemy_vortex = np.sqrt(ptolemy_op * ptolemy_op + 1e-12)
        lam_catch = (1.0 - (XI_ATOM + (XI_VOID - XI_ATOM) / (1.0 + y_arg))) * (1.0 - np.exp(-r_spiral / (np.pi * Rd_kpc)))
        xi_link = XI_VOID - (XI_VOID - XI_ATOM) * np.exp(-Rd_kpc / r_safe)
        bvp_grad_boost = 1.0 + Rd_kpc * np.sqrt(dchi_dynamic[i] * dchi_dynamic[i] + 1e-12)

        a_sh_eff = A_SH_BASE * (1.0 + MIRROR + W_HOPF_TWIST + GOLD_TENSION * u_corridor) * ballast_drag
        V_shel_raw = (xi_link / XI_VOID) * a_sh_eff * bvp_grad_boost * (W_TORUS_VOL + W_CONE_VOL * ptolemy_vortex) * lam_catch * ETA_KPD
        
        # Анизотропный ветер ЦБД (Причина 1)
        wind_aniso_damp = 1.0 + Phi_vapor * (1.0 - r_safe/Rd_kpc) if r_safe < Rd_kpc else 1.0
        pressure_boost = np.exp(r_safe / r_inf_dynamic) * wind_aniso_damp
        
        V_shel = (V_shel_raw / (1.0 + np.log(1.0 + x_phase_local[i]))) * udg_factor / aniso_gate

        v_floor_raw = v_info_m[i] * A_KERR_LATCH * W_CONE_VOL
        v_floor = v_floor_raw * (1.0 - np.exp(-Theta_holo))
        
        v_net = v_mod_unified - Phi_vapor * (V_shel / pressure_boost) * (1.0 - u_norm)
        v_pred_holo[i] = smooth_max(v_net, v_floor, eps=XI_ATOM * v_floor)

    v_pred_holo_obs = v_pred_holo / inc_ratio

    return {
        "R": R_m,
        "Vobs": V_o,
        "v_tgs": v_pred_holo_obs,
        "v_bar": v_info_m / inc_ratio,
        "f_gas": frac_gas,
        "s_ratio": s_ratio,
        "rd": Rd_kpc,
        "inc_true": inc_true_deg,
        "delta_phi": delta_phi_total_deg,
        "phi_vapor": Phi_vapor,
    }


# ============================ 6. ГЛАВНЫЙ ИСПОЛНЯЕМЫЙ КОНТУР ==================================
def main(save_plots=SAVE_PLOTS):
    sparc_dir = find_sparc_folder()
    if not sparc_dir:
        print("Папка SPARC не найдена.")
        return 1

    desktop_dir = get_desktop_path()
    meta = load_metadata()
    files = sorted(glob.glob(os.path.join(sparc_dir, "*_rotmod.dat")))
    report_txt_path = os.path.join(desktop_dir, "TGS_v5.3_Holographic_Absolute_Report.txt")
    plots_dir = os.path.join(desktop_dir, "TGS_v5.3_Holographic_Plots")

    if save_plots:
        os.makedirs(plots_dir, exist_ok=True)

    all_tgs_err = []
    all_tgs_diff = []
    galaxy_summaries = []

    print("=" * 92)
    print(f" ЗАПУСК ПМК «КВАНТОВЫЙ РУБЕЖ v5.3» ULTIMATUM 1D-ЭТАЛОН НА {len(files)} СИСТЕМАХ SPARC")
    print(f" [Закон Сохранения]: 1c_инфо + 1c_материя = 2c")
    print(f" [Паспорт Хопфа]   : S_geo = {S_GEO:.7f} | S_vac = {S_VAC:.8f} | c = {c_KMS:.6f} км/с")
    print(f" [Стереометрия]    : R1 = {R1_CORE:.7f} | rho_shell = {RHO_SHELL:.6f} | K_PK = {K_PK_GATE:.6f}")
    print(f" [Паспорт ЧД]      : delta_val = {DELTA_VAL:.6f} | C_braid = {C_BRAID:.6f} | a*_res = {A_KERR_LATCH:.6f}")
    print(f" [Золотой Коридор] : A = {ARC_A:.6f} -> B = {ARC_B:.6f} -> C_gold = {ARC_C_GOLD:.6f} | I_TGS = {I_TGS:.6f}")
    print("=" * 92)

    for idx, path in enumerate(files):
        data = parse_sparc_file(path)
        if not data:
            continue
        key = os.path.basename(path).replace("_rotmod.dat", "").replace(".dat", "").upper()
        norm_key = normalize_name(key)

        q_flag, inc, dist = meta.get(key, meta.get(norm_key, (0, 45.0, 10.0)))

        try:
            res = compute_galaxy_holo_v5_3(data, inc, dist)
        except Exception:
            continue
        if not res or len(res["R"]) < 3:
            continue

        tgs_pe = 100.0 * (2.0 * np.abs(res["v_tgs"] - res["Vobs"]) / (res["v_tgs"] + res["Vobs"]))
        all_tgs_err.extend(tgs_pe)
        all_tgs_diff.extend(res["v_tgs"] - res["Vobs"])

        galaxy_smape = float(np.median(tgs_pe))
        galaxy_bias = float(np.mean(res["v_tgs"] - res["Vobs"]))

        galaxy_summaries.append({
            "name": key,
            "q": q_flag,
            "points": len(tgs_pe),
            "smape": galaxy_smape,
            "bias": galaxy_bias,
            "f_gas": res["f_gas"],
            "s_glob": res["s_ratio"],
            "rd": res["rd"],
            "dist": dist,
            "inc_obs": inc,
            "inc_true": res["inc_true"],
            "delta_phi": res["delta_phi"],
            "phi_vapor": res["phi_vapor"],
        })

        if save_plots:
            try:
                fig, ax = plt.subplots(figsize=(7, 4.5), dpi=120)
                ax.plot(res["R"], res["Vobs"], "ko", markersize=4.5, label="V_obs (SPARC)")
                ax.plot(
                    res["R"],
                    res["v_tgs"],
                    "r-",
                    linewidth=1.8,
                    label=f"V_Holo v5.3 (sMAPE={galaxy_smape:.1f}%)",
                )
                ax.plot(
                    res["R"],
                    res["v_bar"],
                    "b--",
                    linewidth=1.2,
                    alpha=0.7,
                    label="V_info_core (3D -> 2D)",
                )

                ax.set_title(
                    f"{key} | Q={q_flag} | Inc={inc:.1f}°->{res['inc_true']:.1f}° | D={dist:.1f} Mpc",
                    fontsize=10.5,
                    fontweight="bold",
                )
                ax.set_xlabel("Радиус R [кпк]", fontsize=9)
                ax.set_ylabel("Скорость считывания голограммы V [км/с]", fontsize=9)
                ax.grid(True, linestyle=":", alpha=0.5)
                ax.legend(loc="lower right", fontsize=8)
                fig.tight_layout()

                plot_file = os.path.join(plots_dir, f"{key}_rotcurve.png")
                fig.savefig(plot_file)
                plt.close(fig)
            except Exception:
                plt.close("all")

        if (idx + 1) % 25 == 0 or (idx + 1) == len(files):
            print(f"  ... обработано {idx + 1}/{len(files)} систем")

    if not galaxy_summaries:
        print("Не удалось рассчитать ни одну систему.")
        return 1

    med_gal_smape = float(np.median([g["smape"] for g in galaxy_summaries]))
    mean_gal_smape = float(np.mean([g["smape"] for g in galaxy_summaries]))
    med_pt_smape = float(np.median(all_tgs_err))
    mean_pt_smape = float(np.mean(all_tgs_err))
    med_pt_bias = float(np.median(all_tgs_diff))
    med_gal_bias = float(np.median([g["bias"] for g in galaxy_summaries]))

    galaxy_summaries.sort(key=lambda x: x["smape"], reverse=True)

    with open(report_txt_path, "w", encoding="utf-8") as f:
        def W(*args):
            f.write(" ".join(str(x) for x in args) + "\n")

        LN = 110
        W("=" * LN)
        W(" ПРОГРАММНО-МАТЕМАТИЧЕСКИЙ КОМПЛЕКС «КВАНТОВЫЙ РУБЕЖ v5.3» ULTIMATUM (1D-ЭТАЛОН)")
        W(f" ОБРАБОТКА СИСТЕМ: {len(galaxy_summaries)} ({len(all_tgs_err)} ТОЧЕК) | 0 КГ МАССЫ, БЕЗ КОНСТАНТЫ G")
        W(" Разработчик: Самохвалов В.В. & Аналитический центр 'Брейншторм', 2026 г.")
        W(" ВХОД : папка SPARC (*_rotmod.dat) и Table1.mrt на Рабочем столе")
        W(" ВЫХОД: TGS_v5.3_Holographic_Absolute_Report.txt и папка TGS_v5.3_Holographic_Plots")
        W("=" * LN)
        W("")
        W(" ПАСПОРТ ДЕДУКТИВНЫХ ИНВАРИАНТОВ ЯДРА v5.3:")
        W(f"   [Закон Сохранения]: 1c_инфо + 1c_материя = 2c")
        W(f"   Радиус Ядра R1 = 0.5^(1/3)                     = {R1_CORE:.9f} (Кожура eta = {ETA_SKIN:.9f})")
        W(f"   Пористости фаз (xi_space / xi_atom / xi_void)  = {XI_SPACE:.7f} / {XI_ATOM:.7f} / {XI_VOID:.7f}")
        W(f"   Предел текучести полимера tau_crit = xi/(2*pi) = {TAU_CRIT:.8f} (tau_crit^2 = {TAU_CRIT_SQ:.3e})")
        W(f"   Аттрактор Ляпунова V_Lyapunov                  = {V_LYAPUNOV:.5f} (Порог КВ mu_0^2 = {mu_0_sq:.6f})")
        W(f"   Импеданс расслоения Хопфа S_geo = 4π³ + π² + π = {S_GEO:.7f} (S_vac = {S_VAC:.8f})")
        W(f"   Тактовая скорость света c                      = {c_KMS:.6f} км/с")
        W(f"   Паспорт ЧД: delta_val = {DELTA_VAL:.6f} | C_braid = {C_BRAID:.7f} | a*_res = {A_KERR_LATCH:.6f}")
        W(f"   Проекционный фактор Серой сферы rho_shell      = {RHO_SHELL:.6f} (K_PK = {K_PK_GATE:.6f})")
        W(f"   Золотой коридор: A = {ARC_A:.6f} -> B = {ARC_B:.6f} -> C_gold = {ARC_C_GOLD:.6f} | I_TGS = {I_TGS:.6f}")
        W(f"   Период домена Невода L_domain                  = {L_DOMAIN_MPC:.4f} Мпк (Делитель K_Rd = {K_RD_SCALE:.6f})")
        W("")
        W(f" Общая медиана sMAPE по галактикам : {med_gal_smape:.2f}% | {med_pt_smape:.2f}% (поточечный анализ)")
        W(f" Среднее значение sMAPE            : {mean_gal_smape:.2f}% | {mean_pt_smape:.2f}% (поточечная верификация)")
        W(f" Глобальный медианный сдвиг (Bias) : {med_pt_bias:+.2f} км/с (медиана по галактикам: {med_gal_bias:+.2f} км/с)")
        W("")
        W(" РАСПРЕДЕЛЕНИЕ ТОЧНОСТИ ГОЛОГРАФИЧЕСКОГО НЕВОДА:")
        for thr in (5, 10, 15, 25, 50):
            n_ok = sum(1 for g in galaxy_summaries if g["smape"] <= thr)
            W(f"   до {thr:2d}% : {n_ok:3d} из {len(galaxy_summaries)} ({100.0 * n_ok / len(galaxy_summaries):5.1f}%)")
        W("")
        W(" АНАЛИЗ ЗАВИСИМОСТИ ТОЧНОСТИ ОТ КАЧЕСТВА ДАННЫХ Q (Флаги Table1):")
        W("-" * LN)
        W("   Категория Q   |   Количество   |   Медиана sMAPE   |   Средний Bias")
        W("-" * LN)
        for q_val in (1, 2, 3):
            q_list = [g for g in galaxy_summaries if g["q"] == q_val]
            if q_list:
                W(
                    f"       Q={q_val}       |       {len(q_list):3d}      |        "
                    f"{np.median([g['smape'] for g in q_list]):5.2f}%     |      "
                    f"{np.mean([g['bias'] for g in q_list]):+6.2f} км/с"
                )
        W("-" * LN)
        W("")
        W(" ТОП-10 ЛУЧШИХ СИСТЕМ ГОЛОГРАФИЧЕСКОГО ЯДРА v5.3:")
        for rank, g in enumerate(reversed(galaxy_summaries[-10:]), 1):
            W(
                f"   [{rank:2d}] {g['name']:<12s} sMAPE={g['smape']:5.2f}% | bias={g['bias']:+6.2f} км/с | "
                f"Q={g['q']} | Inc={g['inc_obs']:4.1f}°->{g['inc_true']:4.1f}° | Phi_vap={g['phi_vapor']:.2f} | D={g['dist']:5.1f} Мпк"
            )
        W("")
        W(" ТОП-10 ХУДШИХ СИСТЕМ ВЫВЕРНУТОГО ПОГРАНИЧНОГО СЛОЯ v5.3:")
        for rank, g in enumerate(galaxy_summaries[:10], 1):
            W(
                f"   [{rank:2d}] {g['name']:<12s} sMAPE={g['smape']:6.2f}% | bias={g['bias']:+6.1f} км/с | "
                f"Q={g['q']} | Inc={g['inc_obs']:4.1f}°->{g['inc_true']:4.1f}° | Phi_vap={g['phi_vapor']:.2f} | D={g['dist']:5.1f} Мпк"
            )
        W("")
        W("=" * LN)
        W(" ПОГАЛАКТИЧНАЯ ТАБЛИЦА ГОЛОГРАФИЧЕСКОЙ СБОРКИ СКОРОСТЕЙ (COSMO_OS v5.3)")
        W(" Галактика      |   sMAPE  |      Bias     |  Inc_obs -> Inc_true | Phi_vap |  f_gas |   s_glob |     Rd   |  Q")
        W("-" * LN)
        galaxy_summaries.sort(key=lambda x: x["name"])
        for g in galaxy_summaries:
            W(
                f" {g['name']:<14s} |   {g['smape']:5.2f}  |     {g['bias']:+8.2f}  |   "
                f"{g['inc_obs']:5.1f}° -> {g['inc_true']:5.1f}°  |   {g['phi_vapor']:4.2f}  |  "
                f"{g['f_gas']:4.2f}  |   {g['s_glob']:6.2f} |   {g['rd']:5.2f}  |  {g['q']}"
            )
        W("=" * LN)

    print(f"\n[ГОТОВО]: Общая медиана sMAPE: {med_gal_smape:.2f}% | Поточечная медиана: {med_pt_smape:.2f}%")
    print(f"[ГОТОВО]: Полный отчёт сохранён: {report_txt_path}")
    if save_plots:
        print(f"[ГОТОВО]: Графики сохранены в папку: {plots_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
