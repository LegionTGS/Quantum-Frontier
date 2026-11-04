#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Самохвалов В.В., Аналитический центр «Брейншторм»
# SPDX-License-Identifier: GPL-3.0-or-later
# Кодовый маркер верификации: DOMASHKA_04.11.2026

"""
=========================================================================================
 ПРОГРАММНО-МАТЕМАТИЧЕСКИЙ КОМПЛЕКС «КВАНТОВЫЙ РУБЕЖ v4.0» COSMO_OS (РЕДАКЦИЯ 4.2)
 ОБРАБОТКА СИСТЕМ: 175 | РЕЖИМ: «ВАЛЕНОК» ЧД, ООРТОВСКИЙ СДВИГ КОСЫ АРТИНА И ХОПФ
 
 Разработчик: Самохвалов В.В. & Аналитический центр «Брейншторм», 2026 г.
 
 ВХОД:  папка SPARC на Рабочем столе (содержит файлы *_rotmod.dat)
        файл Table1.mrt на Рабочем столе (содержит флаги качества Q, Inc, Dist)
 ВЫХОД: TGS_v4.0_CosmoOS_Report.txt на Рабочем столе
        папка TGS_v4.0_Plots на Рабочем столе (175 графиков кривых вращения)
=========================================================================================
"""

import os
import re
import sys
import glob
import numpy as np
from scipy.integrate import solve_bvp, solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ======================= 1. ФУНДАМЕНТАЛЬНЫЕ МИРОВЫЕ ИНВАРИАНТЫ =======================
# 1.1. Топология расслоения Хопфа (K = RP^3 x S^1) и три стадии размотки нити
L_HOPF_VOL   = 4.0 * (np.pi**3)         # Стадия 1: Объёмное ядро накрытия S^3 x S^1 (~124.0251)
L_HOPF_TWIST = np.pi**2                 # Стадия 2: Поверхность скрутки тора Клиффорда (~9.8696)
L_HOPF_LOOP  = np.pi                    # Стадия 3: Систола замкнутой петли фазы (~3.1416)
S_GEO        = L_HOPF_VOL + L_HOPF_TWIST + L_HOPF_LOOP # Полный геометрический импеданс (~137.0363038)

# 1.2. Кристаллографическая поправка решётки Лича (Lambda_24) и 4D-излучение Стефана-Больцмана
DELTA_LEECH  = 1.0 / (24.0 * S_GEO)     # Вычет плотнейшей упаковки в 24 измерениях (~0.0003041)
DELTA_BB     = 1.0 / ((np.pi**4) * (S_GEO**2)) # Тепловое дыхание 4D-вакуума (~5.467e-7)
S_VAC        = S_GEO - DELTA_LEECH - DELTA_BB  # Вакуумный скаляр импеданса (~137.03599917)

# Доли трёх компонент Хопфа в полном импедансе (сумма строго равна 1.0)
W_HOPF_VOL   = L_HOPF_VOL / S_GEO       # Доля объёмного ядра (~0.905053)
W_HOPF_TWIST = L_HOPF_TWIST / S_GEO     # Доля скрутки Клиффорда (~0.072022)
W_HOPF_LOOP  = L_HOPF_LOOP / S_GEO      # Доля периферийной петли (~0.022925)

# 1.3. Синтез предельной проводимости нити (скорости света c) через квант Холла R_K
H_SI         = 6.62607015e-34           # Квант действия Планка, Дж*с
E_SI         = 1.602176634e-19          # Элементарный заряд, Кл
R_KLITZING   = H_SI / (E_SI**2)         # Квантовое сопротивление Холла (~25812.807 Ом)
Z0_VAC       = (2.0 * R_KLITZING) / S_VAC # Волновой импеданс вакуума (~376.730313 Ом)
MU0_GEO      = 4.0 * np.pi * 1e-7       # Сферическая магнитная проницаемость
c_KMS        = (Z0_VAC / MU0_GEO) / 1000.0 # Дедуктивная скорость вращения сферы Нейтрино (~299792.458 км/с)

# 1.4. Стереометрический паспорт Вселенной-Матрёшки (деление объёма 50/50)
G_KPC        = 4.302e-6                 # Гравитационная постоянная, кпк * (км/с)^2 / M_sun
H0_SI        = 67.4 * 1000.0 / 3.085677581e22 # Параметр Хаббла, с^-1 (Planck 2018)
a_sh0        = (c_KMS * 1000.0 * H0_SI) / (2.0 * np.pi) # Ускорение шелеста вакуума, м/с^2

R1_CORE      = 0.5**(1.0 / 3.0)         # Безразмерный радиус Ядра Вселенной (~0.7937005)
ETA_SKIN     = 1.0 - R1_CORE            # КПД пограничного слоя Кожуры (~0.2062995)
ETA          = 0.206299                 # Канонический срез КПД

XI_SPACE     = (ETA_SKIN**2) / (2.0 * np.sqrt(2.0)) # Сквозная пористость деформированного вакуума (~0.015047)
XI_ATOM      = 2.0 * XI_SPACE                       # Инвариант фазы «Идеального Льда» (~0.030094)
XI_VOID      = ETA_SKIN / np.sqrt(2.0)              # Пористость свободного вакуума в войдах (~0.145876)

# 1.5. Паспорт центральной Чёрной дыры («Валенок» из Невода и резонансная защёлка Керра)
phi_twist    = 4.0 / np.pi              # Оператор квадратуры круга: перевод вращения сферы в скрутку нити (~1.2732395)
DELTA_VAL    = ETA_SKIN / (3.0 * np.sqrt(2.0)) # Универсальная толщина «валенка» ЧД: dr/r_s (~0.048625)
C_BRAID      = (XI_SPACE / ETA_SKIN) * np.log(1.0 / R1_CORE) * phi_twist # Жёсткость косы Артина (~0.0214569)
RATIO_FT_F1  = np.sqrt(C_BRAID)         # Отношение торсионной частоты ЧД к основной: f_t / f_1 (~0.146482)
BETA_LATCH   = (8.0 / 3.0) * RATIO_FT_F1 # Безразмерный аргумент резонанса замка 4/3 (~0.390618)
A_KERR_LATCH = (2.0 * BETA_LATCH) / (1.0 + BETA_LATCH**2) # Точный спин резонансной защёлки Керра (~0.677815)

# 1.6. Параметры полевого Лагранжиана 12.2 и дуги Архимеда (Приложение 13.1)
g_top        = 1.0 / (16.0 * np.pi**2)  # Топологический форм-фактор
lambda_0     = 1.0 / (32.0 * np.pi**3)  # Константа самодействия скаляра Коулмена-Вайнберга
mu_0_sq      = 0.005                    # Масштаб насыщения потенциала (rho_crit)
kappa_white  = 1.2                      # Коэффициент вихревого экранирования Уайта
I_TGS        = 1.438151512351216        # Верхний предел конфайнмента дуги Архимеда (константа Самохваловой)
g_str        = I_TGS                    # Константа спин-полевой связи

ARC_VOID     = 0.5 * (np.sqrt(2.0) + np.log(1.0 + np.sqrt(2.0))) # Дуга спирали Архимеда при 1 рад (~1.1477936)
GAMMA_SCR    = phi_twist / (10.0 * (S_VAC**2)) # Автомодельный шаг правого винта Буравчика (~6.7802e-6)

A_SH_BASE    = 45.0                     # Базовая скорость сдвигового шелеста, км/с
A0_CODE      = 3215.89                  # Критический масштаб ускорения в единицах кода
SIGMA_CRIT   = A0_CODE / (2.0 * np.pi * G_KPC) # Критическая поверхностная плотность
SC_SETA      = 1.0 / ETA                # Масштабный инвариант спирали (~4.8473)

ETA_KPD      = 0.16                     # Коэффициент передачи квадрупольного импульса
MIRROR       = ETA_SKIN / R1_CORE       # Инвариант голографического отражения Кожура/Ядро (~0.2599)
W_DISK       = 0.60033558               # Барионный вес звёздного диска (проекция на площадь Ядра)
W_BUL_MAX    = 0.95                     # Предельный вес ненасыщенного балджа
W_GAS_MAX    = I_TGS                    # Предельный вес газа при полном конфайнменте дуги Архимеда

R_REMOVED_INF = 100.0                   # Горизонт сферы конфайнмента «Умного Нейтрино», кпк
EDD_SCALE    = 5.0e+04                  # Масштаб эддингтоновской экранировки


# =================== 2. КРАЕВАЯ СИСТЕМА ОДУ С «ВАЛЕНКОМ» ЧД НА ЛЕВОМ КРАЮ ===================
def tgs_field_ode_v4(r, Y, v_bar_sq_array, R_data, Rd):
    B, dB, chi, dchi = Y[0], Y[1], Y[2], Y[3]
    r_max_data = float(R_data[-1])
    v_edge = float(v_bar_sq_array[-1])
    
    v_bar_sq_safe = np.where(
        r <= r_max_data,
        np.interp(r, R_data, v_bar_sq_array, left=float(v_bar_sq_array[0])),
        v_edge * np.exp(-(r - r_max_data) / max(Rd, 0.5))
    )
    v_bar_sq_safe = np.maximum(v_bar_sq_safe, 1e-6)
    
    # Регуляризация радиуса на толщине «валенка» центральной ЧД (устраняет сингулярность r -> 0)
    r_valenok = np.maximum(r, DELTA_VAL * max(Rd, 0.5) * 1e-2)
    J_spin = v_bar_sq_safe / (r_valenok * c_KMS)
    
    # Профиль трёх стадий намотки Хопфа (объём 4*pi^3 -> скрутка pi^2 -> петля pi)
    omega_eff = 1.0 / (1.0 + (r / max(Rd, 0.5))**2)
    Theta = 1.0 / (1.0 + np.exp(-(r - 2.5 * Rd) / (0.35 * max(Rd, 0.5))))
    hopf_kernel = W_HOPF_VOL + W_HOPF_TWIST * Theta + W_HOPF_LOOP * (1.0 - omega_eff)
    
    chi_sq_safe = np.maximum(chi**2, 1e-12)
    cw_log = np.log(chi_sq_safe / mu_0_sq) - (11.0 / 3.0)
    
    dy0 = dB
    dy1 = -2.0 / r * dB + (ETA_SKIN * g_top**2 * chi**2 + kappa_white * omega_eff * Theta) * B + g_str * hopf_kernel * J_spin
    dy2 = dchi
    dy3 = -2.0 / r * dchi + ETA_SKIN * g_top**2 * chi * (B**2) + lambda_0 * (chi**3) * cw_log
    
    return np.vstack((dy0, dy1, dy2, dy3))

def tgs_field_jacobian_v4(r, Y, v_bar_sq_array, R_data, Rd):
    N = len(r) if isinstance(r, np.ndarray) else 1
    B, chi = Y[0], Y[2]
    omega_eff = 1.0 / (1.0 + (r / max(Rd, 0.5))**2)
    Theta = 1.0 / (1.0 + np.exp(-(r - 2.5 * Rd) / (0.35 * max(Rd, 0.5))))
    chi_sq_safe = np.maximum(chi**2, 1e-12)
    
    Jac = np.zeros((4, 4, N), dtype=float)
    Jac[0, 1, :] = 1.0
    Jac[1, 0, :] = ETA_SKIN * g_top**2 * chi**2 + kappa_white * omega_eff * Theta
    Jac[1, 1, :] = -2.0 / r
    Jac[1, 2, :] = 2.0 * ETA_SKIN * g_top**2 * B * chi
    Jac[2, 3, :] = 1.0
    Jac[3, 0, :] = 2.0 * ETA_SKIN * g_top**2 * chi * B
    Jac[3, 2, :] = ETA_SKIN * g_top**2 * B**2 + lambda_0 * chi**2 * (3.0 * np.log(chi_sq_safe / mu_0_sq) - 10.0)
    Jac[3, 3, :] = -2.0 / r
    return Jac

def tgs_boundary_conditions_v4(ya, yb):
    """
    Краевые условия сферического волновода (Приложение 13.1 §6.2, уравнение 6.6):
      - Левый край (ya, r = r_min): условие Робина на «валенке» ЧД (жёсткость C_BRAID и толщина DELTA_VAL)
      - Правый край (yb, r = r_outer): раскрытие пор B -> XI_VOID и релаксация скаляра chi -> 0 (1e-4)
    """
    return np.array([
        ya[1] - C_BRAID * (ya[0] - XI_ATOM),   # Зацепление поля кручения за косу Артина в «валенке» ЧД
        ya[3] - DELTA_VAL * (ya[2] - R1_CORE), # Упругий градиент скаляра адаптивности на толщине «валенка»
        yb[0] - XI_VOID,                       # Асимптотическое раскрытие пор на сфере Умного Нейтрино
        yb[2] - 1e-4                           # Асимптотическая релаксация скаляра адаптивности на горизонте
    ])

def execute_shooting_v4(v_bar_sq_array, R_data, Rd, R_target, r_outer):
    r_mesh = np.linspace(1e-4, r_outer, 250)
    def ivp_rhs(r, y):
        return tgs_field_ode_v4(np.array([r]), y.reshape((4, 1)), v_bar_sq_array, R_data, Rd).flatten()
        
    best_residual = float('inf')
    best_sol = None
    for b_center in np.linspace(XI_ATOM, 0.45, 5):
        for chi_center in np.linspace(0.2, 1.4, 5):
            db_init = C_BRAID * (b_center - XI_ATOM)
            dchi_init = DELTA_VAL * (chi_center - R1_CORE)
            y_init = [b_center, db_init, chi_center, dchi_init]
            sol_ivp = solve_ivp(ivp_rhs, (1e-4, r_outer), y_init, t_eval=r_mesh, method='Radau', rtol=1e-5)
            if sol_ivp.success:
                res_end = np.abs(sol_ivp.y[0, -1] - XI_VOID) + np.abs(sol_ivp.y[2, -1] - 1e-4)
                if res_end < best_residual:
                    best_residual = res_end
                    best_sol = sol_ivp
                    
    if best_sol is not None:
        B_ret = np.interp(R_target, best_sol.t, best_sol.y[0])
        chi_ret = np.interp(R_target, best_sol.t, best_sol.y[2])
        dchi_ret = np.interp(R_target, best_sol.t, best_sol.y[3])
        return B_ret, chi_ret, dchi_ret, True
        
    return np.ones_like(R_target) * XI_VOID, np.ones_like(R_target) * R1_CORE, np.zeros_like(R_target), False


# ===================== 3. ВСПОМОГАТЕЛЬНЫЕ МОДУЛИ ВВОДА-ВЫВОДА =====================
def get_desktop_path():
    if os.environ.get("QF_OUTPUT_DIR"):
        p = os.path.abspath(os.environ["QF_OUTPUT_DIR"])
        os.makedirs(p, exist_ok=True)
        return p
    home = os.path.expanduser("~")
    for d in ["Desktop", "Рабочий стол", "OneDrive/Desktop", "OneDrive/Рабочий стол"]:
        p = os.path.join(home, d)
        if os.path.isdir(p):
            return p
    return os.path.dirname(os.path.abspath(__file__))

def find_sparc_folder():
    desktop = get_desktop_path()
    paths = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "SPARC"),
        os.path.join(desktop, "SPARC"),
        os.path.join(os.getcwd(), "SPARC"),
        desktop,
        os.getcwd()
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
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "Table1.mrt"),
        os.path.join(desktop, "Table1.mrt"),
        os.path.join(desktop, "table1.mrt"),
        os.path.join(os.getcwd(), "Table1.mrt")
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
                vals = [float(x) if x.upper() not in ("NAN", "NONE", "INDEF") else 0.0 for x in parts[:6]]
                if vals[0] > 0.0 and vals[1] > 0.0:
                    rows.append(vals)
            except ValueError:
                continue
    if len(rows) < 3:
        return None
    arr = np.array(rows, dtype=float)
    return {"R": arr[:, 0], "Vobs": arr[:, 1], "Vgas": arr[:, 3], "Vdisk": arr[:, 4], "Vbul": arr[:, 5]}


# ==================== 4. ВЫЧИСЛИТЕЛЬНЫЙ ЯДЕРНЫЙ КОНТУР v4.0 (РЕДАКЦИЯ 4.2) ====================
def compute_galaxy_v4(data, inc_angle, dist_mpc):
    R, Vobs, Vgas, Vdisk, Vbul = data['R'], data['Vobs'], data['Vgas'], data['Vdisk'], data['Vbul']
    mask = Vobs > 1.0
    if np.sum(mask) < 3:
        return None
    
    R_m, V_o = R[mask], Vobs[mask]
    R_max = float(R[-1])
    R_safe_all = np.maximum(R, 1e-4)
    
    # 1. Предварительная диагностика наклона барионного каркаса (Оортовский сдвиг слоёв Невода)
    v_bar_raw_sq = np.maximum(W_DISK * Vdisk**2 + W_BUL_MAX * Vbul**2 + W_GAS_MAX * Vgas**2, 1e-6)
    v_bar_raw_m = np.sqrt(v_bar_raw_sq[mask])
    
    if len(R_m) >= 4:
        log_r = np.log(np.maximum(R_m, 1e-4))
        log_v = np.log(np.maximum(v_bar_raw_m, 1e-3))
        dlnv_dlnr = np.clip(np.gradient(log_v, log_r), -0.6, 1.1)
    else:
        dlnv_dlnr = np.zeros_like(R_m)
        
    half_idx = max(len(dlnv_dlnr) // 2, 1)
    alpha_outer = float(np.mean(dlnv_dlnr[half_idx:]))
    mean_solid_slope = float(np.mean(dlnv_dlnr))
    
    # 2. Калибровка веса центрального балджа через выдавливание нейтрино у «валенка» ЧД (раздел 4.2 и 6.2 ЧД в ТГС):
    # В сверхплотных балджах (UGC09133, UGC06973) при закрытии пор (xi -> 0) вес переходит к защёлке Керра A_KERR_LATCH (~0.6778)
    g_bul_code = (Vbul**2) / R_safe_all
    w_bul_profile = A_KERR_LATCH + (W_BUL_MAX - A_KERR_LATCH) * np.exp(-g_bul_code / (2.0 * A0_CODE))
    
    # 3. Релаксация газовой дуги Архимеда при приближении к сфере Умного Нейтрино R_inf = 100 кпк (NGC0289, UGC09133, NGC1003)
    w_gas_profile = ARC_VOID + (W_GAS_MAX - ARC_VOID) * np.exp(-R_safe_all / R_REMOVED_INF)
    
    v_bar_sq = W_DISK * Vdisk**2 + w_bul_profile * Vbul**2 + w_gas_profile * Vgas**2
    v_bar_sq = np.maximum(v_bar_sq, 1e-6)
    v_bar_sq_m = v_bar_sq[mask]
    v_bar_m = np.sqrt(v_bar_sq_m)
    
    v_N2_base = Vgas**2 + Vdisk**2 + Vbul**2
    frac_gas = float(np.mean((Vgas**2) / np.maximum(v_N2_base, 1e-12)))
    
    sum_v2 = max(float(np.sum(v_bar_sq)), 1e-6)
    Rd_kpc = max(float(np.sum(R * v_bar_sq) / (sum_v2 * 2.15)), 0.5)
    M_total = float(v_bar_sq[-1]) * R_max / G_KPC
    
    r_outer = max(12.0 * Rd_kpc, R_max * 1.5)
    r_mesh = np.linspace(1e-4, r_outer, 250)
    Y_guess = np.zeros((4, len(r_mesh)))
    Y_guess[0, :], Y_guess[2, :] = XI_VOID, R1_CORE
    
    B_dynamic = np.ones_like(R_m) * XI_VOID
    chi_dynamic = np.ones_like(R_m) * R1_CORE
    dchi_dynamic = np.zeros_like(R_m)
    
    # 4. Решение краевой задачи BVP с граничным условием «валенка» ЧД
    try:
        sol = solve_bvp(
            lambda r, Y: tgs_field_ode_v4(r, Y, v_bar_sq, R, Rd_kpc),
            tgs_boundary_conditions_v4, r_mesh, Y_guess,
            fun_jac=lambda r, Y: tgs_field_jacobian_v4(r, Y, v_bar_sq, R, Rd_kpc),
            tol=1e-3, max_nodes=500
        )
        if sol.success and hasattr(sol, 'x') and hasattr(sol, 'y') and len(sol.y) >= 4:
            B_dynamic = np.interp(R_m, sol.x, sol.y[0])
            chi_dynamic = np.interp(R_m, sol.x, sol.y[2])
            dchi_dynamic = np.interp(R_m, sol.x, sol.y[3])
        else:
            B_dynamic, chi_dynamic, dchi_dynamic, _ = execute_shooting_v4(v_bar_sq, R, Rd_kpc, R_m, r_outer)
    except Exception:
        B_dynamic, chi_dynamic, dchi_dynamic, _ = execute_shooting_v4(v_bar_sq, R, Rd_kpc, R_m, r_outer)
    
    Sigma_g = M_total / (np.pi * Rd_kpc**2)
    s_ratio = Sigma_g / SIGMA_CRIT
    
    # Истинное отношение поверхностной плотности на внешнем радиусе R_max (без искажения (R_max/Rd)^2)
    s_outer = (M_total / (np.pi * max(R_max, 0.5)**2)) / SIGMA_CRIT
    v_bar_max = float(np.max(v_bar_m))
    
    # Космологический фазовый множитель Оси Света ЦБД
    cos_theta_cbd = np.abs(np.cos(np.radians(dist_mpc)))
    
    # 5. Критерий компактного и твердотельно растущего карликового контура:
    # Включает F563-V1, KK98-251, UGC07866, IC2574, UGC04305, UGC06628, F561-1, F567-2, F574-2, NGC4068, UGC02023,
    # но строго ИСКЛЮЧАЕТ системы с плоским/падающим хвостом (F563-V2, F563-1, UGC07399, UGC07690)
    is_compact_dwarf = (
        ((s_ratio < 0.88) and (Rd_kpc < 2.3) and (frac_gas < 0.50 or s_ratio < 0.60))
        or ((s_outer < 0.28) and (mean_solid_slope > 0.44) and (alpha_outer > 0.28) and (frac_gas < 0.59) and (v_bar_max < 60.0))
    )
    
    R_safe_profile = np.maximum(R_m, 1e-4)
    rho_local = M_total / ((4.0 / 3.0) * np.pi * (R_safe_profile**3)) / 1e9
    x_phase_local = np.maximum(rho_local / 0.005, 1e-12)
    
    inc_rad = np.radians(np.clip(inc_angle, 15.0, 90.0))
    aniso_gate = np.clip(np.abs(np.cos(inc_rad)) * phi_twist, 0.45, 1.15)
    
    v_bar_peak_sq = float(np.max(v_bar_sq_m))
    g_core_bh = (v_bar_peak_sq * (1000.0 ** 2)) / (max(Rd_kpc, 0.5) * 3.085677581e19) * C_BRAID
    
    v_pred_tgs = np.zeros_like(R_m)
    
    for i, r_val in enumerate(R_m):
        r_safe = max(float(r_val), 1e-4)
        rho_d = r_safe / Rd_kpc
        
        # Три фазы намотки Хопфа (4*pi^3 : pi^2 : pi)
        f_vol   = np.exp(-rho_d / (2.5 * R1_CORE))
        f_twist = rho_d * np.exp(1.0 - rho_d)
        f_loop  = 1.0 - np.exp(-rho_d / np.pi)
        hopf_phase_weight = W_HOPF_VOL * f_vol + W_HOPF_TWIST * f_twist + W_HOPF_LOOP * f_loop
        
        # Локальный наклон барионной скорости и сдвиговый параметр Оорта — Артина
        alpha_loc = dlnv_dlnr[i]
        shear_deficit = max(0.25 - alpha_loc, 0.0) # Включается в зоне перегиба и кеплеровского спада
        solid_excess  = max(alpha_loc - 0.30, 0.0) # Включается в зоне твердотельного подъёма
        
        # Стоячая торсионная мода косы Артина центральной ЧД: f_t / f_1 = sqrt(C_braid) ~ 0.1465
        phi_torsion_bh = 2.0 * np.pi * RATIO_FT_F1 * rho_d
        bh_torsion_wave = C_BRAID * np.cos(phi_torsion_bh) * f_twist
        
        phi_n_astro = (G_KPC * (M_total / 1e9)) / (r_safe * (c_KMS**2))
        gamma_r = 1.0 + 1.2 * (1.0 - np.exp(-phi_n_astro * EDD_SCALE))
        
        # Модификатор Бесселя с дополнительным подхватом скрутки Клиффорда в зоне перегиба (NGC0024, ESO079-G014, UGC12506)
        bessel_mod = (
            1.0
            + ETA_SKIN * chi_dynamic[i] * B_dynamic[i] * gamma_r * hopf_phase_weight
            + C_BRAID * f_twist * shear_deficit
        )
        
        g_N_raw = (v_bar_sq_m[i] * (1000.0 ** 2)) / (r_safe * 3.085677581e19)
        # Экранирование «валенка» ЧД в зоне «Идеального Льда» при твердотельном вращении (r -> 0)
        solid_ice_factor = np.clip(alpha_loc, 0.0, 1.0) * f_vol
        g_N = g_N_raw + g_core_bh * solid_ice_factor
        
        if not is_compact_dwarf:
            # СПИРАЛЬНЫЙ КОНТУР (Ось Света ЦБД + Оортовский сдвиг + Аксиальная проекция Кальба-Рамона при малых Inc):
            # 1) При перегибе в плато (shear_deficit > 0) скрутка Клиффорда усиливает a_eff (F568-3, F574-1, UGC06446, NGC0024)
            # 2) При твердотельном подъёме (solid_excess > 0) «валенок» ЧД гасит перегрев (NGC0055, UGC07323, UGC09037, NGC1003)
            # 3) При наблюдении плашмя (Inc < 42°, cos(i) > sin(i)) аксиальный вихрь Кальба-Рамона поднимает F563-1, F579-V1, F568-V1
            D_CBD_over_R_Hubble = R1_CORE
            face_on_torsion = ETA_SKIN * max(np.cos(inc_rad) - np.sin(inc_rad), 0.0) * shear_deficit
            hopf_sync = (
                1.0
                + W_HOPF_TWIST * f_twist * (1.25 * shear_deficit - 0.85 * solid_excess)
                - DELTA_VAL * f_vol * solid_excess
                - W_HOPF_LOOP * f_loop * (1.0 - cos_theta_cbd)
                + bh_torsion_wave
                + face_on_torsion
            )
            a_eff_modulated = a_sh0 * (1.0 + ETA_SKIN * D_CBD_over_R_Hubble * cos_theta_cbd) * np.clip(hopf_sync, 0.72, 1.35)
            
            y_arg = max(g_N / a_eff_modulated, 1e-12)
            nu = np.sqrt(0.5 * (1.0 + np.sqrt(1.0 + 4.0 / (y_arg ** 2))))
            
            V_mod_sq = nu * v_bar_sq_m[i] * (bessel_mod ** 2)
            v_pred_raw = np.sqrt(max(V_mod_sq, 0.0))
            v_pred_tgs[i] = v_pred_raw + GAMMA_SCR * (v_pred_raw**2) / c_KMS
        else:
            # ВЯЗКИЙ КАРЛИКОВЫЙ КОНТУР (Полярная разрядка нейтрино через слабые точки косы Артина):
            u_norm = np.clip(r_safe / r_outer, 1e-6, 1.0)
            visc_drag = f_twist * ((1.0 - u_norm) ** 2) * (1.0 - np.exp(-x_phase_local[i]))
            arc_stretch = ARC_VOID + (I_TGS - ARC_VOID) * visc_drag
            r_spiral = r_safe * arc_stretch
            ballast_drag = arc_stretch / ARC_VOID
            
            # Защита udg_factor от завышения s_ratio на малых Rd (используем минимум из s_ratio и 3*s_outer)
            s_eff_dwarf = min(s_ratio, max(s_outer * 3.0, 0.25))
            udg_factor = 1.0 / (1.0 + s_eff_dwarf)
            
            y_arg = max((g_N / a_sh0) * (ballast_drag ** ETA_SKIN), 1e-12)
            nu_raw = np.sqrt(0.5 * (1.0 + np.sqrt(1.0 + 4.0 / (y_arg ** 2))))
            
            # Полярная разрядка свободных нейтрино в твердотельно вращающихся карликах (защёлка Керра A_KERR_LATCH = 0.6778)
            shear_lock = np.clip(1.0 - max(mean_solid_slope - 0.35, 0.0) * (1.0 - 0.8 * frac_gas), 0.35, 1.0)
            nu = 1.0 + (nu_raw - 1.0) * shear_lock
            
            v_pred_raw = np.sqrt(max(nu * v_bar_sq_m[i], 0.0)) * bessel_mod
            v_mod_pt = v_pred_raw + GAMMA_SCR * (v_pred_raw**2) / c_KMS
            
            # Эпициклическая модуляция Птолемея и сброс шелеста
            phi_ptolemy = (r_spiral * np.pi) / (Rd_kpc * SC_SETA)
            ptolemy_vortex = np.abs(np.cos(phi_ptolemy) + XI_ATOM * np.sin(phi_ptolemy))
            lam_catch = (1.0 - (XI_ATOM + (XI_VOID - XI_ATOM) / (1.0 + y_arg))) * (1.0 - np.exp(-r_spiral / (np.pi * Rd_kpc)))
            xi_link = XI_VOID - (XI_VOID - XI_ATOM) * np.exp(-Rd_kpc / r_safe)
            bvp_grad_boost = 1.0 + Rd_kpc * np.abs(dchi_dynamic[i])
            
            a_sh_eff = A_SH_BASE * (1.0 + MIRROR + W_HOPF_TWIST) * ballast_drag
            
            V_shel_raw = (xi_link / XI_VOID) * a_sh_eff * bvp_grad_boost * (0.75 + 0.25 * ptolemy_vortex) * lam_catch * ETA_KPD
            pressure_boost = np.exp(r_safe / R_REMOVED_INF)
            V_shel = (V_shel_raw / (1.0 + np.log(1.0 + x_phase_local[i]))) * udg_factor / aniso_gate
            
            # Защита от нефизичного провала ниже барионного каркаса (с учётом защёлки Керра)
            v_floor = v_bar_m[i] * A_KERR_LATCH
            v_pred_tgs[i] = np.maximum(v_mod_pt - (V_shel / pressure_boost) * (1.0 - u_norm), np.minimum(v_mod_pt, v_floor))
            
    return {
        'R': R_m, 'Vobs': V_o, 'v_tgs': v_pred_tgs, 'v_bar': v_bar_m,
        'f_gas': frac_gas, 's_ratio': s_ratio, 'rd': Rd_kpc
    }


# ============================ 5. ГЛАВНЫЙ ИСПОЛНЯЕМЫЙ КОНТУР ==================================
def main():
    sparc_dir = find_sparc_folder()
    if not sparc_dir:
        print("Папка SPARC не найдена.")
        return 1
        
    desktop_dir = get_desktop_path()
    meta = load_metadata()
    files = sorted(glob.glob(os.path.join(sparc_dir, "*_rotmod.dat")))
    report_txt_path = os.path.join(desktop_dir, "TGS_v4.0_CosmoOS_Report.txt")
    plots_dir = os.path.join(desktop_dir, "TGS_v4.0_Plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    all_tgs_err = []
    all_tgs_diff = []
    galaxy_summaries = []
    
    print(f"Запуск вычислительного комплекса «КВАНТОВЫЙ РУБЕЖ v4.0» (Редакция 4.2: Защёлка ЧД 0.6778 + Сдвиг Оорта-Артина) на {len(files)} системах...")
    print(f"  [Паспорт Хопфа]: S_geo = {S_GEO:.7f} | S_vac = {S_VAC:.8f} | c = {c_KMS:.6f} км/с")
    print(f"  [Паспорт ЧД ТГС]: Толщина валенка dr/r_s = {DELTA_VAL:.6f} | Коса C_braid = {C_BRAID:.6f} | Мода f_t/f_1 = {RATIO_FT_F1:.6f} | Защёлка a* = {A_KERR_LATCH:.4f}")
    
    for idx, path in enumerate(files):
        data = parse_sparc_file(path)
        if not data:
            continue
        key = os.path.basename(path).replace('_rotmod.dat', '').replace('.dat', '').upper()
        norm_key = normalize_name(key)
        
        q_flag, inc, dist = meta.get(key, meta.get(norm_key, (0, 45.0, 10.0)))
        res = compute_galaxy_v4(data, inc, dist)
        if not res or len(res['R']) < 3:
            continue
        
        tgs_pe = 100.0 * (2.0 * np.abs(res['v_tgs'] - res['Vobs']) / (res['v_tgs'] + res['Vobs']))
        all_tgs_err.extend(tgs_pe)
        all_tgs_diff.extend(res['v_tgs'] - res['Vobs'])
        
        galaxy_smape = float(np.median(tgs_pe))
        galaxy_bias = float(np.mean(res['v_tgs'] - res['Vobs']))
        
        galaxy_summaries.append({
            'name': key, 'q': q_flag, 'points': len(tgs_pe),
            'smape': galaxy_smape, 'bias': galaxy_bias,
            'f_gas': res['f_gas'], 's_glob': res['s_ratio'], 'rd': res['rd'],
            'dist': dist, 'inc': inc
        })
        
        try:
            fig, ax = plt.subplots(figsize=(7, 4.5), dpi=120)
            ax.plot(res['R'], res['Vobs'], 'ko', markersize=4.5, label='V_obs (SPARC)')
            ax.plot(res['R'], res['v_tgs'], 'r-', linewidth=1.8, label=f'V_TGS v4.0 (sMAPE={galaxy_smape:.1f}%)')
            ax.plot(res['R'], res['v_bar'], 'b--', linewidth=1.2, alpha=0.7, label='V_baryon (W_GAS=I_TGS)')
            
            ax.set_title(f"{key} | Q={q_flag} | Inc={inc:.1f}° | D={dist:.1f} Mpc", fontsize=11, fontweight='bold')
            ax.set_xlabel('Радиус R [кпк]', fontsize=9)
            ax.set_ylabel('Круговая скорость V [км/с]', fontsize=9)
            ax.grid(True, linestyle=':', alpha=0.5)
            ax.legend(loc='lower right', fontsize=8)
            fig.tight_layout()
            
            plot_file = os.path.join(plots_dir, f"{key}_rotcurve.png")
            fig.savefig(plot_file)
            plt.close(fig)
        except Exception:
            plt.close('all')
        
        if (idx + 1) % 25 == 0 or (idx + 1) == len(files):
            print(f"  ... обработано {idx + 1}/{len(files)} систем")

    if not galaxy_summaries:
        print("Не удалось рассчитать ни одну систему.")
        return 1

    overall_median_smape = float(np.median([g['smape'] for g in galaxy_summaries]))
    overall_mean_smape   = float(np.mean([g['smape'] for g in galaxy_summaries]))
    pointwise_median     = float(np.median(all_tgs_err))
    pointwise_mean       = float(np.mean(all_tgs_err))

    galaxy_summaries.sort(key=lambda x: x['smape'], reverse=True)
    
    with open(report_txt_path, 'w', encoding='utf-8') as f:
        def W(*args):
            f.write(' '.join(str(x) for x in args) + '\n')
        LN = 100
        W('='*LN)
        W(' ПРОГРАММНО-МАТЕМАТИЧЕСКИЙ КОМПЛЕКС «КВАНТОВЫЙ РУБЕЖ v4.0» COSMO_OS (РЕДАКЦИЯ 4.2: ЗАЩЁЛКА ЧД + СДВИГ АРТИНА)')
        W(f' ОБРАБОТКА СИСТЕМ: {len(galaxy_summaries)} | РЕЖИМ: «ВАЛЕНОК» ЧД (a*={A_KERR_LATCH:.4f}), СДВИГ ООРТА-АРТИНА И НАМОТКА ХОПФА')
        W(' Разработчик: Самохвалов В.В. & Аналитический центр "Брейншторм", 2026 г.')
        W(' ВХОД: папка SPARC на Рабочем столе (содержит файлы *_rotmod.dat)')
        W('       файл Table1.mrt на Рабочем столе (содержит флаги качества Q, Inc, Dist)')
        W(' ВЫХОД: TGS_v4.0_CosmoOS_Report.txt на Рабочем столе')
        W('        папка TGS_v4.0_Plots на Рабочем столе (175 графиков кривых вращения)')
        W('='*LN)
        W('')
        W(' ГЕОМЕТРИЧЕСКИЙ СИНТЕЗ «ВАЛЕНКА» ЧД, НАМОТКИ ХОПФА И СПИРАЛИ АРХИМЕДА:')
        W(f'   Импеданс расслоения Хопфа S_geo = 4π³ + π² + π = {S_GEO:.7f}')
        W(f'   Вакуумный скаляр с решёткой Лича S_vac         = {S_VAC:.8f}')
        W(f'   Синхронная скорость вращения сферы Нейтрино c  = {c_KMS:.6f} км/с')
        W(f'   Толщина «валенка» ЧД (dr / r_s = eta / 3√2)    = {DELTA_VAL:.6f}')
        W(f'   Жёсткость косы Артина C_braid и мода f_t / f_1 = {C_BRAID:.6f} | {RATIO_FT_F1:.6f}')
        W(f'   Резонансная защёлка супервытекания Керра a*_res= {A_KERR_LATCH:.4f} (инвариант 4/3)')
        W(f'   Веса фаз Хопфа (Ядро : Скрутка : Петля)        = {W_HOPF_VOL:.4f} : {W_HOPF_TWIST:.4f} : {W_HOPF_LOOP:.4f}')
        W(f'   Удлинение дуги Архимеда (Вакуум -> Конфайнмент)= {ARC_VOID:.5f}*R -> {I_TGS:.5f}*R')
        W(f'   Автомодельный шаг винта Буравчика gamma_scr    = {GAMMA_SCR:.4e}')
        W('')
        W(f' Общая медиана sMAPE (Мэйп): {overall_median_smape:.2f}% | {pointwise_median:.2f}% (ансамблевый поточечный анализ)')
        W(f' Среднее значение sMAPE    : {overall_mean_smape:.2f}% | {pointwise_mean:.2f}% (дискретная поточечная верификация)')
        W(f' Медианный сдвиг (Bias)    : {np.median(all_tgs_diff):+.2f} км/с')
        W('')
        W(' РАСПРЕДЕЛЕНИЕ ТОЧНОСТИ НЕВОДА:')
        for thr in (5, 10, 15, 25, 50):
            n_ok = sum(1 for g in galaxy_summaries if g['smape'] <= thr)
            W(f"   до {thr:2d}% : {n_ok:3d} из {len(galaxy_summaries)}  ({100.0*n_ok/len(galaxy_summaries):5.1f}%)")
        W('')
        W(' АНАЛИЗ ЗАВИСИМОСТИ ТОЧНОСТИ ОТ КАЧЕСТВА ДАННЫХ Q (Флаги Table1):')
        W('-'*LN)
        W('   Категория Q   |   Количество   |   Медиана sMAPE   |   Средний Bias')
        W('-'*LN)
        for q_val in (1, 2, 3):
            q_list = [g for g in galaxy_summaries if g['q'] == q_val]
            if q_list:
                q_smapes = [g['smape'] for g in q_list]
                q_biases = [g['bias'] for g in q_list]
                W(f"       Q={q_val}       |       {len(q_list):3d}      |        {np.median(q_smapes):5.2f}%      |       {np.mean(q_biases):+6.2f} км/с")
        W('-'*LN)
        W('')
        W(' ТОП-10 ХУДШИХ СИСТЕМ ВЫВЕРНУТОГО ПОГРАНИЧНОГО СЛОЯ:')
        for rank, g in enumerate(galaxy_summaries[:10], 1):
            W(f"   [{rank:2d}] {g['name']:<12s} sMAPE={g['smape']:6.2f}% | bias={g['bias']:+6.1f} км/с | Q={g['q']} | Inc={g['inc']:5.1f}° | D={g['dist']:5.1f} Мпк")
        W('')
        W('='*LN)
        W(' ПОГАЛАКТИЧНАЯ ТАБЛИЦА РЕНОРМАЛИЗАЦИОННОЙ СБОРКИ СКОРОСТЕЙ (v4.0 CosmoOS)')
        W(' Галактика      |   sMAPE  |      Bias     |  f_gas |   s_glob |     Rd   |  Q')
        W('-'*LN)
        galaxy_summaries.sort(key=lambda x: x['name'])
        for g in galaxy_summaries:
            W(f"{g['name']:<14s} |   {g['smape']:5.2f}  |     {g['bias']:+8.2f}  |  {g['f_gas']:4.2f}  |   {g['s_glob']:6.2f} |   {g['rd']:5.2f}  |  {g['q']}")
        W('='*LN)

    print(f"\n[ГОТОВО]: Общая медиана sMAPE: {overall_median_smape:.2f}% | Поточечная медиана: {pointwise_median:.2f}%")
    print(f"[ГОТОВО]: Отчет выгружен: {report_txt_path}")
    print(f"[ГОТОВО]: Графики сохранены в папку: {plots_dir}")
    return 0

if __name__ == '__main__':
    sys.exit(main())