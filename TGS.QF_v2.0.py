#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Самохвалов В.В., Аналитический центр «Брейншторм»
# SPDX-License-Identifier: GPL-3.0-or-later
# Кодовый маркер верификации: DOMASHKA_04.11.2026

"""
=========================================================================================
 ПРОГРАММНО-МАТЕМАТИЧЕСКИЙ КОМПЛЕКС «КВАНТОВЫЙ РУБЕЖ v2.0» ПТОЛЕМЕЕВСКИЙ РЕФАКТОРИНГ ОДУ2
 ОБРАБОТКА СИСТЕМ: 175 | РЕЖИМ: СТРОГИЙ АСИМПТОТИЧЕСКИЙ BVP И ПТОЛЕМЕЕВЫ ЭПИЦИКЛЫ
 
 Разработчик: Самохвалов В.В. & Аналитический центр «Брейншторм», 2026 г.
 
 ВХОД:  папка SPARC на Рабочем столе (содержит файлы *_rotmod.dat)
        файл Table1.mrt на Рабочем столе (содержит флаги качества Q, Inc, Dist)
 ВЫХОД: TGS_v2.0_Epicycles_Report.txt на Рабочем столе
        папка TGS_v2.0_Plots на Рабочем столе (175 графиков кривых вращения)
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

# ===================== 1. ФУНДАМЕНТАЛЬНЫЕ ИНВАРИАНТЫ ТГС =====================
G_SI       = 6.67430e-11                # Метрическая гравитационная постоянная
c_SI       = 299792458.0                # Скорость света в вакууме, м/с
H0_SI      = 67.4 * 1000.0 / 3.085677581e22 # Пульс Хаббла (Planck 2018)
a_sh0      = (c_SI * H0_SI) / (2.0 * np.pi) # Ускорение шелеста вакуума (~1.0422e-10 м/с²)

ETA        = 0.206299                   # Медианное отношение геометрии
R1_CORE    = 0.5**(1.0 / 3.0)           # ~0.7937 (Радиус архивного Ядра Вселенной)
ETA_SKIN   = 1.0 - R1_CORE              # ~0.2063 (Вычислительный КПД Кожуры)

XI_ATOM    = ETA**2 / (2.0 * np.pi * 0.225079)  # Фиксированная атомарная пористость ядра
XI_VOID    = ETA / np.sqrt(2.0)                 # Пористость свободного вакуумного войда (~0.145876)

g_top      = 1.0 / (16.0 * np.pi**2)   # Топологическая константа связи
lambda_0   = 1.0 / (32.0 * np.pi**3)   # Потенциал Коулмена — Вайнберга
mu_0_sq    = 0.005                      # Масштаб нормировки вакуума (rho_crit)
κ_white    = 1.0                        # Филаментное давление космической паутины
g_str      = 1.43815                    # Константа струнного натяжения

KPC_TO_M   = 3.085677581e19             # Перевод кпк в метры
KMS_TO_MS  = 1000.0                     # Перевод км/с в м/с
M_SUN      = 1.98847e30                 # Масса Солнца, кг

A0         = 3215.89                    # Ускорение шелеста в единицах кода
G_KPC      = 4.302e-6                   # G в астрофизических единицах
SIGMA_CRIT = A0 / (2.0 * np.pi * G_KPC) # Критическая поверхностная плотность
OMEGA_CONST= 5.4e-16                    # Частотный масштаб фазового канала
SC_SETA    = 1.0 / ETA                  # Шкала невода: s_c = 1/ETA = 4.8473

# СТРОГИЕ ЛАГРАНЖЕВЫ КВАНТОВЫЕ КОНСТАНТЫ ДИССИПАЦИИ (Страницы 11–12 препринта)
GAMMA_DISS = 1.483e-61                  # Фундаментальный параметр диссипации
L_PLANCK   = 1.616255e-35               # Планковская длина, м
ETA_KPD    = 0.16                       # Квантовый инвариант КПД передачи импульса
DAMP_P     = 1.45                       # Показатель затухания по фазовой плотности
W_GAS      = 1.43368721   
W_DISK     = 0.60033558   
GAMMA_SCR  = 6.7706e-06   
R_REMOVED_INF = 100.0                   # Сетка ToE-интегрирования (100 кпк)

# ЖЁСТКО ЗАНЕСЕННЫЕ РЕНОРМАЛИЗАЦИОННЫЕ ИНВАРИАНТЫ ПТС-СЕТИ
CW_SUB      = 3.2000                    # 1. Постоянный вычет косы ПТС
EDD_SCALE   = 5.0e+04                   # 2. Масштаб Эддингтона прогиба полотна
FLUID_SCALE = 8.00e-85                  # 3. Фиксированная вязкость реликтового нейтринного океана

# ==================== 2. СИСТЕМА ДИФФЕРЕНЦИАЛЬНЫХ УРАВНЕНИЙ ===================
def tgs_field_ode(r, Y, v_bar_sq_array, R_data, Rd):
    y0, y1, y2, y3 = Y[0], Y[1], Y[2], Y[3]
    r_max_data = float(R_data[-1])
    v_edge = float(v_bar_sq_array[-1])
    
    v_bar_sq_safe = np.where(
        r <= r_max_data,
        np.interp(r, R_data, v_bar_sq_array, left=float(v_bar_sq_array[0])),
        v_edge * np.exp(-(r - r_max_data) / max(Rd, 0.5))
    )
    v_bar_sq_safe = np.maximum(v_bar_sq_safe, 1e-6)
    J_spin = v_bar_sq_safe / (r * 299792.458)
    
    omega_eff = 1.0 / (1.0 + (r / max(Rd, 0.5))**2)
    Theta = 1.0 / (1.0 + np.exp((r - 3.2 * Rd) / 0.2))
    
    dy0 = y1
    dy1 = -2.0 / r * y1 + (0.21 * g_top**2 * y2**2 + κ_white * omega_eff * Theta) * y0 + g_str * J_spin
    
    dy2 = y3
    y2_sq_safe = np.maximum(y2**2, 1e-6)
    cw_log = np.log(y2_sq_safe / mu_0_sq) - CW_SUB
    dy3 = -2.0 / r * y3 + 0.21 * g_top**2 * y2 * y0**2 + lambda_0 * y2**3 * cw_log
    
    return np.vstack((dy0, dy1, dy2, dy3))

def tgs_field_jacobian(r, Y, v_bar_sq_array, R_data, Rd):
    N = len(r) if isinstance(r, np.ndarray) else 1
    y0, y2 = Y[0], Y[2]
    
    omega_eff = 1.0 / (1.0 + (r / max(Rd, 0.5))**2)
    Theta = 1.0 / (1.0 + np.exp((r - 3.2 * Rd) / 0.2))
    
    Jac = np.zeros((4, 4, N), dtype=float)
    Jac[0, 1, :] = 1.0
    Jac[1, 0, :] = 0.21 * g_top**2 * y2**2 + κ_white * omega_eff * Theta
    Jac[1, 1, :] = -2.0 / r
    Jac[1, 2, :] = 0.42 * g_top**2 * y0 * y2
    
    Jac[2, 3, :] = 1.0
    Jac[3, 0, :] = 0.42 * g_top**2 * y0 * y2
    y2_sq_safe = np.maximum(y2**2, 1e-6)
    Jac[3, 2, :] = 0.21 * g_top**2 * y0**2 + lambda_0 * y2**2 * (3.0 * np.log(y2_sq_safe / mu_0_sq) - (3.0 * CW_SUB - 1.0))
    Jac[3, 3, :] = -2.0 / r
    
    return Jac

def tgs_boundary_conditions(ya, yb):
    res = np.zeros(4)
    res[0] = ya[1]  
    res[1] = ya[3]  
    res[2] = yb[1]  
    res[3] = yb[3]  
    return res

def execute_shooting_solver(v_bar_sq_array, R_data, Rd, R_target):
    r_mesh = np.linspace(1e-4, R_REMOVED_INF, 150)
    
    def ivp_rhs(r, y):
        Y_col = y.reshape((4, 1))
        return tgs_field_ode(np.array([r]), Y_col, v_bar_sq_array, R_data, Rd).flatten()
    
    best_residual = float('inf')
    best_sol = None
    
    for b_center in np.linspace(0.01, 0.4, 5):
        for chi_center in np.linspace(0.2, 1.8, 5):
            y_init = [b_center, 0.0, chi_center, 0.0]
            sol_ivp = solve_ivp(ivp_rhs, (1e-4, R_REMOVED_INF), y_init, t_eval=r_mesh, method='Radau', rtol=1e-4)
            if sol_ivp.success:
                res_end = np.sqrt(sol_ivp.y[1, -1]**2 + sol_ivp.y[3, -1]**2)
                if res_end < best_residual:
                    best_residual = res_end
                    best_sol = sol_ivp
                    
    if best_sol is not None:
        B_ret = np.interp(R_target, best_sol.t, best_sol.y[0])
        chi_ret = np.interp(R_target, best_sol.t, best_sol.y[2])
        dchi_ret = np.interp(R_target, best_sol.t, best_sol.y[3])
        return B_ret, chi_ret, dchi_ret, True
        
    return np.ones_like(R_target) * XI_VOID, np.ones_like(R_target) * 0.5, np.zeros_like(R_target), False

# ======================= ВСПОМОГАТЕЛЬНЫЕ МОДУЛИ =======================
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
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            clean = line.strip()
            if not clean or clean.startswith('#') or 'TAS' in clean: continue
            parts = clean.split()
            if len(parts) < 6: continue
            try:
                row_floats = [float(x) if x.upper() not in ('NAN', 'NONE', 'INDEF') else 0.0 for x in parts[:6]]
                if len(row_floats) == 6 and row_floats[0] > 0.0:
                    rows.append(row_floats)
            except ValueError:
                continue
    if len(rows) < 3: return None
    arr = np.array(rows, dtype=float)
    return {'R': arr[:, 0], 'Vobs': arr[:, 1], 'Vgas': arr[:, 3], 'Vdisk': arr[:, 4], 'Vbul': arr[:, 5]}

# ======================= 3. ДВИЖОК МЕТРИЧЕСКОЙ СБОРКИ =======================
def calculate_galaxy_bvp_dynamic(data):
    R, Vobs, Vgas, Vdisk, Vbul = data['R'], data['Vobs'], data['Vgas'], data['Vdisk'], data['Vbul']
    mask = Vobs > 1.0
    if np.sum(mask) < 3:
        return np.zeros_like(Vobs), 999.0, 0.0, 0.5, 0.0, 0.0, False, np.array([]), np.array([])
    
    R_m, V_o = R[mask], Vobs[mask]
    R_max = float(R[-1])
    
    v_bar_sq_array = W_DISK * Vdisk**2 + 0.95 * Vbul**2 + W_GAS * Vgas**2
    v_bar_sq_array = np.maximum(v_bar_sq_array, 1e-6)
    
    sum_v2 = max(float(np.sum(v_bar_sq_array)), 1e-6)
    Rd_kpc = max(float(np.sum(R * v_bar_sq_array) / (sum_v2 * 2.15)), 0.5)
    
    # ИСПРАВЛЕНИЕ: numpy 2.0 совместимость (замена np.trapz на np.trapezoid)
    if hasattr(np, 'trapezoid'):
        M_total = np.trapezoid(v_bar_sq_array * R, x=R) / G_KPC
    else:
        M_total = np.trapz(v_bar_sq_array * R, x=R) / G_KPC
    
    r_mesh = np.linspace(1e-4, R_REMOVED_INF, 150)
    Y_guess = np.zeros((4, len(r_mesh)))
    Y_guess[0, :] = XI_VOID
    Y_guess[2, :] = 0.5
    
    ode_lambda = lambda r, Y: tgs_field_ode(r, Y, v_bar_sq_array, R, Rd_kpc)
    jac_lambda = lambda r, Y: tgs_field_jacobian(r, Y, v_bar_sq_array, R, Rd_kpc)
    
    sol = solve_bvp(ode_lambda, tgs_boundary_conditions, r_mesh, Y_guess, fun_jac=jac_lambda, tol=1e-3, max_nodes=400)
    
    if not sol.success or np.any(np.isnan(sol.y)):
        B_dynamic, chi_dynamic, dchi_dynamic, was_shoot = execute_shooting_solver(v_bar_sq_array, R, Rd_kpc, R)
    else:
        B_dynamic = np.interp(R, sol.x, sol.y[0])
        chi_dynamic = np.interp(R, sol.x, sol.y[2])
        dchi_dynamic = np.interp(R, sol.x, sol.y[3])
        was_shoot = False
    
    Sigma_g = M_total / (np.pi * Rd_kpc**2)
    s_ratio = Sigma_g / SIGMA_CRIT
    
    rho_local = M_total / ((4.0/3.0) * np.pi * (R**3)) / 1e9
    x_phase_local = rho_local / 0.005
    
    v_mod = np.zeros_like(R)
    for i, r_val in enumerate(R):
        r_safe = max(float(r_val), 1e-6)
        phi_n_astro = (G_KPC * (M_total / 1e9)) / (r_safe * (299792.458**2))
        gamma_r = 1.0 + 1.2 * (1.0 - np.exp(-phi_n_astro * EDD_SCALE))
        
        bessel_mod = 1.0 + ETA_SKIN * chi_dynamic[i] * B_dynamic[i] * gamma_r
        
        g_N = (v_bar_sq_array[i] * (KMS_TO_MS ** 2)) / (r_safe * KPC_TO_M)
        y_arg = max(g_N / a_sh0, 1e-12)
        nu = np.sqrt(0.5 * (1.0 + np.sqrt(1.0 + 4.0 / (y_arg ** 2))))
        
        v_mod_point = np.sqrt(max(nu * v_bar_sq_array[i], 0.0)) * bessel_mod
        v_mod[i] = v_mod_point + GAMMA_SCR * (v_mod_point**2) / 299792.458
        
    udg_factor = 1.0 / (1.0 + s_ratio)
    phi_ptolemy = (R * np.pi) / (Rd_kpc * SC_SETA)
    ptolemy_vortex = np.abs(np.cos(phi_ptolemy) + XI_ATOM * np.sin(phi_ptolemy))
    lam_catch = (1.0 - (XI_ATOM + (XI_VOID - XI_ATOM)/(1.0 + y_arg))) * (1.0 - np.exp(-R/(np.pi*Rd_kpc)))
    
    v_pred = np.zeros_like(R)
    for i in range(len(R)):
        xi_link = GAMMA_DISS * (x_phase_local[i] / (L_PLANCK**2)) * FLUID_SCALE
        xi_link = np.clip(xi_link, 1e-5, 0.85)
        
        V_shel_raw = xi_link * np.abs(dchi_dynamic[i]) * ptolemy_vortex[i] * lam_catch[i] * ETA_KPD * 1e6
        V_shel = V_shel_raw * np.exp(-DAMP_P * x_phase_local[i]) * udg_factor
        
        v_pred[i] = np.maximum(v_mod[i] - V_shel, 1e-3)
    
    v_pred_m = v_pred[mask]
    smape_array = 100.0 * (2.0 * np.abs(v_pred_m - V_o) / (v_pred_m + V_o))
    smape = float(np.median(smape_array))
    bias = float(np.mean(v_pred_m - V_o))
    
    v_N2_base = Vgas**2 + Vdisk**2 + Vbul**2
    frac_gas = float(np.mean((Vgas**2) / np.maximum(v_N2_base, 1e-12)))
    
    v_baryon_curve = np.sqrt(v_bar_sq_array[mask])
    
    return v_pred_m, smape, bias, Rd_kpc, frac_gas, s_ratio, was_shoot, R_m, v_baryon_curve

# ========================= ОСНОВНОЙ КОНТУР ================================
def main():
    desktop = get_desktop_path()
    sparc_folder = find_sparc_folder()
    report_path  = os.path.join(desktop, "TGS_v2.0_Epicycles_Report.txt")
    plots_dir    = os.path.join(desktop, "TGS_v2.0_Plots")
    os.makedirs(plots_dir, exist_ok=True)

    if not sparc_folder:
        print(f"ОШИБКА: Папка 'SPARC' не найдена.")
        sys.exit(1)

    meta = load_metadata()
    files = sorted(glob.glob(os.path.join(sparc_folder, '*rotmod.dat')))
    if not files:
        print(f"ОШИБКА: В папке нет файлов *_rotmod.dat.")
        sys.exit(1)

    print(f"Найдено файлов: {len(files)}. Запуск расчёта Квантовый рубеж v2.0...")
    
    rows_log = []
    all_mapes = []
    all_biases = []
    shoot_count = 0

    for idx, path in enumerate(files):
        data = parse_sparc_file(path)
        if not data: continue
        key = os.path.basename(path).replace('_rotmod.dat', '').replace('.dat', '').upper()
        norm_key = normalize_name(key)
        
        q_flag, inc, dist = meta.get(key, meta.get(norm_key, (0, 45.0, 10.0)))
        
        v_pred_m, smape, bias, rd, f_gas, s_glob, was_shoot, R_m, v_bar_m = calculate_galaxy_bvp_dynamic(data)
        if smape >= 999.0: continue
        
        if was_shoot: shoot_count += 1
        all_mapes.append(smape)
        all_biases.append(bias)
        
        rows_log.append({
            'name': key, 'smape': smape, 'bias': bias,
            'f_gas': f_gas, 's_glob': s_glob, 'rd': rd, 'q': q_flag,
            'inc': inc, 'dist': dist
        })
        
        # Построение графиков
        try:
            fig, ax = plt.subplots(figsize=(7, 4.5), dpi=120)
            mask = data['Vobs'] > 1.0
            ax.plot(data['R'][mask], data['Vobs'][mask], 'ko', markersize=4.5, label='V_obs (SPARC)')
            ax.plot(R_m, v_pred_m, 'r-', linewidth=1.8, label=f'V_TGS v2.0 (sMAPE={smape:.1f}%)')
            ax.plot(R_m, v_bar_m, 'b--', linewidth=1.2, alpha=0.7, label='V_baryon')
            
            ax.set_title(f"{key} | Q={q_flag} | Inc={inc:.1f}° | D={dist:.1f} Mpc", fontsize=11, fontweight='bold')
            ax.set_xlabel('Радиус R [кпк]', fontsize=9)
            ax.set_ylabel('Круговая скорость V [км/с]', fontsize=9)
            ax.grid(True, linestyle=':', alpha=0.5)
            ax.legend(loc='lower right', fontsize=8)
            fig.tight_layout()
            
            fig.savefig(os.path.join(plots_dir, f"{key}_rotcurve.png"))
            plt.close(fig)
        except Exception:
            plt.close('all')
        
        if (idx + 1) % 25 == 0:
            print(f"  ... обработано {idx + 1}/{len(files)} систем")

    rows_log.sort(key=lambda x: x['smape'], reverse=True)
    pure_mapes = np.array([r['smape'] for r in rows_log])
    pure_biases = np.array([r['bias'] for r in rows_log])
    
    median_mape = float(np.median(pure_mapes))
    mean_mape   = float(np.mean(pure_mapes))
    median_bias = float(np.median(pure_biases))

    with open(report_path, 'w', encoding='utf-8') as f:
        def W(*args):
            f.write(' '.join(str(x) for x in args) + '\n')
        LN = 100

        W('='*LN)
        W(' ПРОГРАММНО-МАТЕМАТИЧЕСКИЙ КОМПЛЕКС «КВАНТОВЫЙ РУБЕЖ v2.0» ПТОЛЕМЕЕВСКИЙ РЕФАКТОРИНГ ОДУ2')
        W(f' ОБРАБОТКА СИСТЕМ: {len(rows_log)} | РЕЖИМ: СТРОГИЙ АСИМПТОТИЧЕСКИЙ BVP И ПТОЛЕМЕЕВЫ ЭПИЦИКЛЫ')
        W(' Разработчик: Самохвалов В.В. & Аналитический центр "Брейншторм", 2026 г.')
        W(' ВХОД: папка SPARC на Рабочем столе (содержит файлы *_rotmod.dat)')
        W('       файл Table1.mrt на Рабочем столе (содержит флаги качества Q, Inc, Dist)')
        W(f' ВЫХОД: TGS_v2.0_Epicycles_Report.txt на Рабочем столе')
        W(f'        папка TGS_v2.0_Plots на Рабочем столе ({len(rows_log)} графиков кривых вращения)')
        W('='*LN)

        W('')
        W(f' Общая медиана sMAPE (Мэйп): {median_mape:.2f}%')
        W(f' Среднее значение sMAPE    : {mean_mape:.2f}%')
        W(f' Медианный сдвиг (Bias)    : {median_bias:+.2f} км/с')

        W('')
        W(' РАСПРЕДЕЛЕНИЕ ТОЧНОСТИ НЕВОДА:')
        for thr in (5, 10, 15, 25, 50):
            n_ok = int(np.sum(pure_mapes <= thr))
            W(f'   до {thr:2d}% : {n_ok:3d} из {len(rows_log)}  ({100.0*n_ok/len(rows_log):5.1f}%)')

        W('')
        W(' АНАЛИЗ ЗАВИСИМОСТИ ТОЧНОСТИ ОТ КАЧЕСТВА ДАННЫХ Q (Флаги Table1):')
        W('-'*LN)
        W('   Категория Q   |   Количество   |   Медиана sMAPE   |   Средний Bias')
        W('-'*LN)
        for q_val in (1, 2, 3):
            q_list = [r for r in rows_log if r['q'] == q_val]
            if q_list:
                q_smapes = [r['smape'] for r in q_list]
                q_biases = [r['bias'] for r in q_list]
                W(f"       Q={q_val}       |       {len(q_list):3d}      |        {np.median(q_smapes):5.2f}%      |       {np.mean(q_biases):+6.2f} км/с")
        W('-'*LN)

        W('')
        W(' ТОП-10 ХУДШИХ СИСТЕМ ВЫВЕРНУТОГО ПОГРАНИЧНОГО СЛОЯ:')
        for rank, r in enumerate(rows_log[:10], 1):
            W(f"   [{rank:2d}] {r['name']:<12s} sMAPE={r['smape']:6.2f}% | bias={r['bias']:+6.1f} км/с | Q={r['q']} | Inc={r['inc']:5.1f}° | D={r['dist']:5.1f} Мпк")

        W('')
        W('='*LN)
        W(' ПОГАЛАКТИЧНАЯ ТАБЛИЦА РЕНОРМАЛИЗАЦИОННОЙ СБОРКИ СКОРОСТЕЙ (v2.0 Ptol)')
        W(' Галактика      |   sMAPE  |      Bias     |  f_gas |   s_glob |     Rd   |  Q')
        W('-'*LN)
        rows_log.sort(key=lambda x: x['name'])
        for r in rows_log:
            W(f"{r['name']:<14s} |   {r['smape']:5.2f}  |     {r['bias']:+8.2f}  |  {r['f_gas']:4.2f}  |   {r['s_glob']:6.2f} |   {r['rd']:5.2f}  |  {r['q']}")
        W('='*LN)

    print(f"\n[Завершено]: Отчет выгружен на Рабочий стол:\n >> {report_path}")
    print(f"[Завершено]: Графики сохранены в папку: {plots_dir}")

if __name__ == '__main__':
    main()
