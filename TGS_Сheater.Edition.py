#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===========================================================================
 ПРОГРАММНО-МАТЕМАТИЧЕСКИЙ КОМПЛЕКС «КВАНТОВЫЙ РУБЕЖ» Cheater.Edition
 ОБРАБОТКА СИСТЕМ: 175 | РЕЖИМ: БОНУСНАЯ ВЕРСИЯ (СЛЕПОЙ ПОДБОР SCALE)
 
 Разработчик: Самохвалов В.В. & Аналитический центр «Брейншторм», 2026 г.
 Лицензия: GNU General Public License v3 (GNU GPL v3)
 Маркер штурма: DOMASHKA_04.11.2026

 ВХОД:  папка SPARC на Рабочем столе (содержит файлы *_rotmod.dat)
        файл Table1.mrt на Рабочем столе (содержит флаги качества Q, Inc, Dist)
 ВЫХОД: TGS_Cheater_Report.txt на Рабочем столе
        папка TGS_Cheater_Plots на Рабочем столе
===========================================================================
"""

import os
import re
import sys
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ===================== 1. ФУНДАМЕНТАЛЬНЫЕ ИНВАРИАНТЫ ТГС =====================
G_SI    = 6.67430e-11                      # гравитационная постоянная, м³/(кг·с²)
c_SI    = 299792458.0                      # скорость света, м/с
H0_SI   = 67.4 * 1000.0 / 3.085677581e22   # пульс ЦБД (постоянная Хаббла, Planck 2018), 1/с
a_sh0   = (c_SI * H0_SI) / (2.0 * np.pi)   # ускорение шелеста координат, ~1.0422e-10 м/с²

R1_core   = 0.5**(1.0 / 3.0)               # радиус архивного Ядра, ~0.7937
eta_skin  = 1.0 - R1_core                  # вычислительный КПД Кожуры, ~0.2063
g_top     = 1.0 / (16.0 * np.pi**2)        # топологическая константа связи
xi_space  = 0.63460                        # базовая пористость решётки вакуума
gamma_tgs = 1.438151512351216              # фактор инверсии информационной ёмкости (I_TGS)

KPC_TO_M  = 3.085677581e19                 # кпк -> м
KMS_TO_MS = 1000.0                         # км/с -> м/с
M_SUN     = 1.98847e30                     # масса Солнца, кг
G_ASTRO   = G_SI * M_SUN / (KPC_TO_M * 1e6)# G в астрофизических единицах
rho_crit  = 0.005                          # порог деструкции ячеек, М_солнца/пк³


# ===================== 2. ВСПОМОГАТЕЛЬНЫЕ МОДУЛИ =====================
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


# ===================== 3. САМОСОГЛАСОВАННЫЙ ДВИЖОК =====================
def tgs_blind_velocity(data, scale_tgs, inc_angle_obs, dist_mpc, q_flag):
    """Кривая вращения «вслепую» при заданном масштабе сети scale_tgs."""
    scale_tgs = float(np.squeeze(scale_tgs))

    R, Vobs, Vgas, Vdisk, Vbul = data['R'], data['Vobs'], data['Vgas'], data['Vdisk'], data['Vbul']

    v_bar_raw_sq = 0.5 * Vdisk**2 + 0.5 * Vbul**2 + Vgas**2
    v_bar_raw_sq = np.maximum(v_bar_raw_sq, 1e-6)

    sum_v2 = max(float(np.sum(v_bar_raw_sq)), 1e-6)
    Rd_kpc = max(float(np.sum(R * v_bar_raw_sq) / (sum_v2 * 2.15)), 0.5)
    M_total_galaxy = float(np.sum(v_bar_raw_sq) / G_ASTRO)

    V_eff = (4.0 / 3.0) * np.pi * ((3.0 * Rd_kpc)**3)
    rho_avg = (M_total_galaxy / V_eff) / 1e9
    x_phase = rho_avg / rho_crit
    chi = np.log(1.0 + np.sqrt(max(x_phase, 0.0)))

    smooth_gate = 1.0 / (1.0 + np.exp((Rd_kpc - 1.45) / 0.15))
    damping_amplitude = 0.425 * (1.0 - np.exp(-x_phase)) * smooth_gate

    v_mod = np.zeros_like(R)

    for i, r in enumerate(R):
        r_safe = max(float(r), 1e-6)
        rho = r_safe / Rd_kpc

        bessel_arg = rho / (np.pi * xi_space * gamma_tgs * scale_tgs)
        bessel_stretch = g_top * (np.sin(bessel_arg + 1e-6) / (bessel_arg + 1e-6))

        Y_disk_adaptive = 0.5 + eta_skin * chi * np.exp(-rho / (3.2 * scale_tgs)) * (1.0 - np.exp(-rho)) + bessel_stretch
        landau_damping = damping_amplitude * (np.sin(rho / 1.15) / (rho + 1e-6)) * np.exp(-rho / 3.5)
        Y_disk_adaptive = max(Y_disk_adaptive - landau_damping, eta_skin)

        Y_bul_adaptive = 0.5 + 0.20 * chi * np.exp(-rho)
        Y_bul_adaptive = max(Y_bul_adaptive, 0.5)

        v_bar_sq = Y_disk_adaptive * Vdisk[i]**2 + Y_bul_adaptive * Vbul[i]**2 + Vgas[i]**2
        v_bar_sq = max(v_bar_sq, 1e-6)

        R_m = max(r_safe * KPC_TO_M, 1e10)
        g_N = (v_bar_sq * (KMS_TO_MS ** 2)) / R_m

        y = max(g_N / a_sh0, 1e-12)
        nu = np.sqrt(0.5 * (1.0 + np.sqrt(1.0 + 4.0 / (y ** 2))))

        bias_compensator = 1.0 + (g_top * gamma_tgs) * (1.0 - np.exp(-rho / 5.0))
        v_mod[i] = np.sqrt(max(nu * v_bar_sq, 0.0)) * bias_compensator

    v_bar_curve = np.sqrt(v_bar_raw_sq)
    return v_mod, v_bar_curve, chi, x_phase, Rd_kpc

def calculate_galaxy_tgs_cheater(data, inc_angle_obs, dist_mpc, q_flag):
    R, Vobs = data['R'], data['Vobs']
    mask = Vobs > 1.0
    if np.sum(mask) < 3:
        return None

    def internal_theory_loss_log(ln_scale):
        ln_scale_scalar = float(np.squeeze(ln_scale))
        scale = np.exp(ln_scale_scalar)
        _, _, chi, x_phase, Rd_kpc = tgs_blind_velocity(data, scale, inc_angle_obs, dist_mpc, q_flag)
        core_scale_factor = gamma_tgs * eta_skin * scale
        vacuum_tension_leak = np.abs(xi_space * (chi**2) - x_phase * np.exp(-Rd_kpc / (core_scale_factor + 1e-5)))
        penalty = 1e-7 * (ln_scale_scalar ** 2)
        return float(vacuum_tension_leak + penalty)

    res = minimize(internal_theory_loss_log, x0=[1.0], method='Nelder-Mead')
    best_scale = float(np.exp(np.squeeze(res.x)))

    v_mod, v_bar, _, _, Rd_kpc = tgs_blind_velocity(data, best_scale, inc_angle_obs, dist_mpc, q_flag)

    return {
        'R': R[mask], 'Vobs': Vobs[mask], 'v_tgs': v_mod[mask], 'v_bar': v_bar[mask],
        'scale': best_scale, 'rd': Rd_kpc
    }


# ===================== 4. ЗАПУСК И ОТЧЁТ =====================
def main():
    sparc_dir = find_sparc_folder()
    if not sparc_dir:
        print("Папка SPARC не найдена.")
        return 1
        
    desktop_dir = get_desktop_path()
    meta = load_metadata()
    files = sorted(glob.glob(os.path.join(sparc_dir, "*_rotmod.dat")))
    
    report_txt_path = os.path.join(desktop_dir, "TGS_Cheater_Report.txt")
    plots_dir = os.path.join(desktop_dir, "TGS_Cheater_Plots")
    os.makedirs(plots_dir, exist_ok=True)

    print("===========================================================================")
    print(f" ЗАПУСК: ПРОГРАММНО-МАТЕМАТИЧЕСКИЙ КОМПЛЕКС «КВАНТОВЫЙ РУБЕЖ» Cheater.Edition на {len(files)} системах...")
    print("===========================================================================")

    all_tgs_err = []
    all_tgs_diff = []
    galaxy_summaries = []

    for idx, path in enumerate(files):
        data = parse_sparc_file(path)
        if not data:
            continue
        key = os.path.basename(path).replace('_rotmod.dat', '').replace('.dat', '').upper()
        norm_key = normalize_name(key)
        q_flag, inc, dist = meta.get(key, meta.get(norm_key, (0, 45.0, 10.0)))

        res = calculate_galaxy_tgs_cheater(data, inc, dist, q_flag)
        if not res or len(res['R']) < 3:
            continue

        # sMAPE (симметричная ошибка)
        tgs_pe = 100.0 * (2.0 * np.abs(res['v_tgs'] - res['Vobs']) / (res['v_tgs'] + res['Vobs']))
        all_tgs_err.extend(tgs_pe)
        all_tgs_diff.extend(res['v_tgs'] - res['Vobs'])

        galaxy_smape = float(np.median(tgs_pe))
        galaxy_bias = float(np.mean(res['v_tgs'] - res['Vobs']))

        galaxy_summaries.append({
            'name': key, 'q': q_flag, 'points': len(tgs_pe),
            'smape': galaxy_smape, 'bias': galaxy_bias,
            'scale': res['scale'], 'rd': res['rd'],
            'dist': dist, 'inc_obs': inc
        })

        try:
            fig, ax = plt.subplots(figsize=(7, 4.5), dpi=120)
            ax.plot(res['R'], res['Vobs'], 'ko', markersize=4.5, label='V_obs (SPARC)')
            ax.plot(res['R'], res['v_tgs'], 'r-', linewidth=1.8, label=f'V_TGS Cheater (sMAPE={galaxy_smape:.1f}%)')
            ax.plot(res['R'], res['v_bar'], 'b--', linewidth=1.2, alpha=0.7, label='V_baryon (base)')
            
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
    overall_median_scale = float(np.median([g['scale'] for g in galaxy_summaries]))

    galaxy_summaries.sort(key=lambda x: x['smape'], reverse=True)

    with open(report_txt_path, 'w', encoding='utf-8') as f:
        def W(*args):
            f.write(' '.join(str(x) for x in args) + '\n')
        LN = 105
        W('='*LN)
        W(' ПРОГРАММНО-МАТЕМАТИЧЕСКИЙ КОМПЛЕКС «КВАНТОВЫЙ РУБЕЖ» Cheater.Edition')
        W(f' ОБРАБОТКА СИСТЕМ: {len(galaxy_summaries)} | РЕЖИМ: БОНУСНАЯ ВЕРСИЯ (СЛЕПОЙ ПОДБОР)')
        W(' Разработчик: Самохвалов В.В. & Аналитический центр "Брейншторм", 2026 г.')
        W(' ВХОД: папка SPARC на Рабочем столе (содержит файлы *_rotmod.dat)')
        W('       файл Table1.mrt на Рабочем столе (содержит флаги качества Q, Inc, Dist)')
        W(' ВЫХОД: TGS_Cheater_Report.txt на Рабочем столе')
        W('='*LN)
        W('')
        W(f' Общая медиана sMAPE (Мэйп): {overall_median_smape:.2f}% | {pointwise_median:.2f}% (ансамблевый поточечный анализ)')
        W(f' Среднее значение sMAPE    : {overall_mean_smape:.2f}% | {pointwise_mean:.2f}% (дискретная поточечная верификация)')
        W(f' Медианный сдвиг (Bias)    : {np.median(all_tgs_diff):+.2f} км/с')
        W(f' Медианный масштаб scale   : {overall_median_scale:.3f}')
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
        W(' ТОП-10 ХУДШИХ СИСТЕМ (СЛЕПАЯ ОПТИМИЗАЦИЯ):')
        for rank, g in enumerate(galaxy_summaries[:10], 1):
            W(f"   [{rank:2d}] {g['name']:<12s} sMAPE={g['smape']:6.2f}% | bias={g['bias']:+6.1f} км/с | Q={g['q']} | Inc={g['inc_obs']:5.1f}° | scale={g['scale']:6.3f}")
        W('')
        W('='*LN)
        W(' ПОГАЛАКТИЧНАЯ ТАБЛИЦА СБОРКИ СКОРОСТЕЙ (Cheater.Edition)')
        W(' Галактика      |   sMAPE  |      Bias     |  scale |     Rd   |  Q')
        W('-'*LN)
        galaxy_summaries.sort(key=lambda x: x['name'])
        for g in galaxy_summaries:
            W(f"{g['name']:<14s} |   {g['smape']:5.2f}  |     {g['bias']:+8.2f}  |  {g['scale']:5.3f} |   {g['rd']:5.2f}  |  {g['q']}")
        W('='*LN)

    print(f"\n[ГОТОВО]: Общая медиана sMAPE: {overall_median_smape:.2f}% | Поточечная медиана: {pointwise_median:.2f}%")
    print(f"[ГОТОВО]: Отчет выгружен: {report_txt_path}")
    print(f"[ГОТОВО]: Графики сохранены в папку: {plots_dir}")
    return 0

if __name__ == '__main__':
    sys.exit(main())