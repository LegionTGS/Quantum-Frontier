#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===========================================================================
 ПРОГРАММНО-МАТЕМАТИЧЕСКИЙ КОМПЛЕКС «КВАНТОВЫЙ РУБЕЖ v1.0»
 ОБРАБОТКА СИСТЕМ: 175
 
 Разработчик: Самохвалов В.В. & Аналитический центр «Брейншторм», 2026 г.
 Лицензия: GNU General Public License v3 (GNU GPL v3)
 Маркер штурма: DOMASHKA_04.11.2026

 ВХОД:  папка SPARC на Рабочем столе (содержит файлы *_rotmod.dat)
        файл Table1.mrt на Рабочем столе (содержит фотометрию и флаги Q, Inc, Dist)
 ВЫХОД: TGS_QF_Report.txt на Рабочем столе
        папка TGS_QF_Plots на Рабочем столе
===========================================================================
"""

import os
import re
import sys
import glob
import numpy as np
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ===================== 1. ФУНДАМЕНТАЛЬНЫЕ ИНВАРИАНТЫ ТГС =====================
G_SI    = 6.67430e-11                      # гравитационная постоянная, м³/(кг·с²)
c_SI    = 299792458.0                      # скорость света, м/с
H0_SI   = 67.4 * 1000.0 / 3.085677581e22   # пульс ЦБД (постоянная Хаббла, Planck 2018), 1/с
a_sh0   = (c_SI * H0_SI) / (2.0 * np.pi)   # ускорение шелеста координат, ~1.0422e-10 м/с²

R1_core   = 0.5**(1.0 / 3.0)               
eta_skin  = 1.0 - R1_core                  
XI_ATOM   = eta_skin**2 / (2.0*math.sqrt(2.0))
XI_VOID   = eta_skin / math.sqrt(2.0)

A0        = 3215.89                        
G_KPC     = 4.302e-6                       
SIGMA_CRIT= A0 / (2.0*math.pi*G_KPC)       
OMEGA_CONST = 5.4e-16                      
SC_SETA   = 1.0/eta_skin                   

I_TGS     = 1.438151512351216
W_GAS     = I_TGS
W_DISK    = 0.60033558
ANISO     = 0.29614151
PHI_COV   = 4.00000000
GAMMA_SCR = 6.7706e-06
ML36      = 0.50
HEX       = 1.33

ETA_KPD   = 0.16
A_SH_BASE = 45.0
V_CRIT    = 32.4
MIRROR    = 0.2658
DAMP_P    = 1.45

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

def num_or_nan(t):
    try: return float(t)
    except (ValueError, TypeError): return float('nan')

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
                        t = line.split()
                        if len(t) >= 19 and re.match(r'^[A-Za-z]', t[0]):
                            try: Q = int(t[17])
                            except (ValueError, IndexError): Q = None
                            if Q in (1, 2, 3):
                                name = normalize_name(t[0])
                                meta[name] = dict(name=t[0], D=num_or_nan(t[2]), eD=num_or_nan(t[3]),
                                                  Inc=num_or_nan(t[5]), eInc=num_or_nan(t[6]),
                                                  L36=num_or_nan(t[7]), Rdisk=num_or_nan(t[11]),
                                                  MHI=num_or_nan(t[13]), Vflat=num_or_nan(t[15]), Q=Q)
                        else:
                            s = line.ljust(120)
                            def fld(a, b): return s[a-1:b].strip()
                            name_raw = fld(1, 11)
                            if name_raw and re.match(r'^[A-Za-z]', name_raw):
                                try: Q = int(fld(97, 99))
                                except ValueError: Q = None
                                if Q in (1, 2, 3):
                                    name = normalize_name(name_raw)
                                    meta[name] = dict(name=name_raw, D=num_or_nan(fld(14,19)), eD=num_or_nan(fld(20,24)),
                                                      Inc=num_or_nan(fld(27,30)), eInc=num_or_nan(fld(31,34)),
                                                      L36=num_or_nan(fld(35,41)), Rdisk=num_or_nan(fld(62,66)),
                                                      MHI=num_or_nan(fld(75,81)), Vflat=num_or_nan(fld(87,91)), Q=Q)
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


# ===================== 3. САМОСОГЛАСОВАННЫЙ ДВИЖОК v1.0 =====================
def synthesize(g, m, mode='bary'):
    R      = g['R']
    R_max  = float(R[-1])
    src = 'dyn'
    Mb = float('nan')
    if mode == 'bary' and m is not None:
        Mb = ML36*m['L36'] + HEX*m['MHI']          
        if np.isfinite(Mb) and Mb > 0: src = 'bary'
        
    if src == 'bary':
        Rd = m['Rdisk'] if (np.isfinite(m['Rdisk']) and m['Rdisk'] > 0) else R_max/3.2
        M_total = Mb * 1e9
    else:
        vb2_out = g['Vgas'][-1]**2 + ML36*(g['Vdisk'][-1]**2 + g['Vbul'][-1]**2)
        Rd = R_max/3.2
        M_total = max(vb2_out, 1e-6) * R_max / G_KPC
        
    V_flat = (G_KPC * M_total * A0)**0.25

    Sigma_g = M_total / (np.pi * Rd**2)
    s_ratio = Sigma_g / SIGMA_CRIT
    piston  = 1.0 / (1.0 + (SIGMA_CRIT/max(Sigma_g, 1e-3))**2)

    rho_pc3 = M_total / ((4.0/3.0)*np.pi*R_max**3) / 1e9
    x_phase = rho_pc3 / 0.005
    chi     = math.log(1.0 + math.sqrt(max(x_phase, 1e-12)))
    chi_clip= chi / (1.0 + s_ratio**2)

    v_N2 = (W_GAS*g['Vgas'])**2 + (W_DISK*g['Vdisk'])**2 + g['Vbul']**2
    a_N  = v_N2 / R
    phi_n   = a_N / A0
    phi_cov = 1.0 + PHI_COV * chi_clip * np.log1p(phi_n * OMEGA_CONST)

    s_loc   = 2.0*V_flat**2 / (A0*R)
    red_loc = s_loc**4 / (s_loc**4 + SC_SETA**4)
    w       = 1.0 + (W_GAS - 1.0)*red_loc

    v_N2m = (w*g['Vgas'])**2 + (W_DISK*g['Vdisk'])**2 + g['Vbul']**2
    a_Nm  = v_N2m / R
    y_m   = a_Nm / A0

    xi_r   = XI_ATOM + (XI_VOID - XI_ATOM)/(1.0 + phi_n)
    w_mond = np.clip(xi_r / XI_VOID, 0.0, 1.0)
    g_mond = np.sqrt(np.maximum(a_Nm, 1e-12) * A0)
    Fk     = 1.0/np.sqrt(1.0 + y_m**2)     

    a_total = (a_N * phi_cov * (1.0 + piston*0.0009)
               + g_mond * w_mond * Fk * (1.0 + ANISO*(1.0 - w_mond)))
    v_model = np.sqrt(np.maximum(a_total*R, 1e-9))
    v_model = v_model + GAMMA_SCR * (v_model**2) / 2.998e5

    udg_factor = 1.0/(1.0 + s_ratio)
    eps = np.exp(-x_phase) * udg_factor * ETA_KPD
    a_sh = A_SH_BASE * (1.0 + MIRROR)
    lam_catch = (1.0 - xi_r) * (1.0 - np.exp(-R/(np.pi*Rd)))
    f_v = 0.5*(1.0 + np.tanh((v_model - V_CRIT)/10.0))
    V_feded = eps * a_sh * np.tanh(R/(2.0*Rd)) * np.exp(-DAMP_P*x_phase) * f_v * lam_catch
    v_pred = v_model - V_feded

    frac_gas = float(np.mean((g['Vgas']**2) / np.maximum(v_N2, 1e-12)))
    
    return dict(v_pred=v_pred, s_ratio=s_ratio, Rd=Rd,
                xi_mean=float(np.mean(xi_r)), frac_gas=frac_gas,
                src=src, V_flat=V_flat, M_total=M_total,
                y_bar=float(np.mean(y_m)))


# ===================== 4. ЗАПУСК И ОТЧЁТ =====================
def main():
    sparc_dir = find_sparc_folder()
    if not sparc_dir:
        print("Папка SPARC не найдена.")
        return 1
        
    desktop_dir = get_desktop_path()
    meta = load_metadata()
    files = sorted(glob.glob(os.path.join(sparc_dir, "*_rotmod.dat")))
    
    report_txt_path = os.path.join(desktop_dir, "TGS_QF_Report.txt")
    plots_dir = os.path.join(desktop_dir, "TGS_QF_Plots")
    os.makedirs(plots_dir, exist_ok=True)

    print("===========================================================================")
    print(f" ЗАПУСК: ПРОГРАММНО-МАТЕМАТИЧЕСКИЙ КОМПЛЕКС «КВАНТОВЫЙ РУБЕЖ v1.0» на {len(files)} системах...")
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
        m = meta.get(norm_key)
        
        q_flag = m['Q'] if m and 'Q' in m else 0
        inc_angle = m['Inc'] if m and 'Inc' in m else 45.0
        dist_mpc = m['D'] if m and 'D' in m else 10.0

        mask = data['Vobs'] > 1.0
        if np.sum(mask) < 3:
            continue
            
        syn_b = synthesize(data, m, 'bary')
        v_mod = syn_b['v_pred']
        
        # sMAPE
        tgs_pe = 100.0 * (2.0 * np.abs(v_mod[mask] - data['Vobs'][mask]) / (v_mod[mask] + data['Vobs'][mask]))
        all_tgs_err.extend(tgs_pe)
        all_tgs_diff.extend(v_mod[mask] - data['Vobs'][mask])

        galaxy_smape = float(np.median(tgs_pe))
        galaxy_bias = float(np.mean(v_mod[mask] - data['Vobs'][mask]))

        galaxy_summaries.append({
            'name': key, 'q': q_flag, 'points': len(tgs_pe),
            'smape': galaxy_smape, 'bias': galaxy_bias,
            'f_gas': syn_b['frac_gas'], 's_glob': syn_b['s_ratio'], 'rd': syn_b['Rd'],
            'dist': dist_mpc, 'inc': inc_angle, 'src': syn_b['src']
        })

        try:
            fig, ax = plt.subplots(figsize=(7, 4.5), dpi=120)
            ax.plot(data['R'], data['Vobs'], 'ko', markersize=4.5, label='V_obs (SPARC)')
            ax.plot(data['R'], v_mod, 'r-', linewidth=1.8, label=f'V_TGS v1.0 (sMAPE={galaxy_smape:.1f}%)')
            v_bar = np.sqrt(data['Vgas']**2 + ML36*data['Vdisk']**2 + 0.7*data['Vbul']**2)
            ax.plot(data['R'], v_bar, 'b--', linewidth=1.2, alpha=0.7, label='V_baryon (base)')
            
            ax.set_title(f"{key} | Q={q_flag} | Inc={inc_angle:.1f}° | D={dist_mpc:.1f} Mpc", fontsize=11, fontweight='bold')
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
        LN = 105
        W('='*LN)
        W(' ПРОГРАММНО-МАТЕМАТИЧЕСКИЙ КОМПЛЕКС «КВАНТОВЫЙ РУБЕЖ v1.0»')
        W(f' ОБРАБОТКА СИСТЕМ: {len(galaxy_summaries)}')
        W(' Разработчик: Самохвалов В.В. & Аналитический центр "Брейншторм", 2026 г.')
        W(' ВХОД: папка SPARC на Рабочем столе (содержит файлы *_rotmod.dat)')
        W('       файл Table1.mrt на Рабочем столе (содержит флаги качества Q, Inc, Dist)')
        W(' ВЫХОД: TGS_QF_Report.txt на Рабочем столе')
        W('='*LN)
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
        W(' ТОП-10 ХУДШИХ СИСТЕМ:')
        for rank, g in enumerate(galaxy_summaries[:10], 1):
            W(f"   [{rank:2d}] {g['name']:<12s} sMAPE={g['smape']:6.2f}% | bias={g['bias']:+6.1f} км/с | Q={g['q']} | Inc={g['inc']:5.1f}° | D={g['dist']:5.1f} Мпк")
        W('')
        W('='*LN)
        W(' ПОГАЛАКТИЧНАЯ ТАБЛИЦА СБОРКИ СКОРОСТЕЙ (v1.0)')
        W(' Галактика      |   sMAPE  |      Bias     |  Inc |     D    |  f_gas |   s_glob |     Rd   |  Q')
        W('-'*LN)
        galaxy_summaries.sort(key=lambda x: x['name'])
        for g in galaxy_summaries:
            W(f"{g['name']:<14s} |   {g['smape']:5.2f}  |     {g['bias']:+8.2f}  | {g['inc']:4.1f} |  {g['dist']:5.1f}  |  {g['f_gas']:5.2f} |  {g['s_glob']:6.2f}  |   {g['rd']:5.2f}  |  {g['q']}")
        W('='*LN)

    print(f"\n[ГОТОВО]: Общая медиана sMAPE: {overall_median_smape:.2f}% | Поточечная медиана: {pointwise_median:.2f}%")
    print(f"[ГОТОВО]: Отчет выгружен: {report_txt_path}")
    print(f"[ГОТОВО]: Графики сохранены в папку: {plots_dir}")
    return 0

if __name__ == '__main__':
    sys.exit(main())