#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Программный комплекс "Квантовый рубеж" (v17.2 Loss Balance, QFcheater Edition)
# Copyright (c) 2026 Самохвалов В.В., Аналитический центр "Брейншторм"
#
# Данное программное обеспечение является открытым: вы можете распространять
# его и/или модифицировать в соответствии с условиями Стандартной общественной
# лицензии GNU (GNU GPL), опубликованной Фондом свободного программного
# обеспечения, версии 3 этой Лицензии либо (по вашему выбору) любой более
# поздней версии. Программа распространяется БЕЗ КАКИХ-ЛИБО ГАРАНТИЙ.
# Полный текст лицензии: файл LICENSE или https://www.gnu.org/licenses/gpl-3.0.html
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Кодовый маркер верификации: DOMASHKA_04.11.2026
'''
===========================================================================
 «КВАНТОВЫЙ РУБЕЖ» v17.2 Loss Balance — бонусная версия (QFcheater Edition)

 Самосогласованный расчёт кривых вращения на базе кода из первой статьи
 автора (https://doi.org/10.24108/preprints-3115230).

 Расчёт «слепой»: наблюдаемые скорости V_набл в подбор не входят.
 Для каждой галактики свободный масштаб сети scale_tgs ищется методом
 Nelder-Mead в логарифмическом пространстве ln(scale) по внутреннему
 функционалу натяжения вакуума (без ограничений по краям диапазона).
 V_набл используются только на последнем шаге — для расчёта ошибки.

 Метрика: MAPE = среднее |V_мод - V_набл| / V_набл * 100% (по точкам
 с V_набл > 1 км/с). Это не sMAPE основного кода KVR_Final.py, поэтому
 цифры двух версий напрямую не сравниваются.

 ВХОД:  каталог SPARC (файлы *_rotmod.dat) — рядом со скриптом, в подпапке
        SPARC_DATA / SPARC или на рабочем столе.
 ВЫХОД: TGS_v17_2_Loss_Balance_Report.txt — на рабочий стол (если папки
        рабочего стола нет — рядом со скриптом).

 Нужны: numpy, scipy.
===========================================================================
'''

import os
import sys
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ===================== 1. ФУНДАМЕНТАЛЬНЫЕ ИНВАРИАНТЫ ТГС =====================
G_SI    = 6.67430e-11                     # гравитационная постоянная, м³/(кг·с²)
c_SI    = 299792458.0                     # скорость света, м/с
H0_SI   = 67.4 * 1000.0 / 3.085677581e22  # пульс ЦБД (постоянная Хаббла, Planck 2018), 1/с
a_sh0   = (c_SI * H0_SI) / (2.0 * np.pi)  # ускорение шелеста координат, ~1.0422e-10 м/с²

# Паспорт Невода (принцип 50/50, как в KVR_Final.py)
R1_core   = 0.5**(1.0 / 3.0)              # радиус архивного Ядра, ~0.7937
eta_skin  = 1.0 - R1_core                 # вычислительный КПД Кожуры, ~0.2063
g_top     = 1.0 / (16.0 * np.pi**2)       # топологическая константа связи
xi_space  = 0.63460                       # базовая пористость решётки вакуума
# Константа Самохваловой (из работы P vs NP), та же, что I_TGS в KVR_Final.py
gamma_tgs = 1.438151512351216             # фактор инверсии информационной ёмкости

KPC_TO_M  = 3.085677581e19                # кпк -> м
KMS_TO_MS = 1000.0                        # км/с -> м/с
M_SUN     = 1.98847e30                    # масса Солнца, кг
G_ASTRO   = G_SI * M_SUN / (KPC_TO_M * 1e6)  # G в астрофизических единицах
rho_crit  = 0.005                         # порог деструкции ячеек, М_солнца/пк³

REPORT_NAME = 'TGS_v17_2_Loss_Balance_Report.txt'

# ===================== ЗАГРУЗКА КАТАЛОГА SPARC =====================
def parse_sparc_file(filepath):
    """Читает профили скоростей компонент галактики из файла *_rotmod.dat."""
    rows = []
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            clean = line.strip()
            if not clean or clean.startswith('#') or 'TAS' in clean:
                continue
            parts = clean.split()
            if len(parts) < 6: continue
            try: rows.append(list(map(float, parts[:6])))
            except ValueError: continue
    if len(rows) < 3: return None
    arr = np.array(rows, dtype=float)
    return {'R': arr[:, 0], 'Vobs': arr[:, 1], 'Vgas': arr[:, 3], 'Vdisk': arr[:, 4], 'Vbul': arr[:, 5]}

def find_desktop(fallback):
    if os.environ.get('QF_OUTPUT_DIR'):
        os.makedirs(os.environ['QF_OUTPUT_DIR'], exist_ok=True)
        return os.path.abspath(os.environ['QF_OUTPUT_DIR'])
    """Рабочий стол пользователя: Windows (в т.ч. под OneDrive), macOS, Linux."""
    home = os.path.expanduser('~')
    cand = []
    if os.name == 'nt':
        for env in ('OneDrive', 'OneDriveConsumer', 'OneDriveCommercial'):
            if os.environ.get(env):
                cand.append(os.path.join(os.environ[env], 'Desktop'))
                cand.append(os.path.join(os.environ[env], 'Рабочий стол'))
        if os.environ.get('USERPROFILE'):
            cand.append(os.path.join(os.environ['USERPROFILE'], 'Desktop'))
    if os.environ.get('XDG_DESKTOP_DIR'):
        cand.append(os.environ['XDG_DESKTOP_DIR'])
    cand += [os.path.join(home, 'Desktop'), os.path.join(home, 'Рабочий стол')]
    for d in cand:
        if os.path.isdir(d): return d
    return fallback

def find_sparc_files(here):
    """Ищет *_rotmod.dat рядом со скриптом, в подпапках и на рабочем столе."""
    desktop = find_desktop(here)
    for base in (here, desktop):
        for sub in ('', 'SPARC_DATA', 'SPARC', 'sparc', 'data'):
            files = sorted(glob.glob(os.path.join(base, sub, '*_rotmod.dat')))
            if files: return files
    found = []
    for root, _, _ in os.walk(here):
        found += glob.glob(os.path.join(root, '*_rotmod.dat'))
    return sorted(found)

def load_sparc(files):
    gals = {}
    for f in files:
        name = os.path.basename(f).replace('_rotmod.dat', '')
        data = parse_sparc_file(f)
        if data: gals[name] = data
    return gals

# ===================== 2. САМОСОГЛАСОВАННЫЙ ДВИЖОК v17.2 =====================
def tgs_blind_velocity(data, scale_tgs):
    """Кривая вращения «вслепую» при заданном масштабе сети scale_tgs."""
    scale_tgs = float(np.squeeze(scale_tgs))

    R, Vgas, Vdisk, Vbul = data['R'], data['Vgas'], data['Vdisk'], data['Vbul']

    # Барионный вход с базовым M/L = 0.5 для диска и балджа
    v_bar_raw_sq = 0.5 * Vdisk**2 + 0.5 * Vbul**2 + Vgas**2
    v_bar_raw_sq = np.maximum(v_bar_raw_sq, 1e-6)

    # Дисковый масштаб и масса по профилю барионных скоростей
    sum_v2 = max(float(np.sum(v_bar_raw_sq)), 1e-6)
    Rd_kpc = max(float(np.sum(R * v_bar_raw_sq) / (sum_v2 * 2.15)), 0.5)
    M_total_galaxy = float(np.sum(v_bar_raw_sq) / G_ASTRO)

    # Фазовая плотность внутри 3*R_d и глубина связности chi
    V_eff = (4.0 / 3.0) * np.pi * ((3.0 * Rd_kpc)**3)
    rho_avg = (M_total_galaxy / V_eff) / 1e9
    x_phase = rho_avg / rho_crit
    chi = np.log(1.0 + np.sqrt(max(x_phase, 0.0)))

    # Затухание Ландау включается только у компактных дисков (R_d < ~1.45 кпк)
    smooth_gate = 1.0 / (1.0 + np.exp((Rd_kpc - 1.45) / 0.15))
    damping_amplitude = 0.425 * (1.0 - np.exp(-x_phase)) * smooth_gate

    v_mod = np.zeros_like(R)

    for i, r in enumerate(R):
        r_safe = max(float(r), 1e-6)
        rho = r_safe / Rd_kpc

        # Модуляция шага решётки через масштаб scale_tgs (бесселева растяжка)
        bessel_arg = rho / (np.pi * xi_space * gamma_tgs * scale_tgs)
        bessel_stretch = g_top * (np.sin(bessel_arg + 1e-6) / (bessel_arg + 1e-6))

        # Адаптивное M/L диска: рост к центру, растяжка, затухание Ландау
        Y_disk_adaptive = 0.5 + eta_skin * chi * np.exp(-rho / (3.2 * scale_tgs)) * (1.0 - np.exp(-rho)) + bessel_stretch
        landau_damping = damping_amplitude * (np.sin(rho / 1.15) / (rho + 1e-6)) * np.exp(-rho / 3.5)
        Y_disk_adaptive = max(Y_disk_adaptive - landau_damping, eta_skin)

        # Адаптивное M/L балджа (не ниже 0.5)
        Y_bul_adaptive = 0.5 + 0.20 * chi * np.exp(-rho)
        Y_bul_adaptive = max(Y_bul_adaptive, 0.5)

        v_bar_sq = Y_disk_adaptive * Vdisk[i]**2 + Y_bul_adaptive * Vbul[i]**2 + Vgas[i]**2
        v_bar_sq = max(v_bar_sq, 1e-6)

        # Ньютоновское ускорение и отклик шелеста nu(y), y = g_N / a_sh0
        R_m = max(r_safe * KPC_TO_M, 1e10)
        g_N = (v_bar_sq * (KMS_TO_MS ** 2)) / R_m

        y = max(g_N / a_sh0, 1e-12)
        nu = np.sqrt(0.5 * (1.0 + np.sqrt(1.0 + 4.0 / (y ** 2))))

        bias_compensator = 1.0 + (g_top * gamma_tgs) * (1.0 - np.exp(-rho / 5.0))
        v_mod[i] = np.sqrt(max(nu * v_bar_sq, 0.0)) * bias_compensator

    return v_mod, chi, x_phase, Rd_kpc

def calculate_galaxy_tgs_v17_2(data):
    """Подбор scale_tgs по внутреннему функционалу натяжения (без V_набл)."""
    R, Vobs = data['R'], data['Vobs']
    mask = Vobs > 1.0
    if np.sum(mask) < 3:
        return np.zeros_like(Vobs), 999.0, 0.0, 1.0

    # Поиск ведётся по ln(scale), поэтому масштаб не упирается в границы
    def internal_theory_loss_log(ln_scale):
        ln_scale_scalar = float(np.squeeze(ln_scale))
        scale = np.exp(ln_scale_scalar)

        _, chi, x_phase, Rd_kpc = tgs_blind_velocity(data, scale)

        core_scale_factor = gamma_tgs * eta_skin * scale
        vacuum_tension_leak = np.abs(xi_space * (chi**2) - x_phase * np.exp(-Rd_kpc / (core_scale_factor + 1e-5)))

        # Малый штраф для устойчивой сходимости при очень больших |ln(scale)|
        penalty = 1e-7 * (ln_scale_scalar ** 2)
        return float(vacuum_tension_leak + penalty)

    # Старт из ln(scale) = 1.0
    res = minimize(internal_theory_loss_log, x0=[1.0], method='Nelder-Mead')
    best_scale = float(np.exp(np.squeeze(res.x)))

    v_mod, _, _, _ = tgs_blind_velocity(data, best_scale)

    # V_набл используются только здесь, для оценки ошибки
    mape = np.mean(np.abs(v_mod[mask] - Vobs[mask]) / Vobs[mask]) * 100.0
    bias = np.mean(v_mod[mask] - Vobs[mask])
    return v_mod, mape, bias, best_scale

# ===================== 3. ЗАПУСК И ОТЧЁТ =====================

def save_rotation_plot(data, velocity, name, directory):
    os.makedirs(directory, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=120)
    ax.plot(data['R'], data['Vobs'], 'ko', markersize=4, label='SPARC')
    ax.plot(data['R'], velocity, 'r-', label='Model')
    baryon = np.sqrt(np.maximum(0.5 * data['Vdisk']**2 + 0.7 * data['Vbul']**2 + data['Vgas']**2, 0))
    ax.plot(data['R'], baryon, 'b--', label='Baryonic reference (disk 0.5, bulge 0.7)')
    ax.set(xlabel='R [kpc]', ylabel='V [km/s]', title=name)
    ax.grid(alpha=0.3); ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(os.path.join(directory, name + '.png')); plt.close(fig)

if __name__ == '__main__':
    here = os.path.dirname(os.path.abspath(__file__))
    report_path = os.path.join(find_desktop(here), REPORT_NAME)

    print("===========================================================================")
    print(" ЗАПУСК: ПМК «КВАНТОВЫЙ РУБЕЖ» v17.2 Loss Balance (QFcheater Edition)")
    print("===========================================================================")

    files = find_sparc_files(here)
    if not files:
        print('ОШИБКА: не найдены файлы *_rotmod.dat (каталог SPARC).')
        print('Положите скрипт на рабочий стол рядом с каталогом SPARC.')
        sys.exit(1)
    print('Найдено файлов: %d (%s)' % (len(files), os.path.dirname(files[0])))

    gals = load_sparc(files)
    names = sorted(gals.keys())

    records = []
    all_mapes = []
    all_biases = []
    all_scales = []

    for name in names:
        vmod, mape, bias, scale = calculate_galaxy_tgs_v17_2(gals[name])
        if mape >= 999.0: continue
        save_rotation_plot(gals[name], vmod, name, os.path.join(os.path.dirname(report_path), "TGS_cheater_Plots"))
        all_mapes.append(mape)
        all_biases.append(bias)
        all_scales.append(scale)

        log_line = f"{name:<16s} | MAPE = {mape:6.2f}% | Bias = {bias:+6.2f} км/с | scale = {scale:7.3f}"
        records.append(log_line)
        print(log_line)

    pure_mapes = np.array(all_mapes)
    median_mape = np.median(pure_mapes)
    mean_bias = np.mean(all_biases)
    median_scale = np.median(all_scales)
    high_precision = np.sum(pure_mapes < 15.0)

    report_content = (
        "===========================================================================\n"
        "   «КВАНТОВЫЙ РУБЕЖ» v17.2 Loss Balance (QFcheater Edition): ОТЧЁТ\n"
        "===========================================================================\n"
        f" Базовое ускорение шелеста вакуума a_sh0: {a_sh0:.4e} м/с²\n"
        f" Всего рассчитано объектов              : {len(pure_mapes)}/{len(names)}\n"
        "---------------------------------------------------------------------------\n"
        " КРИВЫЕ ВРАЩЕНИЯ ПО ГАЛАКТИКАМ (слепой подбор scale по ln-шкале):\n"
        "---------------------------------------------------------------------------\n"
        + "\n".join(records) + "\n"
        "===========================================================================\n"
        "   ИТОГ ПО КАТАЛОГУ:\n"
        "===========================================================================\n"
        f"  Медианный MAPE                         : {median_mape:.2f}%\n"
        f"  Средний Bias                           : {mean_bias:+.2f} км/с\n"
        f"  Медианный масштаб сети scale           : {median_scale:.3f}\n"
        f"  Галактик с MAPE < 15%                  : {high_precision}/{len(pure_mapes)}\n"
        "===========================================================================\n"
    )

    with open(report_path, "w", encoding="utf-8") as report_file:
        report_file.write(report_content)

    print("\n" + "=" * 75)
    print(" ИТОГ v17.2:")
    print("-" * 75)
    print(f" Медианный MAPE          : {median_mape:.2f}%")
    print(f" Средний Bias            : {mean_bias:+.2f} км/с")
    print(f" Медианный масштаб scale : {median_scale:.3f}")
    print("=" * 75)
    print(f"Отчёт сохранён: {report_path}")
