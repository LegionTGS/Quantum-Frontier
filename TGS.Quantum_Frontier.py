#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Программный комплекс "Квантовый рубеж" (v1.0)
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
 ПРОГРАММНО-МАТЕМАТИЧЕСКИЙ КОМПЛЕКС
 ДЛЯ РАСЧЁТА КРИВЫХ ВРАЩЕНИЯ ГАЛАКТИК «КВАНТОВЫЙ РУБЕЖ»

 Разработчик: Самохвалов В.В., Аналитический центр "Брейншторм"

 ВХОД:  каталог SPARC (файлы *_rotmod.dat) — рядом со скриптом или в
        подпапке SPARC. Если рядом лежит Table1, из него берётся
        фотометрия (барионный вход массы) и флаги качества Q,
        инклинация и расстояние; без Table1 расчёт идёт полностью
        по каталогу SPARC: масса по кривым барионных компонент.
 ВЫХОД: на рабочий стол (если папки рабочего стола нет — рядом со
        скриптом):
          TGS_QF_Report.txt — отчёт по всей базе;
          TGS_QF_Plots/          — графики кривых вращения всех галактик
                                 (нужен matplotlib; без него графики
                                 пропускаются, отчёт формируется).
===========================================================================
'''
import os, sys, glob, math, re
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

# ======================= КОНСТАНТЫ ГЕОМЕТРИИ ==============================
# ETA — шаг активной Кожуры из принципа 50/50: ETA = 1 - (0.5)^(1/3).
ETA        = 0.206299
# Предельные инварианты отклика: атомарная фаза (плотное ядро диска)
# и фаза пустоты (внешний прибой) связываются через ETA.
XI_ATOM    = ETA**2 / (2.0*math.sqrt(2.0))
XI_VOID    = ETA / math.sqrt(2.0)
# a0 — ускорение перехода шелеста: a0 = c*H0/(2*pi) при H0 = 67.4 км/с/Мпк.
# В единицах кода: (км/с)^2/кпк; в СИ: 3215.89*3.2408e-14 = 1.042e-10 м/с^2.
A0         = 3215.89
G_KPC      = 4.302e-6                  # G, кпк*(км/с)^2/солн.масса
# Критическая поверхностная плотность невода (плотность перехода).
SIGMA_CRIT = A0 / (2.0*math.pi*G_KPC)  # солн.масса/кпк^2
OMEGA_CONST= 5.4e-16                   # частотный масштаб фазового канала
SC_SETA    = 1.0/ETA                   # шкала невода: s_c = 1/ETA = 4.8473

# ==================== ДЕДУКТИВНЫЕ ПАРАМЕТРЫ ===============================
# Веса компонент в ньютоновском поле (фазовые множители невода):
# Константа Самохваловой (из работы P vs NP):
I_TGS      = 1.438151512351216
W_GAS      = I_TGS        # вес газа
W_DISK     = 0.60033558   # вес звёздного диска
# Анизотропия отклика невода (доля нецентрального члена при xi->0):
ANISO      = 0.29614151
# Коэффициент ковариации фаз (множитель глубинного канала):
PHI_COV    = 4.00000000
# Релятивистский скрининг скорости (поправка ~ v^2/c):
GAMMA_SCR  = 6.7706e-06
# Звёздная масса-светимость в полосе 3.6 мкм (стандарт SPARC-моделей)
# и множитель гелия для массы нейтрального водорода — оба значения
# зашиты в опубликованных кривых компонент SPARC:
ML36       = 0.50
HEX        = 1.33

# ============= КОНСТАНТЫ ВСТРЕЧНОГО ШЕЛЕСТА (Приложение 12.2) =============
ETA_KPD    = 0.16     # эффективность захвата встречного поля
A_SH_BASE  = 45.0     # базовая амплитуда встречного шелеста
V_CRIT     = 32.4     # порог скорости канала, км/с
MIRROR     = 0.2658   # зеркальный множитель амплитуды
DAMP_P     = 1.45     # показатель затухания по фазовой плотности

# ==================== НОРМАЛИЗАЦИЯ ИМЁН ===================================
def norm_id(name):
    # DDO064 и DDO64, NGC0055 и NGC55 — один объект каталога
    s = str(name).strip().upper()
    m = re.match(r'^([A-Z]+)\s*0*(\d+)$', s)
    if m: return '%s%d' % (m.group(1), int(m.group(2)))
    return s

# ======================= ЗАГРУЗКА SPARC ===================================
def load_galaxy(path):
    # Формат *_rotmod.dat: R  V_набл  e_V  V_газ  V_диск  V_булge (R в кпк)
    R, VG, VD, VB, VO, EV = [], [], [], [], [], []
    with open(path) as f:
        for line in f:
            if line.startswith('#') or not line.strip(): continue
            p = line.split()
            try:
                r, vo, ev = float(p[0]), float(p[1]), float(p[2])
                vg, vd, vb = float(p[3]), float(p[4]), float(p[5])
            except (ValueError, IndexError): continue
            if not np.isfinite(r) or r <= 0: continue
            R.append(r); VO.append(vo)
            EV.append(ev if np.isfinite(ev) else 0.0)
            VG.append(vg if np.isfinite(vg) else 0.0)
            VD.append(vd if np.isfinite(vd) else 0.0)
            VB.append(vb if np.isfinite(vb) else 0.0)
    R = np.array(R)
    if len(R) < 3: return None
    return dict(R=R,
                v_gas=np.abs(np.array(VG)), v_disk=np.abs(np.array(VD)),
                v_bul=np.abs(np.array(VB)), v_obs=np.array(VO),
                e_obs=np.abs(np.array(EV)))

def find_desktop(fallback):
    if os.environ.get('QF_OUTPUT_DIR'):
        os.makedirs(os.environ['QF_OUTPUT_DIR'], exist_ok=True)
        return os.path.abspath(os.environ['QF_OUTPUT_DIR'])
    # Рабочий стол пользователя: Windows (в т.ч. под OneDrive), macOS, Linux
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

def find_data(here):
    # Поиск каталога: рядом со скриптом, в подпапках SPARC/data,
    # затем рекурсивный обход подпапок
    for d in [here, os.path.join(here,'SPARC'), os.path.join(here,'data'),
              os.path.join(here,'sparc'), os.getcwd()]:
        files = sorted(glob.glob(os.path.join(d,'*_rotmod.dat')))
        if files: return files
    found = []
    for root, dirs, _ in os.walk(here):
        for fn in sorted(glob.glob(os.path.join(root,'*_rotmod.dat'))):
            found.append(fn)
        if len(found) > 1000: break
    return sorted(found)

# ===================== МЕТАДАННЫЕ TABLE1 (SPARC) ==========================
def num_or_nan(t):
    try: return float(t)
    except (ValueError, TypeError): return float('nan')

def parse_meta_line(line):
    t = line.split()
    # пробельный формат: имя + 16 чисел + Q + ссылка
    if len(t) >= 19 and re.match(r'^[A-Za-z]', t[0]):
        try: Q = int(t[17])
        except (ValueError, IndexError): Q = None
        if Q in (1, 2, 3):
            return dict(name=t[0], D=num_or_nan(t[2]), eD=num_or_nan(t[3]),
                        Inc=num_or_nan(t[5]), eInc=num_or_nan(t[6]),
                        L36=num_or_nan(t[7]), Rdisk=num_or_nan(t[11]),
                        MHI=num_or_nan(t[13]), Vflat=num_or_nan(t[15]), Q=Q)
    # оригинальный фикс-ширинный .mrt
    s = line.ljust(120)
    def fld(a, b): return s[a-1:b].strip()
    name = fld(1, 11)
    if not name or not re.match(r'^[A-Za-z]', name): return None
    try: Q = int(fld(97, 99))
    except ValueError: return None
    if Q not in (1, 2, 3): return None
    return dict(name=name, D=num_or_nan(fld(14,19)), eD=num_or_nan(fld(20,24)),
                Inc=num_or_nan(fld(27,30)), eInc=num_or_nan(fld(31,34)),
                L36=num_or_nan(fld(35,41)), Rdisk=num_or_nan(fld(62,66)),
                MHI=num_or_nan(fld(75,81)), Vflat=num_or_nan(fld(87,91)), Q=Q)

def load_meta(here):
    seen = []
    for d in [here, os.path.join(here,'SPARC'), os.path.join(here,'data'),
              os.path.join(here,'sparc'), os.getcwd()]:
        for pat in ('Table1.*', 'table1.*', 'TABLE1.*'):
            for fn in sorted(glob.glob(os.path.join(d, pat))):
                if os.path.isfile(fn) and fn not in seen: seen.append(fn)
    for fn in seen:
        rows = {}
        with open(fn, encoding='utf-8', errors='replace') as f:
            for line in f:
                r = parse_meta_line(line)
                if r: rows[norm_id(r['name'])] = r
        if len(rows) >= 20:
            return fn, rows
    return None, {}

# ======================= ЯДРО СИНТЕЗА =====================================
def synthesize(g, m, mode):
    # Наблюдаемые скорости v_obs здесь не используются: масса берётся из
    # фотометрии Table1 или из кривых барионных компонент, а V_flat
    # выводится из массы по Талли-Фишеру V_flat^4 = G*M*a0.
    R      = g['R']
    R_max  = float(R[-1])
    src = 'dyn'
    Mb = float('nan')
    if mode == 'bary' and m is not None:
        Mb = ML36*m['L36'] + HEX*m['MHI']          # 1e9 солн.масс
        if np.isfinite(Mb) and Mb > 0: src = 'bary'
    if src == 'bary':
        Rd = m['Rdisk'] if (np.isfinite(m['Rdisk']) and m['Rdisk'] > 0) else R_max/3.2
        M_total = Mb * 1e9
    else:
        # Без фотометрии: масса по барионной кривой на внешнем радиусе
        vb2_out = g['v_gas'][-1]**2 + ML36*(g['v_disk'][-1]**2 + g['v_bul'][-1]**2)
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

    v_N2 = (W_GAS*g['v_gas'])**2 + (W_DISK*g['v_disk'])**2 + g['v_bul']**2
    a_N  = v_N2 / R
    phi_n   = a_N / A0
    phi_cov = 1.0 + PHI_COV * chi_clip * np.log1p(phi_n * OMEGA_CONST)

    s_loc   = 2.0*V_flat**2 / (A0*R)
    red_loc = s_loc**4 / (s_loc**4 + SC_SETA**4)
    w       = 1.0 + (W_GAS - 1.0)*red_loc

    v_N2m = (w*g['v_gas'])**2 + (W_DISK*g['v_disk'])**2 + g['v_bul']**2
    a_Nm  = v_N2m / R
    y_m   = a_Nm / A0

    xi_r   = XI_ATOM + (XI_VOID - XI_ATOM)/(1.0 + phi_n)
    w_mond = np.clip(xi_r / XI_VOID, 0.0, 1.0)
    g_mond = np.sqrt(np.maximum(a_Nm, 1e-12) * A0)
    Fk     = 1.0/np.sqrt(1.0 + y_m**2)     # чётная переходная функция: F'(0)=0

    a_total = (a_N * phi_cov * (1.0 + piston*0.0009)
               + g_mond * w_mond * Fk * (1.0 + ANISO*(1.0 - w_mond)))
    v_model = np.sqrt(np.maximum(a_total*R, 1e-9))
    v_model = v_model + GAMMA_SCR * (v_model**2) / 2.998e5

    # канал встречного шелеста (Приложение 12.2):
    # торможение модельной скорости встречным полем
    udg_factor = 1.0/(1.0 + s_ratio)
    eps = np.exp(-x_phase) * udg_factor * ETA_KPD
    a_sh = A_SH_BASE * (1.0 + MIRROR)
    lam_catch = (1.0 - xi_r) * (1.0 - np.exp(-R/(np.pi*Rd)))
    f_v = 0.5*(1.0 + np.tanh((v_model - V_CRIT)/10.0))
    V_feded = eps * a_sh * np.tanh(R/(2.0*Rd)) * np.exp(-DAMP_P*x_phase) * f_v * lam_catch
    v_pred = v_model - V_feded

    frac_gas = float(np.mean((g['v_gas']**2) / np.maximum(v_N2, 1e-12)))
    return dict(v_pred=v_pred, s_ratio=s_ratio, Rd=Rd,
                xi_mean=float(np.mean(xi_r)), frac_gas=frac_gas,
                src=src, V_flat=V_flat, M_total=M_total,
                y_bar=float(np.mean(y_m)))

# ============================ МЕТРИКИ =====================================
def metrics(v_pred, v_obs):
    v_pred = np.asarray(v_pred); v_obs = np.asarray(v_obs)
    m = np.isfinite(v_pred) & np.isfinite(v_obs) & (v_obs > 0)
    p, o = v_pred[m], v_obs[m]
    smape = 100.0*np.mean(2.0*np.abs(p-o)/(p+o))
    mape  = 100.0*np.mean(np.abs(p-o)/o)
    bias  = float(np.mean(p-o))
    return smape, mape, bias

# ============================ ГРАФИКИ =====================================
def plot_galaxy(r, path):
    # Кривая вращения: наблюдения с ошибками, модель, ньютоновские барионы
    g = r['g']
    R = g['R']
    v_bar = np.sqrt(g['v_gas']**2 + ML36*g['v_disk']**2 + 0.7*g['v_bul']**2)
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.errorbar(R, g['v_obs'], yerr=g['e_obs'], fmt='o', ms=4, color='black',
                ecolor='0.55', capsize=2, label='V набл. (SPARC)')
    ax.plot(R, r['v_pred'], '-', lw=2.2, color='#c0392b',
            label='V мод. («Квантовый рубеж»)')
    ax.plot(R, v_bar, '--', lw=1.4, color='#2471a3',
            label='V бар. (Ньютон, M/L диск 0.5, балдж 0.7)')
    q = (', Q=%d' % r['Q']) if r['Q'] else ''
    ax.set_title('%s   sMAPE = %.1f%%   биаз = %+.1f км/с%s' %
                 (r['name'], r['sbary'], r['bias'], q))
    ax.set_xlabel('R, кпк')
    ax.set_ylabel('V, км/с')
    ax.set_xlim(0, R[-1]*1.05)
    ax.set_ylim(0, max(np.nanmax(g['v_obs'] + g['e_obs']),
                       np.nanmax(r['v_pred']), np.nanmax(v_bar))*1.12)
    ax.grid(alpha=0.3)
    ax.legend(loc='lower right', fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)

# ========================= ОСНОВНОЙ ПРОГОН ================================
def main(base_dir=None):
    here = base_dir if base_dir else os.path.dirname(os.path.abspath(__file__))
    files = find_data(here)
    if not files:
        print('ОШИБКА: не найдены файлы *_rotmod.dat (каталог SPARC).')
        print('Положите скрипт на рабочий стол рядом с каталогом SPARC.')
        sys.exit(1)
    data_dir = os.path.dirname(files[0])
    print('Найдено файлов: %d (%s)' % (len(files), data_dir), flush=True)
    meta_path, meta = load_meta(here)
    if meta:
        print('Метаданные: %s (%d записей)' %
              (os.path.basename(meta_path), len(meta)), flush=True)
    else:
        print('Table1 не найден — расчёт полностью по каталогу SPARC '
              '(масса по кривым барионных компонент).', flush=True)

    rows = []
    for idx, path in enumerate(files):
        g = load_galaxy(path)
        if g is None: continue
        key = norm_id(os.path.basename(path).replace('_rotmod.dat','').replace('.dat',''))
        m = meta.get(key)
        syn_b = synthesize(g, m, 'bary')
        syn_d = synthesize(g, m, 'dyn')
        s_d, _, b_d = metrics(syn_d['v_pred'], g['v_obs'])
        s_b, _, b_b = metrics(syn_b['v_pred'], g['v_obs'])
        rows.append(dict(name=key, sdyn=s_d, sbary=s_b, bdyn=b_d, bias=b_b,
                         Q=(m['Q'] if m else 0),
                         Inc=(m['Inc'] if m else float('nan')),
                         D=(m['D'] if m else float('nan')),
                         f_gas=syn_b['frac_gas'], s_glob=syn_b['s_ratio'],
                         src=syn_b['src'], Rd=syn_b['Rd'],
                         V_flat=syn_b['V_flat'],
                         g=g, v_pred=syn_b['v_pred']))
        if (idx+1) % 25 == 0:
            print('  ... просчитано %d/%d' % (idx+1, len(files)), flush=True)

    rows.sort(key=lambda r: r['name'])
    def col(k): return np.array([r[k] for r in rows], dtype=float)
    def med(x): return float(np.median(x)) if len(x) else float('nan')
    def sc(v, f):
        return (f % v) if np.isfinite(v) else '   -'

    sm_d, sm_b = col('sdyn'), col('sbary')
    bd, bi     = col('bdyn'), col('bias')
    Q, Inc, D  = col('Q'), col('Inc'), col('D')
    fg, sg     = col('f_gas'), col('s_glob')

    out_dir = find_desktop(here)
    rep = os.path.join(out_dir, 'TGS_QF_Report.txt')
    with open(rep, 'w', encoding='utf-8') as f:
        def P(*a):
            s = ' '.join(str(x) for x in a); print(s); f.write(s + '\n')
        W = 92

        P('='*W)
        P(' ПРОГРАММНО-МАТЕМАТИЧЕСКИЙ КОМПЛЕКС')
        P(' ДЛЯ РАСЧЁТА КРИВЫХ ВРАЩЕНИЯ ГАЛАКТИК «КВАНТОВЫЙ РУБЕЖ»')
        P(' Разработчик: Самохвалов В.В., Аналитический центр "Брейншторм"')
        P(' Галактик в расчёте: %d' % len(rows))
        if meta:
            P(' Метаданные: %s — сопоставлено %d из %d (фотометрия и флаги Q)' %
              (os.path.basename(meta_path), int(np.sum(Q > 0)), len(rows)))
        else:
            P(' Метаданные: Table1 не найден — разбивка по Q недоступна.')
        P('='*W)
        P(' РЕЗУЛЬТАТЫ ПО ВСЕЙ БАЗЕ (%d галактик)' % len(rows))
        P('-'*W)
        P(' Мэйп  (median sMAPE): %.2f%%' % med(sm_b))
        P(' Мэйп  (mean   sMAPE): %.2f%%' % sm_b.mean())
        P(' Биаз  (медиана):      %+.2f км/с' % med(bi))
        P(' Контроль (без Table1, масса по кривым компонент): median %.2f%%  mean %.2f%%  биаз %+.2f км/с' %
          (med(sm_d), sm_d.mean(), med(bd)))
        src_bary = int(np.sum([1 for r in rows if r['src'] == 'bary']))
        P(' Источники массы: фотометрия=%d, кривые компонент=%d' %
          (src_bary, len(rows) - src_bary))

        P('')
        P(' РАСПРЕДЕЛЕНИЕ ТОЧНОСТИ (сколько галактик укладывается в порог):')
        for thr in (5, 10, 15, 25, 50):
            n_ok = int(np.sum(sm_b <= thr))
            P('   до %2d%% : %3d из %d  (%5.1f%%)' %
              (thr, n_ok, len(rows), 100.0*n_ok/len(rows)))
        n_bad = int(np.sum(sm_b > 50))
        P('   свыше 50%%: %3d из %d  (%5.1f%%)' % (n_bad, len(rows), 100.0*n_bad/len(rows)))

        P('')
        P(' ТРИ ХУДШИХ ГАЛАКТИКИ:')
        order = np.argsort(-sm_b)
        for i in order[:3]:
            r = rows[i]
            P('   %-12s %6.1f%%  биаз %+8.1f  Q=%d  Inc=%s  D=%s  f_gas=%.2f  s=%.2f' %
              (r['name'], r['sbary'], r['bias'], r['Q'],
               sc(r['Inc'], '%5.1f'), sc(r['D'], '%5.1f'), r['f_gas'], r['s_glob']))

        P('')
        P('='*W)
        P(' РАЗБИВКА ПО КАЧЕСТВУ НАБЛЮДЕНИЙ (флаг Q каталога SPARC)')
        P('-'*W)
        P('   Q     n    мэйп (median)  мэйп (mean)   биаз')
        for q in (1, 2, 3):
            msk = (Q == q)
            if msk.sum():
                P('   Q=%d  %3d   %8.2f%%    %8.2f%%   %+8.2f' %
                  (q, int(msk.sum()), med(sm_b[msk]), sm_b[msk].mean(), med(bi[msk])))
        m12 = (Q == 1) | (Q == 2)
        if m12.sum() and (Q == 3).sum():
            d_q3 = med(sm_b[(Q == 3)]) - med(sm_b[m12])
            P('   Q<=2 vs Q=3: дельта медиан %+.2f п.п. (Q=3 хуже на эту величину)' % d_q3)
        P('   (Q=0 — без метаданных: в группы качества не входят)')

        P('')
        P('='*W)
        P(' ПОГАЛАКТИЧНАЯ ТАБЛИЦА')
        P('-'*W)
        P('Galaxy        sMAPE%%   биаз     Q    Inc     D      f_gas   s      R_d    src')
        P('-'*W)
        for r in rows:
            P('%-12s %6.1f %+8.1f  %d  %s  %s  %5.2f  %6.2f  %5.2f  %s' %
              (r['name'], r['sbary'], r['bias'], r['Q'],
               sc(r['Inc'], '%5.1f'), sc(r['D'], '%5.1f'),
               r['f_gas'], r['s_glob'], r['Rd'], r['src']))

    print('\nОтчёт сохранён: %s' % rep)

    if plt is None:
        print('Графики пропущены: не установлен matplotlib '
              '(установите: pip install matplotlib).')
        return rep
    gdir = os.path.join(out_dir, 'TGS_QF_Plots')
    os.makedirs(gdir, exist_ok=True)
    print('Строю графики (%d шт.)...' % len(rows), flush=True)
    for i, r in enumerate(rows):
        plot_galaxy(r, os.path.join(gdir, '%s.png' % r['name']))
        if (i+1) % 25 == 0:
            print('  ... графиков %d/%d' % (i+1, len(rows)), flush=True)
    print('Графики сохранены: %s' % gdir)
    return rep

if __name__ == '__main__':
    main()
