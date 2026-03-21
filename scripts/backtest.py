"""
Backtesting del modelo Escasez y Resiliencia
Metodología: train 70% / test 30% por activo
Modelo: V(t) = Capital × (1+r)^t × Φ_L(t)/Φ_L(0)  |  Φ_L(t) = 1 + K/(1+e^(-γ(t-t₀)))
Métricas: RMSE, MAE, R², hit-rate direccional
Test estadístico: ¿mejora Φ_L significativamente sobre baseline (1+r)^t?

Referencia: Vilar, J.A. (2026a). Escasez y Resiliencia. SSRN Working Paper.

─────────────────────────────────────────────────────────────────────────────
DATOS REQUERIDOS
─────────────────────────────────────────────────────────────────────────────
Este script requiere ficheros CSV con precios históricos por activo.
Formato esperado (una fila por día):

  fecha,precio_cierre,fuente,notas
  2022-01-03,43.21,yahoo,
  2022-01-04,43.85,yahoo,

Los CSVs deben estar en el directorio indicado con --hist-dir (default: historico/).
El nombre de cada fichero debe coincidir con el ISIN del activo (ej: GB00B15KXQ89.csv).

Los datos de cartera personal NO se incluyen en este repositorio.
─────────────────────────────────────────────────────────────────────────────

Uso:
  python scripts/backtest.py
  python scripts/backtest.py --hist-dir /ruta/a/mis/csvs --out-dir resultados/
"""
import argparse, csv, os, json, math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime
from pathlib import Path

# ── Rutas por defecto (relativas al repo) ──────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent

# ── Colores ────────────────────────────────────────────────────────────────
BG    = '#0d1117'; PANEL = '#161b22'; GRID  = '#21262d'
WHITE = '#e6edf3'; MUTED = '#8b949e'
C_BASE= '#e3b341'; C_LOG = '#79c0ff'; C_ACT = '#56d364'

plt.rcParams.update({
    'figure.facecolor': BG,  'axes.facecolor': PANEL,
    'axes.edgecolor':  GRID, 'axes.labelcolor': WHITE,
    'xtick.color': MUTED,   'ytick.color': MUTED,
    'text.color':  WHITE,   'grid.color':  GRID,
    'grid.linewidth': 0.6,  'legend.facecolor': '#1c2128',
    'legend.edgecolor': GRID,
})

# ── Activos a testear (ISIN → nombre, pilar) ───────────────────────────────
# Sustituye por los ISINs de tus propios activos si tienes los CSVs.
ACTIVOS = {
    'Cobre':      ('GB00B15KXQ89', 'Escasez Física'),
    'Met. Prec.': ('JE00B1VS3W29', 'Escasez Física'),
    'Robótica':   ('IE00BYZK4552', 'Autorrepl. IA'),
    'AI & Data':  ('IE00BGV5VN51', 'Autorrepl. IA'),
    'MSCI World': ('IE00B03HCZ61', 'Resiliencia'),
    'Small-Cap':  ('IE00B42W4L06', 'Resiliencia'),
    'Bitcoin':    ('BTC-EUR',      'Escasez Digital'),
    'Uranio':     ('IE000M7V94E1', 'Energía/Grid'),
}

# ── Carga de datos ─────────────────────────────────────────────────────────
def load_series(hist_dir, isin):
    path = Path(hist_dir) / f'{isin}.csv'
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            try:
                rows.append((datetime.strptime(row['fecha'], '%Y-%m-%d'),
                             float(row['precio_cierre'])))
            except Exception:
                pass
    rows.sort()
    return rows

# ── Modelos ────────────────────────────────────────────────────────────────
def model_baseline(t, r):
    """Benchmark: crecimiento compuesto puro (1+r)^t"""
    return (1 + r) ** t

def model_logistic(t, r, K, gamma, t0):
    """Modelo Escasez y Resiliencia: (1+r)^t × Φ_L(t)  (no normalizado en train)"""
    phi = 1 + K / (1 + np.exp(-gamma * (t - t0)))
    return (1 + r) ** t * phi

# ── Calibración ────────────────────────────────────────────────────────────
def mse_loss(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

def fit_baseline(t_train, y_train):
    best_r, best_mse = 0.10, 1e18
    for r in np.arange(0.0, 0.60, 0.01):
        pred = y_train[0] * model_baseline(t_train, r)
        loss = mse_loss(pred, y_train)
        if loss < best_mse:
            best_mse, best_r = loss, r
    return {'r': best_r}

def fit_logistic(t_train, y_train, r):
    """Grid search grueso + refinamiento local para K, γ, t₀."""
    best = {'K': 1.0, 'gamma': 0.5, 't0': np.mean(t_train)}
    best_mse = 1e18
    max_t = max(t_train)

    for K in np.arange(0.5, 10.0, 0.5):
        for gamma in np.arange(0.2, 3.0, 0.2):
            for t0 in np.arange(0.5, max_t, max_t / 6):
                pred = y_train[0] * model_logistic(t_train, r, K, gamma, t0)
                loss = mse_loss(pred, y_train)
                if loss < best_mse:
                    best_mse = loss
                    best = {'K': K, 'gamma': gamma, 't0': t0}

    K0, g0, t00 = best['K'], best['gamma'], best['t0']
    for K in np.arange(max(0.1, K0 - 0.5), K0 + 0.6, 0.1):
        for gamma in np.arange(max(0.05, g0 - 0.3), g0 + 0.35, 0.05):
            for t0 in np.arange(max(0.1, t00 - 0.5), t00 + 0.6, 0.1):
                pred = y_train[0] * model_logistic(t_train, r, K, gamma, t0)
                loss = mse_loss(pred, y_train)
                if loss < best_mse:
                    best_mse = loss
                    best = {'K': K, 'gamma': gamma, 't0': t0}
    return {'r': r, **best}

# ── Métricas ───────────────────────────────────────────────────────────────
def metrics(y_pred, y_true, name=''):
    n = len(y_true)
    rmse = math.sqrt(np.mean((y_pred - y_true) ** 2))
    mae  = np.mean(np.abs(y_pred - y_true))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    if n > 1:
        hitrate = np.mean(np.diff(y_pred) > 0 == np.diff(y_true) > 0)
    else:
        hitrate = 0.5
    return {'model': name, 'rmse': rmse, 'mae': mae, 'r2': r2, 'hitrate': hitrate}

# ── Test de Wilcoxon (sin scipy) ───────────────────────────────────────────
def wilcoxon_approx(errors_a, errors_b):
    diffs = np.abs(errors_b) - np.abs(errors_a)
    diffs = diffs[diffs != 0]
    n = len(diffs)
    if n < 5:
        return None, None
    ranks = np.argsort(np.abs(diffs)) + 1
    W_plus  = np.sum(ranks[diffs > 0])
    W_minus = np.sum(ranks[diffs < 0])
    W = min(W_plus, W_minus)
    mu_W    = n * (n + 1) / 4
    sigma_W = math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    z = (W - mu_W) / sigma_W if sigma_W > 0 else 0
    p = 2 * (1 - _norm_cdf(abs(z)))
    return z, p

def _norm_cdf(x):
    t = 1 / (1 + 0.2316419 * abs(x))
    poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 +
           t * (-1.821255978 + t * 1.330274429))))
    return (1 - (1 / math.sqrt(2 * math.pi)) * math.exp(-x * x / 2) * poly
            if x >= 0
            else (1 / math.sqrt(2 * math.pi)) * math.exp(-x * x / 2) * poly)


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description='Backtesting del modelo Escasez y Resiliencia')
    parser.add_argument('--hist-dir', default=str(ROOT / 'historico'),
                        help='Directorio con los CSVs de precios históricos')
    parser.add_argument('--out-dir', default=str(ROOT / 'figuras'),
                        help='Directorio de salida para las figuras y JSON')
    args = parser.parse_args()

    hist_dir = Path(args.hist_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not hist_dir.exists():
        print(f"\n⚠  Directorio de históricos no encontrado: {hist_dir}")
        print("   Proporciona los CSVs con --hist-dir /ruta/a/tus/csvs")
        print("   Formato: fecha,precio_cierre,fuente,notas (ver docstring)")
        return

    results = []
    fit_details = {}

    for nombre, (isin, pilar) in ACTIVOS.items():
        path = hist_dir / f'{isin}.csv'
        if not path.exists():
            print(f"⏭  {nombre} ({isin}): CSV no encontrado — omitido")
            continue
        serie = load_series(hist_dir, isin)
        if len(serie) < 100:
            print(f"⏭  {nombre}: menos de 100 registros — omitido")
            continue

        t0_date = serie[0][0]
        t_all   = np.array([(d - t0_date).days / 365.25 for d, _ in serie])
        p_all   = np.array([p for _, p in serie])
        p_norm  = p_all / p_all[0]

        split       = int(len(t_all) * 0.70)
        t_train, p_train = t_all[:split], p_norm[:split]
        t_test,  p_test  = t_all[split:], p_norm[split:]
        t0_test  = t_all[split]

        params_base = fit_baseline(t_train, p_train)
        r_cal       = params_base['r']
        params_log  = fit_logistic(t_train, p_train, r_cal)

        p0_test  = p_test[0]
        t_rel    = t_test - t0_test

        pred_base = p0_test * model_baseline(t_rel, params_base['r'])
        pred_log  = p0_test * model_logistic(t_rel, params_log['r'],
                                              params_log['K'], params_log['gamma'],
                                              params_log['t0'] - t0_test)

        m_base = metrics(pred_base, p_test, 'Baseline')
        m_log  = metrics(pred_log,  p_test, 'Logístico')

        z_stat, p_val = wilcoxon_approx(pred_base - p_test, pred_log - p_test)

        K, g, t0_fit = params_log['K'], params_log['gamma'], params_log['t0']

        results.append({
            'nombre': nombre, 'isin': isin, 'pilar': pilar,
            'n_train': split, 'n_test': len(t_test),
            'train_years': round(t_train[-1] - t_train[0], 1),
            'test_years':  round(t_test[-1]  - t_test[0],  1),
            'r': r_cal, 'K': K, 'gamma': g, 't0': round(t0_fit, 2),
            'rmse_base': m_base['rmse'], 'r2_base': m_base['r2'], 'hit_base': m_base['hitrate'],
            'rmse_log':  m_log['rmse'],  'r2_log':  m_log['r2'],  'hit_log':  m_log['hitrate'],
            'z_wilcoxon': z_stat, 'p_wilcoxon': p_val,
            'log_beats_base': m_log['rmse'] < m_base['rmse'],
        })
        fit_details[nombre] = {
            't_all': t_all.tolist(), 'p_norm': p_norm.tolist(),
            't_train': t_train.tolist(), 't_test': t_test.tolist(),
            'p_train': p_train.tolist(), 'p_test': p_test.tolist(),
            'pred_base': pred_base.tolist(), 'pred_log': pred_log.tolist(),
            'split_idx': split, 't0_test': float(t0_test), 'pilar': pilar,
        }
        print(f"✅ {nombre:<12} | r={r_cal:.2f} K={K:.1f} γ={g:.2f} t₀={t0_fit:.1f} | "
              f"RMSE(log)={m_log['rmse']:.4f} R²={m_log['r2']:.3f} "
              f"{'🏆' if m_log['rmse'] < m_base['rmse'] else '  '}")

    if not results:
        print("\nNo se procesó ningún activo. Verifica --hist-dir.")
        return

    # ── Figura 1: fits por activo ──────────────────────────────────────────
    n_act = len(results)
    ncols = 3
    nrows = math.ceil(n_act / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4.5 * nrows), facecolor=BG)
    fig.suptitle('Backtesting — Train 70% / Test 30% por activo\nPrecio normalizado (inicio=1)',
                 color=WHITE, fontweight='bold', fontsize=12, y=1.01)

    axes_flat = axes.flat if hasattr(axes, 'flat') else [axes]
    for ax, res in zip(axes_flat, results):
        nombre = res['nombre']
        d = fit_details[nombre]
        ax.set_facecolor(PANEL)
        t_all_arr = np.array(d['t_all'])
        t_test_arr = np.array(d['t_test'])
        ax.plot(t_all_arr, d['p_norm'], color=C_ACT, lw=1.2, alpha=0.7, label='Real')
        ax.axvspan(d['t0_test'], t_all_arr[-1], alpha=0.08, color=WHITE)
        ax.axvline(d['t0_test'], color=MUTED, lw=1, ls='--', alpha=0.5)
        ax.plot(t_test_arr, d['pred_base'], color=C_BASE, lw=1.5, ls=':',
                label=f"Base  R²={res['r2_base']:.2f}")
        ax.plot(t_test_arr, d['pred_log'],  color=C_LOG,  lw=2.0,
                label=f"Log   R²={res['r2_log']:.2f}")
        col_w = C_LOG if res['log_beats_base'] else C_BASE
        ax.text(0.98, 0.04, '★ LOG' if res['log_beats_base'] else '★ BASE',
                transform=ax.transAxes, ha='right', fontsize=8,
                color=col_w, fontweight='bold')
        if res['p_wilcoxon'] is not None:
            sig = '***' if res['p_wilcoxon'] < 0.01 else '**' if res['p_wilcoxon'] < 0.05 \
                  else '*' if res['p_wilcoxon'] < 0.10 else 'ns'
            ax.text(0.02, 0.96, f"p={res['p_wilcoxon']:.3f}{sig}",
                    transform=ax.transAxes, ha='left', va='top', fontsize=7.5, color=MUTED)
        ax.set_title(f"{nombre}  [{res['pilar']}]", color=WHITE, fontsize=9)
        ax.set_xlabel('Años'); ax.set_ylabel('Precio norm.')
        ax.legend(fontsize=7, loc='upper left'); ax.grid(True, alpha=0.25)

    for ax in list(axes_flat)[n_act:]:
        ax.set_visible(False)

    plt.tight_layout()
    fig.savefig(out_dir / 'backtest_fits.png', dpi=140, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"\n✅ Figura 1: {out_dir / 'backtest_fits.png'}")

    # ── Figura 2: métricas resumen ─────────────────────────────────────────
    fig2 = plt.figure(figsize=(14, 8), facecolor=BG)
    gs   = gridspec.GridSpec(2, 3, figure=fig2, hspace=0.45, wspace=0.35,
                             left=0.07, right=0.97, top=0.92, bottom=0.08)
    fig2.suptitle('Resumen estadístico del backtesting — Modelo Escasez y Resiliencia',
                  color=WHITE, fontweight='bold', fontsize=12)

    nombres = [r['nombre'] for r in results]
    x = np.arange(len(nombres))
    w = 0.35

    ax1 = fig2.add_subplot(gs[0, 0]); ax1.set_facecolor(PANEL)
    ax1.bar(x - w/2, [r['rmse_base'] for r in results], w, color=C_BASE, alpha=0.85, label='Baseline')
    ax1.bar(x + w/2, [r['rmse_log']  for r in results], w, color=C_LOG,  alpha=0.85, label='Logístico')
    ax1.set_xticks(x); ax1.set_xticklabels(nombres, rotation=30, ha='right', fontsize=7.5)
    ax1.set_title('RMSE en test (↓ mejor)', color=WHITE); ax1.legend(fontsize=7.5)
    ax1.grid(True, alpha=0.25, axis='y')

    ax2 = fig2.add_subplot(gs[0, 1]); ax2.set_facecolor(PANEL)
    ax2.bar(x - w/2, [r['r2_base'] for r in results], w, color=C_BASE, alpha=0.85, label='Baseline')
    ax2.bar(x + w/2, [r['r2_log']  for r in results], w, color=C_LOG,  alpha=0.85, label='Logístico')
    ax2.axhline(0, color=MUTED, lw=0.8, ls=':')
    ax2.set_xticks(x); ax2.set_xticklabels(nombres, rotation=30, ha='right', fontsize=7.5)
    ax2.set_title('R² en test (↑ mejor)', color=WHITE); ax2.legend(fontsize=7.5)
    ax2.grid(True, alpha=0.25, axis='y')

    ax3 = fig2.add_subplot(gs[0, 2]); ax3.set_facecolor(PANEL)
    ax3.bar(x - w/2, [r['hit_base'] for r in results], w, color=C_BASE, alpha=0.85, label='Baseline')
    ax3.bar(x + w/2, [r['hit_log']  for r in results], w, color=C_LOG,  alpha=0.85, label='Logístico')
    ax3.axhline(0.5, color=MUTED, lw=1, ls='--', label='Azar (50%)')
    ax3.set_xticks(x); ax3.set_xticklabels(nombres, rotation=30, ha='right', fontsize=7.5)
    ax3.set_title('Hit-rate direccional (↑ mejor)', color=WHITE); ax3.legend(fontsize=7.5)
    ax3.grid(True, alpha=0.25, axis='y'); ax3.set_ylim(0, 1)

    ax4 = fig2.add_subplot(gs[1, 0]); ax4.set_facecolor(PANEL)
    K_vals = [r['K'] for r in results]
    g_vals = [r['gamma'] for r in results]
    pilar_colors = {
        'Escasez Física': C_BASE, 'Autorrepl. IA': C_LOG,
        'Resiliencia': MUTED, 'Escasez Digital': '#d2a8ff', 'Energía/Grid': '#f78166'
    }
    for r in results:
        col = pilar_colors.get(r['pilar'], WHITE)
        ax4.scatter(r['gamma'], r['K'], color=col, s=90, zorder=5, alpha=0.9)
        ax4.annotate(r['nombre'], (r['gamma'], r['K']),
                     textcoords='offset points', xytext=(4, 3), fontsize=7, color=col)
    for pilar, col in pilar_colors.items():
        ax4.scatter([], [], color=col, label=pilar, s=60)
    ax4.set_xlabel('γ (velocidad)'); ax4.set_ylabel('K (multiplicador máx.)')
    ax4.set_title('Parámetros calibrados K vs γ', color=WHITE)
    ax4.legend(fontsize=7); ax4.grid(True, alpha=0.25)

    ax5 = fig2.add_subplot(gs[1, 1]); ax5.set_facecolor(PANEL)
    p_vals = [r['p_wilcoxon'] if r['p_wilcoxon'] is not None else 0.99 for r in results]
    cols_p = [C_LOG if p < 0.10 else '#f78166' if p < 0.25 else MUTED for p in p_vals]
    ax5.bar(x, p_vals, color=cols_p, alpha=0.85)
    ax5.axhline(0.05, color=C_LOG,  lw=1.5, ls='--', label='p=0.05')
    ax5.axhline(0.10, color=C_BASE, lw=1.0, ls=':',  label='p=0.10')
    ax5.set_xticks(x); ax5.set_xticklabels(nombres, rotation=30, ha='right', fontsize=7.5)
    ax5.set_title('Wilcoxon: Logístico vs Baseline\n(p < 0.05 → mejora significativa)', color=WHITE)
    ax5.legend(fontsize=7.5); ax5.grid(True, alpha=0.25, axis='y')

    ax6 = fig2.add_subplot(gs[1, 2]); ax6.set_facecolor(PANEL)
    r_vals = [r['r'] for r in results]
    ax6.bar(nombres, r_vals, color=C_ACT, alpha=0.85)
    ax6.axhline(np.mean(r_vals), color=MUTED, lw=1, ls='--',
                label=f'Media r={np.mean(r_vals):.2f}')
    ax6.set_xticklabels(nombres, rotation=30, ha='right', fontsize=7.5)
    ax6.set_title('r calibrado por activo', color=WHITE)
    ax6.legend(fontsize=7.5); ax6.grid(True, alpha=0.25, axis='y')

    plt.savefig(out_dir / 'backtest_metricas.png', dpi=140, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"✅ Figura 2: {out_dir / 'backtest_metricas.png'}")

    # ── Guardar JSON ───────────────────────────────────────────────────────
    json_out = out_dir / 'backtest_results.json'
    with open(json_out, 'w') as f:
        json.dump({'results': results}, f, indent=2, default=str)
    print(f"✅ Resultados JSON: {json_out}")

    # ── Resumen texto ──────────────────────────────────────────────────────
    n_wins = sum(1 for r in results if r['log_beats_base'])
    n_sig  = sum(1 for r in results if r['p_wilcoxon'] and r['p_wilcoxon'] < 0.10)
    print(f"\n{'═'*60}")
    print(f"📊 Logístico gana en RMSE: {n_wins}/{len(results)} activos ({100*n_wins/len(results):.0f}%)")
    print(f"📊 Mejora significativa (p<0.10): {n_sig}/{len(results)} activos")


if __name__ == '__main__':
    main()
