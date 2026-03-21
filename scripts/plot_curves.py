"""
Comparativa de curvas del factor de autorreplicación Φ_L(t)
para el modelo Escasez y Resiliencia.

Modelo (Vilar, 2026a — SSRN):
  V(t) = Capital × (1+r)^t × Φ_L(t)/Φ_L(0)
  Φ_L(t) = 1 + K / (1 + e^(−γ(t−t₀)))   ← Verhulst (1838)

Escenarios calibrados:
  BASE:       K=2,  γ=0.30, t₀=3.0
  ACELERADO:  K=4,  γ=0.40, t₀=3.0
  ÓPTIMO:     K=6,  γ=0.50, t₀=3.0

Uso:
  python scripts/plot_curves.py
  python scripts/plot_curves.py --output figuras/mi_figura.png
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# ── Rutas ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / 'figuras' / 'modelo_curvas_comparativa.png'

# ── Colores ────────────────────────────────────────────────────────────────
BG    = '#0d1117'; PANEL = '#161b22'; GRID  = '#21262d'
WHITE = '#e6edf3'; MUTED = '#8b949e'
C_LOG = '#79c0ff'; C_BASE_CLR = '#e3b341'; C_OPT = '#56d364'

plt.rcParams.update({
    'figure.facecolor': BG,  'axes.facecolor': PANEL,
    'axes.edgecolor':  GRID, 'axes.labelcolor': WHITE,
    'xtick.color': MUTED,   'ytick.color': MUTED,
    'text.color':  WHITE,   'grid.color':  GRID,
    'grid.linewidth': 0.6,  'legend.facecolor': '#1c2128',
    'legend.edgecolor': GRID,
})

# ── Modelo ─────────────────────────────────────────────────────────────────
def phi_logistic(t, K, gamma, t0):
    """Φ_L(t) = 1 + K / (1 + e^(-γ(t-t₀)))  — Verhulst (1838)"""
    return 1 + K / (1 + np.exp(-gamma * (t - t0)))

def V_model(t, Capital, r, K, gamma, t0):
    """V(t) = Capital × (1+r)^t × Φ_L(t)/Φ_L(0)"""
    phi = phi_logistic(t, K, gamma, t0)
    phi0 = phi_logistic(0, K, gamma, t0)
    return Capital * (1 + r)**t * phi / phi0

# ── Escenarios (Tabla 2, Vilar 2026a) ────────────────────────────────────
ESCENARIOS = {
    'BASE':      {'K': 2.0, 'gamma': 0.30, 't0': 3.0, 'r': 0.20, 'color': C_BASE_CLR, 'ls': '-'},
    'ACELERADO': {'K': 4.0, 'gamma': 0.40, 't0': 3.0, 'r': 0.25, 'color': C_LOG,      'ls': '--'},
    'ÓPTIMO':    {'K': 6.0, 'gamma': 0.50, 't0': 3.0, 'r': 0.30, 'color': C_OPT,      'ls': ':'},
}

CAPITAL = 100_000  # €
t = np.linspace(0, 13, 500)  # 13 años desde 2026

def plot(out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(16, 9), facecolor=BG)
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.32,
                            left=0.07, right=0.97, top=0.92, bottom=0.08)

    # Panel 1: Φ_L(t) por escenario
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(PANEL)
    for nombre, p in ESCENARIOS.items():
        phi = phi_logistic(t, p['K'], p['gamma'], p['t0'])
        ax1.plot(t + 2026, phi, color=p['color'], lw=2, ls=p['ls'],
                 label=f"{nombre}  K={p['K']}, γ={p['gamma']}")
    ax1.axvline(2026 + ESCENARIOS['BASE']['t0'], color=MUTED, lw=0.8, ls=':', alpha=0.6)
    ax1.text(2026 + ESCENARIOS['BASE']['t0'] + 0.1, 1.05, 't₀', color=MUTED, fontsize=8)
    ax1.set_title('Φ_L(t) — factor de autorreplicación', color=WHITE)
    ax1.set_xlabel('Año'); ax1.set_ylabel('Φ_L(t)')
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.25)

    # Panel 2: V(t) en k€
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor(PANEL)
    V_pure = CAPITAL * (1 + 0.20)**t
    ax2.plot(t + 2026, V_pure / 1000, color=MUTED, lw=1, ls='--',
             alpha=0.5, label='(1+r)^t puro (sin Φ)')
    for nombre, p in ESCENARIOS.items():
        V = V_model(t, CAPITAL, p['r'], p['K'], p['gamma'], p['t0'])
        ax2.plot(t + 2026, V / 1000, color=p['color'], lw=2.2, ls=p['ls'],
                 label=f"{nombre}  → {V[-1]/1000:.0f}k€")
    ax2.set_title(f'V(t) — Capital inicial {CAPITAL/1000:.0f}k€', color=WHITE)
    ax2.set_xlabel('Año'); ax2.set_ylabel('Valor (k€)')
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.25)

    # Panel 3: derivada dΦ/dt (velocidad de adopción)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_facecolor(PANEL)
    for nombre, p in ESCENARIOS.items():
        phi = phi_logistic(t, p['K'], p['gamma'], p['t0'])
        dphi = np.gradient(phi, t)
        ax3.plot(t + 2026, dphi, color=p['color'], lw=2, ls=p['ls'], label=nombre)
    ax3.set_title('dΦ_L/dt — velocidad de adopción tecnológica', color=WHITE)
    ax3.set_xlabel('Año'); ax3.set_ylabel('dΦ_L/dt')
    ax3.legend(fontsize=8); ax3.grid(True, alpha=0.25)

    # Panel 4: sensibilidad a K
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.set_facecolor(PANEL)
    for K_val, alpha in [(1, 0.4), (2, 0.6), (4, 0.8), (6, 0.9), (8, 1.0)]:
        V = V_model(t, CAPITAL, 0.20, K_val, 0.30, 3.0)
        ax4.plot(t + 2026, V / 1000, color=C_LOG, lw=1.8, alpha=alpha,
                 label=f'K={K_val}  → {V[-1]/1000:.0f}k€')
    ax4.set_title('Sensibilidad a K  (γ=0.30 fijo)', color=WHITE)
    ax4.set_xlabel('Año'); ax4.set_ylabel('Valor (k€)')
    ax4.legend(fontsize=7.5); ax4.grid(True, alpha=0.25)

    # Panel 5: sensibilidad a γ
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.set_facecolor(PANEL)
    for g_val, alpha in [(0.10, 0.4), (0.20, 0.6), (0.30, 0.75), (0.50, 0.9), (1.00, 1.0)]:
        V = V_model(t, CAPITAL, 0.20, 2.0, g_val, 3.0)
        ax5.plot(t + 2026, V / 1000, color=C_OPT, lw=1.8, alpha=alpha,
                 label=f'γ={g_val}  → {V[-1]/1000:.0f}k€')
    ax5.set_title('Sensibilidad a γ  (K=2 fijo)', color=WHITE)
    ax5.set_xlabel('Año'); ax5.set_ylabel('Valor (k€)')
    ax5.legend(fontsize=7.5); ax5.grid(True, alpha=0.25)

    # Panel 6: sensibilidad a t₀
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.set_facecolor(PANEL)
    for t0_val, alpha in [(1.0, 0.5), (2.0, 0.65), (3.0, 0.80), (5.0, 0.90), (8.0, 1.0)]:
        V = V_model(t, CAPITAL, 0.20, 2.0, 0.30, t0_val)
        ax6.plot(t + 2026, V / 1000, color=C_BASE_CLR, lw=1.8, alpha=alpha,
                 label=f't₀={int(t0_val+2026)}  → {V[-1]/1000:.0f}k€')
    ax6.set_title('Sensibilidad a t₀  (K=2, γ=0.30)', color=WHITE)
    ax6.set_xlabel('Año'); ax6.set_ylabel('Valor (k€)')
    ax6.legend(fontsize=7.5); ax6.grid(True, alpha=0.25)

    fig.suptitle(
        'Modelo Escasez y Resiliencia — V(t) = Capital × (1+r)^t × Φ_L(t)/Φ_L(0)  '
        '|  Φ_L(t) = 1 + K/(1+e^(−γ(t−t₀)))  [Verhulst, 1838]',
        color=WHITE, fontweight='bold', fontsize=11)

    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f'Guardado: {out_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Genera curvas del modelo Escasez y Resiliencia')
    parser.add_argument('--output', default=str(DEFAULT_OUT),
                        help='Ruta de salida del PNG (default: figuras/modelo_curvas_comparativa.png)')
    args = parser.parse_args()
    plot(args.output)
