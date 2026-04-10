# Escasez y Resiliencia — Código y Cuadernos

🌐 **Language / Idioma:** [English](README.md) | Español

---

Repositorio companion del paper publicado en SSRN:

> **Vilar, J.A. (2026). *Escasez y Resiliencia: Un Marco Matemático para Invertir en la Era de la Autorreplicación Artificial.* SSRN Working Paper.**
> 📄 [Leer el paper en SSRN](https://papers.ssrn.com/) ← *(DOI pendiente de confirmación)*

---

## El modelo

$$V(t) = \text{Capital} \times (1+r)^t \times \frac{\Phi_L(t)}{\Phi_L(0)}$$

$$\Phi_L(t) = 1 + \frac{K}{1 + e^{-\gamma(t - t_0)}}$$

La curva logística $\Phi_L(t)$ — Verhulst (1838) — captura la difusión de la autorreplicación tecnológica (IA + robótica). La normalización por $\Phi_L(0)$ garantiza $V(0) = \text{Capital}$ con independencia de los parámetros.

**Parámetros:**

| Parámetro | Descripción | Calibración |
|-----------|-------------|-------------|
| `r` | Rentabilidad base compuesta | SOX trailing 12M / 100 |
| `K` | Multiplicador máximo de la tecnología | Amplitud de la ola |
| `γ` | Velocidad de adopción | Ritmo de releases LLM / fabs |
| `t₀` | Año de inflexión (máxima aceleración) | Detector SOX/Cobre |

---

## Estructura del repositorio

```
escasez-resilencia-public/
├── notebooks/
│   └── Laboratorio_Modelo_Escasez_Resiliencia.ipynb   ← análisis interactivo
├── scripts/
│   ├── plot_curves.py     ← genera figuras del modelo (standalone)
│   └── backtest.py        ← backtesting train/test con datos propios
├── figuras/               ← outputs generados por los scripts
├── README.md              ← versión en inglés (por defecto)
└── README.es.md           ← este fichero (español)
```

---

## Instalación

```bash
git clone https://github.com/josevilar-qbioai/escasez-resilencia-public.git
cd escasez-resilencia-public
pip install numpy matplotlib
```

El notebook requiere adicionalmente Jupyter:

```bash
pip install jupyter
jupyter notebook notebooks/Laboratorio_Modelo_Escasez_Resiliencia.ipynb
```

---

## Uso rápido

### Generar las curvas del modelo

```bash
python scripts/plot_curves.py
# Guarda: figuras/modelo_curvas_comparativa.png
```

### Backtesting con tus propios datos

`backtest.py` espera CSVs de precios en el siguiente formato:

```
fecha,precio_cierre,fuente,notas
2022-01-03,43.21,yahoo,
2022-01-04,43.85,yahoo,
```

```bash
python scripts/backtest.py --hist-dir /ruta/a/tus/csvs --out-dir resultados/
```

Los datos de cartera personal **no se incluyen** en este repositorio.

---

## Cuaderno interactivo

`notebooks/Laboratorio_Modelo_Escasez_Resiliencia.ipynb` es autocontenido — no requiere datos externos. Incluye:

- Fundamentos matemáticos del modelo logístico
- Visualización de $\Phi_L(t)$ por escenario (Base / Acelerado / Óptimo)
- Proyecciones $V(t)$ y análisis de sensibilidad a $K$, $\gamma$, $t_0$
- Propiedades del modelo: finitud del multiplicador, comportamiento en los límites, comparativa con modelos exponenciales
- Análisis de robustez del detector SOX/Cobre como estimador empírico de $t_0$

---

## Taxonomía de la cartera

El paper describe una cartera construida sobre cuatro pilares de escasez:

| Pilar | Activos | Justificación |
|-------|---------|---------------|
| **Escasez digital** | Bitcoin | Oferta fija por protocolo |
| **Escasez física** | Cobre, Metales preciosos | Recursos geológicos inelásticos |
| **Energía / Grid** | Uranio, Smart Grid | Cuello de botella de la IA |
| **Autorreplicación IA** | ETFs de AI, Robótica, Cuántica | Exposición directa a $K$ y $\gamma$ |
| **Resiliencia** | MSCI World, fondos índice | Amortiguador de volatilidad |

---

## Cita

```bibtex
@techreport{vilar2026escasez,
  author      = {Vilar, Jose Antonio},
  title       = {Escasez y Resiliencia: Un Marco Matemático para Invertir
                 en la Era de la Autorreplicación Artificial},
  year        = {2026},
  institution = {SSRN},
  note        = {Working Paper}
}
```

---

## Contacto

Jose Antonio Vilar · [qmetrika@proton.me](mailto:qmetrika@proton.me)
Estudiante de Grado en Ciencia de Datos · Universitat Oberta de Catalunya
