"""
ezhook_viewer.py — Streamlit app for ezhook 4-probe resistance measurements.

Filename convention (either separator works):
  {prefix}_{top|bot}{sample}[_-]{pos\d+}[_-]{length}mm.json
  e.g.  ezhook_botc1_pos2_80.6mm.json
        ezhook_topcvd2-pos3-83.11mm.json

Current source : Keithley 2461 (specs below, 1-year, 23°C ±5°C)
Voltage meter  : Keithley 2182A
Method         : 4-probe delta (4probe_delta key only; 2probe ignored)
"""

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ── Settings persistence ──────────────────────────────────────────────────────
_SETTINGS_FILE = Path(__file__).parent / ".ezhook_viewer_settings.json"
_DEFAULTS = {
    "fs_axis_label":  13,
    "fs_tick":        11,
    "fs_legend":      11,
    "fs_inside_text": 10,
    "sigma_L_mm":     0.05,
    "sigma_w_mm":     0.0,
    "sigma_h_mm":     0.0,
}

def _load_settings() -> dict:
    try:
        return {**_DEFAULTS, **json.loads(_SETTINGS_FILE.read_text())}
    except Exception:
        return dict(_DEFAULTS)

def _save_settings(s: dict):
    try:
        _SETTINGS_FILE.write_text(json.dumps(s, indent=2))
    except Exception:
        pass


# ── Keithley 2461 current source accuracy (1-year, 23°C ±5°C) ────────────────
# Range keys = upper bound of range in Amps
CURRENT_ACCURACY_2461 = {
    1e-6:   {'ppm': 250,  'offset': 700e-12},
    10e-6:  {'ppm': 250,  'offset': 1e-9},
    100e-6: {'ppm': 200,  'offset': 10e-9},
    1e-3:   {'ppm': 200,  'offset': 100e-9},
    10e-3:  {'ppm': 200,  'offset': 1e-6},
    100e-3: {'ppm': 200,  'offset': 10e-6},
    1.0:    {'ppm': 500,  'offset': 500e-6},
    4.0:    {'ppm': 1000, 'offset': 2.5e-3},
    5.0:    {'ppm': 1000, 'offset': 2.5e-3},
    7.0:    {'ppm': 1500, 'offset': 5e-3},
    10.0:   {'ppm': 1500, 'offset': 5e-3},
}

# ── Keithley 2182A voltage accuracy (1-year, 23°C ±5°C) ──────────────────────
VOLTAGE_ACCURACY = {
    '10mV':  {'ppm': 50, 'offset': 50e-9},
    '100mV': {'ppm': 30, 'offset': 757e-9},
}


def _current_uncertainty(I_A: float) -> float:
    ranges = sorted(CURRENT_ACCURACY_2461.keys())
    for r in ranges:
        if I_A <= r:
            s = CURRENT_ACCURACY_2461[r]
            return (s['ppm'] / 1e6) * I_A + s['offset']
    s = CURRENT_ACCURACY_2461[ranges[-1]]
    return (s['ppm'] / 1e6) * I_A + s['offset']


def _voltage_uncertainty(V: float):
    key = '10mV' if abs(V) <= 0.01 else '100mV'
    s = VOLTAGE_ACCURACY[key]
    return (s['ppm'] / 1e6) * abs(V) + s['offset'], key


def _resistance_uncertainty(R: float, V: float, I: float, dV: float, dI: float) -> float:
    """σ_R = R · √((σ_V/V)² + (σ_I/I)²)"""
    if V <= 0 or I <= 0 or R <= 0:
        return 0.0
    return R * math.sqrt((dV / V) ** 2 + (dI / I) ** 2)


# ── Filename parser ───────────────────────────────────────────────────────────
# {prefix}_(top|bot){sample}[_-](pos\d+)[_-]{length}mm.json
_FNAME_RE = re.compile(
    r'^(.+?)_(top|bot)([a-zA-Z0-9]+)[_\-](pos\d+)[_\-]([\d.]+)mm\.json$',
    re.IGNORECASE
)


def parse_filename(filename: str) -> dict | None:
    m = _FNAME_RE.match(filename)
    if not m:
        return None
    _, layer, sample, position, length = m.groups()
    layer    = layer.lower()
    sample   = sample.lower()
    position = position.lower()
    return {
        'sample_name': sample,
        'location':    position,
        'length_mm':   float(length),
        'layer':       layer,
        'key':         f"{sample}-{position}",
        'group_key':   f"{sample}-{position}-{length}mm-{layer}",
    }


# ── CSV meta parser (adafruit-style # comment headers) ───────────────────────
def parse_csv_meta(header: dict) -> dict | None:
    """Build the same meta dict as parse_filename but from CSV header fields."""
    try:
        sample = header.get('sample_name', '').lower()
        layer  = header.get('surface', '').lower()
        pos    = str(header.get('position', '')).strip()
        length = float(header.get('length', 0))
        if not sample or layer not in ('top', 'bot') or not pos:
            return None
        position = f"pos{pos}"
        return {
            'sample_name': sample,
            'location':    position,
            'length_mm':   length,
            'layer':       layer,
            'key':         f"{sample}-{position}",
            'group_key':   f"{sample}-{position}-{length}mm-{layer}",
        }
    except Exception:
        return None


# ── CSV reader ────────────────────────────────────────────────────────────────
def read_csv_file(file_content) -> tuple[dict | None, list | None]:
    """Parse adafruit-style CSV (# comment headers + data rows).
    Returns (meta, measurements) matching the JSON reader output format.
    """
    try:
        text  = file_content.decode('utf-8') if isinstance(file_content, bytes) else file_content
        lines = text.splitlines()

        header: dict = {}
        data_lines: list[str] = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith('#'):
                content = line[1:].strip()
                if '=' in content and not content.lower().startswith('columns'):
                    key, _, val = content.partition('=')
                    header[key.strip()] = val.strip()
            else:
                data_lines.append(line)

        rows = []
        for line in data_lines:
            parts = line.split(',')
            if len(parts) < 6:
                continue
            try:
                rows.append({
                    'current_A':       float(parts[2]),
                    'resistance_mOhm': float(parts[3]),
                    'stdev_mOhm':      float(parts[4]),
                    'n_points':        int(parts[5]),
                })
            except ValueError:
                continue

        if not rows:
            return None, None

        df = pd.DataFrame(rows)
        measurements = []
        for current_A, grp in df.groupby('current_A'):
            R_mean      = grp['resistance_mOhm'].mean() * 1e-3
            std_within  = math.sqrt((grp['stdev_mOhm'] ** 2).mean()) * 1e-3
            std_between = grp['resistance_mOhm'].std(ddof=1) * 1e-3 if len(grp) > 1 else 0.0
            measurements.append({
                'current_A':      float(current_A),
                'n_measurements': int(grp['n_points'].sum()),
                'R_Ohm':          R_mean,
                'std_Ohm':        math.sqrt(std_within ** 2 + std_between ** 2),
            })
        measurements.sort(key=lambda x: x['current_A'])

        return parse_csv_meta(header), measurements or None

    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        return None, None


# ── JSON reader ───────────────────────────────────────────────────────────────
def read_json(file_content) -> list | None:
    """Return [{current_A, n_measurements, R_Ohm, std_Ohm}, ...] from 4probe_delta."""
    try:
        raw = file_content.decode('utf-8') if isinstance(file_content, bytes) else file_content
        data = json.loads(raw)
        out = []
        for m in data.get('measurements', []):
            probe = m.get('4probe_delta', {})
            if 'R_mOhm' not in probe:
                continue
            out.append({
                'current_A':      float(m['current_A']),
                'n_measurements': int(m.get('n_measurements', 0)),
                'R_Ohm':          probe['R_mOhm'] * 1e-3,
                'std_Ohm':        probe['std_uOhm'] * 1e-6,
            })
        return out or None
    except Exception as e:
        st.error(f"Error reading JSON: {e}")
        return None


# ── dims.txt parser (same 3-line-block format as kdf_viewer) ──────────────────
def parse_dims_txt(content) -> dict:
    text  = content.decode('utf-8') if isinstance(content, bytes) else content
    lines = [l.strip() for l in text.splitlines()]
    dims  = {}
    i = 0
    while i < len(lines):
        if not lines[i]:
            i += 1
            continue
        key = lines[i]
        if i + 2 < len(lines):
            try:
                widths      = [float(v) for v in lines[i+1].split(',') if v.strip()]
                thicknesses = [float(v) for v in lines[i+2].split(',') if v.strip()]
                w_mean = float(np.mean(widths))
                w_std  = float(np.std(widths, ddof=1)) if len(widths) > 1 else 0.0
                n_t = len(thicknesses)
                if n_t >= 2:
                    pos    = np.linspace(0.0, 1.0, n_t)
                    coeffs = np.polyfit(pos, thicknesses, 1)
                    resid  = np.array(thicknesses) - np.polyval(coeffs, pos)
                    h_start = float(np.polyval(coeffs, 0.0))
                    h_end   = float(np.polyval(coeffs, 1.0))
                    h_std   = float(np.std(resid, ddof=2)) if n_t > 2 else 0.0
                else:
                    h_start = h_end = float(thicknesses[0])
                    h_std = 0.0
                dims[key] = {
                    'height_start': h_start, 'height_end': h_end,
                    'height_mean':  (h_start + h_end) / 2.0,
                    'height_std':   h_std,
                    'width_mean':   w_mean, 'width_std': w_std,
                }
                i += 3
                continue
            except ValueError:
                pass
        i += 1
    return dims


# ── Resistivity / IACS ────────────────────────────────────────────────────────
def calculate_resistivity(R, sigma_R, L_mm, sigma_L_mm, w_mm, sigma_w, h_mm, sigma_h):
    """Return (rho, sigma_rho) in Ω·m with full quadrature propagation."""
    L, w, h = L_mm / 1000, w_mm / 1000, h_mm / 1000
    sL, sw, sh = sigma_L_mm / 1000, sigma_w / 1000, sigma_h / 1000
    A   = w * h
    rho = R * A / L
    if R <= 0 or A <= 0:
        return rho, 0.0
    rel = math.sqrt((sigma_R / R) ** 2 + (sw / w) ** 2 + (sh / h) ** 2 + (sL / L) ** 2)
    return rho, rho * rel


def resistivity_to_iacs(rho: float) -> float:
    return (1.7241e-8 / rho) * 100


# ── Plot helper ───────────────────────────────────────────────────────────────
def apply_font_sizes(fig, axis_label, tick, legend, inside_text):
    fig.update_layout(
        font=dict(size=tick),
        xaxis=dict(title_font=dict(size=axis_label), tickfont=dict(size=tick)),
        yaxis=dict(title_font=dict(size=axis_label), tickfont=dict(size=tick)),
        legend=dict(font=dict(size=legend)),
    )
    fig.update_traces(textfont=dict(size=inside_text))
    return fig


# ── Length group assignment ───────────────────────────────────────────────────
_LENGTH_TARGETS = [45, 80]

def _assign_length_group(L: float) -> str:
    target = min(_LENGTH_TARGETS, key=lambda t: abs(L - t))
    return f"~{target} mm"


# ── IACS comparison figure builder ────────────────────────────────────────────
def _build_iacs_comparison_fig(
    plot_df, all_currents, layer_symbols, colors,
    show_separators, x_tickangle, fs_tick, title,
):
    groups          = plot_df['Group'].unique()
    tickvals        = list(range(len(groups)))
    ticktext        = list(groups)
    sample_by_group = [plot_df[plot_df['Group'] == g]['Sample'].iloc[0] for g in groups]
    n_cur           = len(all_currents)
    SPACING         = 0.15

    fig = go.Figure()
    for i, grp in enumerate(groups):
        sub   = plot_df[plot_df['Group'] == grp]
        color = colors[i % len(colors)]
        for layer, sym in layer_symbols.items():
            sub_l = sub[sub['Layer'] == layer]
            if sub_l.empty:
                continue
            for _, row in sub_l.iterrows():
                cur_idx = all_currents.index(row['I (A)'])
                x_off   = i + (cur_idx - (n_cur - 1) / 2) * SPACING
                fig.add_trace(go.Scatter(
                    x=[x_off],
                    y=[row['IACS (%)']],
                    error_y=dict(type='data', array=[row['σ_IACS (%)']],
                                 visible=True, thickness=2, width=5),
                    mode='markers',
                    marker=dict(
                        size=8 + cur_idx * 4,
                        color=color, symbol=sym,
                        line=dict(width=1, color='black'),
                    ),
                    name=f"{grp} ({layer}) {row['I (A)']:.4g} A",
                    legendgroup=f"{grp}-{layer}",
                    showlegend=True,
                ))

    if show_separators and len(sample_by_group) > 0:
        segs, seg_start, cur_samp = [], 0, sample_by_group[0]
        for j, s in enumerate(sample_by_group[1:], 1):
            if s != cur_samp:
                segs.append((cur_samp, seg_start, j - 1))
                seg_start, cur_samp = j, s
        segs.append((cur_samp, seg_start, len(sample_by_group) - 1))
        for samp, start, end in segs:
            mid      = (start + end) / 2
            samp_df  = plot_df[plot_df['Sample'] == samp]
            iacs_mid = samp_df['IACS (%)'].mean()
            iacs_rng = samp_df['IACS (%)'].max() - samp_df['IACS (%)'].min()
            fig.add_annotation(
                x=mid, y=1.02, xref='x', yref='paper',
                text=f"<b>{samp.upper()}</b><br>{iacs_mid:.1f}% ± {iacs_rng:.1f}%",
                showarrow=False, xanchor='center', yanchor='bottom',
                font=dict(size=fs_tick), align='center',
            )
        for _, start, end in segs[:-1]:
            fig.add_shape(
                type='line', x0=end + 0.5, x1=end + 0.5, xref='x',
                y0=0, y1=1, yref='paper',
                line=dict(color='lightgrey', width=1, dash='dash'),
            )

    fig.add_hline(y=100, line_dash="dash", line_color="orange",
                  annotation_text="100% IACS (Pure Cu)")
    layout_kw = dict(
        title=title,
        xaxis=dict(tickmode='array', tickvals=tickvals, ticktext=ticktext,
                   tickangle=x_tickangle),
        yaxis_title='%IACS',
        height=540,
        template='plotly_white',
    )
    if show_separators:
        layout_kw['margin'] = dict(t=120)
    fig.update_layout(**layout_kw)
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="ezhook 4-probe IACS Analyzer", page_icon="⚡", layout="wide")
    st.title("⚡ ezhook 4-probe %IACS Analyzer")
    st.caption(
        "Current source: Keithley 2461  ·  Voltage meter: Keithley 2182A  ·  4-probe delta method"
    )

    cfg = _load_settings()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("🔤 Chart Font Sizes")
        fs_axis_label  = st.slider("Axis labels",   8, 28, cfg["fs_axis_label"])
        fs_tick        = st.slider("Tick labels",   6, 24, cfg["fs_tick"])
        fs_legend      = st.slider("Legend",        6, 24, cfg["fs_legend"])
        fs_inside_text = st.slider("Text in chart", 6, 24, cfg["fs_inside_text"])

        st.header("📐 X-axis Layout")
        x_tickangle     = st.slider("X-axis label rotation", 0, 90, 45)
        show_separators = st.checkbox("Show sample separators", value=False)

        st.header("⚙️ Dimension Uncertainty")
        sigma_L_mm = st.number_input(
            "σ_L — length (mm)", min_value=0.0, value=cfg["sigma_L_mm"], step=0.01, format="%.3f"
        )
        sigma_w_mm = st.number_input(
            "σ_w — width (mm)", min_value=0.0, value=cfg["sigma_w_mm"], step=0.001, format="%.4f",
            help="Override width uncertainty. 0 = use value from dims.txt."
        )
        sigma_h_mm = st.number_input(
            "σ_h — thickness (mm)", min_value=0.0, value=cfg["sigma_h_mm"], step=0.001, format="%.4f",
            help="Override thickness uncertainty. 0 = use value from dims.txt."
        )

    _save_settings({
        "fs_axis_label": fs_axis_label, "fs_tick": fs_tick,
        "fs_legend": fs_legend, "fs_inside_text": fs_inside_text,
        "sigma_L_mm": sigma_L_mm, "sigma_w_mm": sigma_w_mm, "sigma_h_mm": sigma_h_mm,
    })

    # ── 1. dims.txt ───────────────────────────────────────────────────────────
    st.subheader("1. Upload dims.txt")
    dims_file = st.file_uploader("Sample cross-section dimensions", type=['txt'], key='dims')
    dims_dict: dict = {}
    if dims_file:
        dims_dict = parse_dims_txt(dims_file.read())
        if dims_dict:
            rows = [{'Key': k,
                     'Width mean (mm)':      f"{v['width_mean']:.3f}",
                     'Width σ (mm)':         f"{v['width_std']:.4f}",
                     'Thickness start (mm)': f"{v['height_start']:.3f}",
                     'Thickness end (mm)':   f"{v['height_end']:.3f}",
                     'Taper σ_fit (mm)':     f"{v['height_std']:.4f}"}
                    for k, v in dims_dict.items()]
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        else:
            st.warning("Could not parse dims.txt — check format.")

    # ── 2. JSON / CSV files ───────────────────────────────────────────────────
    st.subheader("2. Upload ezhook JSON or CSV Files")
    uploaded = st.file_uploader(
        "Choose ezhook JSON or CSV files", type=['json', 'csv'], accept_multiple_files=True
    )

    if not uploaded:
        st.info("Upload JSON or CSV files to get started.")
        return

    st.success(f"Uploaded {len(uploaded)} files")

    # Read all files immediately — Streamlit file objects can only be read once
    file_data: dict = {}       # fname -> (meta, measurements_list)
    available_currents: set = set()
    preview_rows = []

    for uf in uploaded:
        content = uf.read()
        if uf.name.lower().endswith('.csv'):
            meta, measurements = read_csv_file(content)
        else:
            meta         = parse_filename(uf.name)
            measurements = read_json(content)

        if measurements:
            file_data[uf.name] = (meta, measurements)
            for m in measurements:
                available_currents.add(m['current_A'])

        if meta is None:
            note = '⚠ CSV header not parsed' if uf.name.lower().endswith('.csv') else '⚠ filename not parsed'
            preview_rows.append({
                'File': uf.name, 'Sample': '?', 'Layer': '?', 'Location': '?',
                'Length (mm)': '?', 'Currents (A)': '?', 'Dims': '?',
                'Note': note,
            })
            continue

        key = meta['key']
        d   = dims_dict.get(key) or dims_dict.get(meta['sample_name'])
        curr_str = ', '.join(f"{m['current_A']:.4g}" for m in (measurements or []))
        preview_rows.append({
            'File':          uf.name,
            'Sample':        meta['sample_name'],
            'Layer':         meta['layer'],
            'Location':      meta['location'],
            'Length (mm)':   meta['length_mm'],
            'Currents (A)':  curr_str,
            'Dims':          'match' if d else f"⚠ '{key}' not in dims.txt",
            'Note':          '',
        })

    st.subheader("📏 Parsed File Metadata")
    st.dataframe(pd.DataFrame(preview_rows), use_container_width=True)

    # ── Length overrides ──────────────────────────────────────────────────────
    length_overrides: dict = {}
    with st.expander("Override lengths (optional)"):
        for uf in uploaded:
            fd_meta = file_data.get(uf.name, (None, None))[0]
            default = fd_meta['length_mm'] if fd_meta else 60.0
            length_overrides[uf.name] = st.number_input(
                uf.name, min_value=0.0, value=default, step=0.01, format="%.2f",
                key=f"len_{uf.name}"
            )

    st.divider()

    # ── Compute IACS for every file × every current level ────────────────────
    results_data = []

    for fname, (meta, measurements) in file_data.items():
        if meta is None:
            continue

        key = meta['key']
        d   = dims_dict.get(key) or dims_dict.get(meta['sample_name'])
        if d:
            w_mm           = d['width_mean']
            h_mm           = d['height_mean']
            h_start, h_end = d['height_start'], d['height_end']
            sigma_w        = sigma_w_mm if sigma_w_mm > 0 else d['width_std']
            sigma_h        = sigma_h_mm if sigma_h_mm > 0 else d['height_std']
        else:
            w_mm, h_mm     = 1.80, 9.00
            h_start = h_end = 9.00
            sigma_w        = sigma_w_mm if sigma_w_mm > 0 else 0.05
            sigma_h        = sigma_h_mm if sigma_h_mm > 0 else 0.05

        L_mm = length_overrides.get(fname, meta['length_mm'])

        for m in measurements:
            I_A      = m['current_A']
            R        = m['R_Ohm']
            std_meas = m['std_Ohm']        # delta-method noise already in the file

            V              = I_A * R
            dV, vrange     = _voltage_uncertainty(V)
            dI             = _current_uncertainty(I_A)
            sigma_R_inst   = _resistance_uncertainty(R, V, I_A, dV, dI)
            sigma_R_total  = math.sqrt(sigma_R_inst ** 2 + std_meas ** 2)

            rho, sigma_rho = calculate_resistivity(
                R, sigma_R_total, L_mm, sigma_L_mm, w_mm, sigma_w, h_mm, sigma_h
            )
            iacs       = resistivity_to_iacs(rho)
            sigma_iacs = iacs * (sigma_rho / rho) if rho > 0 else 0.0

            var_R_inst = (sigma_R_inst / R) ** 2 if R > 0 else 0.0
            var_R_meas = (std_meas     / R) ** 2 if R > 0 else 0.0
            var_w      = (sigma_w      / w_mm)   ** 2
            var_h      = (sigma_h      / h_mm)   ** 2
            var_L      = (sigma_L_mm   / L_mm)   ** 2

            results_data.append({
                'Sample':            meta['sample_name'],
                'Location':          meta['location'],
                'Layer':             meta['layer'],
                'Group':             f"{meta['sample_name']}-{meta['location']}",
                'I (A)':             I_A,
                'n_meas':            m['n_measurements'],
                'V range':           vrange,
                'Length (mm)':       L_mm,
                'Width mean (mm)':   w_mm,
                'Width σ (mm)':      sigma_w,
                'Height start (mm)': h_start,
                'Height end (mm)':   h_end,
                'Height mean (mm)':  h_mm,
                'Height σ_fit (mm)': sigma_h,
                'R (Ω)':             R,
                'σ_R inst (Ω)':      sigma_R_inst,
                'σ_R meas (Ω)':      std_meas,
                'σ_R total (Ω)':     sigma_R_total,
                'Resistivity (Ω·m)': rho,
                'IACS (%)':          iacs,
                'σ_IACS (%)':        sigma_iacs,
                '_var_R_inst':       var_R_inst,
                '_var_R_meas':       var_R_meas,
                '_var_w':            var_w,
                '_var_h':            var_h,
                '_var_L':            var_L,
            })

    if not results_data:
        st.error("No valid files processed.")
        return

    results_df  = pd.DataFrame(results_data)
    results_df['Length group'] = results_df['Length (mm)'].apply(_assign_length_group)
    all_currents = sorted(results_df['I (A)'].unique())
    colors       = px.colors.qualitative.Plotly
    layer_symbols = {'top': 'circle', 'bot': 'diamond'}

    # ── IACS comparison plots (one per length group) ──────────────────────────
    st.subheader("📊 %IACS Comparison")

    cur_options = ["All currents"] + [f"{c:.4g} A" for c in all_currents]
    selected_cur = st.selectbox("Filter by current level", options=cur_options, index=0)

    if selected_cur == "All currents":
        plot_df = results_df.copy()
    else:
        c_val   = float(selected_cur.split()[0])
        plot_df = results_df[np.isclose(results_df['I (A)'], c_val)].copy()

    length_groups = sorted(plot_df['Length group'].unique())
    cols = st.columns(len(length_groups))
    for col, lg in zip(cols, length_groups):
        lg_df = plot_df[plot_df['Length group'] == lg]
        fig = _build_iacs_comparison_fig(
            lg_df, all_currents, layer_symbols, colors,
            show_separators, x_tickangle, fs_tick,
            title=f'%IACS — {lg}  (● top  ◆ bot | size ∝ current)',
        )
        col.plotly_chart(
            apply_font_sizes(fig, fs_axis_label, fs_tick, fs_legend, fs_inside_text),
            use_container_width=True,
        )

    # ── IACS vs current (consistency check) ──────────────────────────────────
    st.subheader("📈 IACS vs Current Level — Consistency Check")
    st.caption("Values should be stable across current levels for a well-behaved measurement.")

    fig_cur = go.Figure()
    for i, grp in enumerate(results_df['Group'].unique()):
        sub   = results_df[results_df['Group'] == grp]
        color = colors[i % len(colors)]
        for layer, sym in layer_symbols.items():
            sub_l = sub[sub['Layer'] == layer].sort_values('I (A)')
            if sub_l.empty:
                continue
            fig_cur.add_trace(go.Scatter(
                x=sub_l['I (A)'],
                y=sub_l['IACS (%)'],
                error_y=dict(type='data', array=sub_l['σ_IACS (%)'].tolist(),
                             visible=True, thickness=1.5, width=4),
                mode='lines+markers',
                marker=dict(size=10, color=color, symbol=sym,
                            line=dict(width=1, color='black')),
                line=dict(color=color, width=1.5, dash='dot'),
                name=f"{grp} ({layer})",
            ))

    fig_cur.add_hline(y=100, line_dash="dash", line_color="orange",
                      annotation_text="100% IACS")
    fig_cur.update_layout(
        title='%IACS vs Current Level',
        xaxis_title='Current (A)',
        yaxis_title='%IACS',
        height=460,
        template='plotly_white',
    )
    st.plotly_chart(
        apply_font_sizes(fig_cur, fs_axis_label, fs_tick, fs_legend, fs_inside_text),
        use_container_width=True,
    )

    # ── Uncertainty breakdown ─────────────────────────────────────────────────
    st.subheader("🔍 Uncertainty Breakdown")
    st.caption("Fraction of total variance from each source.")

    if selected_cur == "All currents":
        best_c = max(all_currents)
        unc_df = results_df[np.isclose(results_df['I (A)'], best_c)].copy()
        st.caption(f"Showing highest current ({best_c:.4g} A). Select a single current above to change.")
    else:
        unc_df = plot_df.copy()

    unc_df['x_label'] = unc_df['Sample'] + '-' + unc_df['Location'] + ' / ' + unc_df['Layer']

    var_cols   = ['_var_R_inst', '_var_R_meas', '_var_w', '_var_h', '_var_L']
    src_labels = ['Instrument σ_I (2461)', 'Delta-method noise',
                  'Width variation', 'Thickness variation', 'Length variation']
    src_colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A']

    total_var = unc_df[var_cols].sum(axis=1)
    pct_df    = unc_df[var_cols].div(total_var, axis=0).mul(100)

    fig_bar = go.Figure()
    for col, label, color in zip(var_cols, src_labels, src_colors):
        y = pct_df[col]
        fig_bar.add_trace(go.Bar(
            x=unc_df['x_label'], y=y,
            name=label, marker_color=color,
            text=[f"{v:.0f}%" if v >= 5 else "" for v in y],
            textposition='inside', insidetextanchor='middle',
        ))
    fig_bar.update_layout(
        barmode='stack',
        title='Uncertainty Sources (% of total variance)',
        xaxis_title='Sample-Location / Layer',
        yaxis_title='% of Total Variance',
        xaxis=dict(tickangle=x_tickangle),
        height=400,
        template='plotly_white',
    )
    st.plotly_chart(
        apply_font_sizes(fig_bar, fs_axis_label, fs_tick, fs_legend, fs_inside_text),
        use_container_width=True,
    )

    # ── Summary statistics ────────────────────────────────────────────────────
    st.subheader("📈 Summary Statistics")
    summary = (
        results_df.groupby(['Sample', 'I (A)'])['IACS (%)']
        .agg(
            Mean=lambda x: round(x.mean(), 3),
            Std=lambda x: round(x.std(), 3) if len(x) > 1 else float('nan'),
            Max=lambda x: round(x.max(), 3),
            Min=lambda x: round(x.min(), 3),
            N='count',
        )
        .reset_index()
    )
    st.dataframe(
        summary.style
        .format({'I (A)': '{:.4g}', 'Mean': '{:.3f}', 'Std': '{:.3f}',
                 'Max': '{:.3f}', 'Min': '{:.3f}'}, na_rep='—')
        .background_gradient(subset=['Mean'], cmap='RdYlGn'),
        use_container_width=True,
        hide_index=True,
    )

    # ── Detailed results ──────────────────────────────────────────────────────
    st.subheader("📋 Detailed Results")
    _internal = ['Group'] + [c for c in results_df.columns if c.startswith('_var')]
    display_df = results_df.drop(columns=_internal).copy()
    for col in ['R (Ω)', 'σ_R inst (Ω)', 'σ_R meas (Ω)', 'σ_R total (Ω)', 'Resistivity (Ω·m)']:
        display_df[col] = display_df[col].apply(lambda x: f"{x:.3e}")
    for col in ['IACS (%)', 'σ_IACS (%)']:
        display_df[col] = display_df[col].apply(lambda x: f"{x:.3f}")
    st.dataframe(display_df, use_container_width=True)

    # ── Downloads ─────────────────────────────────────────────────────────────
    st.subheader("📥 Downloads")
    out_df = results_df.drop(columns=_internal)
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "📥 Full Results (CSV)",
            out_df.to_csv(index=False),
            "ezhook_iacs_results.csv", "text/csv",
        )
    with col2:
        st.download_button(
            "📥 IACS Summary (CSV)",
            results_df[['Sample', 'Location', 'Layer', 'I (A)', 'IACS (%)', 'σ_IACS (%)']].to_csv(index=False),
            "ezhook_iacs_summary.csv", "text/csv",
        )


if __name__ == "__main__":
    main()
