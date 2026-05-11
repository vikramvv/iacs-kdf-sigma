import json
import math
import re
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import eddy_analysis


_SETTINGS_FILE = Path(__file__).parent / ".kdf_viewer_settings.json"
_DEFAULTS = {
    "fs_axis_label":  13,
    "fs_tick":        11,
    "fs_legend":      11,
    "fs_inside_text": 10,
    "current_source": "6220",
    "sigma_L_mm":     0.05,
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

# ── Instrument specs (from iacs-uncertainity-app) ────────────────────────────
# Keithley 2450 current accuracy (1 year, 23°C ±5°C)
CURRENT_ACCURACY_2450 = {
    1e-6:  {'ppm': 250,  'offset': 700e-12},
    10e-6: {'ppm': 250,  'offset': 1e-9},
    100e-6:{'ppm': 200,  'offset': 10e-9},
    1e-3:  {'ppm': 200,  'offset': 100e-9},
    10e-3: {'ppm': 200,  'offset': 1e-6},
    100e-3:{'ppm': 200,  'offset': 10e-6},
    1.0:   {'ppm': 500,  'offset': 500e-6},
    4.0:   {'ppm': 1000, 'offset': 2.5e-3},
    5.0:   {'ppm': 1000, 'offset': 2.5e-3},
    7.0:   {'ppm': 1500, 'offset': 5e-3},
    10.0:  {'ppm': 1500, 'offset': 5e-3},
}

# Keithley 6220 current accuracy (1 year, 23°C ±5°C)
CURRENT_ACCURACY_6220 = {
    2e-9:   {'ppm': 4000, 'offset': 2e-12},
    20e-9:  {'ppm': 3000, 'offset': 10e-12},
    200e-9: {'ppm': 3000, 'offset': 100e-12},
    2e-6:   {'ppm': 1000, 'offset': 1e-9},
    20e-6:  {'ppm': 500,  'offset': 10e-9},
    200e-6: {'ppm': 500,  'offset': 100e-9},
    2e-3:   {'ppm': 500,  'offset': 1e-6},
    20e-3:  {'ppm': 500,  'offset': 10e-6},
    100e-3: {'ppm': 1000, 'offset': 50e-6},
}

VOLTAGE_ACCURACY = {
    '10mV':  {'ppm': 50, 'offset': 50e-9},
    '100mV': {'ppm': 30, 'offset': 757e-9},
}

SOURCE_TABLES = {'2450': CURRENT_ACCURACY_2450, '6220': CURRENT_ACCURACY_6220}


def _current_uncertainty(I_A: float, source: str) -> float:
    table = SOURCE_TABLES[source]
    ranges = sorted(table.keys())
    for r in ranges:
        if I_A <= r:
            s = table[r]
            return (s['ppm'] / 1e6) * I_A + s['offset']
    s = table[ranges[-1]]
    return (s['ppm'] / 1e6) * I_A + s['offset']


def _voltage_uncertainty(V: float):
    key = '10mV' if V <= 0.01 else '100mV'
    s = VOLTAGE_ACCURACY[key]
    return (s['ppm'] / 1e6) * V + s['offset'], key


def _resistance_uncertainty(R: float, V: float, I: float, dV: float, dI: float) -> float:
    """σ_R from voltmeter + ammeter specs via quadrature: σ_R = R·sqrt((σ_V/V)²+(σ_I/I)²)"""
    if V <= 0 or I <= 0 or R <= 0:
        return 0.0
    return R * math.sqrt((dV / V) ** 2 + (dI / I) ** 2)


# ── filename: {name}-{location}-{length}mm-{current}-{run}.kdf ──────────────
_FNAME_RE = re.compile(
    r'^([a-z0-9]+)-(pos\d+)-([\d.]+)mm-(\d+)m[as]-(\d+)\.kdf$', re.IGNORECASE
)

def parse_filename(filename):
    m = _FNAME_RE.match(filename)
    if not m:
        return None
    name, loc, length, current_ma, run = m.groups()
    run_int = int(run)
    return {
        'sample_name': name.lower(),
        'location': loc.lower(),
        'length_mm': float(length),
        'current_A': float(current_ma) * 1e-3,   # mA → A
        'run': run_int,
        'layer': 'top' if run_int <= 3 else 'bot',
        'key': f"{name.lower()}-{loc.lower()}",
        'group_key': f"{name.lower()}-{loc.lower()}-{length}mm-{'top' if run_int <= 3 else 'bot'}",
    }


def parse_dims_txt(content):
    """Three-line blocks: key / widths(csv) / thickness(csv), blank-line separated."""
    text = content.decode('utf-8') if isinstance(content, bytes) else content
    lines = [l.strip() for l in text.splitlines()]
    dims = {}
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
                # Width: average across sample
                w_mean = float(np.mean(widths))
                w_std  = float(np.std(widths, ddof=1)) if len(widths) > 1 else 0.0
                # Thickness: fit linear taper from start to end
                n_t = len(thicknesses)
                if n_t >= 2:
                    pos    = np.linspace(0.0, 1.0, n_t)
                    coeffs = np.polyfit(pos, thicknesses, 1)
                    resid  = np.array(thicknesses) - np.polyval(coeffs, pos)
                    h_start = float(np.polyval(coeffs, 0.0))
                    h_end   = float(np.polyval(coeffs, 1.0))
                    # ddof=2: two fitted parameters (slope + intercept)
                    h_taper_std = float(np.std(resid, ddof=2)) if n_t > 2 else 0.0
                else:
                    h_start = h_end = float(thicknesses[0])
                    h_taper_std = 0.0
                dims[key] = {
                    'height_start': h_start,
                    'height_end':   h_end,
                    'height_mean':  (h_start + h_end) / 2.0,
                    'height_std':   h_taper_std,
                    'width_mean':   w_mean,
                    'width_std':    w_std,
                    'widths':       widths,
                    'thicknesses':  thicknesses,
                }
                i += 3
                continue
            except ValueError:
                pass
        i += 1
    return dims


def read_kdf_file(file_content):
    try:
        content = file_content.decode('utf-8') if isinstance(file_content, bytes) else file_content
        lines = content.strip().split('\n')
        header_idx = 0
        for i, line in enumerate(lines):
            if 'Points' in line and 'Time' in line:
                header_idx = i
                break
        header = lines[header_idx].split('\t')
        data = []
        for line in lines[header_idx + 1:]:
            if line.strip():
                values = line.split('\t')
                if len(values) == len(header):
                    data.append(values)
        df = pd.DataFrame(data, columns=header)
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except Exception:
                pass
        return df
    except Exception as e:
        st.error(f"Error reading KDF file: {e}")
        return None


def calculate_resistivity(R, sigma_R, L_mm, sigma_L_mm, w_mm, sigma_w_mm, h_mm, sigma_h_mm):
    """Return (rho, sigma_rho) Ω·m with full quadrature propagation."""
    L, w, h = L_mm / 1000, w_mm / 1000, h_mm / 1000
    sL, sw, sh = sigma_L_mm / 1000, sigma_w_mm / 1000, sigma_h_mm / 1000
    A = w * h
    rel_A   = math.sqrt((sw/w)**2 + (sh/h)**2) if A > 0 else 0.0
    rho     = R * A / L
    rel_rho = math.sqrt((sigma_R/R)**2 + rel_A**2 + (sL/L)**2) if R > 0 else 0.0
    return rho, rho * rel_rho


def resistivity_to_iacs(rho):
    return (1.7241e-8 / rho) * 100


# ─────────────────────────────────────────────────────────────────────────────
def apply_font_sizes(fig, axis_label, tick, legend, inside_text):
    fig.update_layout(
        font=dict(size=tick),
        xaxis=dict(title_font=dict(size=axis_label), tickfont=dict(size=tick)),
        yaxis=dict(title_font=dict(size=axis_label), tickfont=dict(size=tick)),
        legend=dict(font=dict(size=legend)),
    )
    fig.update_traces(textfont=dict(size=inside_text))
    return fig


def main():
    st.set_page_config(page_title="Multi-KDF IACS Analyzer", page_icon="⚡", layout="wide")
    st.title("⚡ Multi-KDF %IACS Analyzer")

    cfg = _load_settings()

    # ── Sidebar: chart font sizes ─────────────────────────────────────────────
    with st.sidebar:
        # Mode selection
        st.header("📊 Analysis Mode")
        analysis_mode = st.radio(
            "Select data type",
            ["KDF (resistance)", "Eddy current (conductivity)"],
            index=0,
            help="KDF: resistance measurements → resistivity → IACS\nEddy: direct conductivity measurements → IACS"
        )
        is_kdf_mode = analysis_mode == "KDF (resistance)"
        is_eddy_mode = analysis_mode == "Eddy current (conductivity)"

        st.header("🔤 Chart Font Sizes")
        fs_axis_label  = st.slider("Axis labels",   min_value=8,  max_value=28, value=cfg["fs_axis_label"])
        fs_tick        = st.slider("Tick labels",   min_value=6,  max_value=24, value=cfg["fs_tick"])
        fs_legend      = st.slider("Legend",        min_value=6,  max_value=24, value=cfg["fs_legend"])
        fs_inside_text = st.slider("Text in chart", min_value=6,  max_value=24, value=cfg["fs_inside_text"])

        st.header("📐 X-axis Layout")
        x_tickangle = st.slider("X-axis label rotation", min_value=0, max_value=90, value=45,
                                help="Rotate x-axis labels to improve readability for crowded sample names.")
        show_sample_separators = st.checkbox(
            "Show sample separators and labels",
            value=False,
            help="Draw vertical separators between sample groups and label each sample name on the x-axis."
        )

    # ── 1. dims.txt ───────────────────────────────────────────────────────────
    if is_kdf_mode:
        st.subheader("1. Upload dims.txt")
        dims_file = st.file_uploader("Sample cross-section dimensions", type=['txt'], key='dims')
        dims_dict = {}
        if dims_file:
            dims_dict = parse_dims_txt(dims_file.read())
            if dims_dict:
                rows = [{'Key': k,
                         'Width mean (mm)':       f"{v['width_mean']:.3f}",
                         'Width σ (mm)':          f"{v['width_std']:.4f}",
                         'Thickness start (mm)':  f"{v['height_start']:.3f}",
                         'Thickness end (mm)':    f"{v['height_end']:.3f}",
                         'Taper σ_fit (mm)':      f"{v['height_std']:.4f}"}
                        for k, v in dims_dict.items()]
                st.dataframe(pd.DataFrame(rows), width='stretch')
            else:
                st.warning("Could not parse dims.txt — check format.")
    else:
        dims_dict = {}

    # ── 2. Instrument & uncertainty settings ──────────────────────────────────
    if is_kdf_mode:
        st.subheader("2. Instrument Settings")
        col_u1, col_u2 = st.columns(2)
        with col_u1:
            current_source = st.selectbox(
                "Current source",
                options=["6220", "2450"],
                format_func=lambda x: f"Keithley {x}",
                index=["6220", "2450"].index(cfg["current_source"]),
            )
        with col_u2:
            sigma_L_mm = st.number_input(
                "Length variation σ_L (mm)",
                min_value=0.0, value=cfg["sigma_L_mm"], step=0.01, format="%.3f"
            )
    else:
        current_source = cfg["current_source"]  # default
        sigma_L_mm = cfg["sigma_L_mm"]

    _save_settings({
        "fs_axis_label":  fs_axis_label,
        "fs_tick":        fs_tick,
        "fs_legend":      fs_legend,
        "fs_inside_text": fs_inside_text,
        "current_source": current_source,
        "sigma_L_mm":     sigma_L_mm,
    })

    if is_kdf_mode:
        st.caption(
            "Resistance uncertainty: σ_R = R · √((σ_V/V)² + (σ_I/I)²) from manufacturer specs. "
            "Voltage range auto-selected per Keithley 2182A specs."
        )

    # ── 3. Data files ──────────────────────────────────────────────────────────
    if is_kdf_mode:
        st.subheader("3. Upload KDF Files")
        uploaded_files = st.file_uploader(
            "Choose KDF files", type=['kdf', 'txt', 'csv'], accept_multiple_files=True
        )
        eddy_files = []
    else:
        st.subheader("3. Upload Eddy Current CSV Files")
        eddy_files = st.file_uploader(
            "Choose eddy current CSV files", type=['csv'], accept_multiple_files=True
        )
        uploaded_files = []

    data_files = uploaded_files if is_kdf_mode else eddy_files

    if not data_files:
        st.info("Upload data files to get started.")
        return

    st.success(f"Uploaded {len(data_files)} files")

    # ── per-file metadata preview ────────────────────────────────────────────
    if is_kdf_mode:
        st.subheader("📏 Parsed KDF Dimensions")
        preview_rows = []
        length_overrides = {}

        for uf in data_files:
            meta = parse_filename(uf.name)
            if meta is None:
                preview_rows.append({'File': uf.name, 'Sample': '?', 'Location': '?',
                                      'Layer': '?', 'Run': '?', 'I (mA)': '?',
                                      'Length (mm)': '?', 'Width (mm)': '?',
                                      'Thickness (mm)': '?', 'Note': '⚠ filename not parsed'})
                continue
            key = meta['key']
            d = dims_dict.get(key)
            t_str = f"{d['height_start']:.3f} → {d['height_end']:.3f}" if d else 'no match'
            w_str = f"{d['width_mean']:.3f} ± {d['width_std']:.4f}"    if d else 'no match'
            note  = '' if d else f"⚠ '{key}' not in dims.txt"
            preview_rows.append({
                'File': uf.name,
                'Sample': meta['sample_name'],
                'Location': meta['location'],
                'Layer': meta['layer'],
                'Run': meta['run'],
                'I (mA)': f"{meta['current_A']*1e3:.0f}",
                'Length (mm)': meta['length_mm'],
                'Width (mm)': w_str,
                'Thickness start→end (mm)': t_str,
                'Note': note,
            })

        st.dataframe(pd.DataFrame(preview_rows), width='stretch')

        with st.expander("Override lengths (optional)"):
            for uf in data_files:
                meta = parse_filename(uf.name)
                default = meta['length_mm'] if meta else 60.0
                length_overrides[uf.name] = st.number_input(
                    uf.name, min_value=0.0, value=default, step=0.01, format="%.2f",
                    key=f"len_{uf.name}"
                )

    else:
        st.subheader("📏 Parsed Eddy Current Metadata")
        preview_rows = []
        eddy_data = eddy_analysis.load_eddy_files(data_files)

        for filename in data_files:
            fname = filename.name
            meta = eddy_analysis.parse_eddy_filename(fname)
            df = eddy_data.get(fname)
            if meta is None:
                preview_rows.append({'File': fname, 'Sample': '?', 'Location': '?',
                                      'Layer': '?', 'Frequency (kHz)': '?', 'Measurements': '?',
                                      'Note': '⚠ filename not parsed'})
                continue
            freq = df['Frequency (kHz)'].iloc[0] if df is not None and 'Frequency (kHz)' in df.columns else '?'
            count = len(df) if df is not None else 0
            preview_rows.append({
                'File': fname,
                'Sample': meta['sample_name'],
                'Location': meta['location'],
                'Layer': meta['layer'],
                'Frequency (kHz)': freq,
                'Measurements': count,
                'Note': '',
            })

        st.dataframe(pd.DataFrame(preview_rows), width='stretch')
        length_overrides = {}  # not used for eddy

    st.divider()

    # ── Eddy sub-mode selection ───────────────────────────────────────────────
    if is_eddy_mode:
        eddy_analysis_mode = st.radio(
            "Eddy current analysis type",
            ["Statistical (grid mean/std)", "Trend (left-to-right)"],
            index=0,
            help="Statistical: aggregate measurements by row/column positions\nTrend: show conductivity variation across sample width"
        )
        eddy_sub_mode = 'grid' if 'Statistical' in eddy_analysis_mode else 'trend'
    else:
        eddy_sub_mode = None

    # ── process data ──────────────────────────────────────────────────────────
    if is_kdf_mode:
        # ── KDF processing: pool R across runs in each group ─────────────────────────────
        per_group = {}  # group_key → {meta, R_values, I_A, uf_name}

        for uf in data_files:
            meta = parse_filename(uf.name)
            df   = read_kdf_file(uf.read())
            if df is None or 'Resistance' not in df.columns:
                st.error(f"Could not process {uf.name}")
                continue
            R_mean = float(df['Resistance'].mean())
            gk = meta['group_key'] if meta else uf.name
            per_group.setdefault(gk, {
                'meta': meta, 'R_values': [],
                'I_A': meta['current_A'] if meta else 0.0,
                'uf_name': uf.name,
            })
            per_group[gk]['R_values'].append(R_mean)

        results_data = []

        for gk, grp in per_group.items():
            meta     = grp['meta']
            R_values = grp['R_values']
            I_A      = grp['I_A']

            R_pooled  = float(np.mean(R_values))
            sigma_run = float(np.std(R_values, ddof=1) / math.sqrt(len(R_values))) \
                        if len(R_values) > 1 else 0.0

            # Instrument σ_R from specs
            V        = I_A * R_pooled
            dV, vrange = _voltage_uncertainty(V)
            dI       = _current_uncertainty(I_A, current_source)
            sigma_R_inst = _resistance_uncertainty(R_pooled, V, I_A, dV, dI)
            # Total σ_R: instrument + run-to-run spread in quadrature
            sigma_R  = math.sqrt(sigma_R_inst ** 2 + sigma_run ** 2)

            L_mm = length_overrides.get(grp['uf_name'], meta['length_mm'] if meta else 60.0)

            key = meta['key'] if meta else None
            d   = dims_dict.get(key) if key else None
            if d:
                w_mm, sigma_w   = d['width_mean'],  d['width_std']
                h_mm, sigma_h   = d['height_mean'], d['height_std']
                h_start, h_end  = d['height_start'], d['height_end']
            else:
                w_mm, sigma_w   = 1.80, 0.05
                h_mm, sigma_h   = 9.00, 0.05
                h_start = h_end = 9.00

            rho, sigma_rho = calculate_resistivity(
                R_pooled, sigma_R, L_mm, sigma_L_mm, w_mm, sigma_w, h_mm, sigma_h
            )
            iacs       = resistivity_to_iacs(rho)
            sigma_iacs = iacs * (sigma_rho / rho)

            var_R_inst = (sigma_R_inst / R_pooled) ** 2
            var_R_run  = (sigma_run    / R_pooled) ** 2
            var_w      = (sigma_w      / w_mm)     ** 2
            var_h      = (sigma_h      / h_mm)     ** 2
            var_L      = (sigma_L_mm   / L_mm)     ** 2

            results_data.append({
                'Sample':          meta['sample_name'] if meta else gk,
                'Location':        meta['location']    if meta else '?',
                'Layer':           meta['layer']       if meta else '?',
                'Group':           f"{meta['sample_name']}-{meta['location']}" if meta else gk,
                'Runs averaged':   len(R_values),
                'I (mA)':          I_A * 1e3,
                'V range':         vrange,
                'Length (mm)':     L_mm,
                'Width mean (mm)':    w_mm,
                'Width σ (mm)':       sigma_w,
                'Height start (mm)':  h_start,
                'Height end (mm)':    h_end,
                'Height mean (mm)':   h_mm,
                'Height σ_fit (mm)':  sigma_h,
                'Area (mm²)':      w_mm * h_mm,
                'R pooled (Ω)':    R_pooled,
                'σ_R inst (Ω)':    sigma_R_inst,
                'σ_R run (Ω)':     sigma_run,
                'σ_R total (Ω)':   sigma_R,
                'Resistivity (Ω·m)': rho,
                'IACS (%)':        iacs,
                'σ_IACS (%)':      sigma_iacs,
                '_var_R_inst':     var_R_inst,
                '_var_R_run':      var_R_run,
                '_var_w':          var_w,
                '_var_h':          var_h,
                '_var_L':          var_L,
            })

        if not results_data:
            st.error("No valid KDF files processed.")
            return

        results_df = pd.DataFrame(results_data)
    else:
        # ── Eddy processing ───────────────────────────────────────────────────
        eddy_results = eddy_analysis.process_eddy_batch(eddy_data, eddy_sub_mode)

        results_data = []
        for filename, df in eddy_results.items():
            meta = eddy_analysis.parse_eddy_filename(filename)
            if meta is None:
                continue

            if eddy_sub_mode == 'grid':
                # For grid mode, create one row per sample with summary stats
                mean_cond = df['mean_conductivity'].mean()
                std_cond = df['std_conductivity'].mean()  # average std across positions
                iacs = eddy_analysis.conductivity_to_iacs(mean_cond)
                sigma_iacs = eddy_analysis.conductivity_to_iacs(mean_cond + std_cond) - iacs  # approx

                results_data.append({
                    'Sample': meta['sample_name'],
                    'Location': meta['location'],
                    'Layer': meta['layer'],
                    'Group': meta['group_key'],
                    'Positions measured': len(df),
                    'Conductivity mean (MS/m)': mean_cond,
                    'Conductivity σ (MS/m)': std_cond,
                    'IACS (%)': iacs,
                    'σ_IACS (%)': sigma_iacs,
                })
            else:  # trend mode
                # For trend mode, we might want to show the trend data differently
                # For now, just add a summary row
                mean_cond = df['mean_conductivity'].mean()
                iacs = eddy_analysis.conductivity_to_iacs(mean_cond)
                results_data.append({
                    'Sample': meta['sample_name'],
                    'Location': meta['location'],
                    'Layer': meta['layer'],
                    'Group': meta['group_key'],
                    'Positions measured': len(df),
                    'Conductivity mean (MS/m)': mean_cond,
                    'IACS (%)': iacs,
                })

        if not results_data:
            st.error("No valid eddy current files processed.")
            return

        results_df = pd.DataFrame(results_data)

    # ── plot ──────────────────────────────────────────────────────────────────
    if is_kdf_mode:
        st.subheader("📊 %IACS Comparison")
        results_df['x_label'] = (results_df['Sample'] + '-' +
                                  results_df['Location'] + ' / ' +
                                  results_df['Layer'])
        groups = results_df['Group'].unique()
        tickvals = list(range(len(groups)))
        ticktext = list(groups)
        sample_names = [results_df[results_df['Group'] == grp]['Sample'].iloc[0] for grp in groups]
        colors = px.colors.qualitative.Plotly
        layer_symbols = {'top': 'circle', 'bot': 'diamond'}
        OFFSET = 0.15

        fig = go.Figure()
        for i, grp in enumerate(groups):
            sub   = results_df[results_df['Group'] == grp]
            color = colors[i % len(colors)]
            for layer, sym in layer_symbols.items():
                sub_l = sub[sub['Layer'] == layer]
                if sub_l.empty:
                    continue
                xvals = [i - OFFSET if layer == 'top' else i + OFFSET] * len(sub_l)
                fig.add_trace(go.Scatter(
                    x=xvals,
                    y=sub_l['IACS (%)'],
                    error_y=dict(type='data', array=sub_l['σ_IACS (%)'].tolist(),
                                 visible=True, thickness=2, width=5),
                    mode='markers',
                    marker=dict(size=12, color=color, symbol=sym,
                                line=dict(width=1, color='black')),
                    name=f"{grp} ({layer})",
                ))

        if show_sample_separators and len(sample_names) > 0:
            sample_segments = []
            seg_start = 0
            current_sample = sample_names[0]
            for j, sample in enumerate(sample_names[1:], start=1):
                if sample != current_sample:
                    sample_segments.append((current_sample, seg_start, j - 1))
                    seg_start = j
                    current_sample = sample
            sample_segments.append((current_sample, seg_start, len(sample_names) - 1))

            for sample, start, end in sample_segments:
                mid = (start + end) / 2
                sample_df = results_df[results_df['Sample'] == sample]
                iacs_mean = sample_df['IACS (%)'].mean()
                iacs_range = sample_df['IACS (%)'].max() - sample_df['IACS (%)'].min()
                label_text = (
                    f"<b>{sample.upper()}</b><br>"
                    f"{iacs_mean:.1f}% ± {iacs_range:.1f}%"
                )

                fig.add_annotation(
                    x=mid,
                    y=1.02,
                    xref='x',
                    yref='paper',
                    text=label_text,
                    showarrow=False,
                    xanchor='center',
                    yanchor='bottom',
                    font=dict(size=fs_tick, color='rgba(0,0,0,0.8)'),
                    align='center',
                    textangle=0,
                )
                fig.add_annotation(
                    x=mid,
                    y=1.02,
                    xref='x',
                    yref='paper',
                    text=label_text,
                    showarrow=False,
                    xanchor='center',
                    yanchor='bottom',
                    font=dict(size=fs_tick, color='rgba(0,0,0,0.15)'),
                    align='center',
                    textangle=0,
                    xshift=1,
                    yshift=-1,
                )

            for _, start, end in sample_segments[:-1]:
                boundary = end + 0.5
                fig.add_shape(
                    type='line',
                    x0=boundary,
                    x1=boundary,
                    xref='x',
                    y0=0,
                    y1=1,
                    yref='paper',
                    line=dict(color='lightgrey', width=1, dash='dash')
                )

        fig.add_hline(y=100, line_dash="dash", line_color="orange",
                      annotation_text="100% IACS (Pure Cu)")
        layout_kwargs = dict(
            title='%IACS by experiment  (color = sample-location | ● top  ◆ bot)',
            xaxis_title='',
            yaxis_title='%IACS',
            height=520,
            xaxis=dict(
                tickmode='array',
                tickvals=tickvals,
                ticktext=ticktext,
                tickangle=x_tickangle,
            ),
            template='plotly_white',
        )
        if show_sample_separators:
            layout_kwargs['margin'] = dict(t=120)

        fig.update_layout(**layout_kwargs)
        st.plotly_chart(apply_font_sizes(fig, fs_axis_label, fs_tick, fs_legend, fs_inside_text), width='stretch')

        # ── uncertainty breakdown ─────────────────────────────────────────────────
        st.subheader("🔍 Uncertainty Breakdown")
        st.caption("What fraction of total variance comes from each source.")

        var_cols   = ['_var_R_inst', '_var_R_run', '_var_w', '_var_h', '_var_L']
        src_labels = ['Instrument (σ_R)', 'Run-to-run spread',
                      'Width variation', 'Thickness variation', 'Length variation']
        src_colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A']

        total_var = results_df[var_cols].sum(axis=1)
        pct_df    = results_df[var_cols].div(total_var, axis=0).mul(100)

        fig_bar = go.Figure()
        for col, label, color in zip(var_cols, src_labels, src_colors):
            y = pct_df[col]
            fig_bar.add_trace(go.Bar(
                x=results_df['x_label'],
                y=y,
                name=label,
                marker_color=color,
                text=[f"{v:.0f}%" if v >= 5 else "" for v in y],
                textposition='inside',
                insidetextanchor='middle',
            ))

        fig_bar.update_layout(
            barmode='stack',
            title='Uncertainty Sources (% of total variance)',
            xaxis_title='Sample-Location / Layer',
            yaxis_title='Percentage of Total Variance',
            height=400,
            xaxis=dict(tickangle=x_tickangle),
            template='plotly_white',
        )
        st.plotly_chart(apply_font_sizes(fig_bar, fs_axis_label, fs_tick, fs_legend, fs_inside_text), width='stretch')
    else:
        # ── Eddy current plots ─────────────────────────────────────────────────
        if eddy_sub_mode == 'grid':
            st.subheader("📊 Eddy Current IACS Summary")
            results_df['x_label'] = (results_df['Sample'] + '-' +
                                      results_df['Location'] + ' / ' +
                                      results_df['Layer'])

            eddy_colors = px.colors.qualitative.Plotly
            eddy_layer_symbols = {'top': 'circle', 'bot': 'diamond'}
            eddy_groups = results_df['Group'].unique()
            tickvals = list(range(len(eddy_groups)))
            ticktext = list(eddy_groups)
            sample_names = [results_df[results_df['Group'] == grp]['Sample'].iloc[0] for grp in eddy_groups]
            OFFSET = 0.15

            fig = go.Figure()
            for i, grp in enumerate(eddy_groups):
                sub = results_df[results_df['Group'] == grp]
                color = eddy_colors[i % len(eddy_colors)]
                for layer, sym in eddy_layer_symbols.items():
                    sub_l = sub[sub['Layer'] == layer]
                    if sub_l.empty:
                        continue
                    xvals = [i - OFFSET if layer == 'top' else i + OFFSET] * len(sub_l)
                    fig.add_trace(go.Scatter(
                        x=xvals,
                        y=sub_l['IACS (%)'],
                        error_y=dict(type='data', array=sub_l['σ_IACS (%)'].tolist(),
                                     visible=True, thickness=2, width=6),
                        mode='markers',
                        marker=dict(size=12, color=color, symbol=sym,
                                    line=dict(width=1, color='black')),
                        name=f"{grp} ({layer})",
                    ))

            if show_sample_separators and len(sample_names) > 0:
                sample_segments = []
                seg_start = 0
                current_sample = sample_names[0]
                for j, sample in enumerate(sample_names[1:], start=1):
                    if sample != current_sample:
                        sample_segments.append((current_sample, seg_start, j - 1))
                        seg_start = j
                        current_sample = sample
                sample_segments.append((current_sample, seg_start, len(sample_names) - 1))

                for sample, start, end in sample_segments:
                    mid = (start + end) / 2
                    sample_df = results_df[results_df['Sample'] == sample]
                    iacs_mean = sample_df['IACS (%)'].mean()
                    iacs_range = sample_df['IACS (%)'].max() - sample_df['IACS (%)'].min()
                    label_text = (
                        f"<b>{sample.upper()}</b><br>"
                        f"{iacs_mean:.1f}% ± {iacs_range:.1f}%"
                    )

                    fig.add_annotation(
                        x=mid,
                        y=1.02,
                        xref='x',
                        yref='paper',
                        text=label_text,
                        showarrow=False,
                        xanchor='center',
                        yanchor='bottom',
                        font=dict(size=fs_tick, color='rgba(0,0,0,0.8)'),
                        align='center',
                        textangle=0,
                    )
                    fig.add_annotation(
                        x=mid,
                        y=1.02,
                        xref='x',
                        yref='paper',
                        text=label_text,
                        showarrow=False,
                        xanchor='center',
                        yanchor='bottom',
                        font=dict(size=fs_tick, color='rgba(0,0,0,0.15)'),
                        align='center',
                        textangle=0,
                        xshift=1,
                        yshift=-1,
                    )

                for _, start, end in sample_segments[:-1]:
                    boundary = end + 0.5
                    fig.add_shape(
                        type='line',
                        x0=boundary,
                        x1=boundary,
                        xref='x',
                        y0=0,
                        y1=1,
                        yref='paper',
                        line=dict(color='lightgrey', width=1, dash='dash')
                    )

            fig.add_hline(y=100, line_dash="dash", line_color="orange",
                          annotation_text="100% IACS (Pure Cu)")
            layout_kwargs = dict(
                title='Eddy Current %IACS by Sample  (● top  ◆ bot)',
                xaxis_title='Sample-Location',
                yaxis_title='%IACS',
                height=520,
                xaxis=dict(
                    tickmode='array',
                    tickvals=tickvals,
                    ticktext=ticktext,
                    tickangle=x_tickangle,
                ),
                template='plotly_white',
            )
            if show_sample_separators:
                layout_kwargs['margin'] = dict(t=120)

            fig.update_layout(**layout_kwargs)
            st.plotly_chart(apply_font_sizes(fig, fs_axis_label, fs_tick, fs_legend, fs_inside_text), width='stretch')
        else:  # trend mode
            st.subheader("📈 Eddy Current Conductivity Trends")
            colors = px.colors.qualitative.Plotly

            fig = go.Figure()
            for i, (filename, df) in enumerate(eddy_results.items()):
                meta = eddy_analysis.parse_eddy_filename(filename)
                if meta is None:
                    continue
                color = colors[i % len(colors)]
                label = f"{meta['sample_name']}-{meta['location']} ({meta['layer']})"

                fig.add_trace(go.Scatter(
                    x=df['Col'],
                    y=df['mean_conductivity'],
                    error_y=dict(type='data', array=df['std_conductivity'].tolist(),
                                 visible=True, thickness=1, width=3),
                    mode='lines+markers',
                    line=dict(color=color, width=2),
                    marker=dict(size=6, color=color),
                    name=label,
                ))

            fig.update_layout(
                title='Eddy Current Conductivity vs Position (Left-to-Right)',
                xaxis_title='Column Position',
                yaxis_title='Conductivity (MS/m)',
                height=520,
                template='plotly_white',
            )
            st.plotly_chart(apply_font_sizes(fig, fs_axis_label, fs_tick, fs_legend, fs_inside_text), width='stretch')

    # ── Raw profiles: top vs bottom per sample ────────────────────────────────
    if is_eddy_mode:
        st.subheader("📉 Raw %IACS Profiles — Top vs Bottom")

        sample_groups: dict = {}
        for filename, df in eddy_data.items():
            meta = eddy_analysis.parse_eddy_filename(filename)
            if meta is None:
                continue
            sample_groups.setdefault(meta['key'], {})[meta['layer']] = df

        layer_color = {'top': '#636EFA', 'bot': '#EF553B'}
        layer_dash  = {'top': 'solid',   'bot': 'dot'}

        for sample_key, layers in sorted(sample_groups.items()):
            fig_raw = go.Figure()
            for layer, df in sorted(layers.items()):
                y = eddy_analysis.conductivity_to_iacs(df['Conductivity_MS_m'].values)
                fig_raw.add_trace(go.Scatter(
                    x=list(range(len(y))),
                    y=y,
                    mode='lines+markers',
                    marker=dict(size=3),
                    line=dict(color=layer_color.get(layer, 'gray'),
                              dash=layer_dash.get(layer, 'solid'), width=1.5),
                    name=layer,
                ))
            fig_raw.add_hline(y=100, line_dash="dash", line_color="orange",
                              annotation_text="100% IACS")
            fig_raw.update_layout(
                title=f"{sample_key} — raw %IACS (left → right)",
                xaxis_title='Measurement index (left → right)',
                yaxis_title='%IACS',
                height=380,
                template='plotly_white',
            )
            st.plotly_chart(apply_font_sizes(fig_raw, fs_axis_label, fs_tick, fs_legend, fs_inside_text), width='stretch')

    # ── summary stats per sample ──────────────────────────────────────────────
    st.subheader("📈 Summary Statistics")
    if is_kdf_mode:
        summary = (
            results_df.groupby('Sample')['IACS (%)']
            .agg(
                Mean=lambda x: round(x.mean(), 2),
                Std=lambda x: round(x.std(), 2) if len(x) > 1 else float('nan'),
                Max=lambda x: round(x.max(), 2),
                Min=lambda x: round(x.min(), 2),
                N='count',
            )
            .reset_index()
            .rename(columns={
                'Sample': 'Sample',
                'Mean': 'Mean %IACS',
                'Std':  'Std %IACS',
                'Max':  'Max %IACS',
                'Min':  'Min %IACS',
                'N':    'Experiments',
            })
        )
        st.dataframe(
            summary.style.format({
                'Mean %IACS': '{:.2f}',
                'Std %IACS':  '{:.2f}',
                'Max %IACS':  '{:.2f}',
                'Min %IACS':  '{:.2f}',
            }, na_rep='—')
            .background_gradient(subset=['Mean %IACS'], cmap='RdYlGn'),
            width='stretch',
            hide_index=True,
        )
    else:
        # For eddy mode, show conductivity and IACS summary
        summary_cols = ['IACS (%)', 'Conductivity mean (MS/m)'] if 'Conductivity mean (MS/m)' in results_df.columns else ['IACS (%)']
        summary = (
            results_df.groupby('Sample')[summary_cols]
            .agg(lambda x: round(x.mean(), 2))
            .reset_index()
        )
        st.dataframe(summary, width='stretch', hide_index=True)

    # ── results table ─────────────────────────────────────────────────────────
    st.subheader("📋 Detailed Results")
    if is_kdf_mode:
        _internal = ['x_label', 'Group'] + [c for c in results_df.columns if c.startswith('_var')]
        display_df = results_df.drop(columns=_internal).copy()
        for col in ['R pooled (Ω)', 'σ_R inst (Ω)', 'σ_R run (Ω)', 'σ_R total (Ω)', 'Resistivity (Ω·m)']:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.3e}")
        for col in ['IACS (%)', 'σ_IACS (%)']:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.3f}")
    else:
        display_df = results_df.copy()
        for col in ['IACS (%)', 'σ_IACS (%)', 'Conductivity mean (MS/m)', 'Conductivity σ (MS/m)']:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.3f}")
    st.dataframe(display_df, width='stretch')

    # ── downloads ─────────────────────────────────────────────────────────────
    st.subheader("📥 Downloads")
    if is_kdf_mode:
        _internal = ['x_label', 'Group'] + [c for c in results_df.columns if c.startswith('_var')]
        out_df = results_df.drop(columns=_internal)
        st.download_button("📥 Download Full Results (CSV)",
                           out_df.to_csv(index=False), "kdf_iacs_results.csv", "text/csv")
        st.download_button("📥 Download %IACS Summary (CSV)",
                           out_df[['Sample','Location','Layer','IACS (%)','σ_IACS (%)']].to_csv(index=False),
                           "iacs_summary.csv", "text/csv")
    else:
        st.download_button("📥 Download Eddy Results (CSV)",
                           results_df.to_csv(index=False), "eddy_results.csv", "text/csv")

    if is_kdf_mode:
        # ── Hypothetical: 1 A with Keithley 2450 ─────────────────────────────────
        st.divider()
        st.subheader("💡 What if you used 1 A with a Keithley 2450?")
        st.caption(
            "Same measured resistance, same dimensions — only the current source and "
            "current level change. Shows how σ_IACS would differ."
        )

        hyp_rows = []
        hyp_plot = []  # numeric data for the comparison plot
        for gk, grp in per_group.items():
            meta     = grp['meta']
            R_pooled = float(np.mean(grp['R_values']))
            sigma_run = float(np.std(grp['R_values'], ddof=1) / math.sqrt(len(grp['R_values']))) \
                        if len(grp['R_values']) > 1 else 0.0

            I_hyp = 1.0  # 1 A
            V_hyp = I_hyp * R_pooled
            dV_hyp, vrange_hyp = _voltage_uncertainty(V_hyp)
            dI_hyp = _current_uncertainty(I_hyp, '2450')
            sR_hyp = math.sqrt(_resistance_uncertainty(R_pooled, V_hyp, I_hyp, dV_hyp, dI_hyp) ** 2
                               + sigma_run ** 2)

            L_mm = length_overrides.get(grp['uf_name'], meta['length_mm'] if meta else 60.0)
            key  = meta['key'] if meta else None
            d    = dims_dict.get(key) if key else None
            w_mm, sigma_w = (d['width_mean'], d['width_std'])   if d else (1.80, 0.05)
            h_mm, sigma_h = (d['height_mean'], d['height_std']) if d else (9.00, 0.05)

            _, srho_hyp = calculate_resistivity(R_pooled, sR_hyp, L_mm, sigma_L_mm,
                                                 w_mm, sigma_w, h_mm, sigma_h)
            rho_actual   = R_pooled * (w_mm * h_mm / 1e6) / (L_mm / 1000)
            iacs_val     = resistivity_to_iacs(rho_actual)
            siacs_hyp    = iacs_val * (srho_hyp / rho_actual)

            row_actual   = results_df[results_df['Sample'] == (meta['sample_name'] if meta else gk)]
            row_actual   = row_actual[row_actual['Layer'] == (meta['layer'] if meta else '?')]
            siacs_actual = row_actual['σ_IACS (%)'].values[0] if len(row_actual) else float('nan')

            sample  = meta['sample_name'] if meta else gk
            loc     = meta['location']    if meta else '?'
            layer   = meta['layer']       if meta else '?'
            grp_lbl = f"{sample}-{loc}"
            x_lbl   = f"{sample}-{loc} / {layer}"

            hyp_rows.append({
                'Sample':                sample,
                'Location':              loc,
                'Layer':                 layer,
                'σ_IACS actual (%)':     f"{siacs_actual:.3f}",
                'σ_IACS @ 1A/2450 (%)': f"{siacs_hyp:.3f}",
                'Change':                f"{siacs_hyp - siacs_actual:+.3f}",
                'V range (hyp)':         vrange_hyp,
            })
            hyp_plot.append({
                'x_label':      x_lbl,
                'Group':        grp_lbl,
                'Layer':        layer,
                'IACS (%)':     iacs_val,
                'σ_actual (%)': siacs_actual,
                'σ_hyp (%)':    siacs_hyp,
            })

        st.dataframe(pd.DataFrame(hyp_rows), width='stretch')

        # ── comparison plot ───────────────────────────────────────────────────
        if hyp_plot:
            hyp_df  = pd.DataFrame(hyp_plot)
            fig2    = go.Figure()
            OFFSET  = 0.18   # horizontal nudge for hypothetical bars
            xlabels = hyp_df['x_label'].tolist()
            xpos    = list(range(len(xlabels)))  # numeric positions

            for i, grp in enumerate(hyp_df['Group'].unique()):
                sub   = hyp_df[hyp_df['Group'] == grp]
                color = colors[i % len(colors)]
                for layer, sym in layer_symbols.items():
                    sub_l  = sub[sub['Layer'] == layer]
                    if sub_l.empty:
                        continue
                    idx = [xlabels.index(x) for x in sub_l['x_label']]

                    # actual — filled marker + error bar
                    fig2.add_trace(go.Scatter(
                        x=idx,
                        y=sub_l['IACS (%)'].tolist(),
                        error_y=dict(type='data', array=sub_l['σ_actual (%)'].tolist(),
                                     visible=True, thickness=2, width=6),
                        mode='markers',
                        marker=dict(size=13, color=color, symbol=sym,
                                    line=dict(width=1, color='black')),
                        name=f"{grp} ({layer}) — actual",
                        legendgroup=f"{grp}-{layer}",
                    ))
                    # hypothetical — no marker, just the range bar, offset right
                    fig2.add_trace(go.Scatter(
                        x=[i + OFFSET for i in idx],
                        y=sub_l['IACS (%)'].tolist(),
                        error_y=dict(type='data', array=sub_l['σ_hyp (%)'].tolist(),
                                     visible=True, thickness=2, width=6,
                                     color='#888888'),
                        mode='markers',
                        marker=dict(size=4, color='#888888', symbol='line-ew'),
                        name=f"{grp} ({layer}) — 1A/2450",
                        legendgroup=f"{grp}-{layer}-hyp",
                    ))

            fig2.add_hline(y=100, line_dash="dash", line_color="orange",
                           annotation_text="100% IACS (Pure Cu)")
            fig2.update_layout(
                title='Uncertainty comparison — filled: actual  |  grey bar: 1 A / Keithley 2450',
                xaxis=dict(
                    tickmode='array', tickvals=xpos, ticktext=xlabels,
                    tickangle=45,
                ),
                yaxis_title='%IACS',
                height=520,
                template='plotly_white',
            )
            st.plotly_chart(apply_font_sizes(fig2, fs_axis_label, fs_tick, fs_legend, fs_inside_text), width='stretch')


if __name__ == "__main__":
    main()
