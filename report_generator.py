import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
from kdf_viewer import (
    _current_uncertainty, _resistance_uncertainty, _voltage_uncertainty,
    apply_font_sizes, calculate_resistivity, parse_dims_txt,
    parse_filename, read_kdf_file, resistivity_to_iacs,
    SOURCE_TABLES,
)
from eddy_analysis import conductivity_to_iacs, load_eddy_files, parse_eddy_filename

COLORS = px.colors.qualitative.Plotly
LAYER_SYM = {'top': 'circle', 'bot': 'diamond'}
LAYER_COLOR = {'top': '#636EFA', 'bot': '#EF553B'}
LAYER_DASH  = {'top': 'solid',   'bot': 'dot'}


# ── 3-D beam ──────────────────────────────────────────────────────────────────

def _beam_surface(x2, y2, z2, name, color):
    return go.Surface(
        x=x2, y=y2, z=z2,
        name=name,
        colorscale=[[0, color], [1, color]],
        showscale=False,
        opacity=0.88,
        showlegend=True,
    )


def make_beam_fig(sample_key, dims, L_mm, fs):
    ws = dims['widths']
    ts = dims['thicknesses']
    N  = 60
    x  = np.linspace(0, L_mm, N)
    w  = np.interp(np.linspace(0, 1, N), np.linspace(0, 1, len(ws)), ws)
    t  = np.interp(np.linspace(0, 1, N), np.linspace(0, 1, len(ts)), ts)

    X2 = np.vstack([x, x])
    traces = [
        _beam_surface(X2, np.vstack([-w/2,  w/2]), np.vstack([t, t]),              'top face',   '#4C78A8'),
        _beam_surface(X2, np.vstack([-w/2,  w/2]), np.zeros((2, N)),               'bottom face','#9ECAE9'),
        _beam_surface(X2, np.vstack([ w/2,  w/2]), np.vstack([np.zeros(N), t]),    'right face', '#72B7B2'),
        _beam_surface(X2, np.vstack([-w/2, -w/2]), np.vstack([np.zeros(N), t]),    'left face',  '#72B7B2'),
    ]
    fig = go.Figure(traces)
    fig.update_layout(
        title=f"{sample_key} — geometry",
        scene=dict(
            xaxis_title='Length (mm)',
            yaxis_title='Width (mm)',
            zaxis_title='Thickness (mm)',
            aspectmode='data',
        ),
        height=460,
        template='plotly_white',
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def make_profile_fig(dims, L_mm, fs):
    ws = dims['widths']
    ts = dims['thicknesses']
    xw = np.linspace(0, L_mm, len(ws))
    xt = np.linspace(0, L_mm, len(ts))

    # linear fit for thickness
    coeffs = np.polyfit(np.linspace(0, 1, len(ts)), ts, 1)
    x_fit  = np.linspace(0, L_mm, 80)
    t_fit  = np.polyval(coeffs, np.linspace(0, 1, 80))

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=('Width along length', 'Thickness along length'))
    fig.add_trace(go.Scatter(x=xw, y=ws, mode='markers+lines',
                             marker=dict(size=8), line=dict(dash='dot'),
                             name='width (measured)'), row=1, col=1)
    fig.add_hline(y=dims['width_mean'], line_dash='dash', line_color='grey',
                  annotation_text=f"mean {dims['width_mean']:.3f} mm", row=1, col=1)

    fig.add_trace(go.Scatter(x=xt, y=ts, mode='markers',
                             marker=dict(size=8), name='thickness (measured)'), row=2, col=1)
    fig.add_trace(go.Scatter(x=x_fit, y=t_fit, mode='lines',
                             line=dict(dash='dash', color='grey'),
                             name='linear taper fit'), row=2, col=1)

    fig.update_xaxes(title_text='Position along length (mm)', row=2, col=1)
    fig.update_yaxes(title_text='Width (mm)',     row=1, col=1)
    fig.update_yaxes(title_text='Thickness (mm)', row=2, col=1)
    fig.update_layout(height=380, template='plotly_white', showlegend=True)
    return apply_font_sizes(fig, fs[0], fs[1], fs[2], fs[3])


# ── KDF processing (mirrors kdf_viewer.py pipeline) ───────────────────────────

def process_kdf(uploaded_files, dims_dict, current_source, sigma_L_mm):
    per_group = {}
    for uf in uploaded_files:
        meta = parse_filename(uf.name)
        df   = read_kdf_file(uf.read())
        if df is None or 'Resistance' not in df.columns:
            st.warning(f"Could not read {uf.name}")
            continue
        R_mean = float(df['Resistance'].mean())
        gk = meta['group_key'] if meta else uf.name
        per_group.setdefault(gk, {
            'meta': meta, 'R_values': [],
            'I_A': meta['current_A'] if meta else 0.0,
            'uf_name': uf.name,
        })
        per_group[gk]['R_values'].append(R_mean)

    rows = []
    for gk, grp in per_group.items():
        meta     = grp['meta']
        R_values = grp['R_values']
        I_A      = grp['I_A']
        R_pooled  = float(np.mean(R_values))
        sigma_run = float(np.std(R_values, ddof=1) / math.sqrt(len(R_values))) \
                    if len(R_values) > 1 else 0.0
        V        = I_A * R_pooled
        dV, _    = _voltage_uncertainty(V)
        dI       = _current_uncertainty(I_A, current_source)
        sigma_R  = math.sqrt(_resistance_uncertainty(R_pooled, V, I_A, dV, dI)**2 + sigma_run**2)

        key = meta['key'] if meta else None
        d   = dims_dict.get(key) if key else None
        L_mm    = meta['length_mm'] if meta else 80.0
        w_mm    = d['width_mean']  if d else 1.80
        sigma_w = d['width_std']   if d else 0.05
        h_mm    = d['height_mean'] if d else 9.00
        sigma_h = d['height_std']  if d else 0.05

        rho, sigma_rho = calculate_resistivity(
            R_pooled, sigma_R, L_mm, sigma_L_mm, w_mm, sigma_w, h_mm, sigma_h)
        iacs       = resistivity_to_iacs(rho)
        sigma_iacs = iacs * (sigma_rho / rho)

        rows.append({
            'Sample':   meta['sample_name'] if meta else gk,
            'Location': meta['location']    if meta else '?',
            'Layer':    meta['layer']       if meta else '?',
            'key':      f"{meta['sample_name']}-{meta['location']}-{meta['layer']}" if meta else gk,
            'L_mm':     L_mm,
            'IACS (%)':   iacs,
            'σ_IACS (%)': sigma_iacs,
        })
    return pd.DataFrame(rows), per_group


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title='IACS Report', page_icon='📋', layout='wide')
    st.title('📋 IACS Combined Report — KDF + Eddy Current')

    with st.sidebar:
        st.header('📂 Data Files')
        dims_file     = st.file_uploader('dims.txt', type=['txt'])
        kdf_files     = st.file_uploader('KDF files', type=['kdf', 'txt'], accept_multiple_files=True)
        eddy_files    = st.file_uploader('Eddy CSVs', type=['csv'], accept_multiple_files=True)

        st.header('⚙️ Settings')
        current_source = st.selectbox('Current source', ['6220', '2450'],
                                      format_func=lambda x: f'Keithley {x}')
        sigma_L_mm = st.number_input('σ_L (mm)', min_value=0.0, value=0.05, step=0.01, format='%.3f')

        st.header('🔤 Font Sizes')
        fs = (
            st.slider('Axis labels', 8, 28, 13),
            st.slider('Tick labels', 6, 24, 11),
            st.slider('Legend',      6, 24, 11),
            st.slider('Chart text',  6, 24, 10),
        )

    if not dims_file and not kdf_files and not eddy_files:
        st.info('Upload dims.txt, KDF files, and/or eddy CSVs in the sidebar.')
        return

    dims_dict = parse_dims_txt(dims_file.read()) if dims_file else {}

    # ── Section 1: Sample geometry ─────────────────────────────────────────────
    if dims_dict:
        st.header('1 · Sample Geometry')

        # Derive L_mm per key from KDF filenames if available
        L_by_key = {}
        for uf in (kdf_files or []):
            meta = parse_filename(uf.name)
            if meta:
                L_by_key.setdefault(meta['key'], []).append(meta['length_mm'])
        L_by_key = {k: float(np.mean(v)) for k, v in L_by_key.items()}

        for sample_key, dims in sorted(dims_dict.items()):
            st.subheader(sample_key)
            L_mm = L_by_key.get(sample_key, 80.0)
            col1, col2 = st.columns([1, 1])
            with col1:
                st.plotly_chart(make_beam_fig(sample_key, dims, L_mm, fs),
                                width='stretch')
            with col2:
                st.plotly_chart(make_profile_fig(dims, L_mm, fs),
                                width='stretch')

        st.divider()

    # ── Section 2: Eddy current data ──────────────────────────────────────────
    if eddy_files:
        st.header('2 · Eddy Current Data')

        eddy_data = load_eddy_files(eddy_files)

        # Group by sample key
        sample_groups: dict = {}
        for filename, df in eddy_data.items():
            meta = parse_eddy_filename(filename)
            if meta is None:
                continue
            sample_groups.setdefault(meta['key'], {})[meta['layer']] = (df, meta)

        # Raw profile plots
        st.subheader('Raw %IACS Profiles — Top vs Bottom')
        for sample_key, layers in sorted(sample_groups.items()):
            fig = go.Figure()
            for layer, (df, _) in sorted(layers.items()):
                y = conductivity_to_iacs(df['Conductivity_MS_m'].values)
                fig.add_trace(go.Scatter(
                    x=list(range(len(y))), y=y,
                    mode='lines+markers',
                    marker=dict(size=3),
                    line=dict(color=LAYER_COLOR.get(layer, 'gray'),
                              dash=LAYER_DASH.get(layer, 'solid'), width=1.5),
                    name=layer,
                ))
            fig.add_hline(y=100, line_dash='dash', line_color='orange',
                          annotation_text='100% IACS')
            fig.update_layout(
                title=f'{sample_key} — raw %IACS (left → right)',
                xaxis_title='Measurement index (left → right)',
                yaxis_title='%IACS',
                height=360,
                template='plotly_white',
            )
            st.plotly_chart(apply_font_sizes(fig, *fs), width='stretch')

        # Summary table
        st.subheader('Eddy Summary')
        eddy_summary = []
        for sample_key, layers in sorted(sample_groups.items()):
            for layer, (df, meta) in sorted(layers.items()):
                y = conductivity_to_iacs(df['Conductivity_MS_m'].values)
                eddy_summary.append({
                    'Sample':   meta['sample_name'],
                    'Location': meta['location'],
                    'Layer':    layer,
                    'N':        len(y),
                    'Mean %IACS': round(float(np.mean(y)), 3),
                    'Std %IACS':  round(float(np.std(y)),  3),
                })
        st.dataframe(pd.DataFrame(eddy_summary), width='stretch', hide_index=True)
        st.divider()

    # ── Section 3: KDF data ───────────────────────────────────────────────────
    if kdf_files:
        st.header('3 · KDF Data')

        # Re-seek eddy files if also loaded (not needed here, kdf is separate)
        kdf_df, _ = process_kdf(kdf_files, dims_dict, current_source, sigma_L_mm)

        if kdf_df.empty:
            st.error('No valid KDF files processed.')
        else:
            kdf_df['x_label'] = (kdf_df['Sample'] + '-' +
                                  kdf_df['Location'] + ' / ' + kdf_df['Layer'])
            groups = kdf_df.groupby(['Sample', 'Location']).ngroups
            color_map = {grp: COLORS[i % len(COLORS)]
                         for i, grp in enumerate(
                             (kdf_df['Sample'] + '-' + kdf_df['Location']).unique())}

            fig_kdf = go.Figure()
            for _, row in kdf_df.iterrows():
                grp_key = f"{row['Sample']}-{row['Location']}"
                fig_kdf.add_trace(go.Scatter(
                    x=[row['x_label']],
                    y=[row['IACS (%)']],
                    error_y=dict(type='data', array=[row['σ_IACS (%)']],
                                 visible=True, thickness=2, width=6),
                    mode='markers',
                    marker=dict(size=12, color=color_map[grp_key],
                                symbol=LAYER_SYM.get(row['Layer'], 'circle'),
                                line=dict(width=1, color='black')),
                    name=f"{grp_key} ({row['Layer']})",
                ))
            fig_kdf.add_hline(y=100, line_dash='dash', line_color='orange',
                              annotation_text='100% IACS (Pure Cu)')
            fig_kdf.update_layout(
                title='KDF %IACS by Sample  (● top  ◆ bot)',
                xaxis_title='Sample-Location / Layer',
                yaxis_title='%IACS',
                height=480,
                xaxis=dict(tickangle=45),
                template='plotly_white',
            )
            st.plotly_chart(apply_font_sizes(fig_kdf, *fs), width='stretch')

            display_cols = ['Sample', 'Location', 'Layer', 'L_mm', 'IACS (%)', 'σ_IACS (%)']
            disp = kdf_df[display_cols].copy()
            for col in ['IACS (%)', 'σ_IACS (%)']:
                disp[col] = disp[col].apply(lambda x: f'{x:.3f}')
            st.dataframe(disp, width='stretch', hide_index=True)

        st.divider()

    # ── Section 4: Combined comparison ────────────────────────────────────────
    if eddy_files and kdf_files:
        st.header('4 · Combined Comparison — Eddy vs KDF')

        combined = []
        # eddy points
        for sample_key, layers in sorted(sample_groups.items()):
            for layer, (df, meta) in layers.items():
                y = conductivity_to_iacs(df['Conductivity_MS_m'].values)
                combined.append({
                    'label':  f"{meta['sample_name']}-{meta['location']} / {layer}",
                    'group':  f"{meta['sample_name']}-{meta['location']}",
                    'layer':  layer,
                    'IACS':   float(np.mean(y)),
                    'σ_IACS': float(np.std(y)),
                    'method': 'Eddy',
                })
        # kdf points
        for _, row in kdf_df.iterrows():
            combined.append({
                'label':  row['x_label'],
                'group':  f"{row['Sample']}-{row['Location']}",
                'layer':  row['Layer'],
                'IACS':   row['IACS (%)'],
                'σ_IACS': row['σ_IACS (%)'],
                'method': 'KDF',
            })

        cdf = pd.DataFrame(combined)
        all_groups = cdf['group'].unique()
        group_colors = {g: COLORS[i % len(COLORS)] for i, g in enumerate(all_groups)}
        method_sym = {'Eddy': 'circle', 'KDF': 'diamond'}

        fig_comb = go.Figure()
        for method in ['Eddy', 'KDF']:
            sub = cdf[cdf['method'] == method]
            for grp in all_groups:
                sub_g = sub[sub['group'] == grp]
                if sub_g.empty:
                    continue
                for _, row in sub_g.iterrows():
                    fig_comb.add_trace(go.Scatter(
                        x=[row['label']],
                        y=[row['IACS']],
                        error_y=dict(type='data', array=[row['σ_IACS']],
                                     visible=True, thickness=2, width=6),
                        mode='markers',
                        marker=dict(size=13, color=group_colors[grp],
                                    symbol=method_sym[method],
                                    line=dict(width=1, color='black'),
                                    opacity=0.9 if method == 'KDF' else 0.6),
                        name=f"{grp} — {method}",
                        legendgroup=f"{grp}-{method}",
                        showlegend=True,
                    ))

        fig_comb.add_hline(y=100, line_dash='dash', line_color='orange',
                           annotation_text='100% IACS (Pure Cu)')
        fig_comb.update_layout(
            title='%IACS Comparison  (● Eddy  ◆ KDF)',
            xaxis_title='Sample-Location / Layer',
            yaxis_title='%IACS',
            height=520,
            xaxis=dict(tickangle=45),
            template='plotly_white',
        )
        st.plotly_chart(apply_font_sizes(fig_comb, *fs), width='stretch')

        st.download_button('📥 Download Combined CSV',
                           cdf.to_csv(index=False), 'iacs_combined.csv', 'text/csv')


if __name__ == '__main__':
    main()
