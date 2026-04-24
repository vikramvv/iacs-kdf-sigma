import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

st.set_page_config(layout="wide")
st.title("Dogbone Grid: Mean with Bubble-like Error Lines (Fixed Rows, Adjustable Y-axis)")
st.cache_data.clear()

uploaded_files = st.file_uploader(
    "Upload raw CSV files",
    type=["csv"],
    accept_multiple_files=True
)

if uploaded_files:
    sample_data = {}

    # --- Process each CSV ---
    for file in uploaded_files:
        df_raw = pd.read_csv(file)

        # Extract numeric values
        pattern = re.compile(r"RC\[current_value\]=([\d\.]+)")
        df_raw["NumericValue"] = df_raw["Value"].astype(str).str.extract(pattern).astype(float)
        df_raw = df_raw.dropna(subset=["NumericValue"]).copy()

        # Extract Row/Col
        if "Row" not in df_raw or "Col" not in df_raw:
            df_raw["Row"] = df_raw["Cell"].str.extract(r"R(\d+)").astype(int)
            df_raw["Col"] = df_raw["Cell"].str.extract(r"C(\d+)").astype(int)

        # Sample name
        sample_name = str(df_raw["Sample Name"].iloc[0]) if "Sample Name" in df_raw.columns else file.name

        # Stats
        stats = df_raw.groupby(["Row", "Col"])["NumericValue"].agg(["mean", "std"]).reset_index()
        sample_data[sample_name] = stats

    st.sidebar.header("Options")
    unit_choice = st.sidebar.radio("Units", ["IACS", "MS/m"], index=0)
    multiplier = 100/58 if unit_choice == "IACS" else 1.0

    # Display mode selection
    display_mode = st.sidebar.selectbox(
        "Display Mode",
        options=[
            "Error Bars Only",
            "Values Only (No Error)",
            "Values + Error Bars",
            "Interactive Hover",
            "Selective Display (High Error Only)"
        ]
    )

    # Error bar representation
    error_style = st.sidebar.selectbox(
        "Error Bar Style",
        options=[
            "Vertical Lines",
            "Error Bars (±)",
            "Circles (Size = Error)",
            "Opacity (Alpha = 1/Error)"
        ]
    )

    # Value display options
    if display_mode in ["Values Only (No Error)", "Values + Error Bars", "Interactive Hover"]:
        value_precision = st.sidebar.slider("Value Decimal Places", 0, 3, 1)
        font_size = st.sidebar.slider("Font Size", 6, 14, 8)

    # Combine all sample data for global statistics
    combined_stats = pd.concat([stats for stats in sample_data.values()], ignore_index=True)
    combined_stats["mean_scaled"] = combined_stats["mean"] * multiplier
    combined_stats["std_scaled"] = combined_stats["std"] * multiplier

    # Determine max std for slider default
    max_std = combined_stats["std_scaled"].max() if not combined_stats.empty else 1.0

    # Threshold for selective display
    if display_mode == "Selective Display (High Error Only)":
        error_threshold = st.sidebar.slider(
            "Show values/errors above threshold",
            min_value=0.0,
            max_value=float(max_std),
            value=float(max_std * 0.7),
            step=0.1
        )
    else:
        annotated_threshold = st.sidebar.slider(
            "Std threshold for numeric annotation",
            min_value=0.0,
            max_value=float(max_std),
            value=float(max_std),
            step=0.1
        )

    # Error line scale factor (bubble size)
    if error_style in ["Vertical Lines", "Circles (Size = Error)"]:
        error_line_scale = st.sidebar.slider(
            "Error visualization scale factor",
            min_value=10.0,
            max_value=200.0,
            value=50.0,
            step=5.0
        )

    # Colormap selection
    cmap_option = st.sidebar.selectbox(
        "Colormap",
        options=["Viridis", "Excel Yellow→Green"]
    )
    excel_cmap = LinearSegmentedColormap.from_list("excel", ["yellow", "green"])

    # Generate sample colors and convert to proper format (FIX: Convert to tuple)
    sample_colors = [tuple(plt.cm.tab10(i)) for i in np.linspace(0, 1, len(sample_data))]

    # Y-axis min/max sliders (for visual spacing) - account for sample offsets
    global_row_min = min(stats["Row"].min() for stats in sample_data.values())
    global_row_max = max(stats["Row"].max() for stats in sample_data.values())
    
    # Calculate total Y range including offsets (4 units per sample)
    max_sample_offset = (len(sample_data) - 1) * 4
    adjusted_row_min = global_row_min
    adjusted_row_max = global_row_max + max_sample_offset

    y_min = st.sidebar.slider("Min Y-axis", float(adjusted_row_min - 5), float(adjusted_row_max + 5), float(adjusted_row_min - 1))
    y_max = st.sidebar.slider("Max Y-axis", float(adjusted_row_min - 5), float(adjusted_row_max + 5), float(adjusted_row_max + 1))

    # --- Unified plot ---
    fig, ax = plt.subplots(figsize=(12, 32))
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Initialize scatter plot variable for colorbar
    sc = None

    for sample_idx, (sample_name, stats) in enumerate(sample_data.items()):
        sample_stats = stats.copy()
        sample_stats["mean_scaled"] = sample_stats["mean"] * multiplier
        sample_stats["std_scaled"] = sample_stats["std"] * multiplier
        vertical_offset = 4  # adjust as needed spacing between samples
        sample_stats["Row_pos"] = sample_stats["Row"] + sample_idx * vertical_offset

        sample_stats["sample_color"] = [sample_colors[sample_idx]] * len(sample_stats)

        # Error visualization based on style
        if display_mode != "Values Only (No Error)":
            if error_style == "Vertical Lines":
                # Original vertical lines
                for _, row in sample_stats.iterrows():
                    show_this_point = True
                    if display_mode == "Selective Display (High Error Only)":
                        show_this_point = row["std_scaled"] >= error_threshold
                    
                    if show_this_point:
                        half_length = row["std_scaled"] / error_line_scale
                        ax.vlines(
                            x=row["Col"],
                            ymin=row["Row_pos"] - half_length,
                            ymax=row["Row_pos"] + half_length,
                            color='black',
                            alpha=0.7,
                            linewidth=1.5,
                            zorder=1
                        )
            
            elif error_style == "Error Bars (±)":
                # Traditional error bars
                mask = np.ones(len(sample_stats), dtype=bool)
                if display_mode == "Selective Display (High Error Only)":
                    mask = sample_stats["std_scaled"] >= error_threshold
                
                ax.errorbar(
                    sample_stats.loc[mask, "Col"],
                    sample_stats.loc[mask, "Row_pos"],
                    yerr=sample_stats.loc[mask, "std_scaled"] / error_line_scale,
                    fmt='none',
                    ecolor='black',
                    alpha=0.7,
                    capsize=3,
                    zorder=1
                )
            
            elif error_style == "Circles (Size = Error)":
                # Error as circle size
                for _, row in sample_stats.iterrows():
                    show_this_point = True
                    if display_mode == "Selective Display (High Error Only)":
                        show_this_point = row["std_scaled"] >= error_threshold
                    
                    if show_this_point:
                        circle_size = row["std_scaled"] * error_line_scale
                        circle = plt.Circle(
                            (row["Col"], row["Row_pos"]),
                            radius=circle_size/100,
                            fill=False,
                            edgecolor=row["sample_color"][0],
                            alpha=0.5,
                            linewidth=1,
                            zorder=1
                        )
                        ax.add_patch(circle)
            
            elif error_style == "Opacity (Alpha = 1/Error)":
                # Error as opacity - will be applied to scatter plot below
                pass

        # Draw mean-colored dots for this sample
        cmap = excel_cmap if cmap_option == "Excel Yellow→Green" else "viridis"
        
        # Calculate alpha values for opacity-based error display
        if error_style == "Opacity (Alpha = 1/Error)":
            # Normalize std to alpha range (0.3 to 1.0)
            if len(sample_stats) > 1:
                normalized_std = (sample_stats["std_scaled"] - sample_stats["std_scaled"].min()) / (sample_stats["std_scaled"].max() - sample_stats["std_scaled"].min())
            else:
                normalized_std = pd.Series([0.5])
            alpha_values = 1.0 - 0.7 * normalized_std  # High error = low alpha
            
            for i, (_, row) in enumerate(sample_stats.iterrows()):
                sc = ax.scatter(
                    row["Col"],
                    row["Row_pos"],
                    c=row["mean_scaled"],
                    s=120,
                    cmap=cmap,
                    edgecolor='black',
                    alpha=alpha_values.iloc[i],
                    vmin=combined_stats["mean_scaled"].min(),
                    vmax=combined_stats["mean_scaled"].max(),
                    zorder=2
                )
        else:
            sc = ax.scatter(
                sample_stats["Col"],
                sample_stats["Row_pos"],
                c=sample_stats["mean_scaled"],
                s=350,
                cmap=cmap,
                edgecolor='black',
                vmin=combined_stats["mean_scaled"].min(),  # Use global range
                vmax=combined_stats["mean_scaled"].max(),  # Use global range
                zorder=2
            )

        # Value annotations based on display mode
        if display_mode == "Values Only (No Error)":
            for _, row in sample_stats.iterrows():
                ax.text(
                    row["Col"],
                    row["Row_pos"] - 0.3,
                    f"{row['mean_scaled']:.{value_precision}f}",
                    fontsize=font_size,
                    color='black',
                    ha='center',
                    va='center',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8),
                    zorder=3
                )
        
        elif display_mode == "Values + Error Bars":
            for _, row in sample_stats.iterrows():
                ax.text(
                    row["Col"],
                    row["Row_pos"] - 0.4,
                    f"{row['mean_scaled']:.{value_precision}f}\n±{row['std_scaled']:.{value_precision}f}",
                    fontsize=font_size,
                    color='black',
                    ha='center',
                    va='center',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8),
                    zorder=3
                )
        
        elif display_mode == "Selective Display (High Error Only)":
            for _, row in sample_stats.iterrows():
                if row["std_scaled"] >= error_threshold:
                    ax.text(
                        row["Col"] + 0.1,
                        row["Row_pos"] + 0.1,
                        f"{row['mean_scaled']:.1f}±{row['std_scaled']:.2f}",
                        fontsize=8,
                        color='red',
                        weight='bold',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.8),
                        zorder=3
                    )
        
        elif display_mode == "Interactive Hover":
            # Add sample-specific high error points info
            high_error_points = sample_stats[sample_stats["std_scaled"] >= sample_stats["std_scaled"].quantile(0.8)]
            if len(high_error_points) > 0:
                ax.text(
                    0.02, 0.98 - sample_idx * 0.15,  # Offset for each sample
                    f"{sample_name} High Error:\n" + "\n".join([
                        f"R{int(row['Row'])}C{int(row['Col'])}: {row['mean_scaled']:.1f}±{row['std_scaled']:.2f}"
                        for _, row in high_error_points.head(3).iterrows()
                    ]),
                    transform=ax.transAxes,
                    fontsize=8,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=sample_colors[sample_idx], alpha=0.8)
                )

        # Add sample name labels positioned near the data (FIX: Use proper color format)
        sample_center_row = sample_stats["Row_pos"].mean()
        sample_center_col = sample_stats["Col"].mean()
        sample_min_col = sample_stats["Col"].min()
        sample_max_col = sample_stats["Col"].max()
        
        # Position sample name to the right of the data
        ax.text(
            sample_max_col + 1, sample_center_row, sample_name,
            fontsize=12, weight='bold', ha='left', va='center',
            bbox=dict(boxstyle='round,pad=0.4', facecolor=sample_colors[sample_idx], alpha=0.8),
            zorder=4
        )
        
        # Also add a smaller label to the left for reference
        ax.text(
            sample_min_col - 1.5, sample_center_row, f"#{sample_idx + 1}",
            fontsize=10, weight='bold', ha='center', va='center',
            bbox=dict(boxstyle='circle,pad=0.3', facecolor=sample_colors[sample_idx], alpha=0.6),
            zorder=4
        )

        # Draw dogbone outline for this sample
        from matplotlib.patches import Polygon

        # Determine min/max row/col from this sample's data
        sample_row_min, sample_row_max = sample_stats["Row_pos"].min(), sample_stats["Row_pos"].max()
        sample_col_min, sample_col_max = sample_stats["Col"].min(), sample_stats["Col"].max()

        # Dogbone parameters
        total_width = 0.1 * (sample_col_max - sample_col_min) if sample_col_max > sample_col_min else 1.0
        total_height = sample_row_max - sample_row_min

        # Padding
        horizontal_padding = 0.2 * total_width if total_width > 0 else 0.2
        vertical_padding = 0.3

        # Extended boundaries
        extended_col_min = sample_col_min - horizontal_padding
        extended_col_max = sample_col_max + horizontal_padding
        extended_row_min = sample_row_min - vertical_padding
        extended_row_max = sample_row_max + vertical_padding

        # Dogbone section widths
        extended_width = extended_col_max - extended_col_min
        grip_section_width = 0.22 * extended_width
        gauge_transition_width = 0.1 * extended_width

        # Calculate key x-coordinates
        left_grip_end = extended_col_min + grip_section_width
        left_gauge_start = left_grip_end + gauge_transition_width
        right_gauge_end = extended_col_max - grip_section_width - gauge_transition_width
        right_grip_start = extended_col_max - grip_section_width

        # Calculate gauge section height
        gauge_height_reduction = 0.25 * total_height if total_height > 0 else 0.5
        gauge_row_min = sample_row_min + gauge_height_reduction
        gauge_row_max = sample_row_max - gauge_height_reduction

        # Create dogbone coordinates
        x_coords = [
            extended_col_min, extended_col_min, left_grip_end, left_gauge_start,
            right_gauge_end, right_grip_start, extended_col_max, extended_col_max,
            right_grip_start, right_gauge_end, left_gauge_start, left_grip_end,
            extended_col_min
        ]

        y_coords = [
            extended_row_min, extended_row_max, extended_row_max, gauge_row_max,
            gauge_row_max, extended_row_max, extended_row_max, extended_row_min,
            extended_row_min, gauge_row_min, gauge_row_min, extended_row_min,
            extended_row_min
        ]

        dogbone_coords = list(zip(x_coords, y_coords))
        dogbone = Polygon(dogbone_coords, fill=False, edgecolor=sample_colors[sample_idx], 
                         linewidth=2, zorder=0, linestyle='--' if sample_idx > 0 else '-')
        ax.add_patch(dogbone)

    # Colorbar for mean (using global range)
    if sc is not None:
        cbar = plt.colorbar(sc, ax=ax, orientation='horizontal', pad=0.2) 
        cbar.set_label(f"Mean ({unit_choice})")
    else:
        # Create a dummy scatter for colorbar when no scatter exists
        cmap = excel_cmap if cmap_option == "Excel Yellow→Green" else "viridis"
        dummy_sc = ax.scatter([], [], c=[], vmin=combined_stats["mean_scaled"].min(), 
                             vmax=combined_stats["mean_scaled"].max(), cmap=cmap)
        cbar = plt.colorbar(dummy_sc, ax=ax)
        cbar.set_label(f"Mean ({unit_choice})")

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(f"Multi-Sample Comparison ({unit_choice}) - {display_mode}", fontsize=16)
    ax.invert_yaxis()  # Row 1 at top
    ax.grid(True, alpha=0.3)
    ax.set_xticks([])  # remove x-axis ticks
    ax.set_yticks([])  # remove y-axis ticks    


    # Set Y-axis limits from sliders
    ax.set_ylim(y_min, y_max)
    
    # Add legend for samples
    legend_elements = [plt.Line2D([0], [0], color=sample_colors[i], lw=2, 
                                 label=name) for i, name in enumerate(sample_data.keys())]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
    ax.legend().set_visible(False)


        
    st.pyplot(fig)
    plt.close()

else:
    st.info("Upload one or more CSV files to begin.")