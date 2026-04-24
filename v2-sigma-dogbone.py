import streamlit as st
import serial
import time
import threading
import queue
from datetime import datetime
import pandas as pd
import io
import matplotlib.pyplot as plt
import numpy as np
import re
from matplotlib.patches import Polygon
from matplotlib.colors import LinearSegmentedColormap
from sigmatest_serial import SigmatestReader


def get_cell_emoji(cell_type):
    return {
        "grip": "🔵",
        "transition": "🟠",
        "gauge": "🟢"
    }.get(cell_type, "")


def get_cell_type(row, col):
    """
    Determine the cell type for dogbone specimen grid
    Returns: 'grip', 'transition', 'gauge', or 'inactive'
    """
    # Dogbone pattern: grip sections on ends, gauge section in middle, transitions between
    if col in [1, 2, 3, 4, 13, 14, 15, 16]:  # Grip sections (first 4 and last 4 columns)
        return "grip"
    elif col in [5, 12]:  # Transition sections
        return "transition" 
    elif col in [6, 7, 8, 9, 10, 11]:  # Gauge section (middle)
        return "gauge"
    else:
        return "inactive"


def get_cell_color(cell_type):
    """Get the color scheme for different cell types"""
    colors = {
        "grip": "#4A90E2",      # Blue
        "transition": "#F5A623", # Orange  
        "gauge": "#7ED321",     # Green
        "inactive": "#F9F9F9"   # Light gray
    }
    return colors.get(cell_type, "#F9F9F9")


def get_cell_data_df():
    """Convert cell measurements to DataFrame with frequency information"""
    rows = []
    sample_name = st.session_state.get('sample_name', '')
    
    for cell, measurements in st.session_state.cell_measurements.items():
        for entry in measurements:
            parts = cell.split('_')
            row = parts[1]
            col = parts[2]
            rows.append({
                'Sample Name': sample_name,
                'Cell': f'R{row}C{col}',
                'Row': row,
                'Col': col,
                'Timestamp': entry['timestamp'],
                'Frequency (kHz)': entry.get('frequency', st.session_state.get('current_frequency', 60)),
                'Value': entry['data']
            })
    
    return pd.DataFrame(rows)


def set_frequency_and_confirm(reader, frequency, max_retries=2):
    """Set frequency and confirm it was set correctly, managing streaming state"""
    
    # Check if streaming is currently active
    was_streaming = st.session_state.get('is_streaming', False)
    
    try:
        # If streaming is active, stop it temporarily
        if was_streaming:
            reader.stop_streaming()
            st.session_state.is_streaming = False
            time.sleep(0.5)  # Give time for streaming to stop
        
        for attempt in range(max_retries):
            try:
                # Set frequency
                reader.send_command(f"RC[current_frequency]={frequency}")
                time.sleep(0.3)  # Short delay for response
                
                # Read the response - should be "RC[ok]" if successful
                response = reader.read_response()
                
                if response and ("ok" in str(response).lower() or "RC[ok]" in str(response)):
                    st.session_state.current_frequency = frequency
                    
                    # Restart streaming if it was active before
                    if was_streaming:
                        time.sleep(0.3)  # Brief pause before restarting
                        reader.start_streaming()
                        st.session_state.is_streaming = True
                    
                    st.toast(f"Frequency set to {frequency} kHz", icon="✅")
                    return True
                else:
                    # If no "ok" response, try verification
                    reader.send_command("RC[current_frequency]?")
                    time.sleep(0.5)
                    
                    freq_response = reader.read_response()
                    if freq_response and (str(frequency) in str(freq_response) or f"{frequency}000" in str(freq_response)):
                        st.session_state.current_frequency = frequency
                        
                        # Restart streaming if it was active before
                        if was_streaming:
                            time.sleep(0.3)
                            reader.start_streaming()
                            st.session_state.is_streaming = True
                        
                        st.toast(f"Frequency set to {frequency} kHz", icon="✅")
                        return True
                    
                    time.sleep(1.0)  # Wait before retry
                    
            except Exception as e:
                st.toast(f"Frequency setting attempt {attempt + 1} failed", icon="⚠️")
                time.sleep(1.0)
        
        # If we get here, all attempts failed
        # Restart streaming if it was active before, even on failure
        if was_streaming:
            try:
                reader.start_streaming()
                st.session_state.is_streaming = True
            except:
                st.toast("Failed to restart streaming after frequency change failure", icon="❌")
        
        st.toast(f"Failed to set frequency to {frequency} kHz", icon="❌")
        return False
        
    except Exception as e:
        # Emergency cleanup - restart streaming if it was active
        if was_streaming:
            try:
                reader.start_streaming()
                st.session_state.is_streaming = True
            except:
                st.toast("Failed to restart streaming after frequency change error", icon="❌")
        
        st.toast(f"Error during frequency change: {str(e)}", icon="❌")
        return False


def fetch_current_frequency(reader, max_retries=3):
    """Fetch the current frequency from the device"""
    for attempt in range(max_retries):
        try:
            reader.send_command("RC[current_frequency]?")
            time.sleep(1.0)
            
            freq_response = reader.read_response()
            
            # Extract frequency from response
            if freq_response:
                # Look for common frequency patterns
                import re
                freq_match = re.search(r'(\d+(?:\.\d+)?)', str(freq_response))
                if freq_match:
                    detected_freq = float(freq_match.group(1))
                    
                    # Convert to kHz if needed and round to nearest standard frequency
                    if detected_freq >= 1000:  # Assume Hz, convert to kHz
                        detected_freq = detected_freq / 1000
                    
                    # Match to standard frequencies
                    standard_freqs = [60, 120, 240, 480, 960]
                    closest_freq = min(standard_freqs, key=lambda x: abs(x - detected_freq))
                    
                    if abs(closest_freq - detected_freq) < 10:  # Within 10 kHz tolerance
                        st.session_state.current_frequency = closest_freq
                        st.toast(f"Device frequency: {closest_freq} kHz", icon="📡")
                        return closest_freq
            
            time.sleep(1.0)
            
        except Exception as e:
            if attempt == max_retries - 1:  # Only show error on last attempt
                st.toast(f"Frequency fetch failed: {str(e)}", icon="⚠️")
            time.sleep(1.0)
    
    st.toast("Could not fetch current frequency from device", icon="⚠️")
    return st.session_state.get('current_frequency', 60)  # Return default


def start_new_sample():
    """Clear all data and start fresh with new sample"""
    st.session_state.cell_measurements = {}
    st.session_state.measurements = []
    st.session_state.selected_cell = None
    
    # Fetch current frequency from device if connected
    if st.session_state.reader.is_connected:
        current_freq = fetch_current_frequency(st.session_state.reader)
    
    st.toast(f"Started new sample: '{st.session_state.sample_name}'", icon="🆕")
    st.rerun()


def create_status_notice_bar():
    """Create a notice bar showing current status and recent activity"""
    
    # Get current status information
    selected_cell = st.session_state.get('selected_cell', None)
    is_streaming = st.session_state.get('is_streaming', False)
    latest_measurement = None
    
    # Get the most recent measurement
    if st.session_state.measurements:
        latest_measurement = st.session_state.measurements[-1]
    
    # Create three columns for the status bar
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if selected_cell:
            parts = selected_cell.split("_")
            row, col = parts[1], parts[2]
            cell_type = get_cell_type(int(row), int(col))
            st.info(f"🎯 **Selected:** R{row}C{col} ({cell_type.title()})")
        else:
            st.warning("⚪ **No Cell Selected**")
    
    with col2:
        if is_streaming:
            st.success("🔴 **STREAMING ACTIVE**")
        else:
            st.info("⏸️ **Streaming Stopped**")
    
    with col3:
        if latest_measurement:
            freq = st.session_state.get('current_frequency', 60)
            timestamp = latest_measurement['timestamp'][-8:]  # Show only time part
            value = latest_measurement['data'][:20] + "..." if len(str(latest_measurement['data'])) > 20 else latest_measurement['data']
            st.success(f"📊 **Latest:** {timestamp} @ {freq}kHz")
            st.caption(f"Value: {value}")
        else:
            st.warning("📊 **No Data Yet**")


def create_results_dogbone_plot():
    """Create a dogbone visualization showing mean values with error bars"""
    
    # Convert cell measurements to DataFrame and process
    df = get_cell_data_df()
    if df.empty:
        st.info("No data to visualize yet. Start collecting measurements!")
        return
    
    # Extract numeric values from the data
    pattern = re.compile(r"RC\[current_value\]=([\d\.]+)")
    df["NumericValue"] = df["Value"].astype(str).str.extract(pattern).astype(float)
    df = df.dropna(subset=["NumericValue"]).copy()
    
    if df.empty:
        st.warning("No valid numeric values found in the data.")
        return
    
    # Convert columns to proper types
    df["Row"] = df["Row"].astype(int)
    df["Col"] = df["Col"].astype(int)
    
    # Calculate statistics for each cell
    stats = df.groupby(["Row", "Col"])["NumericValue"].agg(["mean", "std", "count"]).reset_index()
    stats = stats.fillna(0)  # Fill NaN std values with 0 for single measurements
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(16, 6))
    
    # Remove spines for cleaner look
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Create colormap (Excel-like yellow to green)
    excel_cmap = LinearSegmentedColormap.from_list("excel", ["yellow", "green"])
    
    # Create the scatter plot with color-coded means
    scatter = ax.scatter(
        stats["Col"],
        stats["Row"],
        c=stats["mean"],
        s=400,  # Size of circles
        cmap=excel_cmap,
        edgecolor='black',
        linewidth=1,
        zorder=3,
        alpha=0.8
    )
    
    # Add error bars
    ax.errorbar(
        stats["Col"],
        stats["Row"],
        yerr=stats["std"],
        fmt='none',
        ecolor='black',
        alpha=0.7,
        capsize=4,
        capthick=1.5,
        elinewidth=1.5,
        zorder=2
    )
    
    # Add value labels on each point
    for _, row in stats.iterrows():
        # Show mean ± std
        label_text = f"{row['mean']:.1f}"
        if row['std'] > 0:
            label_text += f"\n±{row['std']:.2f}"
        
        ax.text(
            row["Col"],
            row["Row"],
            label_text,
            fontsize=8,
            color='black',
            ha='center',
            va='center',
            weight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8),
            zorder=4
        )
    
    # FIXED PRESET RANGES for dogbone (16x3 grid)
    # Standard dogbone specimen dimensions
    preset_col_min, preset_col_max = 0.5, 16.5
    preset_row_min, preset_row_max = 0.5, 3.5
    
    # Draw dogbone outline with PRESET dimensions
    padding = 0.3
    extended_col_min = preset_col_min + padding
    extended_col_max = preset_col_max - padding
    extended_row_min = preset_row_min + padding
    extended_row_max = preset_row_max - padding
    
    # Calculate dogbone sections based on PRESET 16-column layout
    total_width = extended_col_max - extended_col_min
    grip_width = 4.0  # Columns 1-4 and 13-16 (4 columns each)
    transition_width = 1.0  # Columns 5 and 12 (1 column each)
    
    # Fixed section boundaries
    left_grip_end = extended_col_min + grip_width
    left_gauge_start = left_grip_end + transition_width
    right_gauge_end = extended_col_max - grip_width - transition_width
    right_grip_start = extended_col_max - grip_width
    
    # Gauge section height reduction (fixed amount)
    gauge_height_reduction = 0.4
    gauge_row_min = extended_row_min + gauge_height_reduction
    gauge_row_max = extended_row_max - gauge_height_reduction
    
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
    dogbone = Polygon(
        dogbone_coords,
        fill=False,
        edgecolor='blue',
        linewidth=2,
        linestyle='-',
        alpha=0.7,
        zorder=1
    )
    ax.add_patch(dogbone)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, orientation='horizontal', pad=0.15, aspect=30)
    cbar.set_label("Mean Value", fontsize=12)
    
    # Add FIXED section labels at preset positions
    mid_row = 2.0  # Middle of 3-row grid
    
    # Section labels at fixed positions
    ax.text(2.5, mid_row - 0.8, "GRIP", ha='center', va='center', 
           fontsize=10, weight='bold', color='blue',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))
    
    ax.text(5, mid_row - 0.8, "TRANS", ha='center', va='center',
           fontsize=9, weight='bold', color='orange',
           bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow', alpha=0.7))
    
    ax.text(8.5, mid_row - 0.8, "GAUGE", ha='center', va='center',
           fontsize=10, weight='bold', color='green',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))
    
    ax.text(12, mid_row - 0.8, "TRANS", ha='center', va='center',
           fontsize=9, weight='bold', color='orange', 
           bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow', alpha=0.7))
    
    ax.text(14.5, mid_row - 0.8, "GRIP", ha='center', va='center',
           fontsize=10, weight='bold', color='blue',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))
    
    # Customize plot with FIXED ranges
    ax.set_xlabel("Column Position", fontsize=12)
    ax.set_ylabel("Row Position", fontsize=12)
    ax.set_title(f"Dogbone Test Results - {st.session_state.get('sample_name', 'Sample')} @ {st.session_state.get('current_frequency', 60)} kHz", 
                fontsize=14, weight='bold')
    
    # Invert y-axis so Row 1 is at top
    ax.invert_yaxis()
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Set FIXED axis limits (no auto-ranging)
    ax.set_xlim(preset_col_min, preset_col_max)
    ax.set_ylim(preset_row_max + 0.2, preset_row_min - 0.2)
    
    # Add statistics summary
    total_points = len(stats)
    mean_overall = stats["mean"].mean()
    std_overall = stats["std"].mean()
    
    stats_text = f"Points: {total_points} | Overall Mean: {mean_overall:.2f} | Avg Std: {std_overall:.2f}"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Display the plot
    st.pyplot(fig)
    plt.close()


def create_dogbone_grid(reader, is_connected):
    """Create the interactive dogbone specimen grid"""
    st.subheader("🔬 Dogbone Specimen Grid (16×3)")
    
    if not is_connected:
        st.info("Connect to device to enable measurements")
    
    # Create legend
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("🔵 **Grip Section** - Specimen clamping area")
    with col2:
        st.markdown("🟠 **Transition** - Grip to gauge transition")
    with col3:
        st.markdown("🟢 **Gauge Section** - Active testing area")
    
    # Sample configuration section
    st.markdown("---")
    st.subheader("🏷️ Sample Configuration")
    
    # Sample name and frequency controls in columns
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        new_sample_name = st.text_input(
            "Sample Name",
            value=st.session_state.get('sample_name', ''),
            help="Enter a name or ID for this specimen/sample"
        )
        
        # Check if sample name changed
        if new_sample_name != st.session_state.get('sample_name', ''):
            st.session_state.sample_name = new_sample_name
    
    with col2:
        # Frequency selection
        frequency_options = [60, 120, 240, 480, 960]
        current_freq_index = 0
        if 'current_frequency' in st.session_state:
            try:
                current_freq_index = frequency_options.index(st.session_state.current_frequency)
            except ValueError:
                current_freq_index = 0
        
        selected_frequency = st.selectbox(
            "Frequency (kHz)",
            frequency_options,
            index=current_freq_index,
            help="Select measurement frequency"
        )
        
        # Update frequency if changed and connected
        if selected_frequency != st.session_state.get('current_frequency', 60):
            if is_connected:
                set_frequency_and_confirm(reader, selected_frequency)
            else:
                st.session_state.current_frequency = selected_frequency
    
    with col3:
        if st.button("🆕 New Sample", help="Clear all data and start fresh"):
            start_new_sample()
    
    # Show current sample info
    if st.session_state.get('sample_name'):
        total_measurements = sum(len(measurements) for measurements in st.session_state.cell_measurements.values())
        st.info(f"📊 Sample: **{st.session_state.sample_name}** | Frequency: **{st.session_state.get('current_frequency', 60)} kHz** | Total Measurements: **{total_measurements}**")
    
    st.markdown("---")
    
    # Create the dogbone grid
    for row in range(1, 4):  # 3 rows
        cols = st.columns(16)  # 16 columns
        
        for col in range(1, 17):  # 16 columns
            cell_type = get_cell_type(row, col)
            
            with cols[col-1]:
                if cell_type != "inactive":
                    cell_label = f"R{row}C{col}"
                    cell_key = f"cell_{row}_{col}"
                    
                    # Check if this cell has measurements
                    has_data = cell_key in st.session_state.cell_measurements
                    measurement_count = len(st.session_state.cell_measurements.get(cell_key, []))
                    
                    # Button styling based on selection and data
                    if st.session_state.get('selected_cell') == cell_key:
                        button_type = "primary"
                    else:
                        button_type = "secondary"
                    
                    cell_emoji = get_cell_emoji(cell_type)
                    display_label = f"{cell_emoji} R{row}C{col}"
                    
                    if has_data:
                        display_label += f" ({measurement_count})"
                    
                    if st.button(
                        display_label,
                        key=cell_key,
                        disabled=not is_connected,
                        help=f"Row {row}, Column {col} - {cell_type.title()} section" + (f" - {measurement_count} measurements" if has_data else ""),
                        type=button_type
                    ):
                        # Cell selection logic
                        st.session_state.selected_cell = cell_key
                        
                        # Show selection notification via toast
                        cell_type = get_cell_type(row, col)
                        st.toast(f"Selected cell R{row}C{col} ({cell_type.title()} section)", icon="🎯")
                        
                        st.rerun()
                else:
                    # Inactive cell
                    st.markdown(
                        '<div style="height: 40px; margin: 1px; background-color: #f9f9f9; border: 1px dashed #ccc; border-radius: 4px; opacity: 0.3;"></div>',
                        unsafe_allow_html=True
                    )
    
    # Show selected cell info and measurement controls
    if st.session_state.get('selected_cell') and is_connected:
        st.markdown("---")
        selected_parts = st.session_state.selected_cell.split("_")
        row, col = int(selected_parts[1]), int(selected_parts[2])
        cell_type = get_cell_type(row, col)
        
        # Current selection info
        st.subheader(f"🎯 Active Cell: R{row}C{col} ({cell_type.title()} Section)")
        
        # Show latest measurement if available
        if st.session_state.selected_cell in st.session_state.cell_measurements:
            measurements = st.session_state.cell_measurements[st.session_state.selected_cell]
            if measurements:
                latest = measurements[-1]
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Latest Value", latest['data'])
                with col2:
                    st.metric("Timestamp", latest['timestamp'])
                with col3:
                    st.metric("Frequency", f"{latest.get('frequency', st.session_state.get('current_frequency', 60))} kHz")
        
        # Measurement controls
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📊 Get Single Value"):
                reader.send_command("RC[current_value]?")
                st.toast("Single value requested", icon="📊")


def main():
    st.set_page_config(
        page_title="Sigmatest RS232 Reader",
        page_icon="📡",
        layout="wide"
    )
    
    st.title("📡 Sigmatest RS232 Communication")
    st.markdown("---")
    
    # Initialize session state
    if 'reader' not in st.session_state:
        st.session_state.reader = SigmatestReader()
        st.session_state.measurements = []
        st.session_state.is_streaming = False
        st.session_state.cell_measurements = {}
        st.session_state.selected_cell = None
        st.session_state.sample_name = ""
        st.session_state.current_frequency = 60
    
    # Sidebar for connection settings
    with st.sidebar:
        st.header("🔌 Connection Settings")
        
        # Serial port settings
        port = st.text_input("Serial Port", value="COM3", help="e.g., COM3 (Windows) or /dev/ttyUSB0 (Linux)")
        baud_rate = st.selectbox("Baud Rate", [300, 1200, 2400, 4800, 9600, 19200, 38400, 57600, 115200], index=4)
        data_bits = st.selectbox("Data Bits", [7, 8], index=1)
        stop_bits = st.selectbox("Stop Bits", [1, 2], index=0)
        parity = st.selectbox("Parity", ['N', 'E', 'O'], index=0)
        
        st.markdown("---")
        
        # Connection controls
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔌 Connect", disabled=st.session_state.reader.is_connected):
                success, message = st.session_state.reader.connect(port, baud_rate, data_bits, stop_bits, parity)
                if success:
                    st.toast("Connected to device", icon="🔌")
                    # Auto-start streaming after connection
                    st.session_state.reader.start_streaming()
                    st.session_state.is_streaming = True
                    st.toast("Streaming started automatically", icon="📡")
                    # Set initial frequency when connecting
                    set_frequency_and_confirm(st.session_state.reader, st.session_state.current_frequency)
                else:
                    st.toast(f"Connection failed: {message}", icon="❌")
                st.rerun()
        
        with col2:
            if st.button("❌ Disconnect", disabled=not st.session_state.reader.is_connected):
                st.session_state.reader.disconnect()
                st.session_state.is_streaming = False
                st.toast("Disconnected from device", icon="❌")
                st.rerun()
        
        # Connection status
        if st.session_state.reader.is_connected:
            st.success("🟢 Connected")
        else:
            st.error("🔴 Disconnected")
        
        # Streaming control - only show when connected
        if st.session_state.reader.is_connected:
            st.markdown("---")
            st.subheader("📡 Streaming Control")
            
            if st.session_state.get('is_streaming', False):
                if st.button("⏹️ Stop Streaming", key="stop_streaming_sidebar"):
                    st.session_state.reader.stop_streaming()
                    st.session_state.is_streaming = False
                    st.toast("Streaming stopped", icon="⏹️")
                    st.rerun()
                st.info("🔴 Streaming Active")
            else:
                if st.button("▶️ Start Streaming", key="start_streaming_sidebar"):
                    st.session_state.reader.start_streaming()
                    st.session_state.is_streaming = True
                    st.toast("Streaming started", icon="▶️")
                    st.rerun()
                st.warning("⏸️ Streaming Stopped")
        
        # Device info section
        if st.session_state.reader.is_connected:
            st.markdown("---")
            st.subheader("📋 Device Info")
            if st.button("🔧 Get Device Info"):
                commands = [
                    "RC[current_frequency]?",
                    "RC[current_units]?",
                    "RC[current_mode]?",
                    "RC[instrument_serial_number]?"
                ]
                for cmd in commands:
                    st.session_state.reader.send_command(cmd)
    
    # Create dogbone grid (always shown)
    create_dogbone_grid(st.session_state.reader, st.session_state.reader.is_connected)
    
    # Main content area
    if st.session_state.reader.is_connected:
        
        # Download section
        if st.session_state.cell_measurements:
            st.markdown("---")
            st.subheader("💾 Data Export")
            
            df = get_cell_data_df()
            
            col1, col2 = st.columns([2, 1])
            with col1:
                # Generate filename with sample name and timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                sample_part = st.session_state.get('sample_name', 'sample').replace(' ', '_')
                filename = f"{sample_part}_{timestamp}.csv"
                
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label=f"⬇️ Download Measurements CSV ({len(df)} records)",
                    data=csv,
                    file_name=filename,
                    mime='text/csv'
                )
            
            with col2:
                st.metric("Total Measurements", len(df))
        
        st.markdown("---")
        
        # Results Visualization Section
        if st.session_state.cell_measurements:
            st.subheader("📊 Results Visualization")
            
            # Create notice bar with status information
            create_status_notice_bar()
            
            # Create the dogbone visualization
            create_results_dogbone_plot()
        
        st.markdown("---")
        
        # Data display
        st.subheader("📈 Live Data")
        
        # Streaming status indicator
        if st.session_state.get('is_streaming', False):
            st.success("🔴 **STREAMING ACTIVE** - Data is being collected continuously")
        else:
            st.info("⏸️ Streaming stopped - Use 'Get Single Value' for individual measurements")
        
        # Create containers for live updates
        data_container = st.container()
        
        # Read and display new data
        new_data_count = 0
        while not st.session_state.reader.data_queue.empty() and new_data_count < 10:
            timestamp, data = st.session_state.reader.data_queue.get()
            st.session_state.measurements.append({'timestamp': timestamp, 'data': data})
            
            # Store in selected cell measurements if a cell is selected
            if st.session_state.get('selected_cell'):
                if st.session_state.selected_cell not in st.session_state.cell_measurements:
                    st.session_state.cell_measurements[st.session_state.selected_cell] = []
                
                st.session_state.cell_measurements[st.session_state.selected_cell].append({
                    'timestamp': timestamp,
                    'frequency': st.session_state.get('current_frequency', 60),
                    'data': data
                })
            
            new_data_count += 1
        
        # Keep only last 100 general measurements
        if len(st.session_state.measurements) > 100:
            st.session_state.measurements = st.session_state.measurements[-100:]
        
        with data_container:
            if st.session_state.measurements:
                # Display latest measurements in a table
                recent_measurements = st.session_state.measurements[-10:][::-1]  # Last 10, reversed
                
                st.write("**Latest Measurements:**")
                for measurement in recent_measurements:
                    freq_display = f"@ {st.session_state.get('current_frequency', 60)} kHz"
                    st.code(f"{measurement['timestamp']} {freq_display}: {measurement['data']}")
                
                # Clear data button
                if st.button("🗑️ Clear All Data"):
                    st.session_state.measurements = []
                    st.session_state.cell_measurements = {}
                    st.session_state.selected_cell = None
                    st.toast("All data cleared", icon="🗑️")
                    st.rerun()
            else:
                st.info("No data received yet. Select a cell and measurements will appear here automatically.")
        
        # Auto-refresh for streaming
        if st.session_state.get('is_streaming', False):
            time.sleep(0.1)
            st.rerun()
    
    else:
        st.warning("Please connect to the device first using the sidebar settings.")
        
        st.markdown("""
        ### 📝 Quick Start Guide:
        1. **Connect**: Set the correct COM port in the sidebar and click "Connect"
        2. **Configure**: Enter sample name and select frequency
        3. **Select Cell**: Click any cell in the dogbone grid (streaming starts automatically)
        4. **Collect Data**: Data is collected automatically, or use "Get Single Value"
        5. **Download**: Export your measurements as CSV when done
        
        ### 🎯 Features:
        - **Auto-streaming**: Selecting a cell automatically starts data collection
        - **Frequency tracking**: All measurements include frequency information
        - **Sample management**: Easy sample naming and data organization
        - **Real-time feedback**: Instant notifications for selections and status
        """)


if __name__ == "__main__":
    main()