import streamlit as st
import serial
import time
import threading
import queue
from datetime import datetime
import pandas as pd
import io
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
    if col in [1, 2, 3,4,13, 14, 15, 16]:  # Grip sections (first 3 and last 3 columns)
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
                'Frequency': entry.get('frequency', ''),
                'Value': entry['data']
            })
    return pd.DataFrame(rows)


def sweep_frequencies_and_store(reader, cell_key, frequencies=[60, 120, 240, 480, 960], max_retries=3):
    if cell_key not in st.session_state.cell_measurements:
        st.session_state.cell_measurements[cell_key] = []
    for freq in frequencies:
        retries = 0
        while retries < max_retries:
            reader.send_command(f"RC[current_frequency]={freq}")
            time.sleep(0.5)
            reader.send_command("RC[current_frequency]?")
            time.sleep(0.2)
            freq_response = reader.read_response()
            if str(freq) in str(freq_response):
                reader.send_command("RC[current_value]?")
                time.sleep(0.5)
                value = reader.read_response()
                timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                st.session_state.cell_measurements[cell_key].append({
                    'timestamp': timestamp,
                    'frequency': freq,
                    'data': value
                })
                break
            else:
                retries += 1
                time.sleep(1)
        else:
            st.error(f"Failed to set frequency {freq} after {max_retries} attempts. Skipping measurement.")


def create_dogbone_grid(reader, is_connected):
    is_connected = True
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
    
    # Track selected cell
    st.session_state.sample_name = st.text_input(
        "Sample Name",
        value=st.session_state.sample_name,
        help="Enter a name or ID for this specimen/sample"
    )

        
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
                    
                    # Button styling based on selection and data
                    if st.session_state.selected_cell == cell_key:
                        button_type = "primary"
                    else:
                        button_type = "secondary"
                    
                    cell_emoji = get_cell_emoji(cell_type)
                    cell_label = f"{cell_emoji} R{row}C{col}"


                    if st.button(
                        f"{cell_label}" + (" 📊" if has_data else ""),
                        key=cell_key,
                        disabled=not is_connected,
                        help=f"Row {row}, Column {col} - {cell_type.title()} section" + (f" - Has {len(st.session_state.cell_measurements.get(cell_key, []))} measurements" if has_data else ""),
                        type=button_type
                    ):
                        st.session_state.selected_cell = cell_key
                        st.session_state.auto_set_measurement = True
                        st.session_state.auto_start_streaming = True

                        st.rerun()
                else:
                    # Inactive cell
                    st.markdown(
                        '<div style="height: 40px; margin: 1px; background-color: #f9f9f9; border: 1px dashed #ccc; border-radius: 4px; opacity: 0.3;"></div>',
                        unsafe_allow_html=True
                    )
    
    # Show selected cell info and measurement controls
    if st.session_state.selected_cell and is_connected:
        st.markdown("---")
        selected_parts = st.session_state.selected_cell.split("_")
        row, col = int(selected_parts[1]), int(selected_parts[2])
        cell_type = get_cell_type(row, col)
        
        if st.session_state.cell_measurements:
            df = get_cell_data_df()
            if not df.empty:
                latest_row = df.iloc[-1]
                st.subheader("🆕 Latest Measurement Entry")
                st.code(latest_row.to_csv(index=False, header=True).strip())


        col1, col2= st.columns(2)
        with col1:
            if st.button("📊 Get Single Value"):
                reader.send_command("RC[current_value]?")
                # Store measurement for this cell
                if st.session_state.selected_cell not in st.session_state.cell_measurements:
                    st.session_state.cell_measurements[st.session_state.selected_cell] = []
        
        with col2:
            if st.button("🎯 Set Measurement Point"):
                # This could send a command to set measurement coordinates if supported
                st.success(f"Measurement point set to R{row}C{col}")
        

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
    
    if 'cell_measurements' not in st.session_state:
        st.session_state.cell_measurements = {}

    if 'selected_cell' not in st.session_state:
        st.session_state.selected_cell = None
        st.session_state.cell_measurements = {}

        # Sample name input (initialize if not present)
    if 'sample_name' not in st.session_state:
        st.session_state.sample_name = ""
    
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
                    st.success(message)
                else:
                    st.error(message)
                st.rerun()
        
        with col2:
            if st.button("❌ Disconnect", disabled=not st.session_state.reader.is_connected):
                st.session_state.reader.disconnect()
                st.session_state.is_streaming = False
                st.success("Disconnected")
                st.rerun()
        
        # Connection status
        if st.session_state.reader.is_connected:
            st.success("🟢 Connected")
        else:
            st.error("🔴 Disconnected")
    
    # Create dogbone grid (always shown)
    create_dogbone_grid(st.session_state.reader, st.session_state.reader.is_connected)
    st.session_state.reader.is_connected = True
    # Main content area
    if st.session_state.reader.is_connected:
        
        # Control buttons
        st.subheader("📊 Measurement Controls")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("📈 Start Streaming", disabled=st.session_state.is_streaming):
                st.session_state.reader.start_streaming()
                st.session_state.is_streaming = True
                st.rerun()
        
        with col2:
            if st.button("⏹️ Stop Streaming", disabled=not st.session_state.is_streaming):
                st.session_state.reader.stop_streaming()
                st.session_state.is_streaming = False
                st.rerun()
        
        with col3:
            if st.button("📊 Get Current Value"):
                st.session_state.reader.send_command("RC[current_value]?")
        
        with col4:
            if st.button("🔧 Get Device Info"):
                commands = [
                    "RC[current_frequency]?",
                    "RC[current_units]?",
                    "RC[current_mode]?",
                    "RC[instrument_serial_number]?"
                ]
                for cmd in commands:
                    st.session_state.reader.send_command(cmd)
        


        # Add this after the grid and before/after Live Data
        if st.session_state.cell_measurements:
            df = get_cell_data_df()
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Download All Cell Measurements (CSV)",
                data=csv,
                file_name='dogbone_measurements.csv',
                mime='text/csv'
            )

        st.markdown("---")
        
        # Manual command input
        st.subheader("💻 Manual Command")
        col1, col2 = st.columns([3, 1])
        
        with col1:
            manual_command = st.text_input(
                "Enter RC command:", 
                placeholder="e.g., RC[current_frequency]?",
                help="Enter any RC command from the manual"
            )
        
        with col2:
            if st.button("📤 Send Command"):
                if manual_command:
                    success, message = st.session_state.reader.send_command(manual_command)
                    if success:
                        st.success("Command sent!")
                    else:
                        st.error(message)
        
        st.markdown("---")
        
        # Data display
        st.subheader("📈 Live Data")
        
        # Create containers for live updates
        data_container = st.container()
        
        # Read and display new data
        new_data_count = 0
        while not st.session_state.reader.data_queue.empty() and new_data_count < 10:
            timestamp, data = st.session_state.reader.data_queue.get()
            st.session_state.measurements.append({'timestamp': timestamp, 'data': data})
            
            # Also store in selected cell measurements if a cell is selected
            if st.session_state.selected_cell:
                if st.session_state.selected_cell not in st.session_state.cell_measurements:
                    st.session_state.cell_measurements[st.session_state.selected_cell] = []
                st.session_state.cell_measurements[st.session_state.selected_cell].append({
                    'timestamp': timestamp, 
                    'data': data
                })
            
            new_data_count += 1
        
        # Keep only last 100 measurements
        if len(st.session_state.measurements) > 100:
            st.session_state.measurements = st.session_state.measurements[-100:]
        
        with data_container:
            if st.session_state.measurements:
                # Display latest measurements in a table
                recent_measurements = st.session_state.measurements[-10:][::-1]  # Last 10, reversed
                
                st.write("**Latest Measurements:**")
                for measurement in recent_measurements:
                    st.code(f"{measurement['timestamp']}: {measurement['data']}")
                
                # Clear data button
                if st.button("🗑️ Clear Data"):
                    st.session_state.measurements = []
                    st.session_state.cell_measurements = {}
                    st.rerun()
            else:
                st.info("No data received yet. Try getting current value or starting streaming.")
        
        # Auto-refresh for streaming
        if st.session_state.is_streaming:
            time.sleep(0.1)
            st.rerun()
    
    else:
        st.warning("Please connect to the device first using the sidebar settings.")
        
        st.markdown("""
        ### 📝 Quick Start Guide:
        1. Set the correct COM port in the sidebar (check Device Manager on Windows)
        2. Configure baud rate and other serial settings if needed
        3. Click "Connect" 
        4. Select a cell in the dogbone grid above
        5. Use "Get Single Value" for individual readings
        6. Use "Start Streaming" for continuous measurements
        
        ### 🔧 Available Commands:
        - `RC[current_value]?` - Get current measurement
        - `RC[current_frequency]?` - Get measurement frequency
        - `RC[current_units]?` - Get measurement units
        - `RC[current_mode]?` - Get measurement mode
        - `RC[streaming_value]=1` - Start streaming
        - `RC[streaming_value]=0` - Stop streaming
        """)

if __name__ == "__main__":
    main()