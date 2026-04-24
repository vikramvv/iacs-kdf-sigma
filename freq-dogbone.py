import streamlit as st
import serial
import time
import threading
import queue
from datetime import datetime
import pandas as pd
import io

class SigmatestReader:
    def __init__(self):
        self.serial_connection = None
        self.is_connected = False
        self.data_queue = queue.Queue()
        self.reading_thread = None
        self.stop_reading = False
        self.frequencies = [60, 120, 240, 480, 960]
        self.current_sweep_progress = 0
        self.is_sweeping = False
        
    def connect(self, port, baud_rate=9600, data_bits=8, stop_bits=1, parity='N'):
        """Connect to the Sigmatest device"""
        try:
            self.serial_connection = serial.Serial(
                port=port,
                baudrate=baud_rate,
                bytesize=data_bits,
                stopbits=stop_bits,
                parity=parity,
                timeout=1
            )
            self.is_connected = True
            return True, "Connected successfully"
        except Exception as e:
            return False, f"Connection failed: {str(e)}"
    
    def disconnect(self):
        """Disconnect from the device"""
        self.stop_reading = True
        if self.reading_thread and self.reading_thread.is_alive():
            self.reading_thread.join(timeout=2)
        
        if self.serial_connection and self.serial_connection.is_open:
            self.serial_connection.close()
        
        self.is_connected = False
        self.serial_connection = None
        self.is_sweeping = False
    
    def send_command(self, command):
        """Send a command to the device"""
        if not self.is_connected or not self.serial_connection:
            return False, "Not connected"
        
        try:
            self.serial_connection.write(f"{command}\r\n".encode())
            return True, "Command sent"
        except Exception as e:
            return False, f"Error sending command: {str(e)}"
    
    def read_response(self, timeout=2.0):
        """Read a single response from the device with timeout"""
        if not self.is_connected or not self.serial_connection:
            return None
        
        start_time = time.time()
        response = ""
        
        try:
            while (time.time() - start_time) < timeout:
                if self.serial_connection.in_waiting > 0:
                    data = self.serial_connection.read(1).decode()
                    if data == '\n' or data == '\r':
                        if response:
                            return response.strip()
                    else:
                        response += data
                time.sleep(0.01)
        except Exception as e:
            st.error(f"Error reading response: {str(e)}")
        
        return response.strip() if response else None
    
    def set_frequency(self, frequency):
        """Set the measurement frequency"""
        success, message = self.send_command(f"RC[current_frequency]={frequency}")
        if success:
            time.sleep(0.2)  # Allow device to settle
            # Verify frequency was set
            self.send_command("RC[current_frequency]?")
            response = self.read_response()
            return True, response
        return False, message
    
    def get_measurement(self):
        """Get a single measurement value"""
        success, message = self.send_command("RC[current_value]?")
        if success:
            response = self.read_response()
            return response
        return None
    
    def perform_frequency_sweep(self, cell_key):
        """Perform a complete frequency sweep for a cell"""
        self.is_sweeping = True
        self.current_sweep_progress = 0
        sweep_results = []
        measurement_set_id = int(time.time())  # Unique ID for this sweep
        
        try:
            for i, frequency in enumerate(self.frequencies):
                self.current_sweep_progress = i + 1
                
                # Set frequency
                success, freq_response = self.set_frequency(frequency)
                if not success:
                    sweep_results.append({
                        'timestamp': datetime.now().strftime("%H:%M:%S.%f")[:-3],
                        'data': f"Error setting frequency {frequency} Hz",
                        'frequency': frequency,
                        'measurement_set_id': measurement_set_id,
                        'status': 'error'
                    })
                    continue
                
                # Get measurement
                measurement = self.get_measurement()
                if measurement:
                    sweep_results.append({
                        'timestamp': datetime.now().strftime("%H:%M:%S.%f")[:-3],
                        'data': measurement,
                        'frequency': frequency,
                        'measurement_set_id': measurement_set_id,
                        'status': 'success'
                    })
                else:
                    sweep_results.append({
                        'timestamp': datetime.now().strftime("%H:%M:%S.%f")[:-3],
                        'data': f"No response at {frequency} Hz",
                        'frequency': frequency,
                        'measurement_set_id': measurement_set_id,
                        'status': 'error'
                    })
                
                # Small delay between frequencies
                time.sleep(0.1)
        
        except Exception as e:
            sweep_results.append({
                'timestamp': datetime.now().strftime("%H:%M:%S.%f")[:-3],
                'data': f"Sweep error: {str(e)}",
                'frequency': 0,
                'measurement_set_id': measurement_set_id,
                'status': 'error'
            })
        
        finally:
            self.is_sweeping = False
            self.current_sweep_progress = 0
        
        return sweep_results
    
    def start_streaming(self):
        """Start streaming measurements"""
        if self.is_connected:
            # Enable streaming mode
            self.send_command("RC[streaming_value]=1")
            
            # Start reading thread
            self.stop_reading = False
            self.reading_thread = threading.Thread(target=self._read_continuously)
            self.reading_thread.daemon = True
            self.reading_thread.start()
    
    def stop_streaming(self):
        """Stop streaming measurements"""
        if self.is_connected:
            # Disable streaming mode
            self.send_command("RC[streaming_value]=0")
        
        self.stop_reading = True
    
    def _read_continuously(self):
        """Continuously read data from the device"""
        while not self.stop_reading and self.is_connected:
            try:
                if self.serial_connection and self.serial_connection.in_waiting > 0:
                    data = self.serial_connection.readline().decode().strip()
                    if data:
                        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                        # For streaming, we don't know the frequency, so we'll mark it as unknown
                        measurement_data = {
                            'timestamp': timestamp,
                            'data': data,
                            'frequency': 'streaming',
                            'measurement_set_id': 0,
                            'status': 'streaming'
                        }
                        self.data_queue.put(measurement_data)
                time.sleep(0.01)  # Small delay to prevent excessive CPU usage
            except Exception as e:
                if not self.stop_reading:
                    error_data = {
                        'timestamp': datetime.now().strftime("%H:%M:%S.%f")[:-3],
                        'data': f"Error: {str(e)}",
                        'frequency': 'error',
                        'measurement_set_id': 0,
                        'status': 'error'
                    }
                    self.data_queue.put(error_data)
                break


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
                'Measurement_Set': entry.get('measurement_set_id', 0),
                'Frequency': entry.get('frequency', 'unknown'),
                'Timestamp': entry['timestamp'],
                'Value': entry['data'],
                'Status': entry.get('status', 'unknown')
            })
    
    return pd.DataFrame(rows)

def create_dogbone_grid(reader, is_connected):
    """Create the interactive dogbone specimen grid"""
    st.subheader("🔬 Dogbone Specimen Grid (16×3)")
    
    if not is_connected:
        st.info("Connect to device to enable measurements")
    elif reader.is_sweeping:
        st.info(f"🔄 Performing frequency sweep... ({reader.current_sweep_progress}/5 frequencies)")
    
    # Create legend
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("🔵 **Grip Section** - Specimen clamping area")
    with col2:
        st.markdown("🟠 **Transition** - Grip to gauge transition")
    with col3:
        st.markdown("🟢 **Gauge Section** - Active testing area")
    
    # Sample name input
    st.session_state.sample_name = st.text_input(
        "Sample Name",
        value=st.session_state.sample_name,
        help="Enter a name or ID for this specimen/sample"
    )
    
    # Frequency sweep info
    st.info("📊 Each cell measurement will collect data at all frequencies: 60, 120, 240, 480, 960 Hz")
        
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

                    # Count measurement sets for this cell
                    measurement_sets = 0
                    if has_data:
                        set_ids = set(entry.get('measurement_set_id', 0) for entry in st.session_state.cell_measurements[cell_key])
                        measurement_sets = len([s for s in set_ids if s > 0])  # Exclude streaming data (set_id=0)

                    if st.button(
                        f"{cell_label}" + (f" 📊×{measurement_sets}" if measurement_sets > 0 else ""),
                        key=cell_key,
                        disabled=not is_connected or reader.is_sweeping,
                        help=f"Row {row}, Column {col} - {cell_type.title()} section" + (f" - Has {measurement_sets} frequency sweep(s)" if measurement_sets > 0 else ""),
                        type=button_type
                    ):
                        st.session_state.selected_cell = cell_key
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
        
        st.subheader(f"🎯 Selected: R{row}C{col} ({cell_type.title()} Section)")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🌊 Perform Frequency Sweep", disabled=reader.is_sweeping):
                # Perform frequency sweep for selected cell
                with st.spinner(f"Collecting measurements at all frequencies for R{row}C{col}..."):
                    sweep_results = reader.perform_frequency_sweep(st.session_state.selected_cell)
                    
                    # Store results in cell measurements
                    if st.session_state.selected_cell not in st.session_state.cell_measurements:
                        st.session_state.cell_measurements[st.session_state.selected_cell] = []
                    
                    st.session_state.cell_measurements[st.session_state.selected_cell].extend(sweep_results)
                    
                    # Show results summary
                    successful_measurements = [r for r in sweep_results if r['status'] == 'success']
                    st.success(f"✅ Completed frequency sweep: {len(successful_measurements)}/5 frequencies successful")
                    
                    for result in sweep_results:
                        if result['status'] == 'success':
                            st.code(f"{result['frequency']} Hz: {result['data']}")
                        else:
                            st.error(f"{result['frequency']} Hz: {result['data']}")
                
                st.rerun()
        
        with col2:
            # Show latest measurement set for this cell
            if st.session_state.selected_cell in st.session_state.cell_measurements:
                measurements = st.session_state.cell_measurements[st.session_state.selected_cell]
                if measurements:
                    # Get the latest measurement set
                    latest_set_id = max(entry.get('measurement_set_id', 0) for entry in measurements if entry.get('measurement_set_id', 0) > 0)
                    if latest_set_id > 0:
                        latest_set = [m for m in measurements if m.get('measurement_set_id') == latest_set_id]
                        st.subheader("🆕 Latest Frequency Sweep")
                        for measurement in latest_set:
                            if measurement['status'] == 'success':
                                st.code(f"{measurement['frequency']} Hz: {measurement['data']}")


def main():
    st.set_page_config(
        page_title="Sigmatest RS232 Reader - Multi-Frequency",
        page_icon="📡",
        layout="wide"
    )
    
    st.title("📡 Sigmatest RS232 Communication - Multi-Frequency Sweep")
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
        
        # Frequency sweep info
        if st.session_state.reader.is_connected:
            st.markdown("---")
            st.subheader("🌊 Frequency Sweep")
            st.write("Available frequencies:")
            for freq in st.session_state.reader.frequencies:
                st.write(f"• {freq} Hz")
    
    # Create dogbone grid (always shown)
    create_dogbone_grid(st.session_state.reader, st.session_state.reader.is_connected)
    
    # Main content area
    if st.session_state.reader.is_connected:
        
        # Control buttons
        st.subheader("📊 Measurement Controls")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("📈 Start Streaming", disabled=st.session_state.is_streaming or st.session_state.reader.is_sweeping):
                st.session_state.reader.start_streaming()
                st.session_state.is_streaming = True
                st.rerun()
        
        with col2:
            if st.button("⏹️ Stop Streaming", disabled=not st.session_state.is_streaming):
                st.session_state.reader.stop_streaming()
                st.session_state.is_streaming = False
                st.rerun()
        
        with col3:
            if st.button("📊 Get Current Value", disabled=st.session_state.reader.is_sweeping):
                st.session_state.reader.send_command("RC[current_value]?")
        
        with col4:
            if st.button("🔧 Get Device Info", disabled=st.session_state.reader.is_sweeping):
                commands = [
                    "RC[current_frequency]?",
                    "RC[current_units]?",
                    "RC[current_mode]?",
                    "RC[instrument_serial_number]?"
                ]
                for cmd in commands:
                    st.session_state.reader.send_command(cmd)

        # Download data section
        if st.session_state.cell_measurements:
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                df = get_cell_data_df()
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇️ Download All Cell Measurements (CSV)",
                    data=csv,
                    file_name=f'dogbone_measurements_{st.session_state.sample_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                    mime='text/csv'
                )
            
            with col2:
                # Show measurement statistics
                df = get_cell_data_df()
                if not df.empty:
                    total_measurements = len(df)
                    successful_measurements = len(df[df['Status'] == 'success'])
                    unique_cells = df['Cell'].nunique()
                    unique_freq_sets = len(df[df['Measurement_Set'] > 0]['Measurement_Set'].unique())
                    
                    st.metric("Total Measurements", total_measurements)
                    st.metric("Successful", successful_measurements)
                    st.metric("Cells Measured", unique_cells)
                    st.metric("Frequency Sweeps", unique_freq_sets)

        st.markdown("---")
        
        # Manual command input
        st.subheader("💻 Manual Command")
        col1, col2 = st.columns([3, 1])
        
        with col1:
            manual_command = st.text_input(
                "Enter RC command:", 
                placeholder="e.g., RC[current_frequency]?",
                help="Enter any RC command from the manual",
                disabled=st.session_state.reader.is_sweeping
            )
        
        with col2:
            if st.button("📤 Send Command", disabled=st.session_state.reader.is_sweeping):
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
            measurement_data = st.session_state.reader.data_queue.get()
            st.session_state.measurements.append(measurement_data)
            
            # Also store in selected cell measurements if a cell is selected (for streaming data)
            if st.session_state.selected_cell and measurement_data.get('status') == 'streaming':
                if st.session_state.selected_cell not in st.session_state.cell_measurements:
                    st.session_state.cell_measurements[st.session_state.selected_cell] = []
                st.session_state.cell_measurements[st.session_state.selected_cell].append(measurement_data)
            
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
                    freq_info = f" ({measurement.get('frequency', 'unknown')} Hz)" if measurement.get('frequency', 'unknown') != 'unknown' else ""
                    st.code(f"{measurement['timestamp']}: {measurement['data']}{freq_info}")
                
                # Clear data button
                if st.button("🗑️ Clear Data", disabled=st.session_state.reader.is_sweeping):
                    st.session_state.measurements = []
                    st.session_state.cell_measurements = {}
                    st.rerun()
            else:
                st.info("No data received yet. Select a cell and perform a frequency sweep, or start streaming.")
        
        # Auto-refresh for streaming or during sweep
        if st.session_state.is_streaming or st.session_state.reader.is_sweeping:
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
        5. Click "Perform Frequency Sweep" to collect measurements at all 5 frequencies
        6. Download the CSV file with all frequency data
        
        ### 🌊 Multi-Frequency Operation:
        - Each cell measurement automatically sweeps through: **60, 120, 240, 480, 960 Hz**
        - All 5 frequency measurements are grouped together in the CSV
        - Progress indicator shows sweep status
        - Results include frequency, timestamp, and measurement value
        
        ### 🔧 Available Commands:
        - `RC[current_value]?` - Get current measurement
        - `RC[current_frequency]?` - Get current frequency
        - `RC[current_frequency]=XXX` - Set frequency (60, 120, 240, 480, 960)
        - `RC[current_units]?` - Get measurement units
        - `RC[current_mode]?` - Get measurement mode
        - `RC[streaming_value]=1` - Start streaming
        - `RC[streaming_value]=0` - Stop streaming
        """)

if __name__ == "__main__":
    main()