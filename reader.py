import streamlit as st
import serial
import time
import threading
import queue
from datetime import datetime

class SigmatestReader:
    def __init__(self):
        self.serial_connection = None
        self.is_connected = False
        self.data_queue = queue.Queue()
        self.reading_thread = None
        self.stop_reading = False
        
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
    
    def send_command(self, command):
        """Send a command to the device"""
        if not self.is_connected or not self.serial_connection:
            return False, "Not connected"
        
        try:
            self.serial_connection.write(f"{command}\r\n".encode())
            return True, "Command sent"
        except Exception as e:
            return False, f"Error sending command: {str(e)}"
    
    def read_response(self):
        """Read a single response from the device"""
        if not self.is_connected or not self.serial_connection:
            return None
        
        try:
            if self.serial_connection.in_waiting > 0:
                response = self.serial_connection.readline().decode().strip()
                return response
        except Exception as e:
            st.error(f"Error reading response: {str(e)}")
        return None
    
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
                        self.data_queue.put((timestamp, data))
                time.sleep(0.01)  # Small delay to prevent excessive CPU usage
            except Exception as e:
                if not self.stop_reading:
                    self.data_queue.put((datetime.now().strftime("%H:%M:%S.%f")[:-3], f"Error: {str(e)}"))
                break

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
        4. Use "Get Current Value" for single readings
        5. Use "Start Streaming" for continuous measurements
        
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