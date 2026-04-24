# sigmatest_serial.py

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
            print(f"Error reading response: {str(e)}")
        return None

    def start_streaming(self):
        """Start streaming measurements"""
        if self.is_connected:
            self.send_command("RC[streaming_value]=1")
            self.stop_reading = False
            self.reading_thread = threading.Thread(target=self._read_continuously)
            self.reading_thread.daemon = True
            self.reading_thread.start()

    def stop_streaming(self):
        """Stop streaming measurements"""
        if self.is_connected:
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
                time.sleep(0.01)
            except Exception as e:
                if not self.stop_reading:
                    self.data_queue.put((datetime.now().strftime("%H:%M:%S.%f")[:-3], f"Error: {str(e)}"))
                break
