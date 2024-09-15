import cereal.messaging as messaging
import serial
import time
from cereal import custom  # This will import the message from custom.capnp

def main():
    pm = messaging.PubMaster(['advisorySpeed'])
    
    # Open the USB serial port where the tablet is sending advisory speed
    ser = serial.Serial('/dev/ttyGS0', baudrate=9600, timeout=1)
    
    while True:
        try:
            # Read data from the serial port
            line = ser.readline().decode('utf-8').strip()
            if line:
                advisory_speed = float(line)  # Convert string to float
                
                # Create and send the message using custom.AdvisorySpeed
                dat = messaging.new_message('advisorySpeed')
                dat.advisorySpeed.speed = advisory_speed
                pm.send('advisorySpeed', dat)
            
            time.sleep(0.1)  # Adjust delay as needed
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(1)

if __name__ == "__main__":
    main()