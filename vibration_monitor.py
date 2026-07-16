import time
import joblib
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
from datetime import datetime

from mpu6050 import mpu6050
from scipy.stats import kurtosis, skew
from scipy.fft import rfft, rfftfreq
from scipy.signal import butter, filtfilt

# Configuration
SAMPLING_RATE = 100
DT = 1 / SAMPLING_RATE
WINDOW_SIZE = 90

LABELS = ["Ideal", "Cracking", "Offset_Pulley", "Wear"]

# Setup Directories
os.makedirs("logs/plots", exist_ok=True)
os.makedirs("logs/data", exist_ok=True)

# Load ML Model and Scaler
model = joblib.load("models/industrial_rf_model.pkl")
scaler = joblib.load("models/industrial_scaler.pkl")

# Initialize MPU6050 Sensor
sensor = mpu6050(0x68)

# Signal Processing Filters & Integration
def highpass(signal, cutoff=5, order=4):
    nyq = 0.5 * SAMPLING_RATE
    b, a = butter(order, cutoff / nyq, btype='high')
    return filtfilt(b, a, signal)

def integrate_velocity(acc):
    return np.cumsum(acc) * DT

def integrate_displacement(acc):
    vel = integrate_velocity(acc)
    return np.cumsum(vel) * DT

# Feature Extraction
def extract_features(signal):
    acc = np.array(signal)
    acc = acc - np.mean(acc)
    acc = highpass(acc)

    vel = integrate_velocity(acc)
    disp = integrate_displacement(acc)

    fft_vals = np.abs(rfft(acc))
    freqs = rfftfreq(len(acc), DT)

    dom_freq = freqs[np.argmax(fft_vals[1:]) + 1]
    fft_energy = np.sum(fft_vals**2) / len(fft_vals)

    return np.array([
        np.sqrt(np.mean(acc**2)),        # RMS acc
        kurtosis(acc),
        skew(acc),
        np.sqrt(np.mean(vel**2)),        # RMS velocity
        np.sqrt(np.mean(disp**2)),       # RMS displacement
        fft_energy,
        dom_freq
    ])

# Initialize CSV Logging
csv_file = open("logs/data/xyz_acceleration.csv", "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Time", "Ax", "Ay", "Az"])

# Real-Time Plotting Setup
plt.ion()

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 9))

line_acc, = ax1.plot([], [])
line_vel, = ax2.plot([], [])
line_disp, = ax3.plot([], [])

ax1.set_title("Acceleration vs Time (X-axis)")
ax2.set_title("Velocity vs Time (X-axis)")
ax3.set_title("Displacement vs Time (X-axis)")

for ax in [ax1, ax2, ax3]:
    ax.grid(True)

print("🟢 REAL-TIME INDUSTRIAL MONITORING STARTED")

# Data Buffers
bx, by, bz = [], [], []

# Main Real-Time Monitoring Loop
while True:
    start = time.time()

    data = sensor.get_accel_data()

    bx.append(data["x"])
    by.append(data["y"])
    bz.append(data["z"])

    # CSV Logging
    csv_writer.writerow([time.time(), data["x"], data["y"], data["z"]])

    if len(bx) >= WINDOW_SIZE:
        acc_x = np.array(bx[-WINDOW_SIZE:])
        acc_x = acc_x - np.mean(acc_x)
        acc_x = highpass(acc_x)

        t = np.arange(WINDOW_SIZE) * DT

        vel_x = integrate_velocity(acc_x)
        disp_x = integrate_displacement(acc_x)

        # ISO 10816 / 20816 Check
        vel_ms = vel_x * 9.81
        vel_rms_mm = np.sqrt(np.mean(vel_ms**2)) * 1000

        if vel_rms_mm < 1.8:
            iso_state = "ISO GOOD"
        elif vel_rms_mm < 4.5:
            iso_state = "ISO SATISFACTORY"
        elif vel_rms_mm < 11.2:
            iso_state = "ISO UNSATISFACTORY"
        else:
            iso_state = "ISO UNACCEPTABLE"

        # ML Predictions
        fx = extract_features(bx[-WINDOW_SIZE:])
        fy = extract_features(by[-WINDOW_SIZE:])
        fz = extract_features(bz[-WINDOW_SIZE:])

        features = np.hstack([fx, fy, fz]).reshape(1, -1)
        features = scaler.transform(features)

        pred = model.predict(features)[0]
        conf = model.predict_proba(features).max() * 100

        # Plotting Updates
        line_acc.set_data(t, acc_x)
        line_vel.set_data(t, vel_x)
        line_disp.set_data(t, disp_x)

        for ax in [ax1, ax2, ax3]:
            ax.set_xlim(0, t[-1])

        ax1.set_ylim(np.min(acc_x)*1.2, np.max(acc_x)*1.2)
        ax2.set_ylim(np.min(vel_x)*1.2, np.max(vel_x)*1.2)
        ax3.set_ylim(np.min(disp_x)*1.2, np.max(disp_x)*1.2)

        plt.pause(0.001)

        # Save Plots Periodically
        if int(time.time()) % 5 == 0:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fig.savefig(f"logs/plots/vibration_{ts}.png", dpi=300)

        # Groq Analysis Placeholder
        if GROQ_API_KEY:
               groq_analysis = "Groq analysis placeholder"
        else:
               groq_analysis = "Groq disabled"

        print(
            f"Condition: {LABELS[pred]} | "
            f"ISO: {iso_state} | "
            f"Vel RMS: {vel_rms_mm:.2f} mm/s | "
            f"Confidence: {conf:.2f}%"
        )

    elapsed = time.time() - start
    time.sleep(max(0, DT - elapsed))
