import socket
import numpy as np
import onnxruntime as ort
import json

# =========================
# Cargar modelo ONNX
# =========================

session = ort.InferenceSession("supervised_model.onnx")
input_name = session.get_inputs()[0].name

# =========================
# Cargar scalers
# =========================

with open("scalers.json", "r") as f:
    scaler_data = json.load(f)

# Al cargar los scalers, conviértelos de una vez
mean_X = np.array(scaler_data["mean_X"], dtype=np.float32)
scale_X = np.array(scaler_data["scale_X"], dtype=np.float32)

mean_y = np.array(scaler_data["mean_y"], dtype=np.float32)
scale_y = np.array(scaler_data["scale_y"], dtype=np.float32)

# =========================
# Configuración UDP
# =========================

UDP_IP = "127.0.0.1"
UDP_PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

print("Servidor de inferencia listo en puerto 5005...")

# =========================
# Loop principal
# =========================

while True:
    data, addr = sock.recvfrom(4096)

    obs = np.frombuffer(data, dtype=np.float32)
    obs = np.array(obs, dtype=np.float32)
    
    if obs.shape[0] != len(mean_X):
        print("Error: tamaño incorrecto recibido")
        continue

    obs = (obs - mean_X) / scale_X
    obs = obs.reshape(1, -1)

    output = session.run(None, {input_name: obs})[0]

    output = output[0]
    output = output * scale_y + mean_y

    sock.sendto(output.astype(np.float32).tobytes(), addr)