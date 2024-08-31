# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import random
import time
from IPython.display import clear_output

# Function to simulate normal traffic data
def generate_normal_traffic(num_samples=1000):
    normal_traffic = {
        "packet_size": np.random.normal(500, 50, num_samples),  # Mean size 500 bytes
        "num_requests": np.random.normal(20, 5, num_samples),  # Average 20 requests per second
        "source_ip": [f"192.168.1.{random.randint(2, 254)}" for _ in range(num_samples)],
        "label": 0  # Normal traffic label
    }
    return pd.DataFrame(normal_traffic)

# Function to simulate DDoS traffic data
def generate_ddos_traffic(num_samples=300):
    ddos_traffic = {
        "packet_size": np.random.normal(800, 100, num_samples),  # Larger packet size
        "num_requests": np.random.normal(100, 20, num_samples),  # High request rate
        "source_ip": [f"10.0.0.{random.randint(2, 254)}" for _ in range(num_samples)],
        "label": 1  # DDoS traffic label
    }
    return pd.DataFrame(ddos_traffic)

# Generate datasets
normal_traffic = generate_normal_traffic()
ddos_traffic = generate_ddos_traffic()

# Combine datasets
traffic_data = pd.concat([normal_traffic, ddos_traffic], ignore_index=True)

# Feature selection
features = ["packet_size", "num_requests"]

# Splitting the data
X = traffic_data[features]
y = traffic_data["label"]

# Splitting the data into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Isolation Forest model
model = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
model.fit(X_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Map predictions to the same labels (1 for anomaly, 0 for normal)
y_pred = [1 if x == -1 else 0 for x in y_pred]

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Real-time traffic monitoring and detection simulation
def simulate_real_time_monitoring(model, num_steps=50):
    blocked_ips = set()
    
    for i in range(num_steps):
        # Simulate a new batch of traffic
        new_normal = generate_normal_traffic(num_samples=10)
        new_ddos = generate_ddos_traffic(num_samples=2)
        new_traffic = pd.concat([new_normal, new_ddos], ignore_index=True)
        
        # Predict using the trained model
        X_new = new_traffic[features]
        y_new_pred = model.predict(X_new)
        y_new_pred = [1 if x == -1 else 0 for x in y_new_pred]
        
        # Visualization
        clear_output(wait=True)
        sns.countplot(x=y_new_pred)
        plt.title(f"Real-Time DDoS Detection - Step {i + 1}")
        plt.xlabel("Traffic Type (0: Normal, 1: DDoS)")
        plt.ylabel("Count")
        plt.show()
        
        # Mitigation strategy: IP blocking for detected DDoS traffic
        ddos_indices = np.where(np.array(y_new_pred) == 1)[0]
        for idx in ddos_indices:
            ip = new_traffic.iloc[idx]["source_ip"]
            if ip not in blocked_ips:
                blocked_ips.add(ip)
                print(f"Mitigating DDoS Attack: Blocking IP {ip}")
        
        # Pause for a short period to simulate real-time traffic
        time.sleep(1)

# Run real-time simulation
simulate_real_time_monitoring(model)
