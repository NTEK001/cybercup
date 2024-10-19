import pandas as pd
from sklearn.preprocessing import StandardScaler  # or MinMaxScaler
import joblib

# Load your dataset
data = pd.read_csv('Input_Data/vmCloud_data.csv')

# Select the features to normalize
features = ["cpu_usage", "memory_usage", "network_traffic",
            "power_consumption", "num_executed_instructions",
            "execution_time", "energy_efficiency"]

# Initialize and fit the StandardScaler
scaler = StandardScaler()  # or MinMaxScaler()
data[features] = scaler.fit_transform(data[features])

# Save the scaler for future use
joblib.dump(scaler, 'scaler.pkl')

# Save the normalized data to a new CSV file
normalized_data_path = './Output_Data/normalizedvmCloud_data.csv'
data.to_csv(normalized_data_path, index=False)
print(f"Normalized data saved to: {normalized_data_path}")

# Display the first few rows of the normalized data
print(data.head())
