import netCDF4
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from netCDF4 import Dataset as netCDFDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split

print("loading data")
labels = netCDFDataset("./data/labels/GTD_1979-2019_JJAextd_8.nc", mode="r").variables[
    "blocking"
][:500]
data = netCDFDataset(
    "./data/geopotential_height_500hPa_era5_6hourly_z0001_daymean_final.nc",
    mode="r",
).variables["z_0001"][:500, :1, :5, 550:950]

# Split the dataset into training and validation sets
train_data, val_data, train_labels, val_labels = train_test_split(
    data, labels, test_size=0.1
)

# Reshape the input data to be flattened
train_data_flat = train_data.reshape(
    len(train_data), int(train_data.size / len(train_data))
)
val_data_flat = val_data.reshape(len(val_data), int(val_data.size / len(val_data)))

rf = RandomForestClassifier(verbose=1)

param_grid = {
    "n_estimators": [100],
    "max_depth": [None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4, 7],
    "max_features": ["1.0", "sqrt", "log2"],
    "bootstrap": [True],
}

rf_classifier = RandomForestClassifier()

grid_search = GridSearchCV(
    estimator=rf_classifier, param_grid=param_grid, cv=5, scoring="accuracy", n_jobs=-1
)

print("training model")
# Train the SVM model
grid_search.fit(train_data_flat, train_labels)

print("validating model")
# Make predictions on the validation set
val_predictions = grid_search.predict(val_data_flat)

# Calculate accuracy on the validation set
accuracy = accuracy_score(val_labels, val_predictions)
print(f"Validation Accuracy: {accuracy * 100:.2f}%")
