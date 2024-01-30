import netCDF4
from netCDF4 import Dataset as netCDFDataset
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

print("loading data")
labels = netCDFDataset("./data/labels/GTD_1979-2019_JJAextd_8.nc", mode="r").variables[
    "blocking"
][:]
data = netCDFDataset(
    "./data/geopotential_height_500hPa_era5_6hourly_z0001_daymean_final.nc",
    mode="r",
).variables["z_0001"][:, :, :, 550:950]

# Split the dataset into training and validation sets
train_data, val_data, train_labels, val_labels = train_test_split(
    data, labels, test_size=0.1
)

# Reshape the input data to be flattened
train_data_flat = train_data.reshape(
    len(train_data), int(train_data.size / len(train_data))
)
val_data_flat = val_data.reshape(len(val_data), int(val_data.size / len(val_data)))

# Create a Support Vector Machine (SVM) model
svm_model = svm.SVC(kernel="linear", verbose=1)

print("training model")
# Train the SVM model
svm_model.fit(train_data_flat, train_labels)

print("validating model")
# Make predictions on the validation set
val_predictions = svm_model.predict(val_data_flat)

# Calculate accuracy on the validation set
accuracy = accuracy_score(val_labels, val_predictions)
print(f"Validation Accuracy: {accuracy * 100:.2f}%")
