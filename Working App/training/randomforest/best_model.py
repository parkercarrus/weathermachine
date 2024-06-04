import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import time
from scipy.stats import binomtest


starttime = time.time()
# Load the processed dataset
data = pd.read_csv('/Users/parkercarrus/Desktop/Weather/Random Forest/Data/Rain Target/rf2_data.csv')
data = data.sample(n=100000, random_state=42)

# Select features and target
#X = data[['temp', 'pressure', 'humidity', 'time_sin', 'time_cos', 'year_day_sin', 'year_day_cos']]

X = data[['time_sin', 'time_cos', 'year_day_sin','year_day_cos', 'temp', 'pressure', 'humidity', 'temp_1h_ago', 'pressure_1h_ago', 'humidity_1h_ago', 'temp_2h_ago', 'pressure_2h_ago', 'humidity_2h_ago', 'temp_3h_ago', 'pressure_3h_ago', 'humidity_3h_ago']]
y = data['rain']


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=300, 
                                                       max_depth=None, 
                                                       min_samples_split=2, 
                                                       min_samples_leaf=1, 
                                                       random_state=42, n_jobs=-1)
# Train the model
print("Training the model...")
rf_classifier.fit(X_train, y_train)

# Make predictions on the test set
print("Making predictions...")
predictions = rf_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
correct_predictions = accuracy * len(y_test)  # Total correct predictions
total_predictions = len(y_test)  # Total predictions made

# Assuming the majority class proportion (proportion of the most common class in y_test)
majority_class_proportion = max(y_test.value_counts()) / len(y_test)

# Calculate the p-value
p_value = binomtest(int(correct_predictions), n=total_predictions, p=majority_class_proportion, alternative='greater')

print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, predictions))
print(f'\nCompleted in {time.time() - starttime} seconds')

import joblib

joblib.dump(rf_classifier, 'best_model.pkl')