import pandas as pd # handling data in tabular format
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# specify the file path
file_path = "/Users/georgianastan/Desktop/Assignments/GDS2771.tab"

# initialize lists to store extracted data
class_list, sample_id_list, cancer_status_list, additional_feature_list_1, additional_feature_list_2 = [], [], [], [], []

sample_id = None 

def store_parameters(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            # check if the line starts with no cancer/ cancer/ suspect cancer
            line_startswith = line.startswith("no cancer") or line.startswith("cancer") or line.startswith("suspect cancer")
            if line_startswith and not(line.split('\t')[1].strip() == "string"):
                sample_id = line.split('\t')[1].strip()
                # print(sample_id)
                sample_id_list.append(sample_id)
                class_list.append(line.split('\t')[0].strip())
                cancer_status_list.append(line.split('\t')[2].strip())  
                additional_feature_list_1.append(line.split('\t')[4].strip())
                additional_feature_list_2.append(line.split('\t')[5].strip())


store_parameters(file_path)
# print(sample_id_list[0:20])

# create a Pandas DataFrame - a tabular data structure, easy to work with structured data
data = pd.DataFrame({'class': class_list, 'sample_id': sample_id_list, 'cancer_status': cancer_status_list, 'additional_feature_1': additional_feature_list_1, 'additional_feature_2': additional_feature_list_2})

print(data)

total_instances = len(data)
print(f'Total instances in original dataset: {total_instances}')

# separate the features (X) and the target variable (y)
# "class" is my target variable
X = data[['sample_id', 'cancer_status', 'additional_feature_1', 'additional_feature_2']]
Y = data['class']

# split the data into training and testing sets
# 20% of the data used for testing, and 80% for training
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# concatenate the train and test sets to ensure consistent one-hot encoding
# pd.get_dummies used to one-hot encode categorical variables, converting them into a format that can be used for modeling
X_concat = pd.concat([X_train, X_test])

# one-hot encode categorical variables
X_concat_encoded = pd.get_dummies(X_concat)

# split back into train and test sets
X_train_encoded = X_concat_encoded[:len(X_train)]
X_test_encoded = X_concat_encoded[len(X_train):]

# initialize the RandomForestClassifier
# ensemble learning method that constructs a multitude of decision trees and merges them together to get a more accurate and stable prediction
clf = RandomForestClassifier(random_state=42)

# train the model
clf.fit(X_train_encoded, Y_train)

# make predictions on the test set
Y_pred = clf.predict(X_test_encoded)

# evaluate the model
accuracy = accuracy_score(Y_test, Y_pred)
print(f'Accuracy: {accuracy:.2f}')

# display additional evaluation metrics
print('Classification Report:')
print(classification_report(Y_test, Y_pred))

# display confusion matrix
print('Confusion Matrix:')
print(confusion_matrix(Y_test, Y_pred))
