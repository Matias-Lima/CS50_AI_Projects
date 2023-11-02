import csv
import pandas as pd
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    # Read the CSV file into a DataFrame
    df = pd.read_csv(f"{filename}")

    # Create empty lists for evidence and labels
    evidence = []
    labels = []

    # Define a mapping for months
    month_mapping = {
        'Jan': 0, 'Feb': 1, 'Mar': 2, 'Apr': 3, 'May': 4, 'June': 5,
        'Jul': 6, 'Aug': 7, 'Sep': 8, 'Oct': 9, 'Nov': 10, 'Dec': 11
    }

    # Define a mapping for VisitorType
    visitor_mapping = {'Returning_Visitor': 1, 'New_Visitor': 0, 'Other': 0}

    # Process each row in the DataFrame
    for index, row in df.iterrows():
        # Extract relevant columns and convert data types
        evidence_row = [
            int(row['Administrative']),
            float(row['Administrative_Duration']),
            int(row['Informational']),
            float(row['Informational_Duration']),
            int(row['ProductRelated']),
            float(row['ProductRelated_Duration']),
            float(row['BounceRates']),
            float(row['ExitRates']),
            float(row['PageValues']),
            float(row['SpecialDay']),
            int(month_mapping[row['Month']]),
            int(row['OperatingSystems']),
            int(row['Browser']),
            int(row['Region']),
            int(row['TrafficType']),
            visitor_mapping[row['VisitorType']],
            int(row['Weekend'])
        ]

        # Append the evidence and label to their respective lists
        evidence.append(evidence_row)
        labels.append(int(row['Revenue']))

    # Now, evidence and labels lists contain the processed data in the desired format
    # evidence[0] contains the evidence for the first user, and labels[0] contains the label for the first user

    # Example of accessing the first evidence and label
    print("First evidence:", evidence[0])
    print("First label:", labels[0])

    return evidence, labels


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """

    # Create and train the KNeighborsClassifier

    knn_classifier = KNeighborsClassifier(n_neighbors=1)  # You can adjust n_neighbors as needed

    knn_classifier.fit(evidence, labels)

    return knn_classifier

def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    true_labels = labels
    predicted_labels = predictions


    # Calculate the number of true positives, true negatives, false positives, and false negatives
    true_positives = sum((true == 1 and pred == 1) for true, pred in zip(true_labels, predicted_labels))
    true_negatives = sum((true == 0 and pred == 0) for true, pred in zip(true_labels, predicted_labels))
    false_positives = sum((true == 0 and pred == 1) for true, pred in zip(true_labels, predicted_labels))
    false_negatives = sum((true == 1 and pred == 0) for true, pred in zip(true_labels, predicted_labels))

    # Calculate sensitivity (true positive rate)
    sensitivity = true_positives / (true_positives + false_negatives) if (
                                                                                     true_positives + false_negatives) != 0 else 0.0

    # Calculate specificity (true negative rate)
    specificity = true_negatives / (true_negatives + false_positives) if (
                                                                                     true_negatives + false_positives) != 0 else 0.0

    return sensitivity, specificity





if __name__ == "__main__":
    main()



