import csv
import math
import os
def load_csv(filename):
    data = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            data.append([float(x) for x in row[:-1]] + [row[-1]])
    return data
def separate_by_class(data):
    separated = {}
    for row in data:
        label = row[-1]
        if label not in separated:
            separated[label] = []
        separated[label].append(row[:-1])
    return separated
def mean(nums):
    return sum(nums) / len(nums)
def stdev(nums):
    avg = mean(nums)
    variance = sum((x - avg) ** 2 for x in nums) / (len(nums) - 1)
    return math.sqrt(variance)
def summarize_dataset(rows):
    summaries = [(mean(col), stdev(col)) for col in zip(*rows)]
    return summaries
def summarize_by_class(data):
    separated = separate_by_class(data)
    summaries = {}
    for label, rows in separated.items():
        summaries[label] = summarize_dataset(rows)
    return summaries
def gaussian_probability(x, mean_val, stdev_val):
    exponent = math.exp(-((x - mean_val) ** 2) / (2 * stdev_val ** 2))
    return (1 / (math.sqrt(2 * math.pi) * stdev_val)) * exponent
def calculate_class_probabilities(summaries, row, class_priors):
    probabilities = {}
    for label, class_summaries in summaries.items():
        probabilities[label] = class_priors[label]
        for i in range(len(class_summaries)):
            mean_val, stdev_val = class_summaries[i]
            probabilities[label] *= gaussian_probability(row[i], mean_val, stdev_val)
    return probabilities
def get_class_priors(data):
    separated = separate_by_class(data)
    total = len(data)
    priors = {}
    for label, rows in separated.items():
        priors[label] = len(rows) / total
    return priors
def predict(summaries, row, class_priors):
    probabilities = calculate_class_probabilities(summaries, row, class_priors)
    return max(probabilities, key=probabilities.get)
def train_test_split(data, test_ratio=0.2):
    split_index = int(len(data) * (1 - test_ratio))
    return data[:split_index], data[split_index:]
def accuracy(predictions, actuals):
    correct = sum(1 for p, a in zip(predictions, actuals) if p == a)
    return correct / len(actuals)
if __name__ == "__main__":
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset.csv")
    data = load_csv(csv_path)
    train_data, test_data = train_test_split(data)
    summaries = summarize_by_class(train_data)
    priors = get_class_priors(train_data)
    predictions = []
    actuals = []
    for row in test_data:
        features = row[:-1]
        actual_label = row[-1]
        predicted_label = predict(summaries, features, priors)
        predictions.append(predicted_label)
        actuals.append(actual_label)
    print("Predictions:", predictions)
    print("Actuals:", actuals)
    print("Accuracy:", accuracy(predictions, actuals))