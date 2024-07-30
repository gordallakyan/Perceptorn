def load_data(filepath):
    attributes = []
    labels = []
    with open(filepath, 'r') as file:
        for line in file:
            parts = line.strip().split('\t')
            if len(parts) == 5:
                attributes.append([float(parts[i].replace(',', '.')) for i in range(4)])
                labels.append(0 if parts[4] == 'Iris-setosa' else 1)
    return attributes, labels


def perceptron_train(attributes, labels, lr=0.1, epochs=100):
    weights = [0.0 for _ in range(len(attributes[0]))]
    bias = 0.0  # Bias

    for epoch in range(epochs):
        for input_vec, label in zip(attributes, labels):
            weighted_sum = sum(w * x for w, x in zip(weights, input_vec)) + bias
            prediction = 1 if weighted_sum >= 0.0 else 0
            error = label - prediction

            bias += lr * error



            for i in range(len(weights)):
                weights[i] += lr * error * input_vec[i]

        print(f'Epoch {epoch+1}, Weights: {weights}, Bias: {bias}')

    return weights, bias


def predict(input_vec, weights, bias):
    activation = sum(w * x for w, x in zip(weights, input_vec)) + bias
    return 1 if activation >= 0.0 else 0


def manual_input_and_classify(weights, bias):
    print("\nWprowadź wektor atrybutów oddzielony przecinkami (lub wpisz 'exit' aby zakończyć):")
    while True:
        user_input = input()
        if user_input.lower() == 'exit':
            break
        try:
            test_point = list(map(float, user_input.split(',')))
            predicted_label = predict(test_point, weights, bias)
            predicted_class = 'Iris-setosa' if predicted_label == 1 else 'Iris-nie-setosa'
            print(f"Przewidziana klasa dla {test_point} to: {predicted_class}")
        except ValueError:
            print("Niepoprawny format danych. Upewnij się, że wprowadzasz liczby oddzielone przecinkami.")




training_attributes, training_labels = load_data("/Users/Gor/Downloads/iris_training.txt")

weights, bias = perceptron_train(training_attributes, training_labels)

test_attributes, test_labels = load_data("/Users/Gor/Downloads/iris_test.txt")  # Dostosuj ścieżkę

correct_predictions = 0
for test_point, true_label in zip(test_attributes, test_labels):
    predicted_label = predict(test_point, weights, bias)
    if predicted_label == true_label:
        correct_predictions += 1

accuracy = correct_predictions / len(test_labels) * 100
print(f"Accuracy: {accuracy:.2f}%")

manual_input_and_classify(weights, bias)

