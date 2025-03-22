#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define LINE_LENGTH 785
#define TRAIN_LENGTH 60000
#define TEST_LENGTH 10000

// Гиперпараметры нейронной сети
#define NUM_LAYERS 2
#define HIDDEN_LAYER_SIZE 128
#define OUTPUT_LAYER_SIZE 10
#define LEARNING_RATE 0.1
#define EPOCHS 5                // Количество эпох обучения
#define PROGRESS_INTERVAL 1000   // Интервал отображения прогресса
#define BATCH_SIZE 1000            // Размер батча для обучения

#define ALPHA 0.01

typedef double (*ActivationFunc)(double);
typedef double (*ActivationDeriv)(double);

typedef struct {
    double *weights;
    double bias;
    int num_inputs;
    ActivationFunc activation;
    ActivationDeriv deriv;
    int is_softmax; // Флаг для обозначения нейронов в softmax-слое
} Neuron;

typedef struct {
    Neuron **neurons;
    int num_neurons;
    int num_inputs;
} Layer;

typedef struct {
    Layer **layers;
    int num_layers;
} NeuralNetwork;

double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }
double sigmoid_deriv(double x) { double s = sigmoid(x); return s * (1.0 - s); }
double relu(double x) { return x > 0 ? x : 0; }
double relu_deriv(double x) { return x > 0 ? 1.0 : 0.0; }
double lrelu(double x) { return x > 0 ? x : ALPHA*x; }
double lrelu_deriv(double x) { return x > 0 ? 1.0 : ALPHA; }
double identity(double x) { return x; }
double identity_deriv(double x) { return 1.0; }

void softmax_layer(double* inputs, double* outputs, int n) {
    double max_val = inputs[0];
    for (int i = 1; i < n; i++) {
        if (inputs[i] > max_val) max_val = inputs[i];
    }

    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        outputs[i] = exp(inputs[i] - max_val); // Вычитаем максимум для численной стабильности
        sum += outputs[i];
    }

    for (int i = 0; i < n; i++) {
        outputs[i] /= sum;
    }
}

Neuron* create_neuron(int num_inputs, ActivationFunc activation, ActivationDeriv deriv) {
    Neuron* neuron = (Neuron*)malloc(sizeof(Neuron));
    if (!neuron) { fprintf(stderr, "Error allocating neuron\n"); exit(1); }
    neuron->weights = (double*)malloc(num_inputs * sizeof(double));
    if (!neuron->weights) { fprintf(stderr, "Error allocating weights\n"); free(neuron); exit(1); }
    neuron->num_inputs = num_inputs;
    neuron->activation = activation;
    neuron->deriv = deriv;
    neuron->bias = 0.0;
    neuron->is_softmax = 0; // По умолчанию не softmax
    double range = sqrt(6.0 / (num_inputs + 10));
    for (int i = 0; i < num_inputs; i++) {
        neuron->weights[i] = ((double)rand() / RAND_MAX) * 2 * range - range;
    }
    return neuron;
}

void free_neuron(Neuron* neuron) {
    if (neuron) { free(neuron->weights); free(neuron); }
}

Layer* create_layer(int num_neurons, int num_inputs, ActivationFunc activation, ActivationDeriv deriv, int is_softmax) {
    Layer* layer = (Layer*)malloc(sizeof(Layer));
    if (!layer) { fprintf(stderr, "Error allocating layer\n"); exit(1); }
    layer->neurons = (Neuron**)malloc(num_neurons * sizeof(Neuron*));
    if (!layer->neurons) { fprintf(stderr, "Error allocating neurons\n"); free(layer); exit(1); }
    layer->num_neurons = num_neurons;
    layer->num_inputs = num_inputs;
    for (int i = 0; i < num_neurons; i++) {
        layer->neurons[i] = create_neuron(num_inputs, activation, deriv);
        layer->neurons[i]->is_softmax = is_softmax;
    }
    return layer;
}

void free_layer(Layer* layer) {
    if (layer) {
        for (int i = 0; i < layer->num_neurons; i++) { free_neuron(layer->neurons[i]); }
        free(layer->neurons);
        free(layer);
    }
}

NeuralNetwork* create_network(int* layer_sizes, int num_layers, ActivationFunc* activations, ActivationDeriv* derivs) {
    NeuralNetwork* network = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    if (!network) { fprintf(stderr, "Error allocating network\n"); exit(1); }
    network->layers = (Layer**)malloc(num_layers * sizeof(Layer*));
    if (!network->layers) { fprintf(stderr, "Error allocating layers\n"); free(network); exit(1); }
    network->num_layers = num_layers;
    for (int i = 0; i < num_layers; i++) {
        int num_inputs = (i == 0) ? LINE_LENGTH - 1 : layer_sizes[i - 1];
        int is_softmax = (i == num_layers - 1) ? 1 : 0; // Последний слой использует softmax
        network->layers[i] = create_layer(layer_sizes[i], num_inputs, activations[i], derivs[i], is_softmax);
    }
    return network;
}

void free_network(NeuralNetwork* network) {
    if (network) {
        for (int i = 0; i < network->num_layers; i++) { free_layer(network->layers[i]); }
        free(network->layers);
        free(network);
    }
}

double* layer_forward(Layer* layer, double* inputs, double* sums) {
    if (!inputs || !sums) { fprintf(stderr, "Invalid input to layer_forward\n"); return NULL; }
    double* outputs = (double*)malloc(layer->num_neurons * sizeof(double));
    if (!outputs) { fprintf(stderr, "Error allocating layer outputs\n"); return NULL; }

    // Вычисляем суммы для всех нейронов
    for (int i = 0; i < layer->num_neurons; i++) {
        sums[i] = layer->neurons[i]->bias;
        for (int j = 0; j < layer->num_inputs; j++) {
            sums[i] += layer->neurons[i]->weights[j] * inputs[j];
        }
    }

    // Если слой использует softmax, применяем его ко всему слою
    if (layer->neurons[0]->is_softmax) {
        softmax_layer(sums, outputs, layer->num_neurons);
    } else {
        // Для других слоев применяем активацию индивидуально к каждому нейрону
        for (int i = 0; i < layer->num_neurons; i++) {
            outputs[i] = layer->neurons[i]->activation(sums[i]);
        }
    }

    return outputs;
}

double* network_forward(NeuralNetwork* network, double* inputs, double*** sums, double*** outputs) {
    if (!network || !inputs || !sums || !outputs) { fprintf(stderr, "Invalid input to network_forward\n"); return NULL; }
    *sums = (double**)malloc(network->num_layers * sizeof(double*));
    *outputs = (double**)malloc(network->num_layers * sizeof(double*));
    if (!*sums || !*outputs) { fprintf(stderr, "Error allocating sums/outputs\n"); free(*sums); free(*outputs); return NULL; }
    double* current_inputs = inputs;
    for (int i = 0; i < network->num_layers; i++) {
        (*sums)[i] = (double*)malloc(network->layers[i]->num_neurons * sizeof(double));
        (*outputs)[i] = layer_forward(network->layers[i], current_inputs, (*sums)[i]);
        if (!(*outputs)[i]) { fprintf(stderr, "Layer %d forward failed\n", i); return NULL; }

        // Удаляем освобождение current_inputs, чтобы избежать двойного освобождения
        // Оно будет освобождено в main вместе с outputs
        current_inputs = (*outputs)[i];
    }

    // Создаем копию выходных данных последнего слоя
    int last_layer = network->num_layers - 1;
    int output_size = network->layers[last_layer]->num_neurons;
    double* output_copy = (double*)malloc(output_size * sizeof(double));
    if (!output_copy) { fprintf(stderr, "Error allocating output copy\n"); return NULL; }
    memcpy(output_copy, (*outputs)[last_layer], output_size * sizeof(double));

    return output_copy; // Возвращаем копию, которую нужно будет освободить отдельно
}

void network_backward(NeuralNetwork* network, double* inputs, double* targets, double learning_rate, double** sums, double** outputs) {
    if (!network || !inputs || !targets || !sums || !outputs) { fprintf(stderr, "Invalid input to network_backward\n"); return; }
    double** deltas = (double**)malloc(network->num_layers * sizeof(double*));
    if (!deltas) { fprintf(stderr, "Error allocating deltas\n"); return; }

    for (int l = network->num_layers - 1; l >= 0; l--) {
        deltas[l] = (double*)malloc(network->layers[l]->num_neurons * sizeof(double));
        if (!deltas[l]) { fprintf(stderr, "Error allocating deltas[%d]\n", l); return; }

        if (l == network->num_layers - 1) {
            // Для выходного слоя с softmax используем прямое вычисление градиента для кросс-энтропии
            if (network->layers[l]->neurons[0]->is_softmax) {
                for (int i = 0; i < network->layers[l]->num_neurons; i++) {
                    // Градиент для softmax с кросс-энтропией: output[i] - target[i]
                    deltas[l][i] = outputs[l][i] - targets[i];
                }
            } else {
                for (int i = 0; i < network->layers[l]->num_neurons; i++) {
                    deltas[l][i] = (outputs[l][i] - targets[i]) * network->layers[l]->neurons[i]->deriv(sums[l][i]);
                }
            }
        } else {
            for (int i = 0; i < network->layers[l]->num_neurons; i++) {
                double error = 0.0;
                for (int j = 0; j < network->layers[l + 1]->num_neurons; j++) {
                    error += deltas[l + 1][j] * network->layers[l + 1]->neurons[j]->weights[i];
                }
                deltas[l][i] = error * network->layers[l]->neurons[i]->deriv(sums[l][i]);
            }
        }

        double* prev_outputs = (l == 0) ? inputs : outputs[l - 1];
        for (int i = 0; i < network->layers[l]->num_neurons; i++) {
            for (int j = 0; j < network->layers[l]->num_inputs; j++) {
                network->layers[l]->neurons[i]->weights[j] -= learning_rate * deltas[l][i] * prev_outputs[j];
            }
            network->layers[l]->neurons[i]->bias -= learning_rate * deltas[l][i];
        }
    }

    for (int l = 0; l < network->num_layers; l++) { free(deltas[l]); }
    free(deltas);
}

// Новая функция для батчевого обновления весов
void batch_update(NeuralNetwork* network, double*** weight_gradients, double** bias_gradients, double learning_rate, int batch_size) {
    for (int l = 0; l < network->num_layers; l++) {
        for (int i = 0; i < network->layers[l]->num_neurons; i++) {
            // Обновляем bias
            double bias_update = 0.0;
            for (int b = 0; b < batch_size; b++) {
                bias_update += bias_gradients[l][i + b * network->layers[l]->num_neurons];
            }
            network->layers[l]->neurons[i]->bias -= learning_rate * bias_update / batch_size;

            // Обновляем веса
            for (int j = 0; j < network->layers[l]->num_inputs; j++) {
                double weight_update = 0.0;
                for (int b = 0; b < batch_size; b++) {
                    weight_update += weight_gradients[l][i + b * network->layers[l]->num_neurons][j];
                }
                network->layers[l]->neurons[i]->weights[j] -= learning_rate * weight_update / batch_size;
            }
        }
    }
}

// Функция для вычисления градиентов без обновления весов
void compute_gradients(NeuralNetwork* network, double* inputs, double* targets,
                     double** sums, double** outputs,
                     double*** weight_gradients, double** bias_gradients, int batch_idx) {
    double** deltas = (double**)malloc(network->num_layers * sizeof(double*));
    if (!deltas) { fprintf(stderr, "Error allocating deltas\n"); return; }

    for (int l = network->num_layers - 1; l >= 0; l--) {
        deltas[l] = (double*)malloc(network->layers[l]->num_neurons * sizeof(double));
        if (!deltas[l]) { fprintf(stderr, "Error allocating deltas[%d]\n", l); return; }

        if (l == network->num_layers - 1) {
            // Для выходного слоя с softmax используем прямое вычисление градиента
            if (network->layers[l]->neurons[0]->is_softmax) {
                for (int i = 0; i < network->layers[l]->num_neurons; i++) {
                    deltas[l][i] = outputs[l][i] - targets[i];
                }
            } else {
                for (int i = 0; i < network->layers[l]->num_neurons; i++) {
                    deltas[l][i] = (outputs[l][i] - targets[i]) * network->layers[l]->neurons[i]->deriv(sums[l][i]);
                }
            }
        } else {
            for (int i = 0; i < network->layers[l]->num_neurons; i++) {
                double error = 0.0;
                for (int j = 0; j < network->layers[l + 1]->num_neurons; j++) {
                    error += deltas[l + 1][j] * network->layers[l + 1]->neurons[j]->weights[i];
                }
                deltas[l][i] = error * network->layers[l]->neurons[i]->deriv(sums[l][i]);
            }
        }

        double* prev_outputs = (l == 0) ? inputs : outputs[l - 1];
        for (int i = 0; i < network->layers[l]->num_neurons; i++) {
            // Сохраняем градиент bias
            bias_gradients[l][i + batch_idx * network->layers[l]->num_neurons] = deltas[l][i];

            // Сохраняем градиенты весов
            for (int j = 0; j < network->layers[l]->num_inputs; j++) {
                weight_gradients[l][i + batch_idx * network->layers[l]->num_neurons][j] = deltas[l][i] * prev_outputs[j];
            }
        }
    }

    for (int l = 0; l < network->num_layers; l++) {
        free(deltas[l]);
    }
    free(deltas);
}

// Функция для оценки точности модели
double evaluate_accuracy(NeuralNetwork* network, int data[TEST_LENGTH][LINE_LENGTH], int num_samples) {
    int correct = 0;
    for (int i = 0; i < num_samples; i++) {
        double inputs[784];
        for (int j = 0; j < 784; j++) {
            inputs[j] = data[i][j + 1] / 255.0;
        }

        double** sums;
        double** outputs;
        double* output = network_forward(network, inputs, &sums, &outputs);
        if (!output) continue;

        // Находим индекс максимального значения (предсказанный класс)
        int predicted = 0;
        for (int j = 1; j < 10; j++) {
            if (output[j] > output[predicted]) {
                predicted = j;
            }
        }

        // Проверяем, совпадает ли предсказание с истинным классом
        if (predicted == data[i][0]) {
            correct++;
        }

        // Освобождаем память
        for (int j = 0; j < network->num_layers; j++) {
            free(sums[j]);
            free(outputs[j]);
        }
        free(sums);
        free(outputs);
        free(output);
    }

    return (double)correct / num_samples;
}

void importTrain(char*, int [TRAIN_LENGTH][LINE_LENGTH]);
void importTest(char*, int [TEST_LENGTH][LINE_LENGTH]);

int main() {
    int (*train)[LINE_LENGTH] = malloc(TRAIN_LENGTH * sizeof(*train));
    int (*test)[LINE_LENGTH] = malloc(TEST_LENGTH * sizeof(*test));
    if (!train || !test) { fprintf(stderr, "Memory allocation failed\n"); free(train); free(test); return 1; }

    importTrain("mnist/mnist_train.csv", train);
    importTest("mnist/mnist_test.csv", test);

    int layer_sizes[] = {HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE};
    ActivationFunc activations[] = {lrelu, identity}; // identity для выходного слоя с softmax
    ActivationDeriv derivs[] = {lrelu_deriv, identity_deriv};
    NeuralNetwork* network = create_network(layer_sizes, NUM_LAYERS, activations, derivs);

    printf("Начало обучения: %d эпох, %d примеров, размер батча: %d\n", EPOCHS, TRAIN_LENGTH, BATCH_SIZE);

    // Обучение по эпохам
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        printf("Эпоха %d/%d\n", epoch + 1, EPOCHS);

        // Перемешиваем обучающую выборку (опционально)
        // shuffle_data(train, TRAIN_LENGTH);

        // Подготовка структур для хранения градиентов
        double*** weight_gradients = (double***)malloc(network->num_layers * sizeof(double**));
        double** bias_gradients = (double**)malloc(network->num_layers * sizeof(double*));

        for (int l = 0; l < network->num_layers; l++) {
            bias_gradients[l] = (double*)calloc(network->layers[l]->num_neurons * BATCH_SIZE, sizeof(double));
            weight_gradients[l] = (double**)malloc(network->layers[l]->num_neurons * BATCH_SIZE * sizeof(double*));

            for (int i = 0; i < network->layers[l]->num_neurons * BATCH_SIZE; i++) {
                weight_gradients[l][i] = (double*)calloc(network->layers[l]->num_inputs, sizeof(double));
            }
        }

        // Проходим по всем обучающим примерам батчами
        int num_batches = (TRAIN_LENGTH + BATCH_SIZE - 1) / BATCH_SIZE; // Округление вверх

        for (int batch = 0; batch < num_batches; batch++) {
            int batch_start = batch * BATCH_SIZE;
            int batch_end = batch_start + BATCH_SIZE;
            if (batch_end > TRAIN_LENGTH) batch_end = TRAIN_LENGTH;
            int current_batch_size = batch_end - batch_start;

            // Обработка примеров в батче
            for (int i = 0; i < current_batch_size; i++) {
                int sample_idx = batch_start + i;

                // Подготовка входных данных
                double inputs[784];
                for (int j = 0; j < 784; j++) {
                    inputs[j] = train[sample_idx][j + 1] / 255.0;
                }

                // Подготовка целевых выходов (one-hot encoding)
                double targets[10] = {0};
                targets[train[sample_idx][0]] = 1.0;

                // Прямой проход
                double** sums;
                double** outputs;
                double* output = network_forward(network, inputs, &sums, &outputs);
                if (!output) {
                    fprintf(stderr, "Forward pass failed for sample %d\n", sample_idx);
                    continue;
                }

                // Вычисляем градиенты без обновления весов
                compute_gradients(network, inputs, targets, sums, outputs, weight_gradients, bias_gradients, i);

                // Освобождаем память
                for (int j = 0; j < network->num_layers; j++) {
                    free(sums[j]);
                    free(outputs[j]);
                }
                free(sums);
                free(outputs);
                free(output);
            }

            // Обновляем веса после обработки всего батча
            batch_update(network, weight_gradients, bias_gradients, LEARNING_RATE, current_batch_size);

            // Обнуляем градиенты для следующего батча
            for (int l = 0; l < network->num_layers; l++) {
                memset(bias_gradients[l], 0, network->layers[l]->num_neurons * BATCH_SIZE * sizeof(double));
                for (int i = 0; i < network->layers[l]->num_neurons * BATCH_SIZE; i++) {
                    memset(weight_gradients[l][i], 0, network->layers[l]->num_inputs * sizeof(double));
                }
            }

            // Показываем прогресс
            if ((batch + 1) % (PROGRESS_INTERVAL / BATCH_SIZE + 1) == 0) {
                printf("  Обработано: %d/%d примеров (%.2f%%)\n",
                       (batch + 1) * BATCH_SIZE > TRAIN_LENGTH ? TRAIN_LENGTH : (batch + 1) * BATCH_SIZE,
                       TRAIN_LENGTH,
                       ((batch + 1) * BATCH_SIZE > TRAIN_LENGTH ? TRAIN_LENGTH : (batch + 1) * BATCH_SIZE) * 100.0 / TRAIN_LENGTH);
            }
        }

        // Освобождаем память градиентов
        for (int l = 0; l < network->num_layers; l++) {
            for (int i = 0; i < network->layers[l]->num_neurons * BATCH_SIZE; i++) {
                free(weight_gradients[l][i]);
            }
            free(weight_gradients[l]);
            free(bias_gradients[l]);
        }
        free(weight_gradients);
        free(bias_gradients);

        // Оцениваем точность на тестовой выборке после каждой эпохи
        double accuracy = evaluate_accuracy(network, test, TEST_LENGTH);
        printf("  Точность на тестовых данных после эпохи %d: %.2f%%\n", epoch + 1, accuracy * 100);
    }

    printf("Обучение завершено\n");

    // Окончательная оценка на тестовой выборке
    double final_accuracy = evaluate_accuracy(network, test, TEST_LENGTH);
    printf("Итоговая точность на тестовых данных: %.2f%%\n", final_accuracy * 100);

    // Освобождаем память
    free_network(network);
    free(train);
    free(test);

    return 0;
}

void importTrain(char *filepath, int data[TRAIN_LENGTH][LINE_LENGTH]) {
    FILE *file = fopen(filepath, "r");
    if (!file) { fprintf(stderr, "Error opening %s\n", filepath); exit(1); }
    char buffer[4096];
    int counter = 0;
    fscanf(file, "%*[^\n]\n");
    while (fgets(buffer, sizeof(buffer), file) && counter < TRAIN_LENGTH) {
        char *token = strtok(buffer, ",");
        int index = 0;
        while (token && index < LINE_LENGTH) {
            data[counter][index++] = atoi(token);
            token = strtok(NULL, ",");
        }
        counter++;
    }
    fclose(file);
}

void importTest(char *filepath, int data[TEST_LENGTH][LINE_LENGTH]) {
    FILE *file = fopen(filepath, "r");
    if (!file) { fprintf(stderr, "Error opening %s\n", filepath); exit(1); }
    char buffer[4096];
    int counter = 0;
    fscanf(file, "%*[^\n]\n");
    while (fgets(buffer, sizeof(buffer), file) && counter < TEST_LENGTH) {
        char *token = strtok(buffer, ",");
        int index = 0;
        while (token && index < LINE_LENGTH) {
            data[counter][index++] = atoi(token);
            token = strtok(NULL, ",");
        }
        counter++;
    }
    fclose(file);
}
