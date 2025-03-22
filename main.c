#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define LINE_LENGTH 785
#define TRAIN_LENGTH 60000
#define TEST_LENGTH 10000

typedef double (*ActivationFunc)(double);
typedef double (*ActivationDeriv)(double);

typedef struct {
    double *weights;
    double bias;
    int num_inputs;
    ActivationFunc activation;
    ActivationDeriv deriv;
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

Neuron* create_neuron(int num_inputs, ActivationFunc activation, ActivationDeriv deriv) {
    Neuron* neuron = (Neuron*)malloc(sizeof(Neuron));
    if (!neuron) { fprintf(stderr, "Error allocating neuron\n"); exit(1); }
    neuron->weights = (double*)malloc(num_inputs * sizeof(double));
    if (!neuron->weights) { fprintf(stderr, "Error allocating weights\n"); free(neuron); exit(1); }
    neuron->num_inputs = num_inputs;
    neuron->activation = activation;
    neuron->deriv = deriv;
    neuron->bias = 0.0;
    for (int i = 0; i < num_inputs; i++) {
        neuron->weights[i] = ((double)rand() / RAND_MAX) * 2 - 1;
    }
    return neuron;
}

void free_neuron(Neuron* neuron) {
    if (neuron) { free(neuron->weights); free(neuron); }
}

Layer* create_layer(int num_neurons, int num_inputs, ActivationFunc activation, ActivationDeriv deriv) {
    Layer* layer = (Layer*)malloc(sizeof(Layer));
    if (!layer) { fprintf(stderr, "Error allocating layer\n"); exit(1); }
    layer->neurons = (Neuron**)malloc(num_neurons * sizeof(Neuron*));
    if (!layer->neurons) { fprintf(stderr, "Error allocating neurons\n"); free(layer); exit(1); }
    layer->num_neurons = num_neurons;
    layer->num_inputs = num_inputs;
    for (int i = 0; i < num_neurons; i++) {
        layer->neurons[i] = create_neuron(num_inputs, activation, deriv);
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
        network->layers[i] = create_layer(layer_sizes[i], num_inputs, activations[i], derivs[i]);
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
    for (int i = 0; i < layer->num_neurons; i++) {
        sums[i] = layer->neurons[i]->bias;
        for (int j = 0; j < layer->num_inputs; j++) {
            sums[i] += layer->neurons[i]->weights[j] * inputs[j];
        }
        outputs[i] = layer->neurons[i]->activation(sums[i]);
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
        if (i > 0) free(current_inputs); // Освобождаем промежуточные данные
        current_inputs = (*outputs)[i];
    }
    return current_inputs; // Это указатель на outputs[num_layers - 1]
}

void network_backward(NeuralNetwork* network, double* inputs, double* targets, double learning_rate, double** sums, double** outputs) {
    if (!network || !inputs || !targets || !sums || !outputs) { fprintf(stderr, "Invalid input to network_backward\n"); return; }
    double** deltas = (double**)malloc(network->num_layers * sizeof(double*));
    if (!deltas) { fprintf(stderr, "Error allocating deltas\n"); return; }
    for (int l = network->num_layers - 1; l >= 0; l--) {
        deltas[l] = (double*)malloc(network->layers[l]->num_neurons * sizeof(double));
        if (!deltas[l]) { fprintf(stderr, "Error allocating deltas[%d]\n", l); return; }
        if (l == network->num_layers - 1) {
            for (int i = 0; i < network->layers[l]->num_neurons; i++) {
                deltas[l][i] = (outputs[l][i] - targets[i]) * network->layers[l]->neurons[i]->deriv(sums[l][i]);
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

void importTrain(char*, int [TRAIN_LENGTH][LINE_LENGTH]);
void importTest(char*, int [TEST_LENGTH][LINE_LENGTH]);

int main() {
    int (*train)[LINE_LENGTH] = malloc(TRAIN_LENGTH * sizeof(*train));
    int (*test)[LINE_LENGTH] = malloc(TEST_LENGTH * sizeof(*test));
    if (!train || !test) { fprintf(stderr, "Memory allocation failed\n"); free(train); free(test); return 1; }

    importTrain("mnist/mnist_train.csv", train);
    importTest("mnist/mnist_test.csv", test);

    int layer_sizes[] = {16, 10};
    ActivationFunc activations[] = {relu, sigmoid};
    ActivationDeriv derivs[] = {relu_deriv, sigmoid_deriv};
    NeuralNetwork* network = create_network(layer_sizes, 2, activations, derivs);

    double inputs[784];
    for (int i = 0; i < 784; i++) {
        inputs[i] = train[0][i + 1] / 255.0;
    }
    double targets[10] = {0};
    targets[train[0][0]] = 1.0;

    double** sums;
    double** outputs;
    double* output = network_forward(network, inputs, &sums, &outputs);
    if (!output) { fprintf(stderr, "Initial forward pass failed\n"); free_network(network); free(train); free(test); return 1; }
    printf("Before training: ");
    for (int i = 0; i < 10; i++) { printf("%f ", output[i]); }
    printf("\n");

    double learning_rate = 0.1;
    network_backward(network, inputs, targets, learning_rate, sums, outputs);

    double** sums2;
    double** outputs2;
    double* output2 = network_forward(network, inputs, &sums2, &outputs2);
    if (!output2) { fprintf(stderr, "Second forward pass failed\n"); free_network(network); free(train); free(test); return 1; }
    printf("After training: ");
    for (int i = 0; i < 10; i++) { printf("%f ", output2[i]); }
    printf("\n");

    // Free memory properly
    for (int i = 0; i < network->num_layers; i++) {
        free(sums[i]);
        free(outputs[i]);
        free(sums2[i]);
        free(outputs2[i]);
    }
    free(sums);
    free(outputs);
    free(sums2);
    free(outputs2);
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
