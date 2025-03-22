#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define LINE_LENGTH 785 // Количество элементов в строке (1 метка + 784 пикселя)
#define TRAIN_LENGTH 60000
#define TEST_LENGTH 10000


void importTrain(char*, int [TRAIN_LENGTH][LINE_LENGTH]);
void importTest(char*, int [TEST_LENGTH][LINE_LENGTH]);

// Тип для функции активации: принимает double и возвращает double
typedef double (*ActivationFunc)(double);

// Структура нейрона
typedef struct {
    double *weights;      // Динамический массив весов
    double bias;          // Смещение
    int num_inputs;       // Количество входов
    ActivationFunc activation; // Указатель на функцию активации
} Neuron;

// Структура слоя
typedef struct {
    Neuron **neurons;     // Массив указателей на нейроны
    int num_neurons;      // Количество нейронов в слое
    int num_inputs;       // Количество входов для каждого нейрона
} Layer;

// Структура нейронной сети
typedef struct {
    Layer **layers;       // Массив указателей на слои
    int num_layers;       // Количество слоев
} NeuralNetwork;

//функции активации
double sigmoid(double x) {return 1.0 / (1.0 + exp(-x));}
double relu(double x) {return x > 0 ? x : 0;}
double LRelu(double x) {return x > 0 ? x : x * 0.01;}
double linear(double x) {return x;}

Neuron* create_neuron(int num_inputs, ActivationFunc activation) {
    Neuron* neuron = (Neuron*)malloc(sizeof(Neuron));
    if (neuron == NULL) {
        printf("Error allocating memory for neuron\n");
        exit(1);
    }

    neuron->weights = (double*)malloc(num_inputs * sizeof(double));
    if (neuron->weights == NULL) {
        printf("Error allocating memory for weights\n");
        free(neuron);
        exit(1);
    }

    neuron->num_inputs = num_inputs;
    neuron->activation = activation;
    neuron->bias = 0.0; // Инициализация смещения нулем

    // Инициализация весов случайными значениями (например, от -1 до 1)
    for (int i = 0; i < num_inputs; i++) {
        neuron->weights[i] = ((double)rand() / RAND_MAX) * 2 - 1;
    }

    return neuron;
}

// Функция для освобождения памяти нейрона
void free_neuron(Neuron* neuron) {
    if (neuron != NULL) {
        free(neuron->weights);
        free(neuron);
    }
}

// Функция для вычисления выхода нейрона
double neuron_output(Neuron* neuron, double* inputs) {
    if (neuron->num_inputs != sizeof(inputs) / sizeof(double)) {
        printf("Mismatch in number of inputs\n");
        return 0.0;
    }

    double sum = neuron->bias;
    for (int i = 0; i < neuron->num_inputs; i++) {
        sum += neuron->weights[i] * inputs[i];
    }
    return neuron->activation(sum);
}


int main() {
    int (*train)[LINE_LENGTH] = malloc(TRAIN_LENGTH * sizeof(*train));
    int (*test)[LINE_LENGTH] = malloc(TEST_LENGTH * sizeof(*test));

    importTrain("mnist/mnist_train.csv", train); importTest("mnist/mnist_test.csv", test); // printf("%d", train[0][0]); printf("%d", test[0][0]);



    free(train); free(test);
    return 0;
};

void importTest(char *filepath, int data[TEST_LENGTH][LINE_LENGTH]) {
    FILE *file = fopen(filepath, "r");
    if (file == NULL) {
        printf("Error opening file\n");
        exit(1);
    }

    char buffer[4096];
    int counter = 0;

    fscanf(file, "%*[^\n]\n");

    while (fgets(buffer, sizeof(buffer), file) && counter < TEST_LENGTH) {
        char *token = strtok(buffer, ",");
        int index = 0;

        while (token != NULL && index < LINE_LENGTH) {
            data[counter][index++] = atoi(token);
            token = strtok(NULL, ",");
        }
        counter++;
    }
    fclose(file);
};


void importTrain(char *filepath, int data[TRAIN_LENGTH][LINE_LENGTH]) {
    FILE *file = fopen(filepath, "r");
    if (file == NULL) {
        printf("Error opening file\n");
        exit(1);
    }

    char buffer[4096];
    int counter = 0;

    fscanf(file, "%*[^\n]\n"); // Читает и игнорирует всю строку

    while (fgets(buffer, sizeof(buffer), file) && counter < TRAIN_LENGTH) {
        char *token = strtok(buffer, ",");
        int index = 0;

        while (token != NULL && index < LINE_LENGTH) {
            data[counter][index++] = atoi(token);
            token = strtok(NULL, ",");
        }
        counter++;
    }
    fclose(file);
};
