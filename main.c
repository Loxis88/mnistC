#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DATA_LENGTH 785  // Количество элементов в строке (1 метка + 784 пикселя)

int main() {
    FILE *file = fopen("mnist/mnist_train.csv", "r");
    if (file == NULL) {
        printf("Error opening file\n");
        return 1;
    }

    char buffer[4096];  // Буфер для чтения строки
    int data[DATA_LENGTH];  // Массив для хранения чисел

    // Пропуск первой строки (заголовок)
    fscanf(file, "%*[^\n]\n"); // Читает и игнорирует всю строку

    // Чтение строки данных
    if (fgets(buffer, sizeof(buffer), file)) {
        // Разделение строки на числа
        char *token = strtok(buffer, ",");  // Первый вызов strtok
        int index = 0;

        while (token != NULL && index < DATA_LENGTH) {
            // Преобразование строки в число
            data[index++] = atoi(token);
            token = strtok(NULL, ",");  // Следующий вызов strtok
        }

        // Вывод массива для проверки
        printf("Label: %d\n", data[0]);  // Метка (первое число в строке)
        printf("First 10 pixels: ");
        for (int i = 1; i < 11; i++) {
            printf("%d ", data[i]);  // Первые 10 пикселей
        }
        printf("\n");
    } else {
        printf("Error reading data line\n");
    }

    fclose(file);
    return 0;
}
