#include <stdio.h>

// zadani 1
void fill_pyramid_numbers(long * numbers, int size);//1
// zadani 2
long multiples(long * numbers, int size, long factor);
// zadani 3
int factorial(int * numbers, int size);
// zadani 4
void change_array_by_avg(long* array, int N);//4

int main(void)
{
    // zadani 1
    printf("\nzadani 1:\n");
    int size = 10;
    long array[10];

    fill_pyramid_numbers(array, size);
    for (int i = 0; i < size; i++)
    {
        printf("%ld ", array[i]);
    }
    printf("\n\n");

    // zadani 2
    printf("zadani 2:\n");
    long numbers[] = {5, 25, 25, 104};
    long answer2 = multiples(numbers, 4, 5);
    for (int i = 0; i < 4; i++)
    {
        printf("%ld ", numbers[i]);
    }
    printf("\npretekly:%ld\n\n", answer2);

    // zadani 3
    /*printf("zadani 3:\n");
    int size2 = 3;
    int numbers2[size2] = {3, 4, 5};
    int overflow_count = factorial(numbers2, size2);
    for (int i = 0; i < size2; i++)
    {
        printf("%d ", numbers2[i]);
    }
    printf("\n\n");*/

    // zadani 4
    printf("zadani 4:\n");
    int arr_len = 9;
    long arr[9] = {1,2,3,4,5,6,7,8,9};
    change_array_by_avg(arr, arr_len);
    for (int i = 0; i < arr_len; i++)
    {
        printf("%ld\t", arr[i]);
    }
    printf("\n");

    return 0;
}