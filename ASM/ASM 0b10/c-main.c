#include <stdio.h>
//zadani 1
int count;
extern int my_strchr(char * str, char to_find);

//zadani 2
extern void str2int(char *buff, int *num);

//zadani 3
extern int not_bits(long* var, char* array, long size);

int main()
{
    //zadani 1
    char string[] = "Ahoj svete! EeEe";
    char to_find = 'e';
    int first = my_strchr(string, to_find);
    printf("first: %d | count: %d\n", first, count);
    
    //zadani 2
    int num;
	char buf[] = "55";
	str2int(buf, &num);
	printf("str2int: %d\n", num);

    //zadani 3
    long var = 0xf0f0f0;
    char array[] = {0, 1, 2, 10, 15};
    int size = 5;
    int oneCount = not_bits(&var, array, size);
    printf("not.bits: %lx | ones: %d\n", var, oneCount);
    
    return 0;
}