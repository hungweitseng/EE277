#include <stdio.h>
#include <stdlib.h>


int main(int argc, char **argv)
{
    FILE *fp_ascii, *fp_binary;
    int number_of_elements = 100000000, i;
    int current;
    if(argc > 1)
        number_of_elements = atoi(argv[1]);
    fp_ascii = fopen("input.ascii","w");
    fp_binary = fopen("input.bin","wb");
    for(i = 0; i < number_of_elements; i++)
    {
        current = rand();
        fprintf(fp_ascii, "%d ", current);
        fwrite(&current, 1, sizeof(current), fp_binary);
    }
    fclose(fp_ascii);
    fclose(fp_binary);
    return 0;
}