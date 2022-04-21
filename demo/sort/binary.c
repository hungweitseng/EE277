#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
    FILE *fp;
    int i=0,number_of_elements=100000000,current;
    double sum=0;
    if(argc > 2)
        number_of_elements = atoi(argv[2]);
    fp = fopen(argv[1],"r");
    for(i=0;i<number_of_elements;i++)
    {
        fread(&current, 1, sizeof(int), fp);
        sum+=current;
    }
    printf("Sum: %lf\n",sum);
    fclose(fp);
    return 0;
}