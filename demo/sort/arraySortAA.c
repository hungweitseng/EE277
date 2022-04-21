#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "algorithms.h"

int cmpfunc (const void * a, const void * b) {
   return ( *(int*)a - *(int*)b );
}

int main(int argc, char **argv)
{
    FILE *fp;
    int i=0,number_of_elements=100000000,current;
    double sum=0;
    double total_time;
    struct timeval time_start, time_end;
    int *values, *out_values;
    if(argc > 2)
        number_of_elements = atoi(argv[2]);
    fp = fopen(argv[1],"r");
    values = (int *)malloc(sizeof(int)*number_of_elements);
    out_values = (int *)malloc(sizeof(int)*number_of_elements);
    gettimeofday(&time_start, NULL);
    for(i=0;i<number_of_elements;i++)
    {
        fscanf(fp, "%d", &values[i]);
//        sum+=current;
    }
    gettimeofday(&time_end, NULL);
    total_time = ((time_end.tv_sec * 1000000 + time_end.tv_usec) - (time_start.tv_sec * 1000000 + time_start.tv_usec));
    fprintf(stderr,"Scan Latency: %.0lf us, Goodput (Integers Per Second): %lf\n",total_time, number_of_elements*1000000/total_time);
    gettimeofday(&time_start, NULL);
    #ifdef AASORT
    aasort_naive_parallel(number_of_elements, values, out_values);
    #else
    qsort(values, number_of_elements, sizeof(int), cmpfunc);
    #endif
//    printf("Sum: %lf\n",sum);
    gettimeofday(&time_end, NULL);
    total_time = ((time_end.tv_sec * 1000000 + time_end.tv_usec) - (time_start.tv_sec * 1000000 + time_start.tv_usec));
    fprintf(stderr,"Sort Latency: %.0lf us, Goodput (Integers Per Second): %lf\n",total_time, number_of_elements*1000000/total_time);
    #ifdef AASORT
    fprintf(stderr, "Samping points: values[0: %d, mid: %d, end: %d]\n", out_values[0], out_values[number_of_elements>1], out_values[number_of_elements-1]);
    #else
    fprintf(stderr, "Samping points: values[0: %d, mid: %d, end: %d]\n", values[0], values[number_of_elements>1], values[number_of_elements-1]);
    #endif
    fclose(fp);
    return 0;
}