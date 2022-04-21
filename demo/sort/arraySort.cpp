#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <ctime>
#include <iostream>
#include <climits>
#include <sys/time.h>

int main(int argc, char **argv)
{
    // input data
    unsigned array_size = 100000000;
    FILE *fp;
    int *data, *sorted;
    int option = atoi(argv[1]);
    double total_time;
    struct timeval time_start, time_end;
    if(argc > 2)
        array_size = atoi(argv[2]);
    long long sum = 0;
    fp = fopen(argv[1],"r");
    data = (int *)malloc(sizeof(int)*array_size);
    sorted = (int *)malloc(sizeof(int)*array_size);

    gettimeofday(&time_start, NULL);
    for(int i=0;i<array_size;i++)
    {
        fscanf(fp, "%d", &data[i]);
    }
    gettimeofday(&time_end, NULL);
    total_time = ((time_end.tv_sec * 1000000 + time_end.tv_usec) - (time_start.tv_sec * 1000000 + time_start.tv_usec));
    fprintf(stderr,"Scan Latency: %.0lf us, Goodput (Integers Per Second): %lf\n",total_time, array_size*1000000/total_time);

    gettimeofday(&time_start, NULL);
    std::sort(data, data + array_size);
    gettimeofday(&time_end, NULL);
    total_time = ((time_end.tv_sec * 1000000 + time_end.tv_usec) - (time_start.tv_sec * 1000000 + time_start.tv_usec));
    fprintf(stderr,"Sort Latency: %.0lf us, Goodput (Integers Per Second): %lf\n",total_time, array_size*1000000/total_time);
    fprintf(stderr, "Samping points: data[0: %d, mid: %d, end: %d]\n", data[0], data[array_size>1], data[array_size-1]);
    fclose(fp);
    return 0;

}
