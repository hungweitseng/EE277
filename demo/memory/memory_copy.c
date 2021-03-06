#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>	/* for clock_gettime */
#include <malloc.h>
#ifdef SIMD
#include <immintrin.h>
#endif

void write_memory_loop(void* array, size_t size) {
  size_t* carray = (size_t*) array;
  size_t i;
  for (i = 0; i < size / sizeof(size_t); i++) {
    carray[i] = 1;
  }
}

int main(int argc, char **argv)
{
    size_t *array;
    size_t *dest;
    size_t size;
    double total_time;
    struct timespec start, end;
    struct timeval time_start, time_end;
    size = atoi(argv[1])/sizeof(size_t);
    array = (size_t *)memalign(32,sizeof(size_t)*size);
    dest = (size_t *)memalign(32,sizeof(size_t)*size);
    write_memory_loop(array, size*sizeof(size_t));
    clock_gettime(CLOCK_MONOTONIC, &start);	/* mark start time */
    memcpy(dest, array, size*sizeof(size_t));
    clock_gettime(CLOCK_MONOTONIC, &end);	/* mark start time */
    total_time = ((end.tv_sec * 1000000000.0 + end.tv_nsec) - (start.tv_sec * 1000000000.0 + start.tv_nsec));
#ifdef CSV
    fprintf(stderr,"%.0lf, %llu\n",total_time, dest[rand()%size]);
#else
    fprintf(stderr,"%.0lf, %llu\n",total_time, dest[rand()%size]);
#endif
    return 0;
}