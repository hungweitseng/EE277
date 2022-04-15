#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>	/* for clock_gettime */
#include <malloc.h>
#ifdef SIMD
#include <immintrin.h>
#endif

int read_memory_loop(void* array, size_t size) {
  size_t* carray = (size_t*) array;
  size_t sum, i;
  for (i = 0; i < size / sizeof(size_t); i++) {
    sum = carray[i];
  }
  return sum;
}

#ifdef SIMD
int read_memory_avx(void* array, size_t size) {
  __m256i* varray = (__m256i*) array;
  __m256i sum;
  int *ret;
  size_t i;
  for (i = 0; i < size / sizeof(__m256i); i++) {
    sum = _mm256_loadu_si256(&varray[i]);  // This will generate the vmovaps instruction.
  }
  ret = &sum;
  return ret[0];
}
#endif

int main(int argc, char **argv)
{
    size_t *array;
    size_t size, number_of_elements;
    size_t sum;
    double total_time;
    struct timespec start, end;
    int i;
    size = atol(argv[1]);
    number_of_elements = size/sizeof(size_t);
    array = (size_t *)memalign(32,size);
    clock_gettime(CLOCK_MONOTONIC, &start);	/* mark start time */
        #ifdef SIMD
        read_memory_avx(array, size);
        #else
        read_memory_loop(array, size);
        #endif
        clock_gettime(CLOCK_MONOTONIC, &end);	/* mark the end time */
        total_time = ((end.tv_sec * 1000000000.0 + end.tv_nsec) - (start.tv_sec * 1000000000.0 + start.tv_nsec));
        #ifdef CSV
        fprintf(stderr,"%.0lf,\t",total_time);
        #else
        fprintf(stderr,"Read: Latency: %.0lf ns, GBps: %lf\n",total_time, (double)((double)size/(total_time)));
        #endif
    total_time = 0.0;
    for(i = 0 ; i< 10; i++)
    {
    clock_gettime(CLOCK_MONOTONIC, &start);	/* mark start time */
    #ifdef SIMD
    read_memory_avx(array, size);
    #else
    read_memory_loop(array, size);
    #endif
    clock_gettime(CLOCK_MONOTONIC, &end);	/* mark the end time */
    total_time += ((end.tv_sec * 1000000000.0 + end.tv_nsec) - (start.tv_sec * 1000000000.0 + start.tv_nsec));
//    sum += array[rand()%number_of_elements];
    }
#ifdef CSV
    fprintf(stderr,"%.0lf, %llu\n",total_time/10, sum);
#else
    fprintf(stderr,"Read: Latency (10x average): %.0lf ns, GBps (10x average): %lf\n",total_time/10, (double)((double)size*10/(total_time)));
//    fprintf(stderr,"%.0lf, %llu\n",total_time/10, sum);
#endif

    return 0;
}