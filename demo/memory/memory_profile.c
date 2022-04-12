#include <stdio.h>
#include <stdlib.h>
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

#ifdef SIMD
void write_memory_avx(void* array, size_t size) {
  __m256i* varray = (__m256i*) array;

  __m256i vals = _mm256_set1_epi32(1);
  size_t i;
  for (i = 0; i < size / sizeof(__m256i); i++) {
    _mm256_store_si256(&varray[i], vals);  // This will generate the vmovaps instruction.
  }
}
#endif

int main(int argc, char **argv)
{
    size_t *array;
    size_t size;
    double total_time;
    struct timespec start, end;
    int i;
    size = atoi(argv[1])/sizeof(size_t);
    array = (size_t *)memalign(32,sizeof(size_t)*size);
    clock_gettime(CLOCK_MONOTONIC, &start);	/* mark start time */
    #ifdef SIMD
    write_memory_avx(array, size);
    #else
    write_memory_loop(array, size);
    #endif
    clock_gettime(CLOCK_MONOTONIC, &end);	/* mark the end time */
    total_time = ((end.tv_sec * 1000000000.0 + end.tv_nsec) - (start.tv_sec * 1000000000.0 + start.tv_nsec));
    fprintf(stderr,"Latency: %.0lf ns, GBps: %lf\n",total_time, (double)((double)size*sizeof(size_t)/(total_time)));
    clock_gettime(CLOCK_MONOTONIC, &start);	/* mark start time */
    for(i = 0 ; i< 10; i++)
    #ifdef SIMD
    write_memory_avx(array, size);
    #else
    write_memory_loop(array, size);
    #endif
    clock_gettime(CLOCK_MONOTONIC, &end);	/* mark the end time */
    total_time = ((end.tv_sec * 1000000000.0 + end.tv_nsec) - (start.tv_sec * 1000000000.0 + start.tv_nsec));
    fprintf(stderr,"Latency (10x average): %.0lf ns, GBps (10x average): %lf\n",total_time/10, (double)((double)size*10*sizeof(size_t)/(total_time)));
    return 0;
}