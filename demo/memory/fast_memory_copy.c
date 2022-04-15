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

#ifdef SIMD
void memcpy_avx(void* dest, void *src, size_t size)
{
  __m256i* vsrc = (__m256i*) src;
  __m256i* vdest = (__m256i*) dest;
  __m256i vals;
  size_t i;
  size_t end;
  end =size / sizeof(__m256i);
  for (i = 0; i < end; i++) {
    vals = _mm256_loadu_si256(&vsrc[i]);  // This will generate the vmovaps instruction.
    _mm256_store_si256(&vdest[i+1], vals);  // This will generate the vmovaps instruction.
  }
    return;
}
#ifdef BLOCK128
/*void memcpy_avx_128(void* dest, void *src, size_t size)
{
  __m128i* vsrc = (__m128i*) src;
  __m128i* vdest = (__m128i*) dest;
  __m128i vals[2];
  size_t i;
  for (i = 0; i < size / sizeof(__m128i); i+=2) {
    vals[0] = _mm128_loadu_si128(&vsrc[i]);  // This will generate the vmovaps instruction.
    vals[1] = _mm128_loadu_si128(&vsrc[i+1]);  // This will generate the vmovaps instruction.
//    vals[2] = _mm128_loadu_si128(&vsrc[i+2]);  // This will generate the vmovaps instruction.
//    vals[3] = _mm128_loadu_si128(&vsrc[i+3]);  // This will generate the vmovaps instruction.
    _mm128_store_si128(&vdest[i], vals[0]);  // This will generate the vmovaps instruction.
    _mm128_store_si128(&vdest[i+1], vals[1]);  // This will generate the vmovaps instruction.
//    _mm128_store_si128(&vdest[i+2], vals[2]);  // This will generate the vmovaps instruction.
//    _mm128_store_si128(&vdest[i+3], vals[3]);  // This will generate the vmovaps instruction.
  }

    return;
}*/
#endif
#endif

int main(int argc, char **argv)
{
    size_t *array;
    size_t *dest;
    size_t *in_the_middle;
    size_t size;
    int i;
    double total_time;
    struct timespec start, end;
    struct timeval time_start, time_end;
    size = atoi(argv[1])/sizeof(size_t);
    array = (size_t *)memalign(32,sizeof(size_t)*size);
    dest = (size_t *)memalign(32,sizeof(size_t)*size+32);
    clock_gettime(CLOCK_MONOTONIC, &start);	/* mark start time */
    write_memory_loop(array, size*sizeof(size_t));
//    write_memory_loop(in_the_middle, 64);
    #ifdef MEMSET
    memset(dest, 0, size);
    #endif
    clock_gettime(CLOCK_MONOTONIC, &end);	/* mark start time */
    total_time = ((end.tv_sec * 1000000000.0 + end.tv_nsec) - (start.tv_sec * 1000000000.0 + start.tv_nsec));
//    fprintf(stderr,"(Init) Latency: %.0lf ns, Bandwidth: %lf GBps, %llu\n",total_time, size*sizeof(size_t)/total_time, dest[rand()%size]);
    clock_gettime(CLOCK_MONOTONIC, &start);	/* mark start time */
    #ifdef SIMD
    memcpy_avx(dest, array, size*sizeof(size_t));
    #else
       #ifdef BLOCK128
       memcpy_avx_128B(dest, array, size*sizeof(size_t));
       #else
       memcpy(dest, array, size*sizeof(size_t));
       #endif
    #endif
    clock_gettime(CLOCK_MONOTONIC, &end);	/* mark start time */
    total_time = ((end.tv_sec * 1000000000.0 + end.tv_nsec) - (start.tv_sec * 1000000000.0 + start.tv_nsec));
#ifdef CSV
    fprintf(stderr,"%.0lf, %llu\n",total_time, dest[rand()%size]);
#else
    fprintf(stderr,"Latency: %.0lf ns, Bandwidth: %lf GBps, %llu\n",total_time, size*sizeof(size_t)/total_time, dest[rand()%size]);
#endif
    return 0;
}