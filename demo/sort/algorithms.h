#ifndef __ALGORITHMS_H__
#include <stdint.h>

#define __ALGORITHMS_H__


/* ***************************************************************************
 * Prototypes.
 * ***************************************************************************/

void quicksort_recursive(uint32_t, uint32_t*, uint32_t*);
void quicksort_iterative(uint32_t, uint32_t*, uint32_t*);
void quicksort_naive_parallel(uint32_t, uint32_t*, uint32_t*);
void aasort(uint32_t, uint32_t*, uint32_t*);
void aasort_naive_parallel(uint32_t, uint32_t*, uint32_t*);


#endif /* __ALGORITHMS_H__ */
