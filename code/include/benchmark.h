#include "tsc_x86.h"
#include <string.h>
#include <assert.h>

#ifndef BENCHMARK_H

#define MAXN_TIMERS 20
#define MAX_TIMER_NAME_LEN 15

typedef int thandle;

struct timer {
    char name[MAX_TIMER_NAME_LEN];
    int rep;
    bool timing;
    myInt64 accum;
    myInt64 st;
    myInt64 ed;
};

int ntimers = 0;
struct timer timers[MAXN_TIMERS];

inline thandle create_timer(const char* name) {
    timers[ntimers] = {
        .rep = 0,
        .timing = false,
        .accum = 0,
        .st = 0
    };
    strcpy(timers[ntimers].name, name);

    return ntimers++;
}

inline void start(thandle th) {
    assert(!timers[th].timing && "Timer Has Already Started\n");
    timers[th].st = start_tsc();
    timers[th].timing = true;
}

inline void stop(thandle th) {
    assert(timers[th].timing && "Timer Hasn't Started\n");
    timers[th].accum += stop_tsc(timers[th].st);
    timers[th].timing = false;
    timers[th].rep++;
}

inline void benchmark_print() {
    printf("============ Print All Timers ============\n");
    printf("%15s, %10s, %10s, %15s\n", "NAME", "REP", "TOTAL", "AVG");
    for (int i = 0; i < ntimers; i++) {
        printf("%15s, %10d, %10lld, %15.4lf\n", timers[i].name, timers[i].rep, timers[i].accum, (double) timers[i].accum / timers[i].rep);
    }
    printf("============ Print All Timers End ============\n\n");
}

#endif