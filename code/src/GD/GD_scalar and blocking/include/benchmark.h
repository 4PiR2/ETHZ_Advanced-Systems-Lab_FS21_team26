#include "tsc_x86.h"
#include <string.h>
#include <assert.h>

#ifndef BENCHMARK_H
#define BENCHMARK_H

#define MAXN_TIMERS 20
#define MAX_TIMER_NAME_LEN 30

typedef int thandle;

struct timer {
    char name[MAX_TIMER_NAME_LEN];
    int rep;
    bool timing;
    myInt64 accum;
    myInt64 st;
    myInt64 ed;
};

// must be defined in main.c
extern int ntimers;
extern struct timer timers[MAXN_TIMERS];
extern int n_timers_to_show;
extern char timers_to_show[][MAX_TIMER_NAME_LEN];
extern int n_timers_to_compare;
extern char timers_to_compare[][2][MAX_TIMER_NAME_LEN];

static thandle get_timer(const char* name) {
    for (int i = 0; i < ntimers; i++) 
        if (strcmp(name, timers[i].name) == 0)
            return i;
    return -1;
}

inline thandle create_timer(const char* name) {
    thandle th = get_timer(name);
    if (th != -1) {
        return th;
    }
    printf("Creating Timer: %s\n", name);
    timers[ntimers].rep = 0;
    timers[ntimers].timing = false;
    timers[ntimers].accum = 0;
    timers[ntimers].st = 0;
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

static bool if_show(char* name) {
    for (int i = 0; i < n_timers_to_show; i++) 
        if (strcmp(name, timers_to_show[i]) == 0)
            return true;
    return false;
}

static double get_duration(char* name) {
    thandle th = get_timer(name);
    if (th != -1) {
        return (double) timers[th].accum / timers[th].rep;
    }
    return -1;
}

inline void benchmark_print() {
    printf("============ Print All Timers ============\n");
    printf("%30s, %10s, %20s\n", "NAME", "REP", "AVG");
    for (int i = 0; i < ntimers; i++) {
        if (if_show(timers[i].name))
            printf("%30s, %10d, %20.4lf\n", timers[i].name, timers[i].rep, (double) timers[i].accum / timers[i].rep);
    }
    printf("============ Print All Timers End ============\n\n");

    printf("============ Print All Speedups ============\n");
    printf("%30s, %30s, %20s\n", "NAME", "NAME_BASELINE", "SPEEDUP");
    for (int i = 0; i < n_timers_to_compare; i++) {
        double d1, d2;
        d1 = get_duration(timers_to_compare[i][0]);
        d2 = get_duration(timers_to_compare[i][1]);
        if (d1 == -1 || d2 == -1) {
            printf("TIMER ERROR: not found %s or %s\n", timers_to_compare[i][0], timers_to_compare[i][1]);
            continue;
        }
        printf("%30s, %30s, %20.2lf\n", timers_to_compare[i][0], timers_to_compare[i][1], d2 / d1);
    }
    printf("============ Print All Speedups End ============\n\n");
}

#endif