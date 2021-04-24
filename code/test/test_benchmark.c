#include <stdio.h>
#include <benchmark.h>

int main() { 
    const double opr = 1.00001;
    volatile double res = 0.0;
    const int N = 100000;

    thandle t = create_timer("Loop1");
    for (int i = 0; i < 100; i++) {
        start(t);
        for (int j = 0; j < N; j++)
            res = opr * opr;
        stop(t);
    }

    t = create_timer("Loop2");
    for (int i = 0; i < 100; i++) {
        start(t);
        for (int j = 0; j < N; j++)
            res = opr - opr;
        stop(t);
    }

    t = create_timer("Loop3");
    for (int i = 0; i < 100; i++) {
        start(t);
        for (int j = 0; j < N; j++)
            res = opr / opr;
        stop(t);
    }   

    benchmark_print();

    // This would fail
    // t = create_timer("T");
    // stop(t);

    // This would fail
    // t = create_timer("T");
    // start(t);
    // start(t);

    return 0;
}