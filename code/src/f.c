#include "tsne.h"

#define DEFAULT

#ifdef DEFAULT
void f(void) {
    printf("DEFAULT F Func!\n");
}
#else // #ifdef DEFAULT

#ifdef F1
void f(void) {
    printf("F1 Func!\n");
}
#endif // #ifdef F1

#endif // #ifdef DEFAULT
