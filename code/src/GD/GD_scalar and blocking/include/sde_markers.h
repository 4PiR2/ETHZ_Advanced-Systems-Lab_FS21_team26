#ifndef __SDE_MARKERS__
#define __SDE_MARKERS__

#ifndef __SSC_MARK
#define __SSC_MARK(tag) __asm__ __volatile__("movl %0, %%ebx; .byte 0x64, 0x67, 0x90 "::"i"(tag) : "%ebx")
#endif

#ifdef __cplusplus
extern "C" {
#endif

void sde_start_marker(void) {
	__SSC_MARK(0xFACE);
}

void sde_stop_marker(void) {
	__SSC_MARK(0xDEAD);
}

#ifdef __cplusplus
}
#endif

#endif // __SDE_MARKERS__
