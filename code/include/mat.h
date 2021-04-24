#ifndef MAT_H
#define MAT_H

#include <cstdarg>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include "immintrin.h"

#define ALIGNMENT 4
// 4: 16 elements (512-bit) | 3: 8 elements (256-bit) | 0: 1 element
#define GET_N(n) (((n) + ((1 << (ALIGNMENT)) - 1)) & (-1 ^ ((1 << (ALIGNMENT)) - 1)))

template<typename T>
T *mat_alloc(int n, int m) {
	void *p = _mm_malloc(GET_N(n) * GET_N(m) * sizeof(T), 4096);
	if (!p)
		std::cerr << "Failed to allocate " << m << " x " << n << " matrix!" << std::endl;
	return (T *) p;
}

template<typename T>
void mat_free(T *p) {
	_mm_free(p);
}

void mat_free_batch(int num...) {
	va_list ap;
	va_start(ap, num);
	for (int i = 0; i < num; i++)
		_mm_free(va_arg(ap, void*));
	va_end(ap);
}

template<typename T>
void mat_clear(T *p, int n, int m) {
	memset(p, 0, GET_N(n) * GET_N(m) * sizeof(T));
}

template<typename T>
void mat_clear_margin(T *p, int n, int m) {
	int N = GET_N(n), M = GET_N(m), M_m_s = (M - m) * sizeof(T);
	for (int i = 0, iMm = m; i < n; ++i, iMm += M) {
		memset(p + iMm, 0, M_m_s);
	}
	memset(p + n * M, 0, (N - n) * M * sizeof(T));
}

template<typename T>
void mat_transpose(T *pt, T *p, int n, int m) {
	int N = GET_N(n), M = GET_N(m);
	for (int j = 0, jN = 0; j < m; ++j, jN += N) {
		for (int i = 0, iM = 0; i < n; ++i, iM += M) {
			pt[jN + i] = p[iM + j];
		}
	}
}

template<typename T>
void mat_rand_norm(T *p, int n, int m, T mean, T std, bool use_seed, long unsigned int seed) {
	int M = GET_N(m);
	std::random_device rd;
	std::mt19937 gen0{rd()}, gen1{seed};
	std::normal_distribution<T> d(mean, std);
	for (int i = 0, iM = 0; i < n; ++i, iM += M) {
		for (int j = 0; j < m; ++j) {
			p[iM + j] = d(use_seed ? gen1 : gen0);
		}
	}
}

template<typename T>
bool mat_load(T *p, int n, int m, const std::string &filename) {
	std::ifstream f(filename);
	if (!f.is_open()) {
		std::cerr << "Failed to open file: " << filename << "!" << std::endl;
		return false;
	}
	int M = GET_N(m);
	std::string s;
	for (int i = 0, iM = 0; i < n; ++i, iM += M) {
		for (int j = 0; j < m; ++j) {
			f >> p[iM + j];
		}
		getline(f, s);
	}
	f.close();
	return true;
}

template<typename T>
bool mat_store(T *p, int n, int m, const std::string &filename) {
	std::ofstream f(filename);
	if (!f.is_open()) {
		std::cerr << "Failed to open file: " << filename << "!" << std::endl;
		return false;
	}
	int M = GET_N(m);
	for (int i = 0, iM = 0; i < n; ++i, iM += M) {
		for (int j = 0; j < m; ++j) {
			f << p[iM + j] << "\t";
		}
		f << "\n";
	}
	f.close();
	return true;
}

#endif //MAT_H
