#ifndef MAT_H
#define MAT_H

#include <cstdarg>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include "immintrin.h"

#define ALIGN_ELEM 1
// ALIGN_ELEM: 16, 8, ..., 1
#define ALIGNMENT (32 * (ALIGN_ELEM))
// ALIGNMENT: 512, 256, ..., sizeof(T) * 8

#define GET_N(n, T) (((n) + (((ALIGNMENT) >> 3) / sizeof(T) - 1)) & (-1 ^ (((ALIGNMENT) >> 3) / sizeof(T) - 1)))

template<typename T>
inline T *mat_alloc(int n, int m) {
	void *p = _mm_malloc(GET_N(n, T) * GET_N(m, T) * sizeof(T), 4096);
	if (!p)
		std::cerr << "Failed to allocate " << m << " x " << n << " matrix!" << std::endl;
	return (T *) p;
}

template<typename T>
inline void mat_free(T *p) {
	_mm_free(p);
}

inline void mat_free_batch(int num...) {
	va_list ap;
	va_start(ap, num);
	for (int i = 0; i < num; i++)
		_mm_free(va_arg(ap, void *));
	va_end(ap);
}

template<typename T>
inline void mat_clear(T *p, int n, int m) {
	memset(p, 0, GET_N(n, T) * GET_N(m, T) * sizeof(T));
}

template<typename T>
inline void mat_clear_margin(T *p, int n, int m) {
	int N = GET_N(n, T), M = GET_N(m, T), M_m_s = (M - m) * sizeof(T);
	for (int i = 0, iMm = m; i < n; ++i, iMm += M) {
		memset(p + iMm, 0, M_m_s);
	}
	memset(p + n * M, 0, (N - n) * M * sizeof(T));
}

template<typename T>
inline void mat_transpose(T *pt, T *p, int n, int m) {
	int N = GET_N(n, T), M = GET_N(m, T);
	for (int j = 0, jN = 0; j < m; ++j, jN += N) {
		for (int i = 0, iM = 0; i < n; ++i, iM += M) {
			pt[jN + i] = p[iM + j];
		}
	}
}

template<typename T>
inline void mat_rand_norm(T *p, int n, int m, T mean, T std, bool use_seed, long unsigned int seed) {
	int M = GET_N(m, T);
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
inline bool mat_load(T *p, int n, int m, const std::string &filename) {
	std::ifstream f(filename);
	if (!f.is_open()) {
		std::cerr << "Failed to open file: " << filename << "!" << std::endl;
		return false;
	}
	int M = GET_N(m, T);
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
inline bool mat_store(T *p, int n, int m, const std::string &filename) {
	std::ofstream f(filename);
	if (!f.is_open()) {
		std::cerr << "Failed to open file: " << filename << "!" << std::endl;
		return false;
	}
	int M = GET_N(m, T);
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
