#define SYM_AFF_PA_SCALAR_INIT

#ifdef SYM_AFF_PA_SCALAR_INIT

// scalar replacement
void symmetrizeAffinities(float *p, int n_samples) {
	auto p_sum_inv = .5f / float(n_samples);
	for (int i = 0; i < n_samples; i++) {
		int row = i * n_samples;
		for (int j = i + 1; j < n_samples; j++) {
			int ij = row + j;
			int ji = j * n_samples + i;

			float p_ij = p[ij];
			float p_ji = p[ji];

			p[ji] = p[ij] = (p_ij + p_ji) * p_sum_inv;
		}
	}
}

#endif // SYM_AFF_PA_SCALAR_INIT


#ifdef SYM_AFF_PA_SCALAR_HALF_MATRIX

// scalar replacement
void symmetrizeAffinities(float *p, int n_samples) {
	auto p_sum_inv = .5f / float(n_samples);
	for (int i = 0; i < n_samples; i++) {
		int row = i * n_samples;
		for (int j = i + 1; j < n_samples; j++) {
			int ij = row + j;
			int ji = j * n_samples + i;

			float p_ij = p[ij];
			float p_ji = p[ji];

			p[ji] = (p_ij + p_ji) * p_sum_inv;
		}
	}
}

#endif // SYM_AFF_PA_SCALAR_HALF_MATRIX