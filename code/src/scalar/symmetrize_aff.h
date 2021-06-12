#ifdef SYM_AFF_SCALAR

// scalar replacement
void symmetrizeAffinities(float *p, int n_samples) {
	auto p_sum_inv = .5f / (float) n_samples;
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

#endif // SYM_AFF_SCALAR


#ifdef SYM_AFF_SCALAR_HALF_MATRIX

// scalar replacement
void symmetrizeAffinities(float *p, int n_samples) {
	auto p_sum_inv = .5f / (float) n_samples;
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

#endif // SYM_AFF_SCALAR_HALF_MATRIX

#ifdef SYM_AFF_BLOCKING

// scalar replacement
void symmetrizeAffinities(float *p, int n_samples) {
	auto p_sum_inv = .5f / (float) n_samples;
	for (int i = 0; i < n_samples; i+=2) {
		int r0 = i * n_samples;
		int r1 = (i + 1) * n_samples;
		int j = i + 1;
		for (; j < n_samples - 8; j+=8) {
			int ij0 = r0 + j;
			int ij1 = r0 + j + 1;
			int ij2 = r0 + j + 2;
			int ij3 = r0 + j + 3;
			int ij4 = r0 + j + 4;
			int ij5 = r0 + j + 5;
			int ij6 = r0 + j + 6;
			int ij7 = r0 + j + 7;
			int ij8 = r1 + j + 0;
			int ij9 = r1 + j + 1;
			int ij10 = r1 + j + 2;
			int ij11 = r1 + j + 3;
			int ij12 = r1 + j + 4;
			int ij13 = r1 + j + 5;
			int ij14 = r1 + j + 6;
			int ij15 = r1 + j + 7;

			int ji0 = j * n_samples + i;
			int ji1 = (j + 1) * n_samples + i;
			int ji2 = (j + 2) * n_samples + i;
			int ji3 = (j + 3) * n_samples + i;
			int ji4 = (j + 4) * n_samples + i;
			int ji5 = (j + 5) * n_samples + i;
			int ji6 = (j + 6) * n_samples + i;
			int ji7 = (j + 7) * n_samples + i;
			int ji8 = (j) * n_samples + i + 1;
			int ji9 = (j + 1) * n_samples + i + 1;
			int ji10 = (j + 2) * n_samples + i + 1;
			int ji11 = (j + 3) * n_samples + i + 1;
			int ji12 = (j + 4) * n_samples + i + 1;
			int ji13 = (j + 5) * n_samples + i + 1;
			int ji14 = (j + 6) * n_samples + i + 1;
			int ji15 = (j + 7) * n_samples + i + 1;

			float p0 = p[ij0];
			float p1 = p[ij1];
			float p2 = p[ij2];
			float p3 = p[ij3];
			float p4 = p[ij4];
			float p5 = p[ij5];
			float p6 = p[ij6];
			float p7 = p[ij7];
			float p8 = p[ij8];
			float p9 = p[ij9];
			float p10 = p[ij10];
			float p11 = p[ij11];
			float p12 = p[ij12];
			float p13 = p[ij13];
			float p14 = p[ij14];
			float p15 = p[ij15];

			float c0 = p[ji0];
			float c1 = p[ji1];
			float c2 = p[ji2];
			float c3 = p[ji3];
			float c4 = p[ji4];
			float c5 = p[ji5];
			float c6 = p[ji6];
			float c7 = p[ji7];
			float c8 = p[ji8];
			float c9 = p[ji9];
			float c10 = p[ji10];
			float c11 = p[ji11];
			float c12 = p[ji12];
			float c13 = p[ji13];
			float c14 = p[ji14];
			float c15 = p[ji15];

			p0 = (p0 + c0) * p_sum_inv;
			p1 = (p1 + c1) * p_sum_inv;
			p2 = (p2 + c2) * p_sum_inv;
			p3 = (p3 + c3) * p_sum_inv;
			p4 = (p4 + c4) * p_sum_inv;
			p5 = (p5 + c5) * p_sum_inv;
			p6 = (p6 + c6) * p_sum_inv;
			p7 = (p7 + c7) * p_sum_inv;
			p8 = (p8 + c8) * p_sum_inv;
			p9 = (p9 + c9) * p_sum_inv;
			p10 = (p10 + c10) * p_sum_inv;
			p11 = (p11 + c11) * p_sum_inv;
			p12 = (p12 + c12) * p_sum_inv;
			p13 = (p13 + c13) * p_sum_inv;
			p14 = (p14 + c14) * p_sum_inv;
			p15 = (p15 + c15) * p_sum_inv;

			p[ij0] = p0;
			p[ij1] = p1;
			p[ij2] = p2;
			p[ij3] = p3;
			p[ij4] = p4;
			p[ij5] = p5;
			p[ij6] = p6;
			p[ij7] = p7;
			p[ij8] = p8;
			p[ij9] = p9;
			p[ij10] = p10;
			p[ij11] = p11;
			p[ij12] = p12;
			p[ij13] = p13;
			p[ij14] = p14;
			p[ij15] = p15;
		}

		for (; j < n_samples; j++) {
			int ij = row + j;
			int ji = j * n_samples + i;

			float p_ij = p[ij];
			float p_ji = p[ji];

			p[ji] = (p_ij + p_ji) * p_sum_inv;
		}
	}
}

#endif // SYM_AFF_BLOCKING