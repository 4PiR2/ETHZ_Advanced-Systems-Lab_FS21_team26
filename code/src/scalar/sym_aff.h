#ifndef SYM_AFF_H
#define SYM_AFF_H

void getSquaredEuclideanDistances(float *x, int n_samples, int d_in, float *d) {
	for (int i = 0; i < n_samples; i++) {
		for (int j = i + 1; j < n_samples; j++) {
			float tmp;
			float tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;
			tmp = tmp0 = tmp1 = tmp2 = tmp3 = tmp4 = tmp5 = tmp6 = tmp7 = 0.f;
			int k = 0;
			for (; k < d_in; k += 8) {
				float sq0, sq1, sq2, sq3, sq4, sq5, sq6, sq7;
				int id = i * d_in + k, jd = j * d_in + k;
				sq0 = x[id + 0] - x[jd + 0];
				sq1 = x[id + 1] - x[jd + 1];
				sq2 = x[id + 2] - x[jd + 2];
				sq3 = x[id + 3] - x[jd + 3];
				sq4 = x[id + 4] - x[jd + 4];
				sq5 = x[id + 5] - x[jd + 5];
				sq6 = x[id + 6] - x[jd + 6];
				sq7 = x[id + 7] - x[jd + 7];

				tmp0 += sq0 * sq0;
				tmp1 += sq1 * sq1;
				tmp2 += sq2 * sq2;
				tmp3 += sq3 * sq3;
				tmp4 += sq4 * sq4;
				tmp5 += sq5 * sq5;
				tmp6 += sq6 * sq6;
				tmp7 += sq7 * sq7;
			}

			tmp = tmp0 + tmp1 + tmp2 + tmp3 + tmp4 + tmp5 + tmp6 + tmp7;
			for (; k < d_in; k++) {
				float sq = x[i * d_in + k] - x[j * d_in + k];
				tmp += sq * sq;
			}
			d[i * n_samples + j] = d[j * n_samples + i] = tmp;
		}
		d[i * n_samples + i] = 0.f;
	}
}

#endif //SYM_AFF_H
