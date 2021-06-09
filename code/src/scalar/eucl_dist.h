#ifdef SQEU_DIST_OPT_ENR8

void getSquaredEuclideanDistances(float *x, int n_samples, int d_in, float *d) {
	float* norms = (float*)malloc(n_samples * sizeof(float));
	for (int i = 0; i < n_samples; i++) {
		float tmp, tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;
		tmp = tmp0 = tmp1 = tmp2 = tmp3 = tmp4 = tmp5 = tmp6 = tmp7 = 0.f;
		int k = 0;
		for (; k < d_in; k += 8) {
			int id = i * d_in + k;
			tmp0 += x[id + 0] * x[id + 0];
			tmp1 += x[id + 1] * x[id + 1];
			tmp2 += x[id + 2] * x[id + 2];
			tmp3 += x[id + 3] * x[id + 3];
			tmp4 += x[id + 4] * x[id + 4];
			tmp5 += x[id + 5] * x[id + 5];
			tmp6 += x[id + 6] * x[id + 6];
			tmp7 += x[id + 7] * x[id + 7];
		}

		tmp = tmp0 + tmp1 + tmp2 + tmp3 + tmp4 + tmp5 + tmp6 + tmp7;
		for (; k < d_in; k++) {
			tmp += x[i * d_in + k] * x[i * d_in + k];
		}
		norms[i] = tmp;
	}

	for (int i = 0; i < n_samples; i++) {
		for (int j = i + 1; j < n_samples; j++) {
			float tmp, tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;
			tmp = tmp0 = tmp1 = tmp2 = tmp3 = tmp4 = tmp5 = tmp6 = tmp7 = 0.f;
			int k = 0;
			for (; k < d_in; k += 8) {
				int id = i * d_in + k;
				int jd = j * d_in + k;
				tmp0 += x[id + 0] * x[jd + 0];
				tmp1 += x[id + 1] * x[jd + 1];
				tmp2 += x[id + 2] * x[jd + 2];
				tmp3 += x[id + 3] * x[jd + 3];
				tmp4 += x[id + 4] * x[jd + 4];
				tmp5 += x[id + 5] * x[jd + 5];
				tmp6 += x[id + 6] * x[jd + 6];
				tmp7 += x[id + 7] * x[jd + 7];
			}

			tmp = tmp0 + tmp1 + tmp2 + tmp3 + tmp4 + tmp5 + tmp6 + tmp7;
			for (; k < d_in; k++) {
				tmp += x[i * d_in + k] * x[j * d_in + k];
			}

			d[i * n_samples + j] = norms[i] - 2*tmp + norms[j];
		}
	}

	free(norms);
}

#endif // SQEU_DIST_OPT_ENR8

// enroll loops with 16
#ifdef SQEU_DIST_OPT_ENR16

void getSquaredEuclideanDistances(float *x, int n_samples, int d_in, float *d) {
	float* norms = (float*)malloc(n_samples * sizeof(float));
	for (int i = 0; i < n_samples; i++) {
		float tmp, tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12, tmp13, tmp14, tmp15;
		tmp = tmp0 = tmp1 = tmp2 = tmp3 = tmp4 = tmp5 = tmp6 = tmp7 = tmp8 = tmp9 = tmp10 = tmp11 = tmp12 = tmp13 = tmp14 = tmp15 = 0.f;
		int k = 0;
		for (; k < d_in; k += 16) {
            int id = i * d_in + k;
            float xid0 = x[id + 0];
            float xid1 = x[id + 1];
            float xid2 = x[id + 2];
            float xid3 = x[id + 3];
            float xid4 = x[id + 4];
            float xid5 = x[id + 5];
            float xid6 = x[id + 6];
            float xid7 = x[id + 7];
            float xid8 = x[id + 8];
            float xid9 = x[id + 9];
            float xid10 = x[id + 10];
            float xid11 = x[id + 11];
            float xid12 = x[id + 12];
            float xid13 = x[id + 13];
            float xid14 = x[id + 14];
            float xid15 = x[id + 15];
			
			tmp0 += xid0 * xid0;
			tmp1 += xid1 * xid1;
            tmp2 += xid2 * xid2;
            tmp3 += xid3 * xid3;
            tmp4 += xid4 * xid4;
            tmp5 += xid5 * xid5;
            tmp6 += xid6 * xid6;
            tmp7 += xid7 * xid7;
            tmp8 += xid8 * xid8;
            tmp9 += xid9 * xid9;
            tmp10 += xid10 * xid10;
            tmp11 += xid11 * xid11;
            tmp12 += xid12 * xid12;
            tmp13 += xid13 * xid13;
            tmp14 += xid14 * xid14;
            tmp15 += xid15 * xid15;
		}

		tmp = tmp0 + tmp1 + tmp2 + tmp3 + tmp4 + tmp5 + tmp6 + tmp7 + tmp8 + tmp9 + tmp10 + tmp11 + tmp12 + tmp13 + tmp14 + tmp15;
		for (; k < d_in; k++) {
			tmp += x[i * d_in + k] * x[i * d_in + k];
		}
		norms[i] = tmp;
	}

	for (int i = 0; i < n_samples; i++) {
		for (int j = i + 1; j < n_samples; j++) {
			float tmp, tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12, tmp13, tmp14, tmp15;
			tmp = tmp0 = tmp1 = tmp2 = tmp3 = tmp4 = tmp5 = tmp6 = tmp7 = tmp8 = tmp9 = tmp10 = tmp11 = tmp12 = tmp13 = tmp14 = tmp15 = 0.f;
			int k = 0;
			for (; k < d_in; k += 16) {
				int id = i * d_in + k;
                float xid0 = x[id + 0];
                float xid1 = x[id + 1];
                float xid2 = x[id + 2];
                float xid3 = x[id + 3];
                float xid4 = x[id + 4];
                float xid5 = x[id + 5];
                float xid6 = x[id + 6];
                float xid7 = x[id + 7];
                float xid8 = x[id + 8];
                float xid9 = x[id + 9];
                float xid10 = x[id + 10];
                float xid11 = x[id + 11];
                float xid12 = x[id + 12];
                float xid13 = x[id + 13];
                float xid14 = x[id + 14];
                float xid15 = x[id + 15];

				int jd = j * d_in + k;
                float xjd0 = x[jd + 0];
                float xjd1 = x[jd + 1];
                float xjd2 = x[jd + 2];
                float xjd3 = x[jd + 3];
                float xjd4 = x[jd + 4];
                float xjd5 = x[jd + 5];
                float xjd6 = x[jd + 6];
                float xjd7 = x[jd + 7];
                float xjd8 = x[jd + 8];
                float xjd9 = x[jd + 9];
                float xjd10 = x[jd + 10];
                float xjd11 = x[jd + 11];
                float xjd12 = x[jd + 12];
                float xjd13 = x[jd + 13];
                float xjd14 = x[jd + 14];
                float xjd15 = x[jd + 15];


				tmp0 += xid0 * xjd0;
                tmp1 += xid1 * xjd1;
                tmp2 += xid2 * xjd2;
                tmp3 += xid3 * xjd3;
                tmp4 += xid4 * xjd4;
                tmp5 += xid5 * xjd5;
                tmp6 += xid6 * xjd6;
                tmp7 += xid7 * xjd7;
                tmp8 += xid8 * xjd8;
                tmp9 += xid9 * xjd9;
                tmp10 += xid10 * xjd10;
                tmp11 += xid11 * xjd11;
                tmp12 += xid12 * xjd12;
                tmp13 += xid13 * xjd13;
                tmp14 += xid14 * xjd14;
                tmp15 += xid15 * xjd15;
			}

			tmp = tmp0 + tmp1 + tmp2 + tmp3 + tmp4 + tmp5 + tmp6 + tmp7 + tmp8 + tmp9 + tmp10 + tmp11 + tmp12 + tmp13 + tmp14 + tmp15;
			for (; k < d_in; k++) {
				tmp += x[i * d_in + k] * x[j * d_in + k];
			}

			d[i * n_samples + j] = norms[i] - 2*tmp + norms[j];
		}
	}

	free(norms);
}

#endif // SQEU_DIST_OPT_ENR16


#ifdef SQEU_DIST_ENR8

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

#endif // SQEU_DIST_ENR8