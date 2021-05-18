# Development Note

## How to compile
```bash
make # compile, executable file in ./bin & .o files in ./obj (directories automatically created)
make run # compile and run
make clean # delete all the build output

make clean && make # when changing only the -D flag in Makefile, by default it will not compile again :S; so first delete all and then re-build
```

## Possible work distribution for the naive version
1. Implement the Data Stuff (incl. reading / writing data, preprocessing, in python or io.h, validation / visualization) (Jiale)
2. Implement the benchmark stuff, giving interface to count cycles etc. (can also write some tests) (Muyu)
3. Implement the get symmetric affinity part (Levin)
4. Implement the core part of gradient descent (Shengze)

## Things to be careful with
- OK to refer to the github code?

## Datasets
- Please go to our shared Google Drive folder

## Hardware Info Note
### Icelake
fma: latency 4, throughput 0.5
add: latency 4, throughtput 0.5
mul: latency 4, throughtput 0.5

## Optimization Note

### Scalar Optimization

#### EuclideanDistance

baseline version: 719756317.0000
```c
void getSquaredEuclideanDistances(float* X, int n_samples, int dim, float* DD) {
    const float* XnD = X;
    for(int n = 0; n < n_samples; ++n, XnD += dim) {
        const float* XmD = XnD + dim;
        float* curr_elem = &DD[n*n_samples + n];
        *curr_elem = 0.0;
        float* curr_elem_sym = curr_elem + n_samples;
        for(int m = n + 1; m < n_samples; ++m, XmD+=dim, curr_elem_sym+=n_samples) {
            *(++curr_elem) = 0.0;
            for(int d = 0; d < dim; ++d) {
                *curr_elem += (XnD[d] - XmD[d]) * (XnD[d] - XmD[d]);
            }
            *curr_elem_sym = *curr_elem;
        }
    }
}
```

refactoring v1: 82768573.0000 (x8.56)
```c
void getSquaredEuclideanDistances(float* X, int n_samples, int dim, float* DD) {
    for (int i = 0; i < n_samples; i++) {
        for (int j = i + 1; j < n_samples; j++) {
            float tmp = 0.0;
            for (int k = 0; k < dim; k++) {
                float sq = X[i * dim + k] - X[j * dim + k];
                tmp += sq * sq;
            }
            DD[i * n_samples + j] = DD[j * n_samples + i] = tmp;
        }
        DD[i * n_samples + i] = 0.0;
    }
}
```

loop unrolling v2: 88249728.0000 (x8.06)
```c
void getSquaredEuclideanDistances(float* X, int n_samples, int dim, float* DD) {
    for (int i = 0; i < n_samples; i++) {
        for (int j = i + 1; j < n_samples; j++) {
            float tmp;
            float tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;
            tmp = tmp0 = tmp1 = tmp2 = tmp3 = tmp4 = tmp5 = tmp6 = tmp7 = 0.0;
            int k = 0;
            for (; k < dim; k += 8) {
                float sq0, sq1, sq2, sq3, sq4, sq5, sq6, sq7;
                int id = i * dim + k, jd = j * dim + k;
                sq0 = X[id + 0] - X[jd + 0];
                sq1 = X[id + 1] - X[jd + 1];
                sq2 = X[id + 2] - X[jd + 2];
                sq3 = X[id + 3] - X[jd + 3];
                sq4 = X[id + 4] - X[jd + 4];
                sq5 = X[id + 5] - X[jd + 5];
                sq6 = X[id + 6] - X[jd + 6];
                sq7 = X[id + 7] - X[jd + 7];

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
            for (; k < dim; k++) {
                float sq = X[i * dim + k] - X[j * dim + k];
                tmp += sq * sq;
            }
            DD[i * n_samples + j] = DD[j * n_samples + i] = tmp;
        }
        DD[i * n_samples + i] = 0.0;
    }
}
```
