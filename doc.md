# Development Note

## How to compile
```bash
make clean && make run
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
