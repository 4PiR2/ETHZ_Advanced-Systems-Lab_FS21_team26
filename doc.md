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