#!/bin/bash

cd omp
make

cd ../gpu_version
make

cd ../omp_gpu_version
make
