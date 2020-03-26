# Interleaved Multi-Vectorizing (IMV)
  This repository contains a collection of experiments to test the effects of IMV. It is mainly composed of two parts:
  
  (1) Testing the effects of IMV on individual operators, including the hash join probe, binary tree search, hash join build and hash aggregation. The source codes lay at src/imv-operator/. This part is based on the source code of the [work](https://ieeexplore.ieee.org/abstract/document/6544839/).
  
  (2) Comparing IMV with other three state-of-the-art query engines: Compilation, Vectorization, and ROF. The source codes lay at src/imv/. This part is based on the code base [Database Engines: Vectorization vs. Compilation](https://github.com/TimoKersten/db-engine-paradigms)
  

## How to Build on CPUs (Skylake)
A configuration file is provided to build this project with CMake. 
In the project directory run:
```
mkdir -p build/release
cd build/release
cmake -DCMAKE_BUILD_TYPE=Release -DVECTORWISE_BRANCHING=on -DKNL=OFF  ../..
make
```

Generating two main binaries:

(1) operator: 

  (1.1) generating data:
  ```
  ./operator -a GEN -n 32 --r-size=1048576 --s-size=52428800 --r-skew=1 --s-skew=1  
  ```
  (1.2) testing individual operators: -a: Applications:(NPO: hash join probe, BTS: binary tree search, BUILD and AGG), -n threads number.
  ```
  ./operator -a NPO -n 32 --r-file=r_skew=1_size=1M --s-file=s_skew=1_size=50M_max=1M
  ```

(2) engine: 3 parameters: reteated times, data source and threads number
```
./engine 5 ../data/tpch/sf100 32
```

scripts/test.sh can be used to run experiments of the paper.

## UPDATES
(1) reading raw data from DBGEN ends without '|', so change it in Line:146 @ Import.cpp

(2) 
```
terminate called after throwing an instance of 'std::runtime_error'  what():  Buffer 20 not found
```
This error results from the Buffer 20 is not created but it is used at /src/benchmarks/sbb/queries/q4X.cpp. Changes lay at those files using "#define CHANGE".
