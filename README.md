# Pipelined-Cheetah
This repository builds up on the OpenCheetah project (https://github.com/Alibaba-Gemini-Lab/OpenCheetah/commit/1c5cd753e641661fa72cedca34acff45a59dde22).
* The following files are modified for pipeline parallelism and batch inference
	* ./networks/main_sqnet.cpp
    * ./networks/main_resnet50.cpp
    * ./SCI/src/defines.h
    * ./SCI/src/functionalities_uniform.h
    * ./SCI/src/globals.h
    * ./SCI/src/globals.cpp
    * ./SCI/src/library_fixed_uniform.cpp
    * ./SCI/src/library_fixed_uniform.h
    * ./SCI/src/library_fixed.cpp
    * ./SCI/src/library_fixed.h
    * ./SCI/src/library_fixed_uniform_cheetah.cpp
    * ./SCI/src/library_fixed_common.h

* The following files are created for the client-server application and batch inference
	* ./scripts/compile-server-client.sh
    * ./scripts/run-client-optimized.sh
    * ./scripts/run-server-optimized.sh
    * ./scripts/server.cpp
    * ./scripts/client.cpp
    * ./networks/layer_processor.h
    * ./networks/load_input.h
    * ./networks/load_input.cpp

# Cheetah: Lean and Fast Secure Two-Party Deep Neural Network Inference
This repo contains a proof-of-concept implementation for our [Cheetah paper](https://eprint.iacr.org/2022/207).
The codes are still under heavy developments, and **should not** be used in any security sensitive product.

### Cheetah -> Secure Processing Unit
Most of the Cheetah protocols has been re-written in the [SecretFlow](https://github.com/secretflow) project. Check the code [here](https://github.com/secretflow/spu/tree/main/libspu/mpc/cheetah).

### Q&A (Updating)
See [QA.md](QA.md).

### Repo Directory Description
- `include/` Contains implementation of Cheetah's linear protocols.
- `SCI/` A fork of CryptFlow2's SCI library and contains implementation of Cheetah's non-linear protocols.
- `networks/` Auto-generated cpp programs that evaluate some neural networks.
- `pretrained/` Pretrained neural networks and inputs.
- `patch/` Patches applied to the dependent libraries.
- `credits/` Licenses of the dependencies. 
- `scripts/` Helper scripts used to build the programs in this repo.

### Requirements

* openssl 
* c++ compiler (>= 8.0 for the better performance on AVX512)
* cmake >= 3.13
* git
* make
* OpenMP (optional, only needed by CryptFlow2 for multi-threading)

### Building Dependencies
* Run `bash scripts/build-deps.sh` which will build the following dependencies
	* [emp-tool](https://github.com/emp-toolkit/emp-tool) We follow the implementation in SCI that using emp-tool for network io and pseudo random generator.
	* [emp-ot](https://github.com/emp-toolkit/emp-ot) We use Ferret in emp-ot as our VOLE-style OT.
	* [Eigen](https://github.com/libigl/eigen) We use Eigen for tensor operations.
	* [SEAL](https://github.com/microsoft/SEAL) We use SEAL's implementation for the BFV homomorphic encryption scheme.
	* [zstd](https://github.com/facebook/zstd) We use zstd for compressing the ciphertext in SEAL which can be replaced by any other compression library.
	* [hexl](https://github.com/intel/hexl/tree/1.2.2) We need hexl's AVX512 acceleration for achieving the reported numbers in our paper.

* The generated objects are placed in the `build/deps/` folder.
* Build has passed on the following setting
  * MacOS 11.6 with clang 13.0.0, Intel Core i5, cmake 3.22.1
  * Red Hat 7.2.0 with gcc 7.2.1, Intel(R) Xeon(R), cmake 3.12.0
  * Ubuntu 18.04 with gcc 7.5.0 Intel(R) Xeon(R),  cmake 3.13
  * Ubuntu 20.04 with gcc 9.4.0 Intel(R) Xeon(R),  cmake 3.16.3
  
### Building Cheetah and SCI-HE Demo

* Run `bash scripts/build.sh` which will build 2 executables in the `build/bin` folder
	* `resnet50-cheetah` 
	* `sqnet-cheetah`

* These models are modified for pipeline parallelism.

You can change the `SERVER_IP` and `SERVER_PORT` defined in the [scripts/common.sh](scripts/common.sh) to run the demo remotely.
Also, you can use our throttle script to mimic a remote network condition within one Linux machine, see below.

### Mimic an WAN setting within LAN on Linux

* To use the throttle script under [scripts/throttle.sh](scripts/throttle.sh) to limit the network speed and ping latency (require `sudo`)
* For example, run `sudo scripts/throttle.sh wan` on a Linux OS which will limit the local-loop interface to about 400Mbps bandwidth and 40ms ping latency.
  You can check the ping latency by just `ping 127.0.0.1`. The bandwidth can be check using extra `iperf` command.

### Server-client application
Install the Boost.asio library with `sudo apt-get install libboost-all-dev`.
On the terminal run `sudo bash scripts/compile-server-client.sh`.
To start the server, run `sudo bash scripts/server [port]`.
To start the client, run `sudo bash scripts/client [ip-address server] [port] cheetah [resnet50 | sqnet] [batch_size]`.
