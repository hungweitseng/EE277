# Copyright (c) 1993-2017, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


all: TCGemm TCGemm_nothread simpleTensorCoreGEMM.cu dgemm
#	/usr/local/cuda-9.2/bin/nvcc -o TCGemm -arch=sm_70 -lcublas -lcurand simpleTensorCoreGEMM.cu
#	/usr/local/cuda-9.0/bin/nvcc -o TCGemm -arch=sm_70 -lcublas -lcurand simpleTensorCoreGEMM.cu
TCGemm: simpleTensorCoreGEMM.cu
#	/usr/local/cuda/bin/nvcc -g -O3 -DTHREAD -o TCGemm -arch=sm_80 -lcublas -lcurand -lpthread simpleTensorCoreGEMM.cu
	/usr/local/cuda/bin/nvcc -g -O3 -DTHREAD -o TCGemm -arch=sm_80 -lcublas_static -lcublasLt_static -lculibos -lcurand -lpthread simpleTensorCoreGEMM.cu

TCGemm_nothread: simpleTensorCoreGEMM.cu
	/usr/local/cuda/bin/nvcc -g -O3 -o TCGemm_nothread -arch=sm_80 -lcublas_static -lcublasLt_static -lculibos -lcurand -lpthread simpleTensorCoreGEMM.cu

dgemm: dgemm_example.c
	gcc -O3 dgemm_example.c -o dgemm  -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl
#	nvcc -o TCGemm -arch=sm_70 -lcublas -lcurand simpleTensorCoreGEMM.cu
clean:
	rm -f TCGemm TCGemm_nothread dgemm


