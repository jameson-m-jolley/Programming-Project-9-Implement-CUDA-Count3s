#  Shaun Cooper
#  Simple makefile for CUDE
#
#   The executable and include paths may be different in the future
#   If they are , email cs-cog@nmsu.edu

counts: count3.cu
	/usr/local/cuda/bin/nvcc -o counts -I/usr/local/cuda/include count3.cu

racecondition: racecondition.cu
	/usr/local/cuda/bin/nvcc -o racecondition -I/usr/local/cuda/include racecondition.cu