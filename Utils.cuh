#pragma once

#ifndef UTIL_H
#define UTIL_H

#include "Defines.cuh"
#include <limits>
#include <math.h>
#include <curand.h>
#include <curand_kernel.h>
#include <iostream>

// Constants

const double inf = std::numeric_limits<double>::infinity();

#define infinity 1000000000000000
#define pi 3.1415926535897932385

// Utility Functions

HOD inline double degrees_to_radians(double degrees) {
	return degrees * pi / 180.0;
}

HO inline double random_double() {
	// Returns a random real in [0,1).
	return rand() / (RAND_MAX + 1.0);
}

HO inline double random_double(double min, double max) {
	// Returns a random real in [min,max).
	return min + (max - min) * random_double();
}

DEV inline double random_double_unit(curandState localState) {
	return curand_uniform_double(&localState) * 2 - 1;
}


HOD inline double clamp(double x, double min, double max) {
	if (x < min) return min;
	if (x > max) return max;
	return x;
}

void HandleCudaKernelError(const cudaError_t CudaError)
{
	if (CudaError == cudaSuccess)
		return;

	std::cout << "CUDA runtime error : " << cudaGetErrorString(CudaError) << "\n";
}

DEV static double reflectance(double cosine, double ref_idx) {
	// Use Schlick's approximation for reflectance.
	auto r0 = (1 - ref_idx) / (1 + ref_idx);
	r0 = r0 * r0;
	return r0 + (1 - r0) * pow((1 - cosine), 5);
}

#endif