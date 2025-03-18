#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Defines.cuh"
#include "Utils.cuh"
#include "Vec3.cuh"
#include "Color.cuh"

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>
#include <vector>

__global__ void simulate(float* PersonalityField) {

    int j = blockIdx.x * blockDim.x + threadIdx.x;

    float personality = PersonalityField[j];

    PersonalityField[j] = 1.0 - personality;

    return;

}

int main() {
	// Image
	const auto aspect_ratio = 1.0 / 1.0;
	const int image_width = 400;
    const int image_height = 400; // static_cast<int>(image_width / aspect_ratio);
    float* personalityField = nullptr;

    // Image File

    std::ofstream ImageFile_prev, ImageFile;
    ImageFile_prev.open("Image_prev.ppm");
    ImageFile.open("Image.ppm");

    ImageFile_prev << "P3\n" << image_width << " " << image_height << "\n255\n";
    ImageFile << "P3\n" << image_width << " " << image_height << "\n255\n";

    personalityField = (float*) malloc(image_height * image_width * sizeof(float));


    for (int j = 0; j < image_height; j++) {
        //std::cout << "\rScanlines remaining: " << j << ' ' << std::flush;
        for (int i = 0; i < image_width; i++) {

            float paranoia = random_double() * 2.0 - 1.0;
            personalityField[j * image_width + i] = paranoia;
            color PixelColor = mix(color(1.0, 0.0, 0.0), color(0.0, 0.0, 1.0), personalityField[j * image_width + i]);
            write_color(ImageFile_prev, PixelColor);

        }
    }

    float* KPersonalityField = NULL;
    cudaMalloc(&KPersonalityField, image_height * image_width * sizeof(float));

    cudaMemcpy(KPersonalityField, personalityField, image_height * image_width * sizeof(float), cudaMemcpyHostToDevice);

    simulate << <image_height, image_width >> > (KPersonalityField);
    cudaDeviceSynchronize();

    cudaMemcpy(personalityField, KPersonalityField, image_height * image_width * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(KPersonalityField);

    for (int j = image_height - 1; j >= 0; --j) {
        //std::cout << "\rScanlines remaining: " << j << ' ' << std::flush;
        for (int i = 0; i < image_width; ++i) {

            color PixelColor = mix(color(1.0, 0.0, 0.0), color(0.0, 0.0, 1.0), personalityField[j * image_width + i]);

            write_color(ImageFile, PixelColor);

        }
    }

    ImageFile_prev.close();
    ImageFile.close();

    delete personalityField;

	return 0;
}