#include <cuda_runtime.h>
#include <device_launch_parameters.h>
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
#include <string>

#define NUM_SIMULATIONS 100

HOD struct Agent {

    float Kp = 1.0;
    float Reach = 1.0;
    float Ks = 1.0;
    float personality = 0.0;

    float PersonalityPressure = 0.0;
    float InfluenceNumber = 0.0;

};

__global__ void ApplyPersonalityPressure(Agent* AgentField, int FieldHeight, int FieldWidth) {

    int j = blockIdx.x * blockDim.x + threadIdx.x;
    vec3 AgentPos = vec3(blockIdx.x, threadIdx.x, 0);

    for (UINT y = 0; y < FieldHeight; y++) {
        for (UINT x = 0; x < FieldWidth; x++) {
            int index = y * blockDim.x + x;
            if (index == j) {
                continue;
            }

            vec3 pos = vec3(y, x, 0);

            float wieghtedDist = (AgentPos - pos).length() / AgentField[j].Reach;

            float personalityPressure = (AgentField[index].Kp * (AgentField[j].personality - AgentField[index].personality)) / wieghtedDist;

            AgentField[index].PersonalityPressure += personalityPressure;
            AgentField[index].InfluenceNumber += 1.0;
            //atomicAdd((float*)(AgentField + index * sizeof(Agent) + 4 * sizeof(float)), personalityPressure);
            //atomicAdd((float*)(AgentField + index * sizeof(Agent) + 5 * sizeof(float)), 1.0);

        }
    }

    return;

}

__global__ void ShiftPersonality(Agent* AgentField) {

    int j = blockIdx.x * blockDim.x + threadIdx.x;

    float shift = ((float) AgentField[j].PersonalityPressure / (float) AgentField[j].InfluenceNumber) / (float) AgentField[j].Ks;

    AgentField[j].personality += shift;

    return;
}

int main() {
	// Image
	const auto aspect_ratio = 1.0 / 1.0;
	const int image_width = 100;
    const int image_height = 100; // static_cast<int>(image_width / aspect_ratio);
    Agent* AgentField = nullptr;
    Agent* AgentField_debug = nullptr;

    // Image File

    std::ofstream ImageFile_prev, ImageFile, ImageFile_diff;
    ImageFile_prev.open("./images/0.ppm");
    ImageFile.open("Image.ppm");
    ImageFile_diff.open("Image_diff.ppm");

    ImageFile_prev << "P3\n" << image_width << " " << image_height << "\n255\n";
    ImageFile << "P3\n" << image_width << " " << image_height << "\n255\n";
    ImageFile_diff << "P3\n" << image_width << " " << image_height << "\n255\n";

    AgentField = (Agent*) malloc(image_height * image_width * sizeof(Agent));
    AgentField_debug = (Agent*) malloc(image_height * image_width * sizeof(Agent));


    for (int j = 0; j < image_height; j++) {
        //std::cout << "\rScanlines remaining: " << j << ' ' << std::flush;
        for (int i = 0; i < image_width; i++) {

            float paranoia = random_double() * 2.0 - 1.0;
            float Kp = random_double() * 5.0;
            float Reach = random_double() * 5.0;
            float Ks = random_double() * 10.0;
            AgentField[j * image_width + i].personality = paranoia;
            AgentField[j * image_width + i].Kp = Kp;
            AgentField[j * image_width + i].Reach = Reach;
            AgentField[j * image_width + i].Ks = Ks;
            AgentField[j * image_width + i].PersonalityPressure = 0.0;
            AgentField[j * image_width + i].InfluenceNumber = 0.0;
            AgentField_debug[j * image_width + i].personality = paranoia;
            AgentField_debug[j * image_width + i].Kp = Kp;
            AgentField_debug[j * image_width + i].Reach = Reach;
            AgentField_debug[j * image_width + i].Ks = Ks;
            AgentField_debug[j * image_width + i].PersonalityPressure = 0.0;
            AgentField_debug[j * image_width + i].InfluenceNumber = 1.0;
            color PixelColor = mix(color(1.0, 0.0, 0.0), color(0.0, 0.0, 1.0), AgentField[j * image_width + i].personality);
            write_color(ImageFile_prev, PixelColor);

        }
    }

    Agent* KAgentField = NULL;
    cudaMalloc(&KAgentField, image_height * image_width * sizeof(Agent));

    cudaMemcpy(KAgentField, AgentField, image_height * image_width * sizeof(Agent), cudaMemcpyHostToDevice);

    for (UINT t = 0; t < NUM_SIMULATIONS; t++) {
        std::cout << "\rSimulation runs remaining: " << NUM_SIMULATIONS - t << ' ' << std::flush;

        cudaMemcpy(KAgentField, AgentField, image_height * image_width * sizeof(Agent), cudaMemcpyHostToDevice);

        ApplyPersonalityPressure << <image_height, image_width >> > (KAgentField, image_height, image_width);
        cudaDeviceSynchronize();

        ShiftPersonality << <image_height, image_width >> > (KAgentField);
        cudaDeviceSynchronize();

        cudaMemcpy(AgentField, KAgentField, image_height * image_width * sizeof(Agent), cudaMemcpyDeviceToHost);

        if (true) {
            std::ofstream ImageFile_timestep;
            ImageFile_timestep.open("./images/" + std::to_string(t + 1) + ".ppm");
            ImageFile_timestep << "P3\n" << image_width << " " << image_height << "\n255\n";

            for (int j = image_height - 1; j >= 0; --j) {
                //std::cout << "\rScanlines remaining: " << j << ' ' << std::flush;
                for (int i = 0; i < image_width; ++i) {

                    color PixelColor = mix(color(1.0, 0.0, 0.0), color(0.0, 0.0, 1.0), AgentField[j * image_width + i].personality);

                    write_color(ImageFile_timestep, PixelColor);
                }
            }

            //for (UINT i = 0; i < image_height * image_width; i++) {
            //    Agent agent = AgentField[i];
            //    continue;
            //}

            ImageFile_timestep.close();
        }
    }


    cudaMemcpy(AgentField, KAgentField, image_height * image_width * sizeof(Agent), cudaMemcpyDeviceToHost);

    cudaFree(KAgentField);

    for (int j = image_height - 1; j >= 0; --j) {
        //std::cout << "\rScanlines remaining: " << j << ' ' << std::flush;
        for (int i = 0; i < image_width; ++i) {

            color PixelColor = mix(color(1.0, 0.0, 0.0), color(0.0, 0.0, 1.0), AgentField[j * image_width + i].personality);

            write_color(ImageFile, PixelColor);

            PixelColor = mix(color(0.0, 0.0, 0.0), color(1.0, 1.0, 1.0), abs(AgentField_debug[j * image_width + i].personality - AgentField[j * image_width + i].personality));

            write_color(ImageFile_diff, PixelColor);
        }
    }

    ImageFile_prev.close();
    ImageFile.close();

    delete AgentField;

	return 0;
}