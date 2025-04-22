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
#include <ctime>
#include <filesystem>
#include "gif.h"
#include <math.h>

#define NUM_FULL_SIMS 50
#define NUM_SIMULATIONS 50

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

            if ((AgentPos - pos).length() > AgentField[j].Reach) {
                continue;
            }

            float wieghtedDist = pow((AgentPos - pos).length(), 2) / AgentField[j].Reach;

            float personalityPressure = (AgentField[index].Kp * (AgentField[j].personality - AgentField[index].personality)) / wieghtedDist;

            //AgentField[index].PersonalityPressure += personalityPressure;
            //AgentField[index].InfluenceNumber += 1.0;
            atomicAdd(&(AgentField[index].PersonalityPressure), personalityPressure);
            atomicAdd(&(AgentField[index].InfluenceNumber), 1.0);

        }
    }

    return;

}

__global__ void ShiftPersonality(Agent* AgentField) {

    int j = blockIdx.x * blockDim.x + threadIdx.x;

    float shift = ((float) AgentField[j].PersonalityPressure / (float) AgentField[j].InfluenceNumber) / (float) AgentField[j].Ks;

    AgentField[j].personality += shift;

    if (AgentField[j].personality < -1) {
        AgentField[j].personality = -1;
    }
    if (AgentField[j].personality > 1) {
        AgentField[j].personality = 1;
    }

    //AgentField[j].PersonalityPressure = 0.0;
    //AgentField[j].InfluenceNumber = 1.0;

    return;
}

int main() {

    srand(std::time(0));

	// Image
	const auto aspect_ratio = 1.0 / 1.0;
	const int image_width = 100;
    const int image_height = 100; // static_cast<int>(image_width / aspect_ratio);
    Agent* AgentField = nullptr;
    Agent* AgentField_debug = nullptr;

    for (int fullSim = 0; fullSim < NUM_FULL_SIMS; fullSim++) {

        // Log File

        std::ofstream LogFile;
        std::filesystem::create_directory("./logs/" + std::to_string(fullSim));
        LogFile.open("./logs/" + std::to_string(fullSim) + "/log.txt");

        // Image File

        AgentField = (Agent*)malloc(image_height * image_width * sizeof(Agent));
        AgentField_debug = (Agent*)malloc(image_height * image_width * sizeof(Agent));

        int paranoidAgents = 0.0;
        int calmAgents = 0.0;

        float Kp = 1.0;// random_double() * 5.0;
        float Reach = 1.0;// random_double() * 5.0;
        float Ks = fullSim + 1.0;// random_double() * 10.0;

        float center_X = 50;
        float center_Y = 50;

        for (int j = 0; j < image_height; j++) {
            //std::cout << "\rScanlines remaining: " << j << ' ' << std::flush;
            for (int i = 0; i < image_width; i++) {

                float paranoia = random_double();// *2.0 - 1.0;

                if (std::sqrt(pow(i - center_X, 2) + pow(j - center_Y, 2)) < 10) {
                    paranoia = -1 * paranoia;
                }

                if (paranoia < 0.0) {
                    paranoidAgents++;
                }
                if (paranoia > 0.0) {
                    calmAgents++;
                }

                /*float Kp = random_double() * 5.0;
                float Reach = random_double() * 5.0;
                float Ks = random_double() * 10.0;*/
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

            }
        }

        Agent* KAgentField = NULL;
        cudaMalloc(&KAgentField, image_height * image_width * sizeof(Agent));

        cudaMemcpy(KAgentField, AgentField, image_height * image_width * sizeof(Agent), cudaMemcpyHostToDevice);

        std::filesystem::create_directory("./gifs/" + std::to_string(fullSim));
        std::string fileName = "./gifs/" + std::to_string(fullSim) + "/render.gif";

        GifWriter g;
        GifBegin(&g, fileName.c_str(), image_width, image_height, 100);

        for (int t = -1; t < NUM_SIMULATIONS; t++) {
            if (t >= 0) {
                std::cout << "\rSimulation ID : " << fullSim << "\tSimulation runs remaining : " << NUM_SIMULATIONS - t << ' ' << std::flush;

                cudaMemcpy(KAgentField, AgentField, image_height * image_width * sizeof(Agent), cudaMemcpyHostToDevice);

                ApplyPersonalityPressure << <image_height, image_width >> > (KAgentField, image_height, image_width);
                cudaDeviceSynchronize();

                ShiftPersonality << <image_height, image_width >> > (KAgentField);
                cudaDeviceSynchronize();

                cudaMemcpy(AgentField, KAgentField, image_height * image_width * sizeof(Agent), cudaMemcpyDeviceToHost);
            }

            std::ofstream ImageFile_timestep;
            std::filesystem::create_directory("./images/" + std::to_string(fullSim));
            ImageFile_timestep.open("./images/" + std::to_string(fullSim) + "/" + std::to_string(t + 1) + ".ppm");
            ImageFile_timestep << "P3\n" << image_width << " " << image_height << "\n255\n";

            std::vector<uint8_t> frame(image_width * image_height* 4);

            for (int j = image_height - 1; j >= 0; --j) {
                //std::cout << "\rScanlines remaining: " << j << ' ' << std::flush;
                for (int i = 0; i < image_width; ++i) {

                    Agent agent = AgentField[i];

                    color PixelColor = mix(color(1.0, 0.0, 0.0), color(0.0, 0.0, 1.0), (AgentField[j * image_width + i].personality + 1) / 2.0);

                    color frameColor = write_color(ImageFile_timestep, PixelColor);

                    frame[(j * image_width + i) * 4 + 0] = frameColor.x();
                    frame[(j * image_width + i) * 4 + 1] = frameColor.y();
                    frame[(j * image_width + i) * 4 + 2] = frameColor.z();
                    frame[(j * image_width + i) * 4 + 3] = 255;


                }
            }

            GifWriteFrame(&g, frame.data(), image_width, image_height, 1);
            
            ImageFile_timestep.close();

            for (int j = 0; j < image_height; j++) {
                //std::cout << "\rScanlines remaining: " << j << ' ' << std::flush;
                for (int i = 0; i < image_width; i++) {
                    AgentField[j * image_width + i].PersonalityPressure = 0.0;
                    AgentField[j * image_width + i].InfluenceNumber = 1.0;
                }
            }
        }

        GifEnd(&g);

        //cudaMemcpy(AgentField, KAgentField, image_height * image_width * sizeof(Agent), cudaMemcpyDeviceToHost);

        cudaFree(KAgentField);

        float finalParanoia = 0.0;
        int numAgents = 0;

        for (int j = 0; j < image_height; j++) {
            //std::cout << "\rScanlines remaining: " << j << ' ' << std::flush;
            for (int i = 0; i < image_width; i++) {
                numAgents++;
                finalParanoia += AgentField[j * image_width + i].personality;
            }
        }

        LogFile << "SIM NUMBER - " << fullSim << "\n";
        LogFile << "PARAMETERS - \n";
        LogFile << "\t Kp = " << Kp << "\n";
        LogFile << "\t Reach = " << Reach << "\n";
        LogFile << "\t Ks = " << Ks << "\n";
        LogFile << "PARANOID AGENTS = " << paranoidAgents << "\n";
        LogFile << "CALM AGENTS = " << calmAgents << "\n";
        LogFile << "FINAL TOTAL PARANOIA = " << finalParanoia << "\n";
        LogFile << "FINAL AVG PARANOIA = " << finalParanoia / numAgents << "\n";

        LogFile.close();

        delete AgentField;
    }

	return 0;
}