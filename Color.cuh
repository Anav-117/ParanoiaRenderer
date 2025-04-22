#pragma once

#ifndef COLOR_H
#define COLOR_H

#include "Vec3.cuh"

#include <fstream>

color write_color(std::ofstream& out, color pixel_color) {
	float r = pixel_color.x();
	float g = pixel_color.y();
	float b = pixel_color.z();

	int degub_r = static_cast<int>(256 * clamp(r, 0, 0.999));
	int degub_g = static_cast<int>(256 * clamp(g, 0, 0.999));
	int degub_b = static_cast<int>(256 * clamp(b, 0, 0.999));

	if (b < 0.001) {
		std::cout << "";
	}

	// Write the translate [0,255] value of each color component
	out << static_cast<int>(256 * clamp(r, 0, 0.999)) << ' '
		<< static_cast<int>(256 * clamp(g, 0, 0.999)) << ' '
		<< static_cast<int>(256 * clamp(b, 0, 0.999)) << '\n';

	return color(static_cast<int>(256 * clamp(r, 0, 0.999)), static_cast<int>(256 * clamp(g, 0, 0.999)), static_cast<int>(256 * clamp(b, 0, 0.999)));
}

#endif
