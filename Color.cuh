#pragma once

#ifndef COLOR_H
#define COLOR_H

#include "Vec3.cuh"

#include <fstream>

void write_color(std::ofstream& out, color pixel_color) {
	auto r = pixel_color.x();
	auto g = pixel_color.y();
	auto b = pixel_color.z();

	 //Divide the color by the number of samples
	auto scale = 1.0;
	r = sqrt(r * scale);
	g = sqrt(g * scale);
	b = sqrt(b * scale);

	// Write the translate [0,255] value of each color component
	out << static_cast<int>(256 * clamp(r, 0, 0.999)) << ' '
		<< static_cast<int>(256 * clamp(g, 0, 0.999)) << ' '
		<< static_cast<int>(256 * clamp(b, 0, 0.999)) << '\n';
}

#endif
