# pragma once

#ifndef VEC3_DEF
#define Vec3_DEF

#include "Defines.cuh"
#include "Utils.cuh"
#include <math.h>
#include <iostream>
#include <stdio.h>

class vec3 {
public:
	HOD vec3() : e{ 0, 0, 0 } {}
	HOD vec3(double e0, double e1, double e2) : e{e0, e1, e2} {}

	HOD double x() const { return e[0]; }
	HOD double y() const { return e[1]; }
	HOD double z() const { return e[2]; }

	HOD double r() const { return e[0]; }
	HOD double g() const { return e[1]; }
	HOD double b() const { return e[2]; }

	HOD vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }
	HOD double operator[](int i) const { return e[i]; }
	HOD double& operator[](int i) { return e[i]; }

	HOD vec3& operator+=(const vec3& v) {
		e[0] += v.e[0];
		e[1] += v.e[1];
		e[2] += v.e[2];

		return *this;
	}

	HOD vec3& operator*=(const double t) {
		e[0] *= t;
		e[1] *= t;
		e[2] *= t;

		return *this;
	}

	HOD vec3& operator/=(const double t) {
		return *this *= 1 / t;
	}

	HOD double length_squared() const {
		return (e[0] * e[0]) + (e[1] * e[1]) + (e[2] * e[2]);
	}

	HOD double length() const {
		return sqrtf(length_squared());
	}

	HO inline static vec3 random() {
		return vec3(random_double(), random_double(), random_double());
	}

	HO inline static vec3 random(double min, double max) {
		return vec3(random_double(min, max), random_double(min, max), random_double(min, max));
	}

	DEV inline static vec3 random_zero_centered(curandState localState) {
		return vec3(random_double_unit(localState), random_double_unit(localState), random_double_unit(localState));
	}

	HOD bool near_zero() const {
		const auto s = 1e-8;
		return (fabs(e[0]) < s) && (fabs(e[1]) < s) && (fabs(e[2]) < s);
	}

public:
	double e[3];
};

using point3 = vec3; // 3D Point
using color = vec3; // RGB Color

HO inline std::ostream& operator<<(std::ostream& out, const vec3& v) {
	return out << v.e[0] << " : " << v.e[1] << " : " << v.e[2];
}

HOD inline vec3 operator+(const vec3& u, const vec3& v) {
	return vec3(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
}

HOD inline vec3 operator-(const vec3& u, const vec3& v) {
	return vec3(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}

HOD inline vec3 operator*(const vec3& u, const vec3& v) {
	return vec3(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
}

HOD inline vec3 operator*(double t, const vec3& v) {
	return vec3(t * v.e[0], t * v.e[1], t * v.e[2]);
}

HOD inline vec3 operator*(const vec3& v, double t) {
	return t * v;
}

HOD inline vec3 operator/(vec3 v, double t) {
	return (1 / t) * v;
}

HOD inline bool operator==(vec3 u, vec3 v) {
	return (u.e[0] == v.e[0]) && (u.e[1] == v.e[1]) && (u.e[2] == v.e[2]);
}

HOD inline double dot(const vec3& u, const vec3& v) {
	return u.e[0] * v.e[0]
		+ u.e[1] * v.e[1]
		+ u.e[2] * v.e[2];
}

HOD inline vec3 cross(const vec3& u, const vec3& v) {
	return vec3(u.e[1] * v.e[2] - u.e[2] * v.e[1],
		u.e[2] * v.e[0] - u.e[0] * v.e[2],
		u.e[0] * v.e[1] - u.e[1] * v.e[0]);
}

HOD inline vec3 unit_vector(vec3 v) {
	return v / v.length();
}

HO vec3 random_in_unit_sphere() {
	return vec3::random(-1, 1);
	/*while (true) {
		auto p = vec3::random(-1, 1);
		if (p.length_squared() >= 1) continue;
		return p;
	}*/
}

HO inline vec3 random_unit_vector() {
	return unit_vector(random_in_unit_sphere());
}

HOD vec3 reflect(const vec3& v, const vec3& n) {
	return v - 2 * dot(v, n) * n;
}

HOD vec3 refract(const vec3& v, const vec3& n, double etai_over_etat) {
	auto cos_theta = fmin(dot(-v, n), 1.0);
	vec3 r_out_perp = etai_over_etat * (v + cos_theta * n);
	vec3 r_out_par = -sqrt(fabs(1.0 - r_out_perp.length_squared())) * n;
	return r_out_perp + r_out_par;
}

HO vec3 random_in_unit_disk() {
	while (true) {
		auto p = vec3(random_double(-1, 1), random_double(-1, 1), 0);
		if (p.length_squared() >= 1) continue;
		return p;
	}
}

HO color mix(color color1, color color2, float t) {
	color mixedColor = color(0, 0, 0);
	mixedColor.e[0] = (1.0 - t) * color1.x() + t * color2.x();
	mixedColor.e[1] = (1.0 - t) * color1.y() + t * color2.y();
	mixedColor.e[2] = (1.0 - t) * color1.z() + t * color2.z();

	return mixedColor;
}

#endif