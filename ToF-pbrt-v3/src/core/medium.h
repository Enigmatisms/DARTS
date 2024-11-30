
/*
	pbrt source code is Copyright(c) 1998-2016
						Matt Pharr, Greg Humphreys, and Wenzel Jakob.

	This file is part of pbrt.

	Redistribution and use in source and binary forms, with or without
	modification, are permitted provided that the following conditions are
	met:

	- Redistributions of source code must retain the above copyright
	  notice, this list of conditions and the following disclaimer.

	- Redistributions in binary form must reproduce the above copyright
	  notice, this list of conditions and the following disclaimer in the
	  documentation and/or other materials provided with the distribution.

	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
	IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
	TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
	PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
	HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
	SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
	LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
	DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
	THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
	(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
	OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

 */

#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef PBRT_CORE_MEDIUM_H
#define PBRT_CORE_MEDIUM_H

// core/medium.h*
#include "pbrt.h"
#include "geometry.h"
#include "spectrum.h"
#include <memory>

namespace pbrt {

struct GuidedSamplingInfo {
	/**
	 * We store vertex pointer and the arrival time of the vertex
	 * The vertex is by the fault the first vertex from the emitter that enters the medium
	 */
	GuidedSamplingInfo(): time(Float(-1)) {}
	GuidedSamplingInfo(Point3f&& vp, Float time, Float glob_min_time = -1): 
			time(time), glob_min_time(glob_min_time), vertex_p(vp) {}
	
	// Using pointer could be too dangerous, since the pointer is pointing to a vector.
	Float time;           // vertex time of arrival
	Float glob_min_time;  // global shortest path time
	Float time_range;	  // max_time - global_min_time
	Point3f vertex_p;     // vertex position
};

using GuideInfoConstPtr = const GuidedSamplingInfo* const;

class EllipseInfo {
public:
	/**
	 * @brief Elliptical sample is the extension for the method 
	 * ellipsoidal path connection in 2019 MitsubaToF
	 * @param full_init: sometimes, the EllipseInfo is used for EPS and does not need to calculate all terms
	 */
	EllipseInfo (
		GuideInfoConstPtr sampling_info, const Point3f& p, Float min_time, 
		Float p_time, Float interval, Sampler &sampler, bool full_init = true);
	
	Float half_power_diff;						// precomputed 0.5 * (T^2 - d^2)
	Float max_power_diff;						// To calculate max_t
	Float min_t, max_t, target_t, foci_dist;   	// max_T for this time bin, T and d
	Float inv_sampling_pdf;						// PDF for sampling the current T value (reciprocal)
	Float T_div_D;								// target_t / foci_dist
	Vector3f to_emitter;						// direction to the emitter (normalized), length is foci_dist

	bool valid_target() const {
		return target_t > 0;
	}

	bool direct_connect() const {
		// being extremely close to the target time
		return fabsf(foci_dist - target_t) < 1e-5;
	}

	bool in_time_range() const {
		return false;
	}

	/// @brief  whether ell_info is useful for EPS
	bool useful() const {
		return valid_target() && !direct_connect();
	}

	bool ill_defined() const {		// in case we get some ill-defined values
		return fabsf(T_div_D - 1.f) < 1e-3;
	}
};

using EllipseConstPtr = const EllipseInfo* const;

// Media Declarations
class PhaseFunction {
public:
	// PhaseFunction Interface
	virtual ~PhaseFunction();
	virtual Float p(const Vector3f &wo, const Vector3f &wi, EllipseConstPtr ell_info = nullptr) const = 0;
	virtual Float Sample_p(const Vector3f &wo, Vector3f *wi,
						   const Point2f &u, EllipseConstPtr ell_info = nullptr, Float extra_sample = 0) const = 0;
	virtual std::string ToString() const = 0;
	virtual Float get_g() const = 0;
public:
	const bool symmetric = true;
};


inline std::ostream &operator<<(std::ostream &os, const PhaseFunction &p) {
	os << p.ToString();
	return os;
}

bool GetMediumScatteringProperties(const std::string &name, Spectrum *sigma_a,
								   Spectrum *sigma_s);

// Media Inline Functions
inline Float PhaseHG(Float cosTheta, Float g) {
	Float denom = 1 + g * g + 2 * g * cosTheta;
	return Inv4Pi * (1 - g * g) / (denom * std::sqrt(denom));
}

// Medium Declarations
class Medium {
  public:
	// Medium Interface
	virtual ~Medium() {}
	virtual Spectrum Tr(const Ray &ray, Sampler &sampler) const = 0;
	virtual Spectrum Sample(const Ray &ray, Sampler &sampler,
							MemoryArena &arena, MediumInteraction *mi, 
							Float full_path_time = 0.,
							GuideInfoConstPtr guide_info = nullptr,
							Float truncation = 0.f, bool guiding = false
	) const = 0;
	virtual Float elliptical_sample(
		const Ray& ray, Sampler &sampler, MediumInteraction* deter_mi, 
	  	MemoryArena &arena, Spectrum& connection_tr, EllipseConstPtr ell_info
	) const {return 0;}

	virtual PhaseFunction* get_phase(MemoryArena &arena) const {return nullptr;}
	virtual MediumInteraction allocate_it(const Ray& ray, MemoryArena &arena, Float d) const = 0;

	virtual Spectrum equiangular_sample(const Ray &ray, Sampler &sampler, MemoryArena &arena, MediumInteraction *mi, 
		const Point3f& target_p
	) const = 0;

	virtual Spectrum uniform_sample(const Ray &ray, Sampler &sampler, MemoryArena &arena, MediumInteraction *mi, 
		const Point3f& target_p
	) const = 0;
};

// HenyeyGreenstein Declarations
class HenyeyGreenstein : public PhaseFunction {
  public:
	// HenyeyGreenstein Public Methods
	HenyeyGreenstein(Float g) : g(g) {}
	Float p(const Vector3f &wo, const Vector3f &wi, EllipseConstPtr ell_info = nullptr) const;
	Float Sample_p(const Vector3f &wo, Vector3f *wi,
				   const Point2f &sample, EllipseConstPtr ell_info = nullptr, Float extra_sample = 0) const;

	std::string ToString() const {
		return StringPrintf("[ HenyeyGreenstein g: %f ]", g);
	}

	Float get_g() const {return this->g;}

  protected:
	const Float g;
};

// MediumInterface Declarations
struct MediumInterface {
	MediumInterface() : inside(nullptr), outside(nullptr) {}
	// MediumInterface Public Methods
	MediumInterface(const Medium *medium) : inside(medium), outside(medium) {}
	MediumInterface(const Medium *inside, const Medium *outside)
		: inside(inside), outside(outside) {}
	bool IsMediumTransition() const { return inside != outside; }
	const Medium *inside, *outside;
};

/**
 * Elliptical Phase Samlping for efficient elliptical directional sampling
 * EPS fuses the original sampling results and the elliptical results using MIS 
 * The proba for choosnig between different methods is related to |g| |cosine term| and |T/d|
 * and should not be used when g > -0.5 / cosine term > 0 and T / d > 3
*/
class EllipticalPhase: public HenyeyGreenstein {
using HenyeyGreenstein::g;
public:
	EllipticalPhase(Float g, Float alpha = 0.5): HenyeyGreenstein(g), alpha(alpha), g2(g * g) {}

	Float Sample_p(const Vector3f &wo, Vector3f *wi,
				   const Point2f &sample, EllipseConstPtr ell_info = nullptr, Float extra_sample = 0) const;

	/**
	 * @brief get value for C2 and normalization factor Z 
	 * @param ratio_d_T: d / T
	 */
	std::pair<Float, Float> get_c2_z(Float k1, Float k2, Float ratio_d_T, Float& p1_nn) const;

	/**
	 * @brief get the probability of using EPS  
	 * @param ratio_d_T: d / T
	 */
	Float get_ell_proba(Float ratio_d_T, Float cos_io) const {
		if (cos_io >= 0 || g > -0.4 || ratio_d_T < 0.25) {
			return 0.0;			// only when backwards
		}
		return 0.5 * (cos_io - 1) * g / (alpha / ratio_d_T - g);
	}

	/**
	 * @brief get the slope for the piecewise linear model
	 * @param r: r = d / T
	*/
	template<bool first_term>
	Float get_k(Float r) const {
		if constexpr (first_term) {
			return 2.f * r * r / (r + 1);
		} else {
			return -2.f * r * r  / (1 - r);
		}
	}

	/// @brief non-normalized CDF 
	template<bool part_1>
	Float cdf_nnorm(Float k, Float cos_sample) const {
		Float a = 2 * g * k;
		Float input_cos = 0., second_term = 0.;
		if constexpr (part_1) {
			input_cos = k * cos_sample + k - 1;
			second_term = (1 - g) / a;
		} else {
			input_cos = -(1 + k - k * cos_sample);
		}
		Float first_term = (1 - g2) / a / sqrtf(1 + g2 - 2 * g * input_cos);
		return first_term - second_term;
	}

	std::string ToString() const {
		return StringPrintf("[ Elliptical isotropic medium ]");
	}
	
	/// @brief inverse CDF sampling for the cosine value
	Float inverse_cdf_sample(EllipseConstPtr ell_info, Float sample, Float& pdf) const;

	/**
	 * @brief piece-wise linear inverse CDF mapping
	 * @param k1k2: slope for the piece wise linear func
	 * @param Z: normalization factor Z
	 * 
	*/
	template<bool is_part1>
	Float inverse_map(Float k1, Float k2, Float Z, Float C2, Float rd_sample, Float& pdf) const {
		Float f2 = 1 - g2, a = 2 * g, nominator = 0.0;
		if constexpr (is_part1) {		// piece-wise linear first part
			nominator = 1 + g2 + a * (1 - k1);
			a *= k1;
			f2 /= a * Z * rd_sample + 1 - g;
		} else {					// piece-wise linear second part
			nominator = 1 + g2 + a * (1 + k2);
			a *= k2;				
			f2 /= a * (Z * rd_sample - C2);
		}
		f2 *= f2;
		Float result = (nominator - f2) / a;
		// ATTENTION: cosine input of phase function should be negated.
		if constexpr (is_part1)
			pdf = PhaseHG(1 - k1 * (result + 1), g);
		else
			pdf = PhaseHG(1 - k2 * (result - 1), g);
		return result;
	}
public:
	const bool symmetric = false;
	const Float alpha;
	const Float g2;
};

};  // namespace pbrt

#endif  // PBRT_CORE_MEDIUM_H
