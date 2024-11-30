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

/**
 * Guided homogeneous medium for Residual Time Sampling
 * @date: 2023-5-29
 */

#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef PBRT_MEDIA_GUIDED_HOMOGENEOUS_H
#define PBRT_MEDIA_GUIDED_HOMOGENEOUS_H

#include <fstream>
#include "stats.h"
#include "media/homogeneous.h"
#include "sse/vectorclass.h"

#define RIS_V8
#ifdef RIS_V8
using VecType = Vec8f;
# define CASCADE_NUM 8
#else 
using VecType = Vec4f;
# define CASCADE_NUM 4
#endif

#define EXPONENTIAL_SAMPLING
#define USE_SSE				// SSE version for Sampling

namespace pbrt {

// GuidedHomogeneousMedium Declarations
template <typename PhaseType = HenyeyGreenstein>
class GuidedHomogeneousMedium : public HomogeneousMedium<PhaseType> {
  public:
    // GuidedHomogeneousMedium Public Methods
    GuidedHomogeneousMedium(const Spectrum &_sigma_a, const Spectrum &_sigma_s, Float _g, bool debug = false)
        : HomogeneousMedium<PhaseType>(_sigma_a, _sigma_s, _g),
          	diffusion_d((_sigma_a + _sigma_s * (1. - _g)).inv() / 3.), inv_sigma_t(sigma_t.inv()),
			inv_diffusion(diffusion_d.inv()), debug_output(debug)
	{
#ifdef GUIDE_HOMO_DEBUG
			if (debug) {
        		line_cnt = 0;
				viz_output.open("./viz_output.json", std::ios::out);
				viz_output << "{" << std::endl;
				viz_output << "\t\"values\":[" << std::endl;
			}
#endif 
    }

	~GuidedHomogeneousMedium() {
#ifdef GUIDE_HOMO_DEBUG
		if (debug_output && viz_output.is_open()) {
			viz_output << "\t]" << std::endl << "}" << std::endl;
			viz_output.close();
		}
#endif 
	}

    int get_max_index() const {return sigma_t.MaxComponentIndex();}

	VecType full_diffusion_sse(const VecType& length2, const VecType& res_ts, Float inv_d, Float sigma_a) const;

	/**
	 * Decide whether the current event is medium interaction
	 * TODO: tr_das should be able to correctly reflect the value of in-scattered radiance
	 * @return if True: medium interaction, otherwise surface interaction
	*/
	bool sample_interaction_event(const Ray& ray, Float sample, Float sig_t) const {
		ProfilePhase _(Prof::EventSampling);
		Float dist = -std::log(1 - sample) / sig_t;
	    Float t = std::min(dist / ray.d.Length(), ray.tMax);
	    return t < ray.tMax;
	}

	/**
	 * @brief resampled importance sampling for DA-based path guiding (SSE implementation)
	 * @return weight_sum 
	 */
	Spectrum importance_resample_sse(
		const Ray& ray, Sampler &sampler, MemoryArena &arena, 
		MediumInteraction *mi, Float diff_len2, Float ray_len,
		Float truncation, Float dot_val, Float target_t, int channel_idx = 0
	) const;

    /**
     * remaining_time: time to sample from the current vertex to the target vertex
     * guide_info: information about the vertex and its arrival time
     */
    Spectrum Sample(const Ray &ray, Sampler &sampler, MemoryArena &arena, 
		MediumInteraction *mi, Float remaining_time,
		GuideInfoConstPtr guide_info, 
        Float truncation = 0.f, bool guiding = false
    ) const;

	// This function is used when we DO use UniformSampleOneLight in UDPT (volpath)
	Float elliptical_sample(
		const Ray& ray, Sampler &sampler, MediumInteraction* deter_mi, 
      	MemoryArena &arena, Spectrum& connection_tr, EllipseConstPtr ell_info
	) const;

  private:
	using HomogeneousMedium<PhaseType>::sigma_a;
	using HomogeneousMedium<PhaseType>::sigma_s;
	using HomogeneousMedium<PhaseType>::sigma_t;
	using HomogeneousMedium<PhaseType>::g;
    // GuidedHomogeneousMedium Private Data

	// Precompute some information
	const Spectrum diffusion_d;
	const Spectrum inv_sigma_t;
	const Spectrum inv_diffusion;

	static constexpr int cascade_num = CASCADE_NUM;
	static constexpr Float inv_cascasde = 1.f / CASCADE_NUM;

	bool debug_output;
#ifdef GUIDE_HOMO_DEBUG
	mutable std::ofstream viz_output;
	mutable size_t line_cnt;
#endif
};

template class GuidedHomogeneousMedium<HenyeyGreenstein>;
template class GuidedHomogeneousMedium<EllipticalPhase>;

}  // namespace pbrt

#endif  // PBRT_MEDIA_GUIDED_HOMOGENEOUS_H
