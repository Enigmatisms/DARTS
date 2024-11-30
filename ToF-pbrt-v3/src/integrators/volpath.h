
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

#ifndef PBRT_INTEGRATORS_VOLPATH_H
#define PBRT_INTEGRATORS_VOLPATH_H

// integrators/volpath.h*
#include "pbrt.h"
#include "integrator.h"
#include "lightdistrib.h"

namespace pbrt {

// VolPathIntegrator Declarations
class VolPathIntegrator : public SamplerIntegrator {
  public:
	// VolPathIntegrator Public Methods
	VolPathIntegrator(int maxDepth, std::shared_ptr<const Camera> camera,
					  std::shared_ptr<Sampler> sampler,
					  const Bounds2i &pixelBounds, Float rrThreshold = 1,
					  const std::string &lightSampleStrategy = "spatial",
					  bool da_guide = false, bool log_time = false, 
					  bool time_sample = false, bool use_truncate = false, 
					  bool fuse_original = false, bool rseed_time = false, 
					  bool sample_flag = false, int nee_nums = 1)
		: SamplerIntegrator(camera, sampler, pixelBounds, log_time, rseed_time, da_guide, sample_flag),
		  maxDepth(maxDepth),
		  nee_nums(nee_nums),
		  nee_div(1.f / (nee_nums + 1.f)),
		  time_sample(time_sample),
		  use_truncate(use_truncate),
		  fuse_original(fuse_original),
		  rrThreshold(rrThreshold),
		  lightSampleStrategy(lightSampleStrategy) {
			printf("volpath initialized with log_time = %d, rseed = %d\n", int(log_time), int(rseed_time));
		  }
	void Preprocess(const Scene &scene, Sampler &sampler);
	Spectrum Li(const RayDifferential &ray, const Scene &scene,
				Sampler &sampler, MemoryArena &arena, int depth, 
				Float origin_remain_t = 0., Float bin_pdf = 1., FilmInfoConstPtr film_info = nullptr, 
				FilmTilePixel* const tfilm = nullptr, bool verbose = false) const;
  private:
	TransientStatus accumulate_radiance(
		const Spectrum& nee_le, FilmInfoConstPtr film_info, 
		FilmTilePixel* const tfilm, Float nee_time, bool transient_valid = true
	) const;

	/**
	 * A connection scheme that utilizes path reuse, taking both
	 * surface interaction and medium interaction into account
	*/
	bool path_reuse_connection(
		const Scene &scene, const Interaction &it, const Ray& ray_wo, Spectrum& inout,
		Sampler& sampler, MemoryArena &arena, Float& nee_time, Float min_remain_t, Float interval
	) const;
  private:
	// VolPathIntegrator Private Data
	const int maxDepth;
	// this field is particularly used in our DARTS volpath to reuse multiple scattering path
	const int nee_nums;				// number of NEE events
	const Float nee_div;			// 1. / (nee_nums + 1.)
	bool time_sample;				// temporal elliptical sampling
	bool use_truncate;				// truncated sampling
	bool fuse_original;				// if true, time-gated  / transient rendering will try to fuse the original method
	const Float rrThreshold;
	const std::string lightSampleStrategy;
	std::unique_ptr<LightDistribution> lightDistribution;

	// This should be mutable since we are going to modify the info block inside `Li`
};

VolPathIntegrator *CreateVolPathIntegrator(
	const ParamSet &params, std::shared_ptr<Sampler> sampler,
	std::shared_ptr<const Camera> camera);

}  // namespace pbrt

#endif  // PBRT_INTEGRATORS_VOLPATH_H
