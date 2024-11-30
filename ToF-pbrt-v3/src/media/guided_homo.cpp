
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


// media/homogeneous.cpp*
#include "media/guided_homo.h"
#include "sampler.h"
#include "sampling.h"
#include "light.h"
#include "interaction.h"
#include "paramset.h"

#include "media/low_discrep.h"
#include "sse/vectormath_exp.h" 
#include "sse/vectormath_trig.h" 
#include <numeric>

namespace pbrt {

template <typename PhaseType>
VecType GuidedHomogeneousMedium<PhaseType>::full_diffusion_sse(const VecType& length2, const VecType& res_ts, Float inv_d, Float sigma_a) const {
	// SSE acceleration
	ProfilePhase _(Prof::DiffusionCalculation);
	VecType denom_inv = 0.25f * inv_d / res_ts;
	VecType coeff = denom_inv * sqrt(denom_inv);
	return coeff * exp(-length2 * denom_inv - sigma_a * res_ts);
}

	// We might use AVX2-SSE to further accelerate
template <typename PhaseType>
Spectrum GuidedHomogeneousMedium<PhaseType>::importance_resample_sse(
	const Ray& ray, Sampler &sampler, MemoryArena &arena, 
	MediumInteraction *mi, Float diff_len2, Float ray_len,
	Float truncation, Float dot_val, Float target_t, int channel
) const {
	ProfilePhase _(Prof::ImportanceResample);
	const Float proj_len = dot_val / ray_len;
	const Float sqr_dist = diff_len2 - proj_len * proj_len;
	Float proposals[cascade_num], da_vals[cascade_num], weights[cascade_num];

	Float sig_t = sigma_t[channel], tr_alpha = 1.f - expf(-sig_t * truncation),
		inv_sig_t = inv_sigma_t[channel], prop_coeff = sig_t / tr_alpha;
	// For truncation = ray.tMax, it is possible that we will sample inapproriate values
	#ifdef EXPONENTIAL_SAMPLING
	{
		ProfilePhase _ip(Prof::ImportanceProposal);
		Float inv_d = inv_diffusion[channel], sig_a = sigma_a[channel];
		auto rands = sampler.Get2D();
		int arr_idx = std::min((int)(rands[0] * 32.f), 31);
		
		VecType samples;
		samples.load(low_discrep[arr_idx]);
		samples += rands[1];
		samples = select(samples >= 1, samples - 1, samples);			// float mod

		const VecType dists = -log(1 - samples * tr_alpha) * inv_sig_t;
		const VecType trans = exp(-sig_t * dists), prop_pdf = prop_coeff * trans;
		const VecType dlens = proj_len - dists, length2 = sqr_dist + dlens * dlens;
		const VecType res_ts = target_t - dists;
		const auto valid_flags = (res_ts > 0) && ((res_ts * res_ts) > length2);
		const VecType das   = select(valid_flags, full_diffusion_sse(length2, res_ts, inv_d, sig_a) * trans, VecType(1));
		const VecType ws    = select(valid_flags, das / prop_pdf, VecType(0));
		ws.store(weights);
		das.store(da_vals);
		dists.store(proposals);
	}
	#else			// this else branch is for equiangular sampling proposal
	{
		ProfilePhase _ip(Prof::ImportanceProposal);
		// Try equiangular sampling
		Float inv_d = inv_diffusion[channel], sig_a = sigma_a[channel];
		const Float proj_dist = std::sqrt(sqr_dist), foci_dist = std::sqrt(diff_len2);
		Float theta_a = std::acos(proj_len / foci_dist) - PiOver2,
			  delta_theta = std::atan2(ray.tMax - proj_len, proj_dist) - theta_a;
		// We can try equiangular sampling (even with MIS)
		Float sample_vs[cascade_num];
		VecType samples;
		for (int i = 0; i < cascade_num; i++) {
			sample_vs[i] = sampler.Get1D();
		}
		samples.load(sample_vs);

		const VecType ts = proj_dist * tan(theta_a + samples * delta_theta);
		const VecType dists = ts + proj_len;
		const VecType trans = exp(-sig_t * dists);
		const VecType ts2 = ts * ts, length2 = sqr_dist + ts2;
		const VecType prop_pdf = proj_dist / delta_theta / (sqr_dist + ts2);
		const VecType res_ts = target_t - dists;
		const auto valid_flags = (res_ts > 0) && ((res_ts * res_ts) > length2);
		const VecType das   = select(valid_flags, full_diffusion_sse(length2, res_ts, inv_d, sig_a) * trans, VecType(1));
		const VecType ws    = select(valid_flags, das / prop_pdf, VecType(0));
		ws.store(weights);
		das.store(da_vals);
		dists.store(proposals);
	}
	#endif
	// if all samples have zero proba (almost impossible), we will use the first sample
	std::partial_sum(weights, weights + cascade_num, weights);
	Float weight_sum = weights[cascade_num - 1];
	int selected_id = std::lower_bound(weights, weights + cascade_num, sampler.Get1D() * weight_sum) - weights;
	
	Float samp_distance = proposals[selected_id], mean_weight = inv_cascasde * weight_sum;
	*mi = HomogeneousMedium<PhaseType>::allocate_it(ray, arena, samp_distance / ray_len);
	// RIS one sample evaluation
	// formula: sigma_s * Tr / target_da_pdf * weight / (proba of scattering event)
	return sigma_s * Exp(-sigma_t * samp_distance) * (mean_weight / da_vals[selected_id]);
}

// GuidedHomogeneousMedium Method Definitions
template <typename PhaseType>
Spectrum GuidedHomogeneousMedium<PhaseType>::Sample(const Ray &ray, Sampler &sampler,
	MemoryArena &arena, MediumInteraction *mi, Float remaining_time,
	GuideInfoConstPtr guide_info,
	Float truncation, bool guiding
) const {
    ProfilePhase _(Prof::MediumSample);
	// Equiangular sampling is not so useful / MIS can not be done since we are using RIS
	// Note that the only difference between DARTS and original method is the medium used

	if (guiding) {
		Float ray_len = ray.d.Length(), pdf = 1.0;
		int channel = std::min((int)(sampler.Get1D() * Spectrum::nSamples),
                           Spectrum::nSamples - 1);

		// let's try sampling event every time but with trucated PDF
		if (truncation < 1e-5) {				
			// if truncation is given, it would mean that the event is medium event
			Float sig_t = sigma_t[channel];
			bool medium_interaction = sample_interaction_event(ray, sampler.Get1D(), sig_t);
			Float actual_d = std::min(ray.tMax, MaxFloat) * ray_len;
			if (!medium_interaction) {			// surface interaction
				Spectrum Tr = Exp(-sigma_t * actual_d);
				pdf = Tr[channel];
				if (pdf == 0.) return Spectrum();
				return Tr / pdf;
			}
			// then the truncation value should be set a valid value 
			truncation = ray.tMax;
			pdf = 1.f - expf(-sig_t * actual_d);		// proba for having medium event (not 1 if truncation is zero, since sampling is used)
		}

		Vector3f diff_vec = guide_info->vertex_p - ray.o; 
		Float dot_val = Dot(diff_vec, ray.d);
		// Here during proposal phase we can use equiangular sampling or MIS equiangular sampling
		Spectrum beta = importance_resample_sse(ray, sampler, arena, mi, diff_vec.LengthSquared(), ray_len, truncation, dot_val, remaining_time, channel);
		return beta / pdf;
	}
	// We choose not to use guiding or we are sampling traditional sampling
	Spectrum result = HomogeneousMedium<PhaseType>::Sample(ray, sampler, arena, mi, 0, nullptr, 0, truncation);
	return result;
}

template <typename PhaseType>
Float GuidedHomogeneousMedium<PhaseType>::elliptical_sample(
	const Ray& ray, Sampler &sampler, MediumInteraction* deter_mi, 
	MemoryArena &arena, Spectrum& connection_tr, EllipseConstPtr ell_info
) const {
	// this might be improved
    ProfilePhase _(Prof::EllipticalSample);
	// Since elliptical sample will utilize UniformSampleOneLight
	// So we don't need to calculate phase function and the transmittance for the second segment
	const Float dot_val = Dot(ray.d, ell_info->to_emitter) / ray.d.Length(),
		  	    dot_foci = dot_val * ell_info->foci_dist;
	Float target_time = ell_info->target_t;
	Float t_m_dotf = target_time - dot_foci;
	Float sample_dist = ell_info->half_power_diff / t_m_dotf;
	if (sample_dist >= ray.tMax) {		// the vertex on the equi-time ellipse is not in the scattering medium
		// unable to sample target time in the medium, therefore we can opt for surface interaction
		return 0;
	}
	Float max_dist = ell_info->max_power_diff / (ell_info->max_t - dot_foci);	// max time ellipse
	*deter_mi = HomogeneousMedium<PhaseType>::allocate_it(ray, arena, sample_dist);
	connection_tr = Exp(-sigma_t * sample_dist) * sigma_s;
	// this is measure conversion Jacobian
	// Erroneous thing could occur: sample_dist is creater than target time
	connection_tr *= (target_time - sample_dist) * ell_info->inv_sampling_pdf / t_m_dotf;
	// Only when the whole time bin is inside the scattering medium, will we use truncated sampling
	return (max_dist >= ray.tMax) ? 0 : max_dist;
}

}  // namespace pbrt
