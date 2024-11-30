
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
#include "media/homogeneous.h"
#include "sampler.h"
#include "interaction.h"
#include "paramset.h"
#include "stats.h"

namespace pbrt {

// HomogeneousMedium Method Definitions
template <typename PhaseType>
Spectrum HomogeneousMedium<PhaseType>::Tr(const Ray &ray, Sampler &sampler) const {
    ProfilePhase _(Prof::MediumTr);
    return Exp(-sigma_t * std::min(ray.tMax * ray.d.Length(), MaxFloat));
}

template <typename PhaseType>
Spectrum HomogeneousMedium<PhaseType>::sample_truncated(
	const Ray &ray, Sampler &sampler, 
	MemoryArena &arena, MediumInteraction *mi, Float truncation
) const {
	// sample truncated: we already know that the sample is in the medium
	int channel = std::min((int)(sampler.Get1D() * Spectrum::nSamples),
                           Spectrum::nSamples - 1);
	Float sig_t = sigma_t[channel], tr_alpha = 1.f - expf(-sig_t * truncation);
    Float dist = -std::log(1.f - sampler.Get1D() * tr_alpha) / sig_t;
    Float d_length = ray.d.Length(), normalized_len = dist / d_length;
	*mi = allocate_it(ray, arena, normalized_len); 

    // Compute the transmittance and sampling density
    // Note that we can not modify the sigma_t here, since it's calculating contribution
    Spectrum Tr = Exp(-sigma_t * dist);

    // Return weighting factor for scattering from homogeneous medium
    Float pdf = sig_t * Tr[channel] / tr_alpha;     // truncated PDF
    return Tr * sigma_s / pdf;
}

template <typename PhaseType>
Spectrum HomogeneousMedium<PhaseType>::Sample(
  	const Ray &ray, Sampler &sampler, MemoryArena &arena, MediumInteraction *mi, 
	Float, GuideInfoConstPtr, Float truncation, bool
) const {
    ProfilePhase _(Prof::MediumSample);
	if (truncation > 1e-5)
		return sample_truncated(ray, sampler, arena, mi, truncation);
    // Sample a channel and distance along the ray
    int channel = std::min((int)(sampler.Get1D() * Spectrum::nSamples),
                           Spectrum::nSamples - 1);
    Float dist = -std::log(1 - sampler.Get1D()) / sigma_t[channel];
    Float d_length = ray.d.Length(), t = std::min(dist / d_length, ray.tMax);
    bool sampledMedium = t < ray.tMax;
    if (sampledMedium)
        *mi = allocate_it(ray, arena, t);

    // Compute the transmittance and sampling density
    // Note that we can not modify the sigma_t here, since it's calculating contribution
    Float actual_d = std::min(t, MaxFloat) * d_length;
    Spectrum Tr = Exp(-sigma_t * actual_d);

    // Return weighting factor for scattering from homogeneous medium
    Spectrum density = sampledMedium ? (sigma_t * Tr) : Tr;
    Float pdf = 0;
    
    for (int i = 0; i < Spectrum::nSamples; ++i) pdf += density[i];
    pdf *= 1 / (Float)Spectrum::nSamples;
    if (pdf == 0) {
        CHECK(Tr.IsBlack());
        pdf = 1;
    }
    return sampledMedium ? (Tr * sigma_s / pdf) : (Tr / pdf);
}

template <typename PhaseType>
Spectrum HomogeneousMedium<PhaseType>::equiangular_sample(
    const Ray &ray, Sampler &sampler, MemoryArena &arena, MediumInteraction *mi, const Point3f& target_p
) const {
    ProfilePhase _(Prof::MediumSample);
    // Sample a channel and distance along the ray
    int channel = std::min((int)(sampler.Get1D() * Spectrum::nSamples),
                           Spectrum::nSamples - 1);
    Float dist = -std::log(1 - sampler.Get1D()) / sigma_t[channel];
    Float d_length = ray.d.Length(), t = std::min(dist / d_length, ray.tMax);
    bool sampledMedium = t < ray.tMax;

    Float theta_a = 0, theta_b = 0, D = 0, sampled_t = 0, sample_pdf = 1.0, sample = 0;
    if (sampledMedium) {
        Vector3f diff_vec = target_p - ray.o;
        Float dot_val = Dot(diff_vec, ray.d);
        Float proj_len = dot_val / d_length;
        D = (ray(proj_len) - target_p).Length();
        sample = sampler.Get1D();

        Float near = 1e-5 - proj_len, far = ray.tMax - proj_len - 1e-5;
        if (D > 1e-5) {
		    theta_a = atan2f(near, D);
            theta_b = atan2f(far, D);
            sampled_t = D * tanf(theta_a * (1 - sample) + sample * theta_b);
            sample_pdf = D / fabsf(theta_b - theta_a) / (D * D + sampled_t * sampled_t);
        } else {
            sampled_t = near * far / (theta_a * (1 - sample) + sample * theta_b);
            sample_pdf = near * far / (far - near) / (sampled_t * sampled_t);
        }
        t = (sampled_t + proj_len) / d_length;
        *mi = allocate_it(ray, arena, t);
        
    }
    // Compute the transmittance and sampling density
    // Note that we can not modify the sigma_t here, since it's calculating contribution
    Float actual_d = std::min(t, MaxFloat) * d_length;
    Float wall_pdf = expf(-sigma_t[channel] * std::min(ray.tMax, MaxFloat) * d_length);
    Spectrum Tr = Exp(-sigma_t * actual_d);
    
    Float pdf = (1.f - wall_pdf) * sample_pdf;
    if (sampledMedium) {
        if (pdf < 1e-8) {
            printf("Numerical error, fall back to exponential sampling. PDF: %f\n", pdf);
            return Sample(ray, sampler, arena, mi);
        }
    } else {
        if (wall_pdf < 1e-8) {
            printf("Numerical error, fall back to exponential sampling. Wall PDF: %f\n", wall_pdf);
            return Sample(ray, sampler, arena, mi);
        }
    }
    return sampledMedium ? (Tr * sigma_s / pdf) : (Tr / wall_pdf);
}

template <typename PhaseType>
Spectrum HomogeneousMedium<PhaseType>::uniform_sample(
    const Ray &ray, Sampler &sampler, MemoryArena &arena, MediumInteraction *mi, const Point3f& target_p
) const {
    ProfilePhase _(Prof::MediumSample);
    // Sample a channel and distance along the ray
    int channel = std::min((int)(sampler.Get1D() * Spectrum::nSamples),
                           Spectrum::nSamples - 1);
    Float dist = -std::log(1 - sampler.Get1D()) / sigma_t[channel], sample_pdf = 1.0;
    Float d_length = ray.d.Length(), t = std::min(dist / d_length, ray.tMax);
    bool sampledMedium = (ray.tMax > 0) && (t < ray.tMax);
    
    if (sampledMedium) {
        t = sampler.Get1D() * ray.tMax;
        *mi = allocate_it(ray, arena, t);
        sample_pdf = 1.f / ray.tMax;
    }
    // Compute the transmittance and sampling density
    // Note that we can not modify the sigma_t here, since it's calculating contribution
    Float actual_d = std::min(t, MaxFloat) * d_length;
    Float wall_pdf = expf(-sigma_t[channel] * std::min(ray.tMax, MaxFloat) * d_length);
    Spectrum Tr = Exp(-sigma_t * actual_d);
    
    Float pdf = (1.f - wall_pdf) * sample_pdf;
    if (sampledMedium) {
        if (pdf < 1e-8) {
            printf("Numerical error, fall back to exponential sampling. PDF: %f\n", pdf);
            return Sample(ray, sampler, arena, mi);
        }
    } else {
        if (wall_pdf < 1e-8) {
            printf("Numerical error, fall back to exponential sampling. Wall PDF: %f\n", wall_pdf);
            return Sample(ray, sampler, arena, mi);
        }
    }
    return sampledMedium ? (Tr * sigma_s / pdf) : (Tr / wall_pdf);
}

template <typename PhaseType>
MediumInteraction HomogeneousMedium<PhaseType>::allocate_it(const Ray& ray, MemoryArena &arena, Float d) const {
    return MediumInteraction(ray(d), -ray.d, ray.time + d, this, 
                                        ARENA_ALLOC(arena, PhaseType)(g));
}

template <typename PhaseType>
PhaseFunction* HomogeneousMedium<PhaseType>::get_phase(MemoryArena &arena) const
{
    return ARENA_ALLOC(arena, PhaseType)(g);
}

}  // namespace pbrt
