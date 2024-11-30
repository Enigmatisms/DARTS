
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

// integrators/volpath.cpp*
#include "integrators/volpath.h"
#include "bssrdf.h"
#include "camera.h"
#include "film.h"
#include "interaction.h"
#include "paramset.h"
#include "scene.h"
#include "stats.h"

namespace pbrt {

STAT_INT_DISTRIBUTION("Integrator/Path length", pathLength);
STAT_COUNTER("Integrator/Volume interactions", volumeInteractions);
STAT_COUNTER("Integrator/Surface interactions", surfaceInteractions);

enum ScatteringEvent: uint8_t {
    NONE      = 0,
    SURFACE   = 1,
    MEDIUM    = 2,
    NULL_BSDF = 4,        // null BSDF boundary, elliptical sampling should skip this
    NO_INIT   = 8
};

enum DistanceSamplingType {
    EXPONENTIAL = 0,
    UNIFORM     = 1,
    EQUIANGULAR = 2
};

constexpr DistanceSamplingType _dist_samp_ty = EXPONENTIAL;

/**
 * A note for some simple tests:
 * (1) for baseline testing:
 *  - To avoid surface event, please set 'break' in the surface branch
 *  - To exclude direct surface event, set skip component to SURFACE
 * (2) for ellipse sampling testing
 *  - Note that elliptical sampling will add an extra vertex, therefore the maxdepth is 1 less than baseline
 *  - for 1 bounce (and one elliptical vertex) path (totalling 2 bounces), to remove all surface events, set skip component to SURFACE
*/

// if we OR the following with NO_INIT, we can see a biased result
constexpr ScatteringEvent skip_component = ScatteringEvent::NONE;
// if we choose to sample more directions for NEE (might not have good results)
constexpr bool enable_multi_nee = false;
// if we wish to discard direct illumination from surface, set this as false
constexpr bool enable_direct_lum = true;
// to visualize all the bounces, set this to a negative number. Otherwise this should be non-negative
constexpr int visualize_bounce = -1;
// path sample counting
constexpr bool sample_counting = false;

// Spectrum remapRadiance(Spectrum input, Float path_time) {
//     // simulate special types of transient response weights
//     Float time_diff = 0.0f, weight = 1.0;
//     if (path_time > 14.2) {
//         time_diff = path_time - 16.0;
//     } else {
//         time_diff = path_time - 12.5;
//         weight = 0.5;
//     }
//     return input * exp(-(time_diff * time_diff) / 0.0162) * weight; 
// }

// VolPathIntegrator Method Definitions
void VolPathIntegrator::Preprocess(const Scene &scene, Sampler &sampler) {
    lightDistribution =
        CreateLightSampleDistribution(lightSampleStrategy, scene);
    if (!IsOnlyOnePointSource(scene)) {
        // For more complex scene configuration (other sources, not in scattering medium)
        da_guide = false;
        time_sample = false;
        sampling_info.reset(nullptr);
    } else {
        if (da_guide || time_sample) {
            const std::shared_ptr<Light> &light = scene.lights[0];
            sampling_info.reset(new GuidedSamplingInfo(light->get_pos(), 0.));
        }
    }
}

TransientStatus VolPathIntegrator::accumulate_radiance(
    const Spectrum& nee_le, FilmInfoConstPtr film_info, 
    FilmTilePixel* const tfilm, Float nee_time, bool transient_valid
) const {
    if (transient_valid && !nee_le.IsBlack()) {
        if (nee_time > film_info->min_time && nee_time < film_info->max_time) {
            int local_idx = int(std::floor((nee_time - film_info->min_time) / film_info->interval));
            // Spectrum result = remapRadiance(nee_le, nee_time);
            if constexpr (sample_counting)
                tfilm[local_idx].contribSum += 1.f;
            else
                tfilm[local_idx].contribSum += nee_le;
            return TransientStatus::OK;
        }
        return TransientStatus::OOR;
    }
    return TransientStatus::ZERO_VAL;
}

bool VolPathIntegrator::path_reuse_connection(
    const Scene &scene, const Interaction &it, const Ray& ray_wo, Spectrum& inout,
    Sampler& sampler, MemoryArena &arena, Float& nee_time, Float min_remain_t, Float interval
) const {
    // low order scattering might not have good elliptical samples (for highly-scattering medium)
    Ray input_ray = ray_wo;
    input_ray.o = it.p;
    Vector3f wo = -ray_wo.d, new_dir, v1, v2;
    EllipseInfo new_info(sampling_info.get(), it.p, min_remain_t, it.time, interval, sampler);
    if (new_info.ill_defined() || !new_info.valid_target()) return false;
    CoordinateSystem(wo, &v1, &v2);
    Spectrum avg_nee_le(0.f);
    for (int k = 0; k < nee_nums; k++) {
        auto u = sampler.Get2D();
        // Float d_sample = 1.f / (2.f / (new_info.target_t - new_info.foci_dist) - u[0] / alpha);
        // Float phi = 2 * Pi * u[1], cos_sample = new_info.T_div_D - 2.f * alpha / (new_info.target_t - d_sample);
        // Float sinTheta = std::sqrt(std::max((Float)0, 1 - cos_sample * cos_sample));
        // // Then we calculate PDF
        // Float p_cos = new_info.target_t / d_sample - 1.f;
        // p_cos *= p_cos * Inv4Pi;               // 0.5 (T - d)^2 / d^2

        Float cos_sample = 1 - 2 * u[0];        // for g = 0 case
        // Compute direction _wi_ for Henyey-Greenstein sample
        Float sinTheta = std::sqrt(std::max((Float)0, 1 - cos_sample * cos_sample));
        Float phi = 2 * Pi * u[1];

        // Note that p is just for sampling the cosine theta, what about Phi? It is uniform so we need not worry about it
        new_dir = SphericalDirection(sinTheta, cos_sample, phi, v1, v2, wo);
        // new_dir = SphericalDirection(sinTheta, cos_sample, phi, v1, v2, new_info.to_emitter);
        // The above logic get a path-reuse sample

        Spectrum local_throughput(1.f), local_att(1.f);
        if (it.IsMediumInteraction()) {
            // const MediumInteraction &mit = (const MediumInteraction &)it;
            // Float cos_wi_wo = Dot(new_dir, wo);
            // Float hg_eval = PhaseHG(cos_wi_wo, mit.phase->get_g());
            // local_throughput = Spectrum(hg_eval / p_cos);
        } else {
            return false;          // do not do this for surface
            const SurfaceInteraction &sit = (const SurfaceInteraction &)it;
            // Do not allow the ray to penetrate into the surface (so test the inner product with geometry normal)
            if (Dot(new_dir, sit.n) < 0) continue;
            Spectrum f = sit.bsdf->f(wo, new_dir);
            if (f.IsBlack()) continue;
            // Specular bounce problem is already resolved
            // local_throughput = f * AbsDot(new_dir, sit.shading.n) / p_cos;
        }
        SurfaceInteraction isect;
        MediumInteraction local_mit;
        input_ray.d = new_dir;
        scene.Intersect(input_ray, &isect); // we will definitely find intersection
        ray_wo.medium->elliptical_sample(input_ray, sampler, 
                                &local_mit, arena, local_att, &new_info);
        if (local_mit.IsValid()) {                   // if not sampled, then the result will be 0 (should be accumulated)
            const Distribution1D *lightDistrib =
                    lightDistribution->Lookup(local_mit.p);
            Float dummy_time = 0.f;
            Spectrum direct_comp = UniformSampleOneLight(local_mit, scene, arena, sampler, &dummy_time,
                                            true, lightDistrib) * local_att * local_throughput;
            if (direct_comp.IsBlack()) continue;
            if(direct_comp.HasNaNorInf() || direct_comp.y() < 0) {
                return false;
            }
            avg_nee_le += direct_comp;
        }
    }
    nee_time = new_info.target_t + it.time;
    inout = avg_nee_le * nee_div;
    return true;
}

Spectrum VolPathIntegrator::Li(const RayDifferential &r, const Scene &scene,
                               Sampler &sampler, MemoryArena &arena,
                               int depth, Float min_remain_t, Float bin_pdf, FilmInfoConstPtr film_info, FilmTilePixel* const tfilm, bool verbose) const {
    ProfilePhase p(Prof::SamplerIntegratorLi);
    Spectrum L(0.f), beta(1.f);
    RayDifferential ray(r);
    bool specularBounce = false;
    int bounces, medium_bounces = 0;
    // Added after book publication: etaScale tracks the accumulated effect
    // of radiance scaling due to rays passing through refractive
    // boundaries (see the derivation on p. 527 of the third edition). We
    // track this value in order to remove it from beta when we apply
    // Russian roulette; this is worthwhile, since it lets us sometimes
    // avoid terminating refracted rays that are about to be refracted back
    // out of a medium and thus have their beta value increased.
    Float etaScale = 1;
    Float remaining_time = 0;

    const bool transient_valid = (film_info != nullptr);
    // not using elliptical samples / not using transient rendering / not doing time-gated sampling
    const bool enable_origin_sample = (!time_sample) || (!transient_valid) || fuse_original;
    bool valid_reused_radiance = false;

    // We can opt for more point sources inside the medium, but it's not meaningful since this is a simple test
    // we use last_event to skip some elliptical sampling, like null BSDF boundary, or when we don't want to visualize
    // one or more particular components, like we don't want surface scattering
    ScatteringEvent last_event = ScatteringEvent::NO_INIT;
    Float interval = -1;  
    if (transient_valid)
        interval = film_info->interval;
    for (bounces = 0;; ++bounces) {
        SurfaceInteraction isect;
        bool foundIntersection = scene.Intersect(ray, &isect);
        Float sample_truncation = 0.0f;

        EllipseInfo ell_info(sampling_info.get(), ray.o, min_remain_t, ray.time, interval, sampler);
        remaining_time = ell_info.target_t;

        // if 2-bounce-path-wo-surface differs, this is the only place where the difference can get injected
        if (time_sample && ((last_event & skip_component) == 0) && ray.medium != nullptr) {
            MediumInteraction mit;
            // this will account for 3 vertex path (originally we only have vertex >= 4 path)
            if (!ell_info.valid_target()) {
                // break if the current target can't be reached 
                break;
            }
            // Direct connect can only be used if the last event is medium and TO AVOID NUMERICAL SINGULARITY
            if (last_event == ScatteringEvent::MEDIUM && ell_info.direct_connect()) {
                // this will make the program go to the direct connection branch
                mit = ray.medium->allocate_it(ray, arena, 0);
                const Distribution1D *lightDistrib =
                        lightDistribution->Lookup(mit.p);
                Float to_emitter_time = 0;
                Spectrum nee_le = beta * UniformSampleOneLight(mit, scene, arena, sampler, &to_emitter_time,
                                                true, lightDistrib);
                Float nee_time = mit.time + to_emitter_time;
                // Note that direct connection is original NEE!
                if (visualize_bounce < 0 || (bounces == visualize_bounce + 1)) {
                    accumulate_radiance(nee_le, film_info, tfilm, nee_time, transient_valid);
                }
                L += nee_le;
            } else {
                // There is alternative method
                // For example, we first sample direction (thus, new ray), then send it to the elliptical_sample function
                // if bounce > 0 (bounce = 0 is camera ray, we can not replace the direction of camera ray)
                // we will definitely use the reused direction since it is efficient
                Spectrum nee_le(0), local_att(1);
                Float local_trunc = ray.medium->elliptical_sample(ray, sampler, 
                                                &mit, arena, local_att, &ell_info);
                if (use_truncate)   
                    sample_truncation = local_trunc;    // we compute two ellipses now

                if (mit.IsValid()) {
                    const Distribution1D *lightDistrib =
                            lightDistribution->Lookup(mit.p);
                    Float to_emitter_time = 0;
                    nee_le += beta * UniformSampleOneLight(mit, scene, arena, sampler, &to_emitter_time,
                                                    true, lightDistrib) * local_att / bin_pdf;
                    if (enable_multi_nee && bounces > 0 && valid_reused_radiance)                // since the first bounce will never use path reuse, we are not going to average anything
                        nee_le *= nee_div;
                    Float nee_time = mit.time + to_emitter_time;
                    if (visualize_bounce < 0 || bounces == visualize_bounce) {
                        accumulate_radiance(nee_le, film_info, tfilm, nee_time, transient_valid);
                    }
                    L += nee_le;
                }
            }
            if (bounces >= maxDepth) break;
        }

        MediumInteraction mi;
        // Sample the participating medium, if present
        // printf("bounce = %d, ray.medium = %d\n", bounces, int(ray.medium == nullptr));
        if (ray.medium && foundIntersection) {  // We do not deal with infinite scene medium
            // Only if we are inside the valid zone of the medium will we start to sample
            // ray.o too close to the surface can lead to extended path

            if constexpr (_dist_samp_ty == DistanceSamplingType::EQUIANGULAR) {
                beta *= ray.medium->equiangular_sample(ray, sampler, arena, &mi, sampling_info->vertex_p);
            } else if constexpr (_dist_samp_ty == DistanceSamplingType::UNIFORM) {
                beta *= ray.medium->uniform_sample(ray, sampler, arena, &mi, sampling_info->vertex_p);
            } else {
                beta *= ray.medium->Sample(ray, sampler, arena, &mi, remaining_time, 
                                    sampling_info.get(), sample_truncation, da_guide);
            }
        }
        if (beta.IsBlack()) break;

        // Handle an interaction with a medium or a surface
        // printf("Bounces = %d\n", bounces);

        if (mi.IsValid()) {
            // Terminate path if ray escaped or _maxDepth_ was reached
            if (bounces >= maxDepth) break;
            // if (medium_bounces > 1) {
            //     break;
            // }
            // medium_bounces++;

            if constexpr ((skip_component & ScatteringEvent::MEDIUM) == 0) {
                if (enable_origin_sample) {
                    // only when elliptical sampling succeeds, will NEE get used
                    const Distribution1D *lightDistrib =
                        lightDistribution->Lookup(mi.p);
                    Float to_emitter_time = 0.;
                    Spectrum nee_le = beta * UniformSampleOneLight(mi, scene, arena, sampler, &to_emitter_time, true,
                                                    lightDistrib);// / bin_pdf;
                    // If we are doing transient rendering, we should use nee sample as an added temporal sample
                    Float nee_time = mi.time + to_emitter_time;
                    if (visualize_bounce < 0 || bounces == visualize_bounce)
                        accumulate_radiance(nee_le, film_info, tfilm, nee_time, transient_valid);
                    L += nee_le;
                }
            }

            Vector3f wo = -ray.d, wi;
            Float pdf_fwd = mi.phase->Sample_p(wo, &wi, sampler.Get2D(), nullptr);
            // // Note that currently we only use HG to build paths (EllipticalPhase for NEE), so eval / pdf = 1
            // Float f = mi.phase->p(mi.wo, wi);
            // beta *= f / pdf_fwd;
            if constexpr (enable_multi_nee) {
                if (time_sample && transient_valid && nee_nums) {
                    Float nee_time = 0.f;
                    Spectrum reused_le(0.f);
                    valid_reused_radiance = path_reuse_connection(scene, mi, ray, reused_le,
                                            sampler, arena, nee_time, min_remain_t, interval);
                    reused_le *= beta / bin_pdf;
                    if (valid_reused_radiance && (visualize_bounce < 0 || bounces == visualize_bounce + 1)) {
                        accumulate_radiance(reused_le, film_info, tfilm, nee_time, transient_valid);
                    }
                    L += reused_le;
                }
            }

            ray = mi.SpawnRay(wi);

            ++volumeInteractions;
            last_event = ScatteringEvent::MEDIUM;

            // Handle scattering at point in medium for volumetric path tracer
            specularBounce = false;
        } else {
            // Handle scattering at point on surface for volumetric path tracer
            // Possibly add emitted light at intersection
            if (bounces == 0 || specularBounce) {
                // Add emitted light at path vertex or from the environment
                if (foundIntersection)
                    L += beta * isect.Le(-ray.d);
                else
                    for (const auto &light : scene.infiniteLights)
                        L += beta * light->Le(ray);
            }
            isect.time += ray.tMax;

            // Terminate path if ray escaped or _maxDepth_ was reached
            if (!foundIntersection || bounces >= maxDepth) break;

            // Compute scattering functions and skip over medium boundaries
            isect.ComputeScatteringFunctions(ray, arena, true);
            if (!isect.bsdf) {
                ray = isect.SpawnRay(ray.d);
                last_event = ScatteringEvent::NULL_BSDF;
                bounces--;
                continue;
            }
            
            if constexpr (((skip_component & ScatteringEvent::SURFACE) == 0) && enable_direct_lum) {
                // Sample illumination from lights to find attenuated path contribution
                const Distribution1D *lightDistrib =
                    lightDistribution->Lookup(isect.p);
                bool in_time_range = false;
                if (transient_valid && time_sample) {
                    DCHECK(sampling_info != nullptr);
                    Float min_target = min_remain_t - isect.time - sampling_info->time,
                          max_target = min_target + interval;
                    Float to_emitter_d = (isect.p - sampling_info->vertex_p).Length();
                    in_time_range = (to_emitter_d > min_target && to_emitter_d < max_target);
                }
                if (enable_origin_sample || in_time_range) {
                    Float to_emitter_time = 0.;
                    // This branch will be used when we don't use time sampling
                    // also, if elliptical_sampled the path should go to medium sampling right away
                    Spectrum nee_le = beta * UniformSampleOneLight(isect, scene, arena, sampler, &to_emitter_time,
                                                    true, lightDistrib);// / bin_pdf;
                    // Do not try DA without elliptical sampling for transient rendering
                    if (transient_valid && film_info->time_gated) {
                        if (sampling_info != nullptr)
                            nee_le /= bin_pdf;
                    } else {
                        if (in_time_range && time_sample) {
                            Float scaler = 1.f / bin_pdf;
                            if (fuse_original) scaler += 1.f;       // this works for time-gated rendering
                            // why should we add an extra one? (DARTS medium) + (original surface sample / bin_pdf)
                            // And (original medium) + (original surface sample), thus: original surface sample * (1 + 1 / bin_pdf)
                            nee_le *= scaler;
                        }
                    } 
                    if (visualize_bounce < 0 || bounces == visualize_bounce)
                        accumulate_radiance(nee_le, film_info, tfilm, isect.time + to_emitter_time, transient_valid);
                    L += nee_le;
                }
            }

            ++surfaceInteractions;
            last_event = ScatteringEvent::SURFACE;

            // Sample BSDF to get new path direction
            Vector3f wo = -ray.d, wi;
            Float pdf;
            BxDFType flags;
            Spectrum f = isect.bsdf->Sample_f(wo, &wi, sampler.Get2D(), &pdf,
                                              BSDF_ALL, &flags);
            if (f.IsBlack() || pdf == 0.f) break;
            beta *= f * AbsDot(wi, isect.shading.n) / pdf;
            DCHECK(std::isinf(beta.y()) == false);
            specularBounce = (flags & BSDF_SPECULAR) != 0;
            if ((flags & BSDF_SPECULAR) && (flags & BSDF_TRANSMISSION)) {
                Float eta = isect.bsdf->eta;
                // Update the term that tracks radiance scaling for refraction
                // depending on whether the ray is entering or leaving the
                // medium.
                etaScale *=
                    (Dot(wo, isect.n) > 0) ? (eta * eta) : 1 / (eta * eta);
            }
            if (!(flags & BSDF_TRANSMISSION) && Dot(wi, isect.n) < 0) {
                // penetrate from the inside of the surface, and the surface is not transmissive
                break;
            }
            if constexpr (enable_multi_nee) {
                if (time_sample && transient_valid && nee_nums) {
                    Float nee_time = 0.f;
                    Spectrum reused_le(0.f);
                    valid_reused_radiance = path_reuse_connection(scene, isect, ray, reused_le,
                                            sampler, arena, nee_time, min_remain_t, interval);
                    reused_le *= beta / bin_pdf;
                    if (valid_reused_radiance && (visualize_bounce < 0 || bounces == visualize_bounce + 1))
                        accumulate_radiance(reused_le, film_info, tfilm, nee_time, transient_valid);
                    L += reused_le;
                }
            }

            ray = isect.SpawnRay(wi);
            // BSSRDF is not supported by DARTS-pbrt-v3 but in pbrt-v3, it is supported
        }

        // Possibly terminate the path with Russian roulette
        // Factor out radiance scaling due to refraction in rrBeta.

        Spectrum rrBeta = beta * etaScale;
        if (rrBeta.MaxComponentValue() < rrThreshold && bounces > 200) {
            Float q = std::max((Float).05, 1 - rrBeta.MaxComponentValue());
            if (sampler.Get1D() < q) break;
            beta /= 1 - q;
            DCHECK(std::isinf(beta.y()) == false);
        }

        // exceed required time range 
        if (transient_valid && ray.time > film_info->max_time) break;
    }
    ReportValue(pathLength, bounces);
    return L;
}

VolPathIntegrator *CreateVolPathIntegrator(
    const ParamSet &params, std::shared_ptr<Sampler> sampler,
    std::shared_ptr<const Camera> camera) {
    int maxDepth      = params.FindOneInt("maxdepth", 5);
    int nee_nums      = std::max(params.FindOneInt("extra_nee_nums", 0), 0);
    bool rseed_time   = params.FindOneInt("rseed_time", 0) != 0;
    bool log_time     = params.FindOneInt("logTime", 0) != 0;
    bool use_guiding  = params.FindOneInt("guiding", 0) != 0;
    bool time_sample  = params.FindOneInt("time_sample", 0) != 0;
    bool use_truncate = params.FindOneInt("use_truncate", 0) != 0;
    bool fuse_origin  = params.FindOneBool("fuse_origin", false);
    bool sample_flag  = params.FindOneBool("sample_flag", false);
    use_truncate &= time_sample;

    bool da_guide = false;
    if (use_guiding) {
        da_guide = params.FindOneBool("da_guide", false);
    }

    int np;
    const int *pb = params.FindInt("pixelbounds", &np);
    Bounds2i pixelBounds = camera->film->GetSampleBounds();
    if (pb) {
        if (np != 4)
            Error("Expected four values for \"pixelbounds\" parameter. Got %d.",
                  np);
        else {
            pixelBounds = Intersect(pixelBounds,
                                    Bounds2i{{pb[0], pb[2]}, {pb[1], pb[3]}});
            if (pixelBounds.Area() == 0)
                Error("Degenerate \"pixelbounds\" specified.");
        }
    }
    Float rrThreshold = params.FindOneFloat("rrthreshold", 1.);
    std::string lightStrategy =
        params.FindOneString("lightsamplestrategy", "spatial");
    std::string nee_status;
    if constexpr (enable_multi_nee) {
        nee_status = std::to_string(nee_nums) + " extra shadow ray(s)";
    } else {
        nee_status = "disabled";
    }
    printf("volpath settings: rr threshold: %f, use_guiding: %d, time_sample: %d, use_truncate: %d, DA guiding: %d, sample: %d, NEE: %s\n", 
                                rrThreshold, int(use_guiding), int(time_sample), int(use_truncate), int(da_guide), int(sample_flag), nee_status.c_str());
    if (!time_sample)
        printf("Time sample disabled. Remember to set 'use_elliptical_phase' as FALSE for guided_homo medium.");
    return new VolPathIntegrator(maxDepth, camera, sampler, pixelBounds, rrThreshold, lightStrategy, 
                da_guide, log_time, time_sample, use_truncate, fuse_origin, rseed_time, sample_flag, nee_nums);
}

}  // namespace pbrt
