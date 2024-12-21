#include "PathTracer.hpp"

#include "bsdfs/TransparencyBsdf.hpp"

namespace Tungsten {

constexpr bool pt_use_tof = true;
constexpr bool enable_darts  = true;           // skip non-darts part

PathTracer::PathTracer(TraceableScene *scene, const PathTracerSettings &settings, uint32 threadId)
: TraceBase(scene, settings, threadId),
  _settings(settings),
  _trackOutputValues(!scene->rendererSettings().renderOutputs().empty())
{
}

Vec3f PathTracer::traceSample(
    Vec2u pixel, PathSampleGenerator &sampler, GuideInfoConstPtr sampling_info, 
    Vec3f* const transients, float bin_pdf, float min_remaining_t
) {
    // TODO: Put diagnostic colors in JSON?
    const Vec3f nanDirColor = Vec3f(0.0f);
    const Vec3f nanEnvDirColor = Vec3f(0.0f);
    const Vec3f nanBsdfColor = Vec3f(0.0f);

    try {

    PositionSample point;
    if (!_scene->cam().samplePosition(sampler, point, pixel))
        return Vec3f(0.0f);
    DirectionSample direction;
    if (!_scene->cam().sampleDirection(sampler, point, pixel, direction))
        return Vec3f(0.0f);

    Vec3f throughput = point.weight*direction.weight;
    Ray ray(point.p, direction.d);
    ray.setPrimaryRay(true);

    MediumSample mediumSample;
    SurfaceScatterEvent surfaceEvent;
    IntersectionTemporary data;
    Medium::MediumState state;
    state.reset();
    IntersectionInfo info;
    Vec3f emission(0.0f);
    const Medium *medium = _scene->cam().medium().get();

    bool recordedOutputValues = false;
    float hitDistance = 0.f, remaining_time = 0.f;

    int mediumBounces = 0, bounceSinceSurface = 0;
    int bounce = 0;
    std::unique_ptr<TransientInfo> tof_info{nullptr};                // TODO: it is possible that we will put a transient vector pointer in this Info struct
    if constexpr (pt_use_tof) {
        tof_info = std::make_unique<TransientInfo>(
            _settings.transientTimeBeg, _settings.transientTimeEnd, 
            _settings.invTransientTimeWidth, _settings.vaccumSpeedOfLight, bin_pdf, transients
        );
        tof_info->enable_origin_sample = !_settings.enable_elliptical || !tof_info->valid();
    }
    bool didHit = _scene->intersect(ray, data, info);
    bool wasSpecular = true;
    EllipseInfo ell_info(sampling_info, ray.pos(), min_remaining_t, tof_info->elapsed_t, _settings.transientTimeWidth, sampler);
    while ((didHit || medium) && bounce < _settings.maxBounces) {
        bool hitSurface = true;
        bounceSinceSurface++;
        if (medium) {
            mediumSample.continuedWeight = throughput;

            if constexpr (enable_darts && pt_use_tof) {
                remaining_time = ell_info.target_t;
                if (_settings.enable_elliptical && medium && _settings.enableVolumeLightSampling) {            // elliptical sampling logic (might only work with photon points)
                    // this will account for 3 vertex path (originally we only have vertex >= 4 path)
                    if (!ell_info.valid_target())
                        break;
                    // This is not for avoiding numerical singularity: we don't want to account for surface photons twice
                    if (bounceSinceSurface > 1 && ell_info.direct_connect()) {          // this could be extremely rare
                        // elapsed (path time before elliptical sampling), remaining_time: sampled connection time, sampling_info->time: emitter time
                        emission += throughput * volumeEstimateDirect(sampler, mediumSample, medium, bounce, ray, tof_info.get()) * bin_pdf;
                    } else if (bounce < _settings.maxBounces - 1) {
                        MediumSample mit;
                        mit.t = -1;
                        medium->elliptical_sample(ray, sampler, mit, &ell_info);
                        if (mit.t > 0) {
                            // elapsed (path time before elliptical sampling), remaining_time: sampled connection time, sampling_info->time: emitter time
                            float aux_elliptical_time = medium->timeTraveled(mit.t);
                            Vec3f radiance = throughput * volumeEstimateDirect(sampler, mit, medium, bounce + 1, ray, tof_info.get(), aux_elliptical_time) * mit.weight;
                            emission += radiance;
                        }
                    }
                }
            }   // this closing-bracket tail is toooo ugly


            // we have simple extension here
            if (!medium->sampleDistance(sampler, ray, state, mediumSample, remaining_time, sampling_info, _settings.enable_guiding))
                return emission;
            emission += throughput*mediumSample.emission;

            throughput *= mediumSample.weight;
            hitSurface = mediumSample.exited;
            if (hitSurface && !didHit)
                break;
            if constexpr (pt_use_tof) {
                if (tof_info->valid()) {
                    tof_info->elapsed_t += medium->timeTraveled(mediumSample.t);
                    if (tof_info->elapsed_t > tof_info->time_end)
                        break;
                }
            }
        } else {
            if constexpr (pt_use_tof) {
                if (tof_info->valid()) {
                    tof_info->elapsed_t += ray.farT() / tof_info->vacuum_sol;
                    if (tof_info->elapsed_t > tof_info->time_end)
                        break;
                }
            }
        }

        if (hitSurface) {
            bounceSinceSurface = 0;
            hitDistance += ray.farT();

            if (mediumBounces == 1 && !_settings.lowOrderScattering)
                return emission;

            surfaceEvent = makeLocalScatterEvent(data, info, ray, &sampler);

            ell_info = EllipseInfo(sampling_info, info.p, min_remaining_t, tof_info->elapsed_t, _settings.transientTimeWidth, sampler);

            Vec3f transmittance(-1.0f);
            bool terminate = !handleSurface(surfaceEvent, data, info, medium, bounce, false,
                    _settings.enableLightSampling && (mediumBounces > 0 || _settings.includeSurfaces), 
                    ray, throughput, emission, wasSpecular, state, &transmittance, tof_info.get());

            if (!info.bsdf->lobes().isPureDirac())
                if (mediumBounces == 0 && !_settings.includeSurfaces)
                    return emission;

            if (_trackOutputValues && !recordedOutputValues && (!wasSpecular || terminate)) {
                if (_scene->cam().depthBuffer())
                    _scene->cam().depthBuffer()->addSample(pixel, hitDistance);
                if (_scene->cam().normalBuffer())
                    _scene->cam().normalBuffer()->addSample(pixel, info.Ns);
                if (_scene->cam().albedoBuffer()) {
                    Vec3f albedo;
                    if (const TransparencyBsdf *bsdf = dynamic_cast<const TransparencyBsdf *>(info.bsdf))
                        albedo = (*bsdf->base()->albedo())[info];
                    else
                        albedo = (*info.bsdf->albedo())[info];
                    if (info.primitive->isEmissive())
                        albedo += info.primitive->evalDirect(data, info);
                    _scene->cam().albedoBuffer()->addSample(pixel, albedo);
                }
                if (_scene->cam().visibilityBuffer() && transmittance != -1.0f)
                    _scene->cam().visibilityBuffer()->addSample(pixel, transmittance.avg());
                recordedOutputValues = true;
            }

            if (terminate)
                return emission;
        } else {
            mediumBounces++;
            bool skip_volume_eval = pt_use_tof && !tof_info->enable_origin_sample;
            // either when when shut off ToF rendering in compile-time, or we decide to use original sample will we use this handleVolume

            // we should of course update the elliptical info struct in surface pass
            ell_info = EllipseInfo(sampling_info, mediumSample.p, min_remaining_t, tof_info->elapsed_t, _settings.transientTimeWidth, sampler);

            if (!handleVolume(sampler, mediumSample, medium, bounce, false,
                _settings.enableVolumeLightSampling && (mediumBounces > 1 || _settings.lowOrderScattering), 
                ray, throughput, emission, wasSpecular, tof_info.get(), &ell_info, skip_volume_eval))
                return emission;
        }

        if (throughput.max() == 0.0f)
            break;
        if (_settings.enable_rr) {
            float roulettePdf = std::abs(throughput).max();
            if (bounce > 2 && roulettePdf < 0.1f) {
                if (sampler.nextBoolean(roulettePdf))
                    throughput /= roulettePdf;
                else
                    return emission;
            }
        }

        if constexpr (!pt_use_tof) {
            // Russian Roulette is disabled for Path Tracing
            float roulettePdf = std::abs(throughput).max();
            if (bounce > 2 && roulettePdf < 0.1f) {
                if (sampler.nextBoolean(roulettePdf))
                    throughput /= roulettePdf;
                else
                    return emission;
            }
        }

        if (std::isnan(ray.dir().sum() + ray.pos().sum()))
            return nanDirColor;
        if (std::isnan(throughput.sum() + emission.sum()))
            return nanBsdfColor;

        bounce++;
        if (bounce < _settings.maxBounces)
            didHit = _scene->intersect(ray, data, info);
    }
    if (bounce >= _settings.minBounces && bounce < _settings.maxBounces)
        handleInfiniteLights(data, info, _settings.enableLightSampling, ray, throughput, wasSpecular, emission);
    if (std::isnan(throughput.sum() + emission.sum()))
        return nanEnvDirColor;

    if (_trackOutputValues && !recordedOutputValues) {
        if (_scene->cam().depthBuffer() && bounce == 0)
            _scene->cam().depthBuffer()->addSample(pixel, 0.0f);
        if (_scene->cam().normalBuffer())
            _scene->cam().normalBuffer()->addSample(pixel, -ray.dir());
        if (_scene->cam().albedoBuffer() && info.primitive && info.primitive->isInfinite())
            _scene->cam().albedoBuffer()->addSample(pixel, info.primitive->evalDirect(data, info));
    }

    return emission;

    } catch (std::runtime_error &e) {
        std::cout << tfm::format("Caught an internal error at pixel %s: %s", pixel, e.what()) << std::endl;

        return Vec3f(0.0f);
    }
}

}
