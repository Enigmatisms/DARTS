#include "Medium.hpp"

#include "transmittances/ExponentialTransmittance.hpp"

#include "phasefunctions/IsotropicPhaseFunction.hpp"

#include "io/JsonObject.hpp"
#include "io/Scene.hpp"

namespace Tungsten {

Medium::Medium()
: _transmittance(std::make_shared<ExponentialTransmittance>()),
  _phaseFunction(std::make_shared<IsotropicPhaseFunction>()),
  _maxBounce(1024)
{
}

void Medium::fromJson(JsonPtr value, const Scene &scene)
{
    JsonSerializable::fromJson(value, scene);

    if (auto phase = value["phase_function"])
        _phaseFunction = scene.fetchPhase(phase);
    if (auto trans = value["transmittance"])
        _transmittance = scene.fetchTransmittance(trans);

    value.getField("max_bounces", _maxBounce);
}

rapidjson::Value Medium::toJson(Allocator &allocator) const
{
    return JsonObject{JsonSerializable::toJson(allocator), allocator,
        "phase_function", *_phaseFunction,
        "transmittance", *_transmittance,
        "max_bounces", _maxBounce
    };
}

bool Medium::invertDistance(WritablePathSampleGenerator &/*sampler*/, const Ray &/*ray*/, bool /*onSurface*/) const
{
    FAIL("Medium::invert not implemented!");
}

Vec3f Medium::transmittanceAndPdfs(PathSampleGenerator &sampler, const Ray &ray, bool startOnSurface,
        bool endOnSurface, float &pdfForward, float &pdfBackward) const
{
    pdfForward  = pdf(sampler, ray, startOnSurface, endOnSurface);
    pdfBackward = pdf(sampler, ray.scatter(ray.hitpoint(), -ray.dir(), 0.0f, ray.farT()), endOnSurface, startOnSurface);
    return transmittance(sampler, ray, startOnSurface, endOnSurface);
}

const PhaseFunction *Medium::phaseFunction(const Vec3f &/*p*/) const
{
    return _phaseFunction.get();
}

bool Medium::isDirac() const
{
    return _transmittance->isDirac();
}

float Medium::timeTraveled(float distance) const
{
    FAIL("Medium::timeTraveled not implemented!");
    return 0.0f;
}

float Medium::timeTraveled(const Vec3f &pStart, const Vec3f &pEnd) const
{
    FAIL("Medium::timeTraveled not implemented!");
    return 0.0f;
}

Vec3f Medium::travel(const Vec3f &o, const Vec3f &d, float time) const
{
    FAIL("Medium::travel not implemented!");
    return Vec3f();
}

float Medium::speedOfLight(const Vec3f &p) const
{
    FAIL("Medium::speedOfLight not implemented!");
    return 0.0f;
}

EllipseInfo::EllipseInfo (
    GuideInfoConstPtr sampling_info, const Vec3f& p, float min_time, 
    float p_time, float interval, PathSampleGenerator &sampler, bool full_init):
    target_t(-1)
{	
    // TODO: note that in Tungsten, the elliptical sampling does not consider the speed of light
    // so, do not set speed of light other than 1 (for now)
    if (interval < 0 || sampling_info == nullptr) return;
    Vec3f diff_vec = sampling_info->vertex_p - p;
    foci_dist = diff_vec.length();
    min_t = min_time - p_time - sampling_info->time;
    max_t = min_t + interval;
    if (max_t > foci_dist + 1e-6f) {
        min_t = std::max(min_t, foci_dist);
        interval = max_t - min_t;
        target_t = min_t + interval * sampler.next1D();
        to_emitter = diff_vec / foci_dist;
        T_div_D = target_t / foci_dist;
        if (full_init) {                        // the PDF of the sample must match the sampling procedure
            half_power_diff = 0.5f * (target_t + foci_dist) * (target_t - foci_dist);
            max_power_diff = 0.5f * (max_t + foci_dist) * (max_t - foci_dist);
            inv_sampling_pdf = interval;
        }
    }
}

}
