#include "TraceBase.hpp"

namespace Tungsten {

TraceBase::TraceBase(TraceableScene *scene, const TraceSettings &settings, uint32 threadId)
: _scene(scene),
  _settings(settings),
  _threadId(threadId)
{
    _scene = scene;
    _lightPdf.resize(scene->lights().size());

    std::vector<float> lightWeights(scene->lights().size());
    for (size_t i = 0; i < scene->lights().size(); ++i) {
        scene->lights()[i]->makeSamplable(*_scene, _threadId);
        lightWeights[i] = 1.0f; // TODO: Use light power here
    }
    _lightSampler.reset(new Distribution1D(std::move(lightWeights)));

    for (const auto &prim : scene->lights())
        prim->makeSamplable(*_scene, _threadId);
}

SurfaceScatterEvent TraceBase::makeLocalScatterEvent(IntersectionTemporary &data, IntersectionInfo &info,
        Ray &ray, PathSampleGenerator *sampler) const
{
    TangentFrame frame;
    info.primitive->setupTangentFrame(data, info, frame);

    bool hitBackside = frame.normal.dot(ray.dir()) > 0.0f;
    bool isTransmissive = info.bsdf->lobes().isTransmissive();

    bool flipFrame = _settings.enableTwoSidedShading && hitBackside && !isTransmissive;

    if (flipFrame) {
        // TODO: Should we flip info.Ns here too? It doesn't seem to be used at the moment,
        // but it may be in the future. Modifying the intersection info itself could be a bad
        // idea though
        frame.normal = -frame.normal;
        frame.tangent = -frame.tangent;
    }

    return SurfaceScatterEvent(
        &info,
        sampler,
        frame,
        frame.toLocal(-ray.dir()),
        BsdfLobes::AllLobes,
        flipFrame
    );
}

bool TraceBase::isConsistent(const SurfaceScatterEvent &event, const Vec3f &w) const
{
    if (!_settings.enableConsistencyChecks)
        return true;
    bool geometricBackside = (w.dot(event.info->Ng) < 0.0f);
    bool shadingBackside = (event.wo.z() < 0.0f) ^ event.flippedFrame;
    return geometricBackside == shadingBackside;
}

template<bool ComputePdfs>
inline Vec3f TraceBase::generalizedShadowRayImpl(PathSampleGenerator &sampler,
                           Ray &ray,
                           const Medium *medium,
                           const Primitive *endCap,
                           int bounce,
                           bool startsOnSurface,
                           bool endsOnSurface,
                           float &pdfForward,
                           float &pdfBackward) const
{
    IntersectionTemporary data;
    IntersectionInfo info;

    float initialFarT = ray.farT();
    Vec3f throughput(1.0f);
    do {
        bool didHit = _scene->intersect(ray, data, info) && info.primitive != endCap;
        if (didHit) {
            if (!info.bsdf->lobes().hasForward())
                return Vec3f(0.0f);

            SurfaceScatterEvent event = makeLocalScatterEvent(data, info, ray, nullptr);

            // For forward events, the transport direction does not matter (since wi = -wo)
            Vec3f transparency = info.bsdf->eval(event.makeForwardEvent(), false);
            if (transparency == 0.0f)
                return Vec3f(0.0f);

            if (ComputePdfs) {
                float transparencyScalar = transparency.avg();
                pdfForward  *= transparencyScalar;
                pdfBackward *= transparencyScalar;
            }

            throughput *= transparency;
            bounce++;

            if (bounce >= _settings.maxBounces)
                return Vec3f(0.0f);
        }

        if (medium) {
            if (ComputePdfs) {
                float forward, backward;
                throughput *= medium->transmittanceAndPdfs(sampler, ray, startsOnSurface, didHit || endsOnSurface, forward, backward);
                pdfForward *= forward;
                pdfBackward *= backward;
            } else {
                throughput *= medium->transmittance(sampler, ray, startsOnSurface, endsOnSurface);
            }
        }
        if (info.primitive == nullptr || info.primitive == endCap)
            return bounce >= _settings.minBounces ? throughput : Vec3f(0.0f);
        medium = info.primitive->selectMedium(medium, !info.primitive->hitBackside(data));
        startsOnSurface = true;

        ray.setPos(ray.hitpoint());
        initialFarT -= ray.farT();
        ray.setNearT(info.epsilon);
        ray.setFarT(initialFarT);
    } while(true);
    return Vec3f(0.0f);
}

Vec3f TraceBase::generalizedShadowRay(PathSampleGenerator &sampler, Ray &ray, const Medium *medium,
            const Primitive *endCap, bool startsOnSurface, bool endsOnSurface, int bounce) const
{
    float dummyA, dummyB;
    return generalizedShadowRayImpl<false>(sampler, ray, medium, endCap, bounce,
            startsOnSurface, endsOnSurface, dummyA, dummyB);
}

Vec3f TraceBase::generalizedShadowRayAndPdfs(PathSampleGenerator &sampler, Ray &ray, const Medium *medium,
           const Primitive *endCap, int bounce, bool startsOnSurface, bool endsOnSurface,
           float &pdfForward, float &pdfBackward) const
{
    pdfForward = pdfBackward = 1.0f;
    return generalizedShadowRayImpl<true>(sampler, ray, medium, endCap, bounce,
            startsOnSurface, endsOnSurface, pdfForward, pdfBackward);
}

Vec3f TraceBase::attenuatedEmission(PathSampleGenerator &sampler,
                         const Primitive &light,
                         const Medium *medium,
                         float expectedDist,
                         IntersectionTemporary &data,
                         IntersectionInfo &info,
                         int bounce,
                         bool startsOnSurface,
                         Ray &ray,
                         Vec3f *transmittance, float* distance)
{
    CONSTEXPR float fudgeFactor = 1.0f + 1e-3f;

    if (light.isDirac()) {
        ray.setFarT(expectedDist);
        Vec3f point;
        if (distance != nullptr && light.getPoint(point))
            *distance = (ray.pos() - point).length();
    } else {
        if (!light.intersect(ray, data) || ray.farT()*fudgeFactor < expectedDist)
            return Vec3f(0.0f);
        if (distance != nullptr)
            *distance = ray.farT();
    }
    info.p = ray.pos() + ray.dir()*ray.farT();
    info.w = ray.dir();
    light.intersectionInfo(data, info);

    Vec3f shadow = generalizedShadowRay(sampler, ray, medium, &light, startsOnSurface, true, bounce);
    if (transmittance)
        *transmittance = shadow;
    if (shadow == 0.0f)
        return Vec3f(0.0f);

    return shadow*light.evalDirect(data, info);
}

bool TraceBase::volumeLensSample(const Camera &camera,
                                 PathSampleGenerator &sampler,
                                 MediumSample &mediumSample,
                                 const Medium *medium,
                                 int bounce,
                                 const Ray &parentRay,
                                 Vec3f &weight,
                                 Vec2f &pixel)
{
    LensSample lensSample;
    if (!camera.sampleDirect(mediumSample.p, sampler, lensSample))
        return false;

    Vec3f f = mediumSample.phase->eval(parentRay.dir(), lensSample.d);
    if (f == 0.0f)
        return false;

    Ray ray = parentRay.scatter(mediumSample.p, lensSample.d, 0.0f);
    ray.setPrimaryRay(false);
    ray.setFarT(lensSample.dist);

    Vec3f transmittance = generalizedShadowRay(sampler, ray, medium, nullptr, false, true, bounce);
    if (transmittance == 0.0f)
        return false;

    weight = f*transmittance*lensSample.weight;
    pixel = lensSample.pixel;

    return true;
}

bool TraceBase::surfaceLensSample(const Camera &camera,
                                  SurfaceScatterEvent &event,
                                  const Medium *medium,
                                  int bounce,
                                  const Ray &parentRay,
                                  Vec3f &weight,
                                  Vec2f &pixel)
{
    LensSample sample;
    if (!camera.sampleDirect(event.info->p, *event.sampler, sample))
        return false;

    event.wo = event.frame.toLocal(sample.d);
    if (!isConsistent(event, sample.d))
        return false;

    bool geometricBackside = (sample.d.dot(event.info->Ng) < 0.0f);
    medium = event.info->primitive->selectMedium(medium, geometricBackside);

    event.requestedLobe = BsdfLobes::AllButSpecular;

    Vec3f f = event.info->bsdf->eval(event, true);
    if (f == 0.0f)
        return false;

    Ray ray = parentRay.scatter(event.info->p, sample.d, event.info->epsilon);
    ray.setPrimaryRay(false);
    ray.setFarT(sample.dist);

    Vec3f transmittance = generalizedShadowRay(*event.sampler, ray, medium, nullptr, true, true, bounce);
    if (transmittance == 0.0f)
        return false;

    weight = f*transmittance*sample.weight;
    pixel = sample.pixel;

    return true;
}

Vec3f TraceBase::lightSample(const Primitive &light,
                             SurfaceScatterEvent &event,
                             const Medium *medium,
                             int bounce,
                             const Ray &parentRay,
                             Vec3f *transmittance, float* distance)
{
    LightSample sample;
    if (!light.sampleDirect(_threadId, event.info->p, *event.sampler, sample))
        return Vec3f(0.0f);

    event.wo = event.frame.toLocal(sample.d);
    if (!isConsistent(event, sample.d))
        return Vec3f(0.0f);

    bool geometricBackside = (sample.d.dot(event.info->Ng) < 0.0f);
    medium = event.info->primitive->selectMedium(medium, geometricBackside);

    event.requestedLobe = BsdfLobes::AllButSpecular;

    Vec3f f = event.info->bsdf->eval(event, false);
    if (f == 0.0f)
        return Vec3f(0.0f);

    Ray ray = parentRay.scatter(event.info->p, sample.d, event.info->epsilon);
    ray.setPrimaryRay(false);

    IntersectionTemporary data;
    IntersectionInfo info;
    Vec3f e = attenuatedEmission(*event.sampler, light, medium, sample.dist, data, info, bounce, true, ray, transmittance);
    if (e == 0.0f)
        return Vec3f(0.0f);

    Vec3f lightF = f*e/sample.pdf;

    if (!light.isDirac())
        lightF *= SampleWarp::powerHeuristic(sample.pdf, event.info->bsdf->pdf(event));
    if (distance != nullptr)
        *distance = sample.dist;

    return lightF;
}

Vec3f TraceBase::bsdfSample(const Primitive &light,
                            SurfaceScatterEvent &event,
                            const Medium *medium,
                            int bounce,
                            const Ray &parentRay, float* distance)
{
    event.requestedLobe = BsdfLobes::AllButSpecular;
    if (!event.info->bsdf->sample(event, false))
        return Vec3f(0.0f);
    if (event.weight == 0.0f)
        return Vec3f(0.0f);

    Vec3f wo = event.frame.toGlobal(event.wo);
    if (!isConsistent(event, wo))
        return Vec3f(0.0f);

    bool geometricBackside = (wo.dot(event.info->Ng) < 0.0f);
    medium = event.info->primitive->selectMedium(medium, geometricBackside);

    Ray ray = parentRay.scatter(event.info->p, wo, event.info->epsilon);
    ray.setPrimaryRay(false);

    IntersectionTemporary data;
    IntersectionInfo info;
    Vec3f e = attenuatedEmission(*event.sampler, light, medium, -1.0f, data, info, bounce, true, ray, nullptr, distance);

    if (e == Vec3f(0.0f))
        return Vec3f(0.0f);

    Vec3f bsdfF = e*event.weight;

    bsdfF *= SampleWarp::powerHeuristic(event.pdf, light.directPdf(_threadId, data, info, event.info->p));

    return bsdfF;
}

Vec3f TraceBase::volumeLightSample(PathSampleGenerator &sampler,
                    MediumSample &mediumSample,
                    const Primitive &light,
                    const Medium *medium,
                    int bounce,
                    const Ray &parentRay, float* distance)
{
    LightSample lightSample;
    if (!light.sampleDirect(_threadId, mediumSample.p, sampler, lightSample))
        return Vec3f(0.0f);

    Vec3f f = mediumSample.phase->eval(parentRay.dir(), lightSample.d);
    if (f == 0.0f)
        return Vec3f(0.0f);

    Ray ray = parentRay.scatter(mediumSample.p, lightSample.d, 0.0f);
    ray.setPrimaryRay(false);

    IntersectionTemporary data;
    IntersectionInfo info;
    Vec3f e = attenuatedEmission(sampler, light, medium, lightSample.dist, data, info, bounce, false, ray, nullptr);
    if (e == 0.0f)
        return Vec3f(0.0f);

    if (distance != nullptr)
        *distance = lightSample.dist;

    Vec3f lightF = f*e/lightSample.pdf;

    if (!light.isDirac())
        lightF *= SampleWarp::powerHeuristic(lightSample.pdf, mediumSample.phase->pdf(parentRay.dir(), lightSample.d));

    return lightF;
}

Vec3f TraceBase::volumePhaseSample(const Primitive &light,
                    PathSampleGenerator &sampler,
                    MediumSample &mediumSample,
                    const Medium *medium,
                    int bounce,
                    const Ray &parentRay, float* distance)
{
    PhaseSample phaseSample;
    if (!mediumSample.phase->sample(sampler, parentRay.dir(), phaseSample))
        return Vec3f(0.0f);

    Ray ray = parentRay.scatter(mediumSample.p, phaseSample.w, 0.0f);
    ray.setPrimaryRay(false);

    IntersectionTemporary data;
    IntersectionInfo info;
    Vec3f e = attenuatedEmission(sampler, light, medium, -1.0f, data, info, bounce, false, ray, nullptr, distance);

    if (e == Vec3f(0.0f))
        return Vec3f(0.0f);

    Vec3f phaseF = e*phaseSample.weight;

    phaseF *= SampleWarp::powerHeuristic(phaseSample.pdf, light.directPdf(_threadId, data, info, mediumSample.p));

    return phaseF;
}

Vec3f TraceBase::sampleDirect(const Primitive &light,
                              SurfaceScatterEvent &event,
                              const Medium *medium,
                              int bounce,
                              const Ray &parentRay,
                              Vec3f *transmittance, TransInfotPtr tof_info)
{
    Vec3f result(0.0f);

    if (event.info->bsdf->lobes().isPureSpecular() || event.info->bsdf->lobes().isForward())
        return Vec3f(0.0f);

    float distance = 0.0, time = -1;
    auto light_val = lightSample(light, event, medium, bounce, parentRay, transmittance, &distance);
    const bool valid_tof_info = (tof_info != nullptr) && tof_info->valid();
    if (valid_tof_info) {
        if (distance > 0) {
            time = tof_info->elapsed_t + (medium ? medium->timeTraveled(distance) : distance / tof_info->vacuum_sol);
            if (tof_info->in_range(time))
                result += light_val;
            // FIXME: throughput problem, the transient here does not account for throughput
            tof_info->addTransient(light_val / tof_info->bin_pdf, time);
        }
    } else {
        result += light_val;
    }
    if (!light.isDirac()) {
        float distance = 0.0;
        auto bsdf_val = bsdfSample(light, event, medium, bounce, parentRay, &distance);
        if (valid_tof_info) {
            if (distance > 0) {
                time = tof_info->elapsed_t + (medium ? medium->timeTraveled(distance) : distance / tof_info->vacuum_sol);
                if (tof_info->in_range(time))
                    result += bsdf_val;
                tof_info->addTransient(bsdf_val / tof_info->bin_pdf, time);
            }
        } else {
            result += bsdf_val;
        }
    }
    return valid_tof_info ? result / tof_info->bin_pdf : result;
}

Vec3f TraceBase::volumeSampleDirect(const Primitive &light,
                    PathSampleGenerator &sampler,
                    MediumSample &mediumSample,
                    const Medium *medium,
                    int bounce,
                    const Ray &parentRay, TransInfotPtr tof_info, float aux_time)
{
    float distance = 0, time = -1;
    Vec3f result = volumeLightSample(sampler, mediumSample, light, medium, bounce, parentRay, &distance);
    const bool valid_tof_info = (tof_info != nullptr) && tof_info->valid();
    if (valid_tof_info) {
        time = tof_info->elapsed_t + (medium ? medium->timeTraveled(distance) : distance / tof_info->vacuum_sol) + aux_time;
        if (distance <= 0 || !tof_info->in_range(time)) {
            result = Vec3f(0.0f);
        }
        tof_info->addTransient(result / tof_info->bin_pdf, time);
    }
    if (!light.isDirac()) {
        float distance = 0;
        auto phase_val = volumePhaseSample(light, sampler, mediumSample, medium, bounce, parentRay, &distance);
        if (valid_tof_info) {
            if (distance > 0) {
                time = tof_info->elapsed_t + (medium ? medium->timeTraveled(distance) : distance / tof_info->vacuum_sol) + aux_time;
                if (tof_info->in_range(time))
                    result += phase_val;
                tof_info->addTransient(phase_val / tof_info->bin_pdf, time);
            }
        } else {
            result += phase_val;
        }
    }
    return valid_tof_info ? result / tof_info->bin_pdf : result;
}

const Primitive *TraceBase::chooseLight(PathSampleGenerator &sampler, const Vec3f &p, float &weight)
{
    if (_scene->lights().empty())
        return nullptr;
    if (_scene->lights().size() == 1) {
        weight = 1.0f;
        return _scene->lights()[0].get();
    }

    float total = 0.0f;
    unsigned numNonNegative = 0;
    for (size_t i = 0; i < _lightPdf.size(); ++i) {
        _lightPdf[i] = _scene->lights()[i]->approximateRadiance(_threadId, p);
        if (_lightPdf[i] >= 0.0f) {
            total += _lightPdf[i];
            numNonNegative++;
        }
    }
    if (numNonNegative == 0) {
        for (size_t i = 0; i < _lightPdf.size(); ++i)
            _lightPdf[i] = 1.0f;
        total = _lightPdf.size();
    } else if (numNonNegative < _lightPdf.size()) {
        for (size_t i = 0; i < _lightPdf.size(); ++i) {
            float uniformWeight = (total == 0.0f ? 1.0f : total)/numNonNegative;
            if (_lightPdf[i] < 0.0f) {
                _lightPdf[i] = uniformWeight;
                total += uniformWeight;
            }
        }
    }
    if (total == 0.0f)
        return nullptr;
    float t = sampler.next1D()*total;
    for (size_t i = 0; i < _lightPdf.size(); ++i) {
        if (t < _lightPdf[i] || i == _lightPdf.size() - 1) {
            weight = total/_lightPdf[i];
            return _scene->lights()[i].get();
        } else {
            t -= _lightPdf[i];
        }
    }
    return nullptr;
}

const Primitive *TraceBase::chooseLightAdjoint(PathSampleGenerator &sampler, float &pdf)
{
    float u = sampler.next1D();
    int lightIdx;
    _lightSampler->warp(u, lightIdx);
    pdf = _lightSampler->pdf(lightIdx);
    return _scene->lights()[lightIdx].get();
}

Vec3f TraceBase::volumeEstimateDirect(PathSampleGenerator &sampler,
                    MediumSample &mediumSample,
                    const Medium *medium,
                    int bounce,
                    const Ray &parentRay, TransInfotPtr tof_info, float aux_time)
{
    float weight;
    const Primitive *light = chooseLight(sampler, mediumSample.p, weight);
    if (light == nullptr)
        return Vec3f(0.0f);
    return volumeSampleDirect(*light, sampler, mediumSample, medium, bounce, parentRay, tof_info, aux_time)*weight;
}

Vec3f TraceBase::estimateDirect(SurfaceScatterEvent &event,
                                const Medium *medium,
                                int bounce,
                                const Ray &parentRay,
                                Vec3f *transmittance, TransInfotPtr tof_info)
{
    float weight;
    const Primitive *light = chooseLight(*event.sampler, event.info->p, weight);
    if (light == nullptr)
        return Vec3f(0.0f);
    return sampleDirect(*light, event, medium, bounce, parentRay, transmittance, tof_info)*weight;
}

bool TraceBase::handleVolume(PathSampleGenerator &sampler, MediumSample &mediumSample,
           const Medium *&medium, int bounce, bool adjoint, bool enableLightSampling,
           Ray &ray, Vec3f &throughput, Vec3f &emission, bool &wasSpecular, TransInfotPtr tof_info, EllipseConstPtr ell_info, bool skip_eval)
{
    wasSpecular = !enableLightSampling;

    if (!skip_eval && !adjoint && enableLightSampling && bounce < _settings.maxBounces - 1) {
        auto radiance = throughput * volumeEstimateDirect(sampler, mediumSample, medium, bounce + 1, ray, tof_info);
        if (tof_info && tof_info->valid()) radiance *= tof_info->bin_pdf;
        emission += radiance;
    }

    PhaseSample phaseSample;
    if (!mediumSample.phase->sample(sampler, ray.dir(), phaseSample, ell_info))
        return false;

    ray = ray.scatter(mediumSample.p, phaseSample.w, 0.0f);
    ray.setPrimaryRay(false);
    throughput *= phaseSample.weight;

    return true;
}

bool TraceBase::handleSurfaceHelper(SurfaceScatterEvent &event, IntersectionTemporary &data,
                                    IntersectionInfo &info, const Medium *&medium,
                                    int bounce, bool adjoint, bool enableLightSampling, Ray &ray,
                                    Vec3f &throughput, Vec3f &emission, bool &wasSpecular,
                                    Medium::MediumState &state, Vec3f *transmittance, bool &geometricBackside, TransInfotPtr tof_info)
{
    const Bsdf &bsdf = *info.bsdf;

    // For forward events, the transport direction does not matter (since wi = -wo)
    Vec3f transparency = bsdf.eval(event.makeForwardEvent(), false);
    float transparencyScalar = transparency.avg();

    Vec3f wo;
    if (event.sampler->nextBoolean(transparencyScalar))
    {
        wo = ray.dir();
        event.pdf = transparencyScalar;
        event.weight = transparency / transparencyScalar;
        event.sampledLobe = BsdfLobes::ForwardLobe;
        throughput *= event.weight;
    }
    else
    {
        if (!adjoint)
        {
            if (enableLightSampling && bounce < _settings.maxBounces - 1)
                emission += estimateDirect(event, medium, bounce + 1, ray, transmittance, tof_info) * throughput;

            if (info.primitive->isEmissive() && bounce >= _settings.minBounces)
            {
                if (!enableLightSampling || wasSpecular || !info.primitive->isSamplable()) {
                    if ((tof_info != nullptr) && tof_info->valid()) {              // evaluate transients and the current elasped time is in range
                        if (tof_info->self_in_range()) {
                            Vec3f radiance = info.primitive->evalDirect(data, info) * throughput / tof_info->bin_pdf;
                            emission += radiance;
                            tof_info->addTransient(radiance, tof_info->elapsed_t);
                        }
                    } else {
                        emission += info.primitive->evalDirect(data, info) * throughput;
                    }
                }
            }
        }

        event.requestedLobe = BsdfLobes::AllLobes;
        if (!bsdf.sample(event, adjoint))
            return false;

        wo = event.frame.toGlobal(event.wo);

        if (!isConsistent(event, wo))
            return false;

        throughput *= event.weight;
        wasSpecular = event.sampledLobe.hasSpecular();
        if (!wasSpecular)
            ray.setPrimaryRay(false);
    }

    geometricBackside = (wo.dot(info.Ng) < 0.0f);
    medium = info.primitive->selectMedium(medium, geometricBackside);
    state.reset();

    ray = ray.scatter(ray.hitpoint(), wo, info.epsilon);

    return true;
}

bool TraceBase::handleSurface(SurfaceScatterEvent &event, IntersectionTemporary &data,
                              IntersectionInfo &info, const Medium *&medium,
                              int bounce, bool adjoint, bool enableLightSampling, Ray &ray,
                              Vec3f &throughput, Vec3f &emission, bool &wasSpecular,
                              Medium::MediumState &state, Vec3f *transmittance, TransInfotPtr tof_info)
{
    bool geometricBackside = false;
    return handleSurfaceHelper(event, data, info, medium, bounce, adjoint, enableLightSampling, 
                ray, throughput, emission, wasSpecular, state, transmittance, geometricBackside, tof_info);
}

bool TraceBase::handleSurface(SurfaceScatterEvent &event, IntersectionTemporary &data,
                              IntersectionInfo &info, const Medium *&medium, float &speedOfLight,
                              int bounce, bool adjoint, bool enableLightSampling, Ray &ray,
                              Vec3f &throughput, Vec3f &emission, bool &wasSpecular,
                              Medium::MediumState &state, Vec3f *transmittance, TransInfotPtr tof_info)
{
    bool geometricBackside = false;
    bool succeed = handleSurfaceHelper(event, data, info, medium, bounce, adjoint, enableLightSampling, 
            ray, throughput, emission, wasSpecular, state, transmittance, geometricBackside, tof_info);
    if (!succeed)
    {
        return false;
    }
    speedOfLight = info.primitive->selectSpeedOfLight(speedOfLight, info.p, geometricBackside);
    return true;
}

void TraceBase::handleInfiniteLights(IntersectionTemporary &data,
        IntersectionInfo &info, bool enableLightSampling, Ray &ray,
        Vec3f throughput, bool wasSpecular, Vec3f &emission)
{
    if (_scene->intersectInfinites(ray, data, info)) {
        if (!enableLightSampling || wasSpecular || !info.primitive->isSamplable()) {
            emission += throughput*info.primitive->evalDirect(data, info);
        }
    }
}

}
