#ifndef TRACEBASE_HPP_
#define TRACEBASE_HPP_

#include "TraceSettings.hpp"

#include "samplerecords/SurfaceScatterEvent.hpp"
#include "samplerecords/MediumSample.hpp"
#include "samplerecords/LightSample.hpp"

#include "sampling/PathSampleGenerator.hpp"
#include "sampling/UniformSampler.hpp"
#include "sampling/Distribution1D.hpp"
#include "sampling/SampleWarp.hpp"

#include "renderer/TraceableScene.hpp"

#include "cameras/Camera.hpp"

#include "media/Medium.hpp"

#include "math/TangentFrame.hpp"
#include "math/MathUtil.hpp"
#include "math/Angle.hpp"

#include "bsdfs/Bsdf.hpp"

#include <vector>
#include <memory>
#include <cmath>

namespace Tungsten {

struct TransientInfo {
    const float time_beg;
    const float time_end;
    const float inv_interval;
    const float vacuum_sol;
    const float bin_pdf;
    bool enable_origin_sample{true};
    float elapsed_t;
    Vec3f* const transients;
    TransientInfo(
        float time_beg, float time_end, float inv_interval, float vacuum_sol, 
        float bin_pdf = 1.0f, Vec3f* const transients = nullptr
    ):  time_beg(time_beg), time_end(time_end), inv_interval(inv_interval), vacuum_sol(vacuum_sol), 
        bin_pdf(bin_pdf), elapsed_t(0.0f), transients(transients) {}
    bool valid() const {
        return (time_beg >= 0) && (time_end > 0);
    }
    bool in_range(float time) const {
        return (time <= time_end) && (time >= time_beg);
    }
    bool self_in_range() const {
        return in_range(elapsed_t);
    }

    template <typename RadianceType>
    void addTransient(RadianceType&& radiance, float path_time) {
        if (!valid() || transients == nullptr) return;
        if (path_time < time_beg || path_time >= time_end) return;
        int index = static_cast<int>(std::floor((path_time - time_beg) * inv_interval));
        transients[index] += radiance;
    }   
};
using TransInfotPtr = TransientInfo* const;

class TraceBase
{
protected:
    const TraceableScene *_scene;
    TraceSettings _settings;
    uint32 _threadId;

    // For computing direct lighting probabilities
    std::vector<float> _lightPdf;
    // For sampling light sources in adjoint light tracing
    std::unique_ptr<Distribution1D> _lightSampler;

    TraceBase(TraceableScene *scene, const TraceSettings &settings, uint32 threadId);

    bool isConsistent(const SurfaceScatterEvent &event, const Vec3f &w) const;

    template<bool ComputePdfs>
    inline Vec3f generalizedShadowRayImpl(PathSampleGenerator &sampler,
                               Ray &ray,
                               const Medium *medium,
                               const Primitive *endCap,
                               int bounce,
                               bool startsOnSurface,
                               bool endsOnSurface,
                               float &pdfForward,
                               float &pdfBackward) const;

    Vec3f attenuatedEmission(PathSampleGenerator &sampler,
                             const Primitive &light,
                             const Medium *medium,
                             float expectedDist,
                             IntersectionTemporary &data,
                             IntersectionInfo &info,
                             int bounce,
                             bool startsOnSurface,
                             Ray &ray,
                             Vec3f *transmittance, float* distance = nullptr);

    bool volumeLensSample(const Camera &camera,
                    PathSampleGenerator &sampler,
                    MediumSample &mediumSample,
                    const Medium *medium,
                    int bounce,
                    const Ray &parentRay,
                    Vec3f &weight,
                    Vec2f &pixel);

    bool surfaceLensSample(const Camera &camera,
                    SurfaceScatterEvent &event,
                    const Medium *medium,
                    int bounce,
                    const Ray &parentRay,
                    Vec3f &weight,
                    Vec2f &pixel);

    Vec3f lightSample(const Primitive &light,
                      SurfaceScatterEvent &event,
                      const Medium *medium,
                      int bounce,
                      const Ray &parentRay,
                      Vec3f *transmittance, float* distance = nullptr);

    Vec3f bsdfSample(const Primitive &light,
                     SurfaceScatterEvent &event,
                     const Medium *medium,
                     int bounce,
                     const Ray &parentRay, float* distance = nullptr);

    Vec3f volumeLightSample(PathSampleGenerator &sampler,
                        MediumSample &mediumSample,
                        const Primitive &light,
                        const Medium *medium,
                        int bounce,
                        const Ray &parentRay, float* distance = nullptr);

    Vec3f volumePhaseSample(const Primitive &light,
                        PathSampleGenerator &sampler,
                        MediumSample &mediumSample,
                        const Medium *medium,
                        int bounce,
                        const Ray &parentRay, float* distance = nullptr);

    Vec3f sampleDirect(const Primitive &light,
                       SurfaceScatterEvent &event,
                       const Medium *medium,
                       int bounce,
                       const Ray &parentRay,
                       Vec3f *transmittance, TransInfotPtr tof_info = nullptr);

    Vec3f volumeSampleDirect(const Primitive &light,
                        PathSampleGenerator &sampler,
                        MediumSample &mediumSample,
                        const Medium *medium,
                        int bounce,
                        const Ray &parentRay, TransInfotPtr tof_info = nullptr, float aux_time = 0.0f);

    const Primitive *chooseLight(PathSampleGenerator &sampler, const Vec3f &p, float &weight);
    const Primitive *chooseLightAdjoint(PathSampleGenerator &sampler, float &pdf);

    Vec3f volumeEstimateDirect(PathSampleGenerator &sampler,
                        MediumSample &mediumSample,
                        const Medium *medium,
                        int bounce,
                        const Ray &parentRay, TransInfotPtr tof_info = nullptr, float aux_time = 0.0f);

    Vec3f estimateDirect(SurfaceScatterEvent &event,
                         const Medium *medium,
                         int bounce,
                         const Ray &parentRay,
                         Vec3f *transmission, TransInfotPtr tof_info = nullptr);

    bool handleSurfaceHelper(SurfaceScatterEvent &event, IntersectionTemporary &data,
                             IntersectionInfo &info, const Medium *&medium,
                             int bounce, bool adjoint, bool enableLightSampling, Ray &ray,
                             Vec3f &throughput, Vec3f &emission, bool &wasSpecular,
                             Medium::MediumState &state, Vec3f *transmittance, bool &geometricBackside, TransInfotPtr tof_info = nullptr);

public:
    SurfaceScatterEvent makeLocalScatterEvent(IntersectionTemporary &data, IntersectionInfo &info,
                                              Ray &ray, PathSampleGenerator *sampler) const;

    Vec3f generalizedShadowRay(PathSampleGenerator &sampler,
                               Ray &ray,
                               const Medium *medium,
                               const Primitive *endCap,
                               bool startsOnSurface,
                               bool endsOnSurface,
                               int bounce) const;
    Vec3f generalizedShadowRayAndPdfs(PathSampleGenerator &sampler,
                               Ray &ray,
                               const Medium *medium,
                               const Primitive *endCap,
                               int bounce,
                               bool startsOnSurface,
                               bool endsOnSurface,
                               float &pdfForward,
                               float &pdfBackward) const;

    bool handleVolume(PathSampleGenerator &sampler, MediumSample &mediumSample,
               const Medium *&medium, int bounce, bool adjoint, bool enableLightSampling,
               Ray &ray, Vec3f &throughput, Vec3f &emission, bool &wasSpecular, 
               TransInfotPtr tof_info = nullptr, EllipseConstPtr ell_info = nullptr, bool skip_eval = false);

    bool handleSurface(SurfaceScatterEvent &event, IntersectionTemporary &data,
               IntersectionInfo &info, const Medium *&medium,
               int bounce, bool adjoint, bool enableLightSampling, Ray &ray,
               Vec3f &throughput, Vec3f &emission, bool &wasSpecular,
               Medium::MediumState &state, Vec3f *transmittance = nullptr, TransInfotPtr tof_info = nullptr);

    // handleSurface with speedOfLight
    bool handleSurface(SurfaceScatterEvent &event, IntersectionTemporary &data,
               IntersectionInfo &info, const Medium *&medium, float &speedOfLight,
               int bounce, bool adjoint, bool enableLightSampling, Ray &ray,
               Vec3f &throughput, Vec3f &emission, bool &wasSpecular,
               Medium::MediumState &state, Vec3f *transmittance = nullptr, TransInfotPtr tof_info = nullptr);

    void handleInfiniteLights(IntersectionTemporary &data,
            IntersectionInfo &info, bool enableLightSampling, Ray &ray,
            Vec3f throughput, bool wasSpecular, Vec3f &emission);
};

}

#endif /* TRACEBASE_HPP_ */
