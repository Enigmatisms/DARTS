#include "ProgressivePhotonMapIntegrator.hpp"

#include "integrators/photon_map/PhotonTracer.hpp"

#include "sampling/UniformPathSampler.hpp"
#include "sampling/SobolPathSampler.hpp"

#include "cameras/Camera.hpp"

#include "thread/ThreadUtils.hpp"
#include "thread/ThreadPool.hpp"

#include "bvh/BinaryBvh.hpp"

namespace Tungsten {

template<bool isTransient>
ProgressivePhotonMapIntegrator<isTransient>::ProgressivePhotonMapIntegrator()
: _iteration(0)
{
}

template<bool isTransient>
void ProgressivePhotonMapIntegrator<isTransient>::fromJson(JsonPtr value, const Scene &scene)
{
    PhotonMapIntegrator<isTransient>::fromJson(value, scene);
    _progressiveSettings.fromJson(value);
}

template<bool isTransient>
rapidjson::Value ProgressivePhotonMapIntegrator<isTransient>::toJson(Allocator &allocator) const
{
    return _progressiveSettings.toJson(this->_settings, allocator);
}

template<bool isTransient>
void ProgressivePhotonMapIntegrator<isTransient>::prepareForRender(TraceableScene &scene, uint32 seed)
{
    _iteration = 0;
    PhotonMapIntegrator<isTransient>::prepareForRender(scene, seed);

    for (size_t i = 0; i < this->_tracers.size(); ++i)
        _shadowSamplers.emplace_back(this->_sampler.nextI());
}

template<bool isTransient>
void ProgressivePhotonMapIntegrator<isTransient>::renderSegment(std::function<void()> completionCallback)
{
    this->_totalTracedSurfacePaths = 0;
    this->_totalTracedVolumePaths  = 0;
    this->_totalTracedPaths        = 0;
    this->_pathPhotonCount         = 0;
    this->_scene->cam().setSplatWeight(1.0/this->_nextSpp);

    using namespace std::placeholders;

    ThreadUtils::pool->yield(*ThreadUtils::pool->enqueue(
        std::bind(&ProgressivePhotonMapIntegrator::tracePhotons, this, _1, _2, _3, this->_iteration*this->_settings.photonCount),
        this->_tracers.size(),
        [](){}
    ));

    float gamma = 1.0f, volume_gamma = 1.0f;
    for (uint32 i = 1; i <= this->_iteration; ++i)
        gamma *= (i + this->_progressiveSettings.alpha)/(i + 1.0f);
    if (this->_progressiveSettings.alpha == this->_progressiveSettings.volumeAlpha)
        volume_gamma = gamma;
    else {
        for (uint32 i = 1; i <= this->_iteration; ++i)
            volume_gamma *= (i + this->_progressiveSettings.volumeAlpha)/(i + 1.0f);
    }


    float gamma1D = gamma;
    float gamma2D = std::sqrt(gamma);
    float gamma3D = std::cbrt(volume_gamma);
    float gamma2D_volume = std::sqrt(volume_gamma);

    float volumeScale = 1.f;
    float beamTemporalScale = 1.f;
    bool useBeams = this->_settings.volumePhotonType == VOLUME_BEAMS
                 || this->_settings.volumePhotonType == VOLUME_VOLUMES_BALLS
                 || this->_settings.volumePhotonType == VOLUME_MIS_VOLUMES_BALLS
                 ;
    bool beamTemporalProgressive = false;

    if (this->_settings.volumePhotonType == VOLUME_POINTS)
    {
        volumeScale = gamma3D;
    }
    else if (isTransient && this->_settings.deltaTimeGate && useBeams)
    {
        volumeScale = gamma2D_volume;
        beamTemporalScale = gamma2D_volume;
        beamTemporalProgressive = true;
    }
    else
    {
        volumeScale = gamma1D;
    }

    float surfaceRadius = this->_settings.gatherRadius * gamma2D;
    float volumeRadius = this->_settings.volumeGatherRadius * volumeScale;
    float beamTimeWidth = this->_settings.transientTimeWidth * beamTemporalScale;

    this->buildPhotonDataStructures(volumeScale, beamTemporalScale);

    if (this->_settings.frame_num > 1) {
        this->_scene->cam().transientBuffer()->initTransientBuffer(this->_settings.frame_num);
    }

    ThreadUtils::pool->yield(*ThreadUtils::pool->enqueue(
        std::bind(&ProgressivePhotonMapIntegrator::tracePixels, this, _1, _3, surfaceRadius, volumeRadius, beamTimeWidth),
        this->_tiles.size(),
        []() {}));
    if (this->_useFrustumGrid) {
        // WARNING: changing time width progressively does not currently support frustum grid
        ThreadUtils::pool->yield(*ThreadUtils::pool->enqueue(
            [&](uint32 tracerId, uint32 numTracers, uint32) {
                uint32 start = intLerp(0, this->_pathPhotonCount, tracerId,     numTracers);
                uint32 end   = intLerp(0, this->_pathPhotonCount, tracerId + 1, numTracers);
                this->_tracers[tracerId]->evalPrimaryRays(this->_beams.get(), this->_planes0D.get(), this->_planes1D.get(),
                        start, end, volumeRadius, this->_depthBuffer.get(), *this->_samplers[tracerId], this->_nextSpp - this->_currentSpp);
            }, this->_tracers.size(), [](){}
        ));
    }

    this->_currentSpp = this->_nextSpp;
    this->advanceSpp();
    _iteration++;

    this->_beams.reset();
    this->_planes0D.reset();
    this->_planes1D.reset();
    this->_volumes.reset();
    this->_hyperVolumes.reset();
    this->_balls.reset();
    this->_surfaceTree.reset();
    this->_volumeTree.reset();
    this->_volumeGrid.reset();
    this->_volumeBvh.reset();
    for (typename PhotonMapIntegrator<isTransient>::SubTaskData &data : this->_taskData) {
        data.surfaceRange.reset();
        data.volumeRange.reset();
        data.pathRange.reset();
    }
    printf("Completed, gamma: %f, volume gamma: %f\n", gamma, volume_gamma);
    completionCallback();
}

template<bool isTransient>
void ProgressivePhotonMapIntegrator<isTransient>::startRender(std::function<void()> completionCallback)
{
    if (this->done()) {
        completionCallback();
        return;
    }

    this->_group = ThreadUtils::pool->enqueue([&, completionCallback](uint32, uint32, uint32) {
        renderSegment(completionCallback);
    }, 1, [](){});
}

// explicit instantiations, required for separating implementations to cpp file.
template class ProgressivePhotonMapIntegrator<true>;
template class ProgressivePhotonMapIntegrator<false>;
}
