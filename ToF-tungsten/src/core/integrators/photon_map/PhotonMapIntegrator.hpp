#ifndef PHOTONMAPINTEGRATOR_HPP_
#define PHOTONMAPINTEGRATOR_HPP_

#include "PhotonMapSettings.hpp"
#include "PhotonTracer.hpp"
#include "GridAccel.hpp"
#include "KdTree.hpp"
#include "Photon.hpp"
#include "bvh/BinaryBvh.hpp"

#include "integrators/Integrator.hpp"
#include "integrators/ImageTile.hpp"

#include "sampling/PathSampleGenerator.hpp"

#include "thread/TaskGroup.hpp"

#include "math/MathUtil.hpp"

#include <atomic>
#include <memory>
#include <vector>

namespace Tungsten {

namespace Bvh {
class BinaryBvh;
}

template<bool isTransient>
class PhotonTracer;

template<bool isTransient>
class PhotonMapIntegrator : public Integrator
{
protected:
    static CONSTEXPR uint32 TileSize = 16;
    static CONSTEXPR bool isSteadyState = !isTransient;

    typedef typename std::conditional<isSteadyState, SteadyVolumePhotonRange, Transient::TransientVolumePhotonRange>::type VolumePhotonRange;
    typedef typename std::conditional<isSteadyState, SteadyPathPhotonRange, Transient::TransientPathPhotonRange>::type PathPhotonRange;
    typedef typename std::conditional<isSteadyState, SteadyVolumePhoton, Transient::TransientVolumePhoton>::type VolumePhoton;
    typedef typename std::conditional<isSteadyState, SteadyPathPhoton, Transient::TransientPathPhoton>::type PathPhoton;
    typedef typename std::conditional<isSteadyState, SteadyPhotonBeam, Transient::TransientPhotonBeam>::type PhotonBeam;
    typedef typename std::conditional<isSteadyState, SteadyPhotonPlane0D, Transient::TransientPhotonPlane0D>::type PhotonPlane0D;
    typedef typename std::conditional<isSteadyState, SteadyPhotonVolume, Transient::TransientPhotonVolume>::type PhotonVolume;
    typedef typename std::conditional<isSteadyState, SteadyPhotonHyperVolume, Transient::TransientPhotonHyperVolume>::type PhotonHyperVolume;
    typedef typename std::conditional<isSteadyState, SteadyPhotonBall, Transient::TransientPhotonBall>::type PhotonBall;

    struct SubTaskData
    {
        SurfacePhotonRange surfaceRange;
        VolumePhotonRange volumeRange;
        PathPhotonRange pathRange;
    };
    std::vector<ImageTile> _tiles;


    PhotonMapSettings _settings;
    uint32 _w;
    uint32 _h;
    UniformSampler _sampler;

    std::shared_ptr<TaskGroup> _group;
    std::unique_ptr<GuidedSamplingInfo> _sampling_info;
    std::unique_ptr<Ray[]> _depthBuffer;

    std::atomic<uint32> _totalTracedSurfacePaths;
    std::atomic<uint32> _totalTracedVolumePaths;
    std::atomic<uint32> _totalTracedPaths;

    std::vector<Photon> _surfacePhotons;
    std::vector<VolumePhoton> _volumePhotons;
    std::vector<PathPhoton> _pathPhotons;
    std::unique_ptr<PhotonBeam[]> _beams;
    std::unique_ptr<PhotonPlane0D[]> _planes0D;
    std::unique_ptr<PhotonPlane1D[]> _planes1D;
    std::unique_ptr<PhotonVolume[]> _volumes;
    std::unique_ptr<PhotonHyperVolume[]> _hyperVolumes;
    std::unique_ptr<PhotonBall[]> _balls;

    uint32 _pathPhotonCount;

    std::unique_ptr<KdTree<Photon>> _surfaceTree;
    std::unique_ptr<KdTree<VolumePhoton>> _volumeTree;
    std::unique_ptr<Bvh::BinaryBvh> _volumeBvh;
    std::unique_ptr<GridAccel> _volumeGrid;

    std::vector<std::unique_ptr<PhotonTracer<isTransient>>> _tracers;
    std::vector<SubTaskData> _taskData;
    std::vector<std::unique_ptr<PathSampleGenerator>> _samplers;

    bool _useFrustumGrid;

    void diceTiles();

    virtual void saveState(OutputStreamHandle &out) override;
    virtual void loadState(InputStreamHandle &in) override;

    void tracePhotons(uint32 taskId, uint32 numSubTasks, uint32 threadId, uint32 sampleBase);
    void tracePixels(uint32 tileId, uint32 threadId, float surfaceRadius, float volumeRadius, float beamTimeWidth);

    bool canBuildVolume(const PathPhoton &p0, const PathPhoton &p1, const PathPhoton &p2, const PathPhoton &p3);

    void buildPointBvh(uint32 tail, float volumeRadiusScale);
    void buildBeamBvh(uint32 tail, float volumeRadiusScale);
    void buildBeamGrid(uint32 tail, float volumeRadiusScale);
    void buildPlaneBvh(uint32 tail, float volumeRadiusScale);
    void buildPlaneGrid(uint32 tail, float volumeRadiusScale);
    void buildVolumeBvh(uint32 tail, float volumeRadiusScale);
    void buildVolumeGrid(uint32 tail, float volumeRadiusScale);
    void buildHyperVolumeBvh(uint32 tail, float volumeRadiusScale);
    void buildHyperVolumeGrid(uint32 tail, float volumeRadiusScale);
    void buildBallBvh(uint32 tail, float volumeRadiusScale);
    void buildVolumeBallBvh(uint32 tail, float volumeRadiusScale, bool enableMIS);

    void buildPhotonDataStructures(float volumeRadiusScale, float beamTemporalRadiusScale = 1.f);

    void renderSegment(std::function<void()> completionCallback);

    bool precomputeVolume(PhotonVolume &volume, const PathPhoton &p0, const PathPhoton &p1, const PathPhoton &p2, const PathPhoton &p3);
    bool precomputeBall(PhotonBall &ball, const PathPhoton &p0, const PathPhoton &p1);

    static void precomputeBeam(PhotonBeam &beam, const PathPhoton &p0, const PathPhoton &p1);
    static void precomputePlane0D(PhotonPlane0D &plane, const PathPhoton &p0, const PathPhoton &p1, const PathPhoton &p2);
    static bool precomputeUnslicedVolume(PhotonVolume &volume, const PathPhoton &p0, const PathPhoton &p1, const PathPhoton &p2, const PathPhoton &p3);
    static void precomputeHyperVolume(PhotonHyperVolume &hyperVolume, const PathPhoton &p0, const PathPhoton &p1, const PathPhoton &p2, const PathPhoton &p3, const PathPhoton &p4);
    static bool precomputeUnslicedBall(PhotonBall &ball, const PathPhoton &p0, const PathPhoton &p1);
    static void insertDicedBeam(Bvh::PrimVector &beams, PhotonBeam &beam, uint32 i, const PathPhoton &p0, const PathPhoton &p1, float radius);

    static bool precomputeSlicedVolume(Transient::TransientPhotonVolume &volume,
                                       const PathPhoton &p0,
                                       const PathPhoton &p1,
                                       const PathPhoton &p2,
                                       const PathPhoton &p3,
                                       float timeGate,
                                       float timeGateBeg,
                                       float timeGateEnd,
                                       bool enableTimeCull = true);
    static bool precomputeSlicedBall(Transient::TransientPhotonBall &ball,
                                     const PathPhoton &p0,
                                     const PathPhoton &p1,
                                     float timeGate,
                                     float timeGateBeg,
                                     float timeGateEnd,
                                     bool enableTimeCull = true);

public:
    PhotonMapIntegrator();
    ~PhotonMapIntegrator();

    virtual void fromJson(JsonPtr value, const Scene &scene) override;
    virtual rapidjson::Value toJson(Allocator &allocator) const override;

    virtual void prepareForRender(TraceableScene &scene, uint32 seed) override;
    virtual void teardownAfterRender() override;

    virtual void startRender(std::function<void()> completionCallback) override;
    virtual void waitForCompletion() override;
    virtual void abortRender() override;

    virtual void setTimeCenter(float timeCenter) override;
};

}

#endif /* PHOTONMAPINTEGRATOR_HPP_ */
