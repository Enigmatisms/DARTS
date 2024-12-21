#include "PhotonMapIntegrator.hpp"
#include "PhotonTracer.hpp"

#include "sampling/UniformPathSampler.hpp"
#include "sampling/SobolPathSampler.hpp"

#include "cameras/PinholeCamera.hpp"

#include "thread/ThreadUtils.hpp"
#include "thread/ThreadPool.hpp"

#include <atomic>
#include <chrono>
static std::atomic<long> photon_pass_time = 0;
static std::atomic<long> sensor_pass_time = 0;
static std::atomic<int> photon_pass_call_cnt = 0;
static std::atomic<int> sensor_pass_call_cnt = 0;

namespace Tungsten {

template<bool isTransient>
PhotonMapIntegrator<isTransient>::PhotonMapIntegrator()
: _w(0),
  _h(0),
  _sampler(0xBA5EBA11)
{
}

template <bool isTransient>
PhotonMapIntegrator<isTransient>::~PhotonMapIntegrator()
{
}

template<bool isTransient>
void PhotonMapIntegrator<isTransient>::diceTiles()
{
    for (uint32 y = 0; y < _h; y += TileSize) {
        for (uint32 x = 0; x < _w; x += TileSize) {
            _tiles.emplace_back(
                x,
                y,
                min(TileSize, _w - x),
                min(TileSize, _h - y),
                _scene->rendererSettings().useSobol() ?
                    std::unique_ptr<PathSampleGenerator>(new SobolPathSampler(MathUtil::hash32(_sampler.nextI()))) :
                    std::unique_ptr<PathSampleGenerator>(new UniformPathSampler(MathUtil::hash32(_sampler.nextI())))
            );
        }
    }
}

template<bool isTransient>
void PhotonMapIntegrator<isTransient>::saveState(OutputStreamHandle &/*out*/)
{
}

template <bool isTransient>
void PhotonMapIntegrator<isTransient>::loadState(InputStreamHandle & /*in*/)
{
}

template <bool isTransient>
void PhotonMapIntegrator<isTransient>::tracePhotons(uint32 taskId, uint32 numSubTasks, uint32 threadId, uint32 sampleBase)
{
    SubTaskData &data = _taskData[taskId];
    PathSampleGenerator &sampler = *_samplers[taskId];

    uint32 photonBase    = intLerp(0, _settings.photonCount, taskId + 0, numSubTasks);
    uint32 photonsToCast = intLerp(0, _settings.photonCount, taskId + 1, numSubTasks) - photonBase;

    uint32 totalSurfaceCast = 0;
    uint32 totalVolumeCast = 0;
    uint32 totalPathsCast = 0;
    for (uint32 i = 0; i < photonsToCast; ++i) {
        sampler.startPath(0, sampleBase + photonBase + i);
        // sample a time point here
        float bin_pdf = 1.f, remaining_time = 0.0;
        if (_sampling_info) {
            bin_pdf = _settings.transientTimeWidth / _sampling_info->time_range;        // normally, 1
            remaining_time = _sampling_info->sample_time_point(sampler, bin_pdf, _settings.transientTimeBeg, _settings.transientTimeWidth);
        }
        auto start_time = std::chrono::steady_clock::now();
        _tracers[threadId]->tracePhotonPath(
            data.surfaceRange,
            data.volumeRange,
            data.pathRange,
            sampler,
            _sampling_info.get(),
            bin_pdf,
            remaining_time
        );
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start_time).count();
        photon_pass_time += duration;
        photon_pass_call_cnt += 1;

        if (!data.surfaceRange.full())
            totalSurfaceCast++;
        if (!data.volumeRange.full())
            totalVolumeCast++;
        if (!data.pathRange.full())
            totalPathsCast++;
        if (data.surfaceRange.full() && data.volumeRange.full() && data.pathRange.full())
            break;

        if (_group->isAborting())
            break;
    }

    _totalTracedSurfacePaths += totalSurfaceCast;
    _totalTracedVolumePaths += totalVolumeCast;
    _totalTracedPaths += totalPathsCast;
}

template <bool isTransient>
void PhotonMapIntegrator<isTransient>::tracePixels(uint32 tileId, uint32 threadId, float surfaceRadius, float volumeRadius, float beamTimeWidth)
{
    int spp = _nextSpp - _currentSpp;
    
    // initialize transientBuffer here
    ImageTile &tile = _tiles[tileId];
    for (uint32 y = 0; y < tile.h; ++y) {
        for (uint32 x = 0; x < tile.w; ++x) {
            Vec2u pixel(tile.x + x, tile.y + y);
            uint32 pixelIndex = pixel.x() + pixel.y()*_w;

            Ray dummyRay;
            Ray *depthRay = _depthBuffer ? &_depthBuffer[pixel.x() + pixel.y()*_w] : &dummyRay;
            for (int i = 0; i < spp; ++i) {
                tile.sampler->startPath(pixelIndex, _currentSpp + i);
                std::unique_ptr<Vec3f[]> transients;
                if (_settings.frame_num > 1) {
                    transients.reset(new Vec3f[_settings.frame_num]);
                    for (int i = 0; i < _settings.frame_num; i++)
                        transients[i] = Vec3f(0.0f);
                }
                auto start_time = std::chrono::steady_clock::now();
                Vec3f c = _tracers[threadId]->traceSensorPath(pixel,
                    *_surfaceTree,
                    _volumeTree.get(),
                    _volumeBvh.get(),
                    _volumeGrid.get(),
                    _beams.get(),
                    _planes0D.get(),
                    _planes1D.get(),
                    _volumes.get(),
                    _hyperVolumes.get(),
                    _balls.get(),
                    *tile.sampler,
                    surfaceRadius,
                    volumeRadius,
                    beamTimeWidth,
                    _settings.volumePhotonType,
                    *depthRay,
                    _useFrustumGrid,
                    transients.get()
                );
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start_time).count();
                sensor_pass_time     += duration;
                sensor_pass_call_cnt += 1;
                _scene->cam().colorBuffer()->addSample(pixel, c);

                if (transients)
                    _scene->cam().tryAddTransient(transients.get(), pixel);
                // this can be done, with... some effort
                // ASSERT(c.min() >= 0.0f, "c is negative");
                // ASSERT(!std::isnan(c), "c is nan");
            }
            if (_group->isAborting())
                break;
        }
    }
}

template<typename PhotonType>
std::unique_ptr<KdTree<PhotonType>> streamCompactAndBuild(std::vector<PhotonRange<PhotonType>> ranges,
        std::vector<PhotonType> &photons, uint32 totalTraced)
{
    uint32 tail = streamCompact(ranges);

    float scale = 1.0f/totalTraced;
    for (uint32 i = 0; i < tail; ++i)
        photons[i].power *= scale;

    return std::unique_ptr<KdTree<PhotonType>>(new KdTree<PhotonType>(&photons[0], tail));
}

template<bool isTransient>
void PhotonMapIntegrator<isTransient>::precomputeBeam(PhotonBeam &beam, const PathPhoton &p0, const PathPhoton &p1)
{
    beam.p0 = p0.pos;
    beam.p1 = p1.pos;
    beam.dir = p0.dir;
    beam.length = p0.length;
    beam.power = p1.power;
    if constexpr (isTransient)
        beam.timeTraveled = p0.timeTraveled;
    beam.bounce = p0.bounce();
    beam.valid = true;
}

template<bool isTransient>
void PhotonMapIntegrator<isTransient>::precomputePlane0D(PhotonPlane0D &plane, const PathPhoton &p0, const PathPhoton &p1, const PathPhoton &p2)
{
    Vec3f d1 = p1.dir*p1.sampledLength;

    plane.p0 = p0.pos;
    plane.p1 = p1.pos;
    plane.p2 = p1.pos + d1;
    plane.p3 = p0.pos + d1;
    plane.power = p0.length * p1.sampledLength * p2.power;
    plane.d1 = p1.dir;
    plane.l1 = p1.sampledLength;
    plane.bounce = int(p1.bounce());
    plane.valid = true;
    if constexpr (isTransient)
        plane.timeTraveled = p0.timeTraveled;
}

static void precomputePlane1D(PhotonPlane1D &plane, const SteadyPathPhoton &p0, const SteadyPathPhoton &p1, const SteadyPathPhoton &p2, float radius)
{
    Vec3f a = p1.pos - p0.pos;
    Vec3f b = p1.dir*p1.sampledLength;
    Vec3f c = 2.0f*a.cross(p1.dir).normalized()*radius;
    float det = std::abs(a.dot(b.cross(c)));

    if (std::isnan(c.sum()) || det < 1e-8f)
        return;

    float invDet = 1.0f/det;
    Vec3f u = invDet*b.cross(c);
    Vec3f v = invDet*c.cross(a);
    Vec3f w = invDet*a.cross(b);

    plane.p = p0.pos - c*0.5f;
    plane.invDet = invDet;
    plane.invU = u;
    plane.invV = v;
    plane.invW = w;
    plane.binCount = a.length()/(2.0f*radius);
    plane.valid = true;

    plane.center = p0.pos + a*0.5f + b*0.5f;
    plane.a = a*0.5f;
    plane.b = b*0.5f;
    plane.c = c*0.5f;

    plane.d1 = p1.dir;
    plane.l1 = p1.sampledLength;
    plane.power = p0.length*p1.sampledLength*p2.power*std::abs(invDet);
    plane.bounce = p1.bounce();
}

template <bool isTransient>
bool PhotonMapIntegrator<isTransient>::precomputeUnslicedVolume(PhotonVolume &volume, const PathPhoton &p0, const PathPhoton &p1, const PathPhoton &p2, const PathPhoton &p3)
{
    Vec3f a = p0.dir * p0.sampledLength;
    Vec3f b = p1.dir * p1.sampledLength;
    Vec3f c = p2.dir * p2.sampledLength;
    // TODO: optimization: avoid normalizing determinant (like precomputePlane1D())
    float det = std::abs(a.dot(b.cross(c))) / (a.length() * b.length() * c.length());
    if (!std::isfinite(det) || det < 1e-4f)
    {
        volume.valid = false;
        return false;
    }

    float invDet = 1.0f / det;

    volume.p = p0.pos;
    volume.a = a;
    volume.b = b;
    volume.c = c;

    volume.aDir = p0.dir;
    volume.bDir = p1.dir;
    volume.cDir = p2.dir;
    volume.aLen = p0.sampledLength;
    volume.bLen = p1.sampledLength;
    volume.cLen = p2.sampledLength;

    if (volume.aLen == 0.f || volume.bLen == 0.f || volume.cLen == 0.f)
    {
        volume.valid = false;
        return false;
    }

    volume.power = p3.power;
    volume.invDet = invDet;
    volume.bounce = int(p2.bounce());
    volume.valid = true;

    if constexpr (isTransient)
    {
        volume.timeTraveled = p0.timeTraveled;
        volume.n = volume.aDir.cross(volume.bDir) + volume.bDir.cross(volume.cDir) + volume.cDir.cross(volume.aDir);
        volume.isDeltaSliced = false;
    }

    return true;
}

template <bool isTransient>
bool PhotonMapIntegrator<isTransient>::precomputeSlicedVolume(Transient::TransientPhotonVolume &volume,
                                                              const PathPhoton &p0,
                                                              const PathPhoton &p1,
                                                              const PathPhoton &p2,
                                                              const PathPhoton &p3,
                                                              float timeGate,
                                                              float timeGateBeg,
                                                              float timeGateEnd,
                                                              bool enableTimeCull)
{
    bool valid = precomputeUnslicedVolume(volume, p0, p1, p2, p3);
    if (!volume.valid)
    {
        return false;
    }

    if (enableTimeCull && timeGate < volume.timeTraveled)
    {
        return false;
    }

    const Medium *medium = p1.scatter().medium();
    volume.isDeltaSliced = true;
    float dt = timeGate - volume.timeTraveled;
    volume.sampledDeltaTime = dt;
    volume.p0 = medium->travel(p0.pos, p0.dir, dt);
    volume.p1 = medium->travel(p0.pos, p1.dir, dt);
    volume.p2 = medium->travel(p0.pos, p2.dir, dt);

    float maxDist = volume.aLen + volume.bLen + volume.cLen;
    if (enableTimeCull && medium->speedOfLight(p0.pos) * dt > maxDist)
    {
        return false;
    }

    if (enableTimeCull && volume.bounds().empty())
    {
        return false;
    }
    return true;
}

template <bool isTransient>
bool PhotonMapIntegrator<isTransient>::precomputeVolume(PhotonVolume &volume, const PathPhoton &p0, const PathPhoton &p1, const PathPhoton &p2, const PathPhoton &p3)
{
    bool valid = true;
    if constexpr (!isTransient)
    {
        valid = precomputeUnslicedVolume(volume, p0, p1, p2, p3);
    }
    else
    {
        float timeGate = lerp(_settings.transientTimeBeg, _settings.transientTimeEnd, _sampler.next1D());
        valid = precomputeSlicedVolume(volume, p0, p1, p2, p3, timeGate, _settings.transientTimeBeg, _settings.transientTimeEnd);
    }
    return valid;
}

template <bool isTransient>
void PhotonMapIntegrator<isTransient>::precomputeHyperVolume(PhotonHyperVolume &hyperVolume, const PathPhoton &p0, const PathPhoton &p1, const PathPhoton &p2, const PathPhoton &p3, const PathPhoton &p4)
{
    Vec3f a = p0.dir * p0.sampledLength;
    Vec3f b = p1.dir * p1.sampledLength;
    Vec3f c = p2.dir * p2.sampledLength;
    Vec3f d = p3.dir * p3.sampledLength;
    Vec3f aDir = p0.dir;
    Vec3f bDir = p1.dir;
    Vec3f cDir = p2.dir;
    Vec3f dDir = p3.dir;

    hyperVolume.p = p0.pos;
    hyperVolume.a = a;
    hyperVolume.b = b;
    hyperVolume.c = c;
    hyperVolume.d = d;
    hyperVolume.aDir = aDir;
    hyperVolume.bDir = bDir;
    hyperVolume.cDir = cDir;
    hyperVolume.dDir = dDir;
    hyperVolume.aLen = p0.sampledLength;
    hyperVolume.bLen = p1.sampledLength;
    hyperVolume.cLen = p2.sampledLength;
    hyperVolume.dLen = p3.sampledLength;

    hyperVolume.bounce = int(p3.bounce());
    hyperVolume.power = p4.power;
    hyperVolume.valid = true;

    if constexpr (isTransient)
    {
        hyperVolume.timeTraveled = p0.timeTraveled;
        float jacobian = abs(
            +aDir.dot(bDir.cross(cDir))
            -bDir.dot(cDir.cross(dDir))
            +cDir.dot(dDir.cross(aDir))
            -dDir.dot(aDir.cross(bDir))
        );
        hyperVolume.invJacobian = 1.0f / jacobian;
        if (jacobian < 1e-9f)
        {
            hyperVolume.valid = false;
            return;
        }
    }
}

template <bool isTransient>
bool PhotonMapIntegrator<isTransient>::precomputeUnslicedBall(PhotonBall &ball, const PathPhoton &p0, const PathPhoton &p1)
{
    ball.p = p0.pos;
    ball.r = p0.sampledLength;
    if (ball.r == 0.f)
    {
        ball.valid = false;
        return false;
    }
    ball.power = p1.power;
    ball.scatter = p0.scatter();
    ball.valid = true;
    ball.bounce = p0.bounce();
    if constexpr (isTransient)
    {
        ball.isDeltaSliced = false;
        ball.timeTraveled = p0.timeTraveled;
    }
    return true;
}

template <bool isTransient>
bool PhotonMapIntegrator<isTransient>::precomputeSlicedBall(Transient::TransientPhotonBall &ball,
                                                            const PathPhoton &p0,
                                                            const PathPhoton &p1,
                                                            float timeGate,
                                                            float timeGateBeg,
                                                            float timeGateEnd,
                                                            bool enableTimeCull)
{
    bool valid = precomputeUnslicedBall(ball, p0, p1);
    if (!valid)
    {
        return false;
    }

    if (enableTimeCull && timeGate < ball.timeTraveled)
    {
        return false;
    }

    const Medium *medium = p1.inMedium;
    ball.isDeltaSliced = true;
    float dt = timeGate - ball.timeTraveled;
    ball.temporalRadius = medium->speedOfLight(p0.pos) * dt;
    if (enableTimeCull && ball.temporalRadius > ball.r)
    {
        return false;
    }

    return true;
}

template <bool isTransient>
bool PhotonMapIntegrator<isTransient>::precomputeBall(PhotonBall &ball, const PathPhoton &p0, const PathPhoton &p1)
{
    bool valid = true;
    if constexpr (!isTransient)
    {
        valid = precomputeUnslicedBall(ball, p0, p1);
    }
    else
    {
        float timeGate = lerp(_settings.transientTimeBeg, _settings.transientTimeEnd, _sampler.next1D());
        valid = precomputeSlicedBall(ball, p0, p1, timeGate, _settings.transientTimeBeg, _settings.transientTimeEnd);
    }
    return valid;
}

template <bool isTransient>
void PhotonMapIntegrator<isTransient>::insertDicedBeam(Bvh::PrimVector &beams, PhotonBeam &beam, uint32 i, const PathPhoton &p0, const PathPhoton &p1, float radius)
{
    precomputeBeam(beam, p0, p1);

    Vec3f absDir = std::abs(p0.dir);
    int majorAxis = absDir.maxDim();
    int numSteps = min(64, max(1, int(absDir[majorAxis]*16.0f)));

    Vec3f minExtend = Vec3f(radius);
    for (int j = 0; j < 3; ++j) {
        minExtend[j] = std::copysign(minExtend[j], p0.dir[j]);
        if (j != majorAxis)
            minExtend[j] /= std::sqrt(max(0.0f, 1.0f - sqr(p0.dir[j])));
    }
    for (int j = 0; j < numSteps; ++j) {
        Vec3f v0 = p0.pos + p0.dir*p0.length*(j + 0)/numSteps;
        Vec3f v1 = p0.pos + p0.dir*p0.length*(j + 1)/numSteps;
        for (int k = 0; k < 3; ++k) {
            if (k != majorAxis || j ==            0) v0[k] -= minExtend[k];
            if (k != majorAxis || j == numSteps - 1) v1[k] += minExtend[k];
        }
        Box3f bounds;
        bounds.grow(v0);
        bounds.grow(v1);

        beams.emplace_back(Bvh::Primitive(bounds, bounds.center(), i));
    }
}

template <bool isTransient>
bool PhotonMapIntegrator<isTransient>::canBuildVolume(const PathPhoton &p0, const PathPhoton &p1, const PathPhoton &p2, const PathPhoton &p3)
{
    bool onSurface = p2.onSurface() || p1.onSurface();
    bool isSampledLengthValid = p1.sampledLength > 0.0f && p2.sampledLength > 0.0f;
    bool canSpawnVolume = p3.bounce() > 2 && !onSurface && isSampledLengthValid;
    return canSpawnVolume;
}

template <bool isTransient>
void PhotonMapIntegrator<isTransient>::buildPointBvh(uint32 tail, float volumeRadiusScale)
{
    float radius = _settings.volumeGatherRadius*volumeRadiusScale;

    Bvh::PrimVector points;
    for (uint32 i = 0; i < tail; ++i) {
        Box3f bounds(_pathPhotons[i].pos);
        bounds.grow(radius);
        points.emplace_back(Bvh::Primitive(bounds, _pathPhotons[i].pos, i));
    }

    _volumeBvh.reset(new Bvh::BinaryBvh(std::move(points), 1));
}
template <bool isTransient>
void PhotonMapIntegrator<isTransient>::buildBeamBvh(uint32 tail, float volumeRadiusScale)
{
    float radius = _settings.volumeGatherRadius*volumeRadiusScale;

    Bvh::PrimVector beams;
    for (uint32 i = 0; i < tail; ++i) {
        if (_pathPhotons[i].bounce() == 0)
            continue;

        if (_settings.excludeNonMIS)
        {
            if (_pathPhotons[i].bounce() <= 2)
                continue;

            if (!canBuildVolume(_pathPhotons[i - 3], _pathPhotons[i - 2], _pathPhotons[i - 1], _pathPhotons[i]))
                continue;
        }

        bool lowOrderValid = _settings.lowOrderScattering && !_pathPhotons[i - 1].isGhostBounce();
        if (!_pathPhotons[i - 1].onSurface() || lowOrderValid)
            insertDicedBeam(beams, _beams[i], i, _pathPhotons[i - 1], _pathPhotons[i], radius);
    }

    _volumeBvh.reset(new Bvh::BinaryBvh(std::move(beams), 1));
}
template <bool isTransient>
void PhotonMapIntegrator<isTransient>::buildPlaneBvh(uint32 tail, float volumeRadiusScale)
{
    float radius = _settings.volumeGatherRadius*volumeRadiusScale;

    Bvh::PrimVector planes;
    for (uint32 i = 0; i < tail; ++i) {
        const PathPhoton &p1 = _pathPhotons[i - 1];
        const PathPhoton &p2 = _pathPhotons[i - 0];

        if (p2.bounce() > 0 && p2.bounce() > p1.bounce() && p1.onSurface() && _settings.lowOrderScattering)
            insertDicedBeam(planes, _beams[i], i, p1, p2, radius);
        if (p2.bounce() > 1 && !p1.onSurface() && p1.sampledLength > 0.0f) {
            const PathPhoton &p0 = _pathPhotons[i - 2];
            if (_settings.volumePhotonType == VOLUME_PLANES) {
                precomputePlane0D(_planes0D[i], p0, p1, p2);
                Box3f bounds = _planes0D[i].bounds();
                planes.emplace_back(Bvh::Primitive(bounds, bounds.center(), i));
            } else {
                precomputePlane1D(_planes1D[i], p0, p1, p2, radius);
                if (_planes1D[i].valid) {
                    Box3f bounds = _planes1D[i].bounds();
                    planes.emplace_back(Bvh::Primitive(bounds, bounds.center(), i));
                }
            }
        }
    }

    _volumeBvh.reset(new Bvh::BinaryBvh(std::move(planes), 1));
}
template <bool isTransient>
void PhotonMapIntegrator<isTransient>::PhotonMapIntegrator::buildVolumeBvh(uint32 tail, float volumeRadiusScale)
{
    Bvh::PrimVector prims;
    for (uint32 i = 3; i < tail; ++i)
    {
        // TODO: support lowOrderScattering
        const PathPhoton &p0 = _pathPhotons[i - 3];
        const PathPhoton &p1 = _pathPhotons[i - 2];
        const PathPhoton &p2 = _pathPhotons[i - 1];
        const PathPhoton &p3 = _pathPhotons[i - 0];

        bool onSurface = p2.onSurface() || p1.onSurface();
        bool isSampledLengthValid = p1.sampledLength > 0.0f && p2.sampledLength > 0.0f;
        if (p3.bounce() > 2 && !onSurface && isSampledLengthValid)
        {
            bool valid = precomputeVolume(_volumes[i], p0, p1, p2, p3);
            if (valid)
            {
                Box3f bounds = _volumes[i].bounds();
                prims.emplace_back(Bvh::Primitive(bounds, bounds.center(), i));
            }
        }
    }

    _volumeBvh.reset(new Bvh::BinaryBvh(std::move(prims), 1));
}

// TODO: deduplicate with PhotonTracer.cpp
static inline bool occuluded(const TraceableScene *scene, Vec3f origin, Vec3f dir, const float len, float nearT = 1e-4f)
{
    Ray shadowRay = Ray(origin, dir, nearT, len);
    shadowRay.setNearT(nearT);
    return scene->occluded(shadowRay);
}

template <bool isTransient>
void PhotonMapIntegrator<isTransient>::PhotonMapIntegrator::buildVolumeBallBvh(uint32 tail, float volumeRadiusScale, bool enableMIS)
{
    Bvh::PrimVector prims;
    for (uint32 i = 0; i < tail; ++i)
    {
        const PathPhoton &p3 = _pathPhotons[i - 0];
        if (p3.bounce() > 0 && _settings.lowOrderScattering && !_settings.excludeNonMIS)
        {
            const PathPhoton &p2 = _pathPhotons[i - 1];
            bool prevTwoOnSurface = i == 1 ? p2.onSurface() : p2.onSurface() || _pathPhotons[i - 2].onSurface();
            bool samePath = p3.bounce() > p2.bounce();
            bool isSpecular = p2.scatter().isSpecular();
            bool isGhostBounce = p2.isGhostBounce();
            if (prevTwoOnSurface && samePath && !isGhostBounce)
            {
                if (!isSpecular)
                {
                    bool valid = precomputeBall(_balls[i], p2, p3);
                    if (valid)
                    {
                        Box3f bounds = _balls[i].bounds();
                        TypedPhotonIdx idx(VOLUME_BALLS, i);
                        prims.emplace_back(Bvh::Primitive(bounds, bounds.center(), static_cast<uint32>(idx)));
                    }
                }
                else
                {
                    // create beam
                    float radius = _settings.volumeGatherRadius * volumeRadiusScale;
                    TypedPhotonIdx idx(VOLUME_BEAMS, i);
                    insertDicedBeam(prims, _beams[i], static_cast<uint32>(idx), p2, p3, radius);
                }
            }
        }

        if (p3.bounce() > 2)
        {
            const PathPhoton &p0 = _pathPhotons[i - 3];
            const PathPhoton &p1 = _pathPhotons[i - 2];
            const PathPhoton &p2 = _pathPhotons[i - 1];

            if (!canBuildVolume(p0, p1, p2, p3))
            {
                continue;
            }

            if (!enableMIS)
            {
                if (precomputeVolume(_volumes[i], p0, p1, p2, p3))
                {
                    Box3f bounds = _volumes[i].bounds();
                    TypedPhotonIdx idx(VOLUME_VOLUMES, i);
                    prims.emplace_back(Bvh::Primitive(bounds, bounds.center(), static_cast<uint32>(idx)));
                }
            }
            else
            {
                if constexpr (isTransient)
                {
                    float timeGate = lerp(_settings.transientTimeBeg, _settings.transientTimeEnd, _sampler.next1D());

                    bool isectVolume = _sampler.next1D() < 0.5f;
                    bool volumeEnableTimeCull = isectVolume;
                    bool ballEnableTimeCull = !isectVolume;

                    bool volumeValid = precomputeSlicedVolume(_volumes[i], p0, p1, p2, p3, timeGate, _settings.transientTimeBeg, _settings.transientTimeEnd, volumeEnableTimeCull);
                    _volumes[i].valid = volumeValid;
                    bool ballValid = precomputeSlicedBall(_balls[i], p2, p3, timeGate, _settings.transientTimeBeg, _settings.transientTimeEnd, ballEnableTimeCull);
                    _balls[i].valid = ballValid;

                    if (isectVolume)
                    {
                        if (volumeValid)
                        {
                            Box3f bounds = _volumes[i].bounds();
                            TypedPhotonIdx idx(VOLUME_VOLUMES, i);
                            prims.emplace_back(Bvh::Primitive(bounds, bounds.center(), static_cast<uint32>(idx)));
                        }
                    }
                    else
                    {
                        bool prevPathOcculuded = false;
                        if (p3.isGhostBounce())
                        {
                            prevPathOcculuded = occuluded(_scene, p0.pos, p0.dir, p0.sampledLength)
                                             || occuluded(_scene, p1.pos, p1.dir, p1.sampledLength)
                                              ;
                        }

                        if (ballValid && !prevPathOcculuded)
                        {
                            Box3f bounds = _balls[i].bounds();
                            TypedPhotonIdx idx(VOLUME_BALLS, i);
                            prims.emplace_back(Bvh::Primitive(bounds, bounds.center(), static_cast<uint32>(idx)));
                        }
                    }
                }
                else
                {
                    FAIL("MIS does not support non transient mode");
                }

            }
        }
    }

    _volumeBvh.reset(new Bvh::BinaryBvh(std::move(prims), 1));
}

template <bool isTransient>
void PhotonMapIntegrator<isTransient>::PhotonMapIntegrator::buildHyperVolumeBvh(uint32 tail, float volumeRadiusScale)
{
    Bvh::PrimVector hyperVolumes;
    for (uint32 i = 4; i < tail; ++i) {
        // TODO: support lowOrderScattering
        const PathPhoton &p0 = _pathPhotons[i - 4];
        const PathPhoton &p1 = _pathPhotons[i - 3];
        const PathPhoton &p2 = _pathPhotons[i - 2];
        const PathPhoton &p3 = _pathPhotons[i - 1];
        const PathPhoton &p4 = _pathPhotons[i - 0];

        bool onSurface = p1.onSurface() || p2.onSurface() || p3.onSurface();
        bool isSampledLengthValid = p1.sampledLength > 0.0f && p2.sampledLength > 0.0f && p3.sampledLength > 0.0f;
        if (p4.bounce() > 3 && !onSurface && isSampledLengthValid) {
            precomputeHyperVolume(_hyperVolumes[i], p0, p1, p2, p3, p4);
            Box3f bounds = _hyperVolumes[i].bounds();
            hyperVolumes.emplace_back(Bvh::Primitive(bounds, bounds.center(), i));
        }
    }

    _volumeBvh.reset(new Bvh::BinaryBvh(std::move(hyperVolumes), 1));
}

template <bool isTransient>
void PhotonMapIntegrator<isTransient>::buildBallBvh(uint32 tail, float volumeRadiusScale)
{
    ASSERT(isTransient, "only support ball for transient for now");
    Bvh::PrimVector prims;

    for (uint32 i = 1; i < tail; ++i)
    {
        const PathPhoton &p0 = _pathPhotons[i - 1];
        const PathPhoton &p1 = _pathPhotons[i - 0];

        bool samePath = p1.bounce() > p0.bounce();
        bool isSpecular = p0.scatter().isSpecular();
        bool isGhostBounce = p0.isGhostBounce();
        bool pathValid = samePath && !isGhostBounce && !isSpecular;
        if (!pathValid)
        {
            continue;
        }

        if (p0.bounce() <= 0)
        {
            continue;
        }
        if (!precomputeBall(_balls[i], p0, p1))
        {
            continue;
        }

        Box3f bounds = _balls[i].bounds();
        prims.emplace_back(Bvh::Primitive(bounds, bounds.center(), i));
    }

    _volumeBvh.reset(new Bvh::BinaryBvh(std::move(prims), 1));
}

template <bool isTransient>
void PhotonMapIntegrator<isTransient>::buildBeamGrid(uint32 tail, float volumeRadiusScale)
{
    float radius = _settings.volumeGatherRadius*volumeRadiusScale;

    std::vector<GridAccel::Primitive> beams;
    for (uint32 i = 0; i < tail; ++i) {
        if (_pathPhotons[i].bounce() == 0)
            continue;

        const PathPhoton &p0 = _pathPhotons[i - 1];
        const PathPhoton &p1 = _pathPhotons[i - 0];

        if (!_pathPhotons[i - 1].onSurface() || _settings.lowOrderScattering) {
            precomputeBeam(_beams[i], p0, p1);
            beams.emplace_back(GridAccel::Primitive(i, p0.pos, p1.pos, Vec3f(0.0f), Vec3f(0.0f), radius, VOLUME_BEAMS));
        }
    }

    _volumeGrid.reset(new GridAccel(_scene->bounds(), _settings.gridMemBudgetKb, std::move(beams)));
}
template <bool isTransient>
void PhotonMapIntegrator<isTransient>::buildPlaneGrid(uint32 tail, float volumeRadiusScale)
{
    float radius = _settings.volumeGatherRadius*volumeRadiusScale;

    std::vector<GridAccel::Primitive> prims;
    for (uint32 i = 0; i < tail; ++i) {
        const PathPhoton &p1 = _pathPhotons[i - 1];
        const PathPhoton &p2 = _pathPhotons[i - 0];

        if (p2.bounce() > 0 && p2.bounce() > p1.bounce() && p1.onSurface() && _settings.lowOrderScattering) {
            precomputeBeam(_beams[i], p1, p2);
            prims.emplace_back(GridAccel::Primitive(i, p1.pos, p2.pos, Vec3f(0.0f), Vec3f(0.0f), radius, VOLUME_BEAMS));
        }
        if (p2.bounce() > 1 && !p1.onSurface() && p1.sampledLength > 0.0f) {
            const PathPhoton &p0 = _pathPhotons[i - 2];
            if (_settings.volumePhotonType == VOLUME_PLANES) {
                precomputePlane0D(_planes0D[i], p0, p1, p2);
                prims.emplace_back(GridAccel::Primitive(i, _planes0D[i].p0, _planes0D[i].p1, _planes0D[i].p2, _planes0D[i].p3, 0.0f, VOLUME_PLANES));
            } else {
                precomputePlane1D(_planes1D[i], p0, p1, p2, radius);
                if (_planes1D[i].valid) {
                    Vec3f p = _planes1D[i].center, a = _planes1D[i].a, b = _planes1D[i].b;
                    prims.emplace_back(GridAccel::Primitive(i, p - a - b, p + a - b, p + a + b, p - a + b, radius, VOLUME_PLANES));
                }
            }
        }
    }

    _volumeGrid.reset(new GridAccel(_scene->bounds(), _settings.gridMemBudgetKb, std::move(prims)));
}
template <bool isTransient>
void PhotonMapIntegrator<isTransient>::PhotonMapIntegrator::buildVolumeGrid(uint32 tail, float volumeRadiusScale)
{
    // TODO: support lowOrderScattering
    std::vector<GridAccel::Primitive> prims;
    for (uint32 i = 0; i < tail; ++i) {
        // TODO: support lowOrderScattering
        const PathPhoton &p3 = _pathPhotons[i - 0];
        if (p3.bounce() > 2) {
            const PathPhoton &p0 = _pathPhotons[i - 3];
            const PathPhoton &p1 = _pathPhotons[i - 2];
            const PathPhoton &p2 = _pathPhotons[i - 1];
            bool onSurface = p2.onSurface() || p1.onSurface();
            bool isSampledLengthValid = p1.sampledLength > 0.0f && p2.sampledLength > 0.0f;
            if (p3.bounce() > 2 && !onSurface && isSampledLengthValid)
            {
                precomputeUnslicedVolume(_volumes[i], p0, p1, p2, p3);
                prims.emplace_back(GridAccel::Primitive(i, _volumes[i].p, _volumes[i].a, _volumes[i].b, _volumes[i].c, 0.0f, VOLUME_VOLUMES));
            }
        }

    }

    _volumeGrid.reset(new GridAccel(_scene->bounds(), _settings.gridMemBudgetKb, std::move(prims)));
}
template <bool isTransient>
void PhotonMapIntegrator<isTransient>::PhotonMapIntegrator::buildHyperVolumeGrid(uint32 tail, float volumeRadiusScale)
{
    FAIL("PhotonMapIntegrator::buildHyperVolumeGrid() not implemented");
}

template <bool isTransient>
void PhotonMapIntegrator<isTransient>::buildPhotonDataStructures(float volumeRadiusScale, float beamTemporalRadiusScale)
{
    std::vector<SurfacePhotonRange> surfaceRanges;
    std::vector<VolumePhotonRange> volumeRanges;
    std::vector<PathPhotonRange> pathRanges;
    for (const SubTaskData &data : _taskData) {
        surfaceRanges.emplace_back(data.surfaceRange);
        volumeRanges.emplace_back(data.volumeRange);
        pathRanges.emplace_back(data.pathRange);
    }

    _surfaceTree = streamCompactAndBuild(surfaceRanges, _surfacePhotons, _totalTracedSurfacePaths);

    if (!_volumePhotons.empty()) {
        _volumeTree = streamCompactAndBuild(volumeRanges, _volumePhotons, _totalTracedVolumePaths);
        float volumeRadius = _settings.fixedVolumeRadius ? _settings.volumeGatherRadius : 1.0f;
        _volumeTree->buildVolumeHierarchy(_settings.fixedVolumeRadius, volumeRadius*volumeRadiusScale);
    } else if (!_pathPhotons.empty()) {
        uint32 tail = streamCompact(pathRanges);
        for (uint32 i = 0; i < tail; ++i)
        {
            _pathPhotons[i].power *= (1.0/_totalTracedPaths);
            _pathPhotons[i].surfPower *= (1.0/_totalTracedPaths);
        }

        for (uint32 i = 0; i < tail; ++i) {
            if (_pathPhotons[i].bounce() > 0) {
                Vec3f dir = _pathPhotons[i].pos - _pathPhotons[i - 1].pos;
                _pathPhotons[i - 1].length = dir.length();
                _pathPhotons[i - 1].dir = dir/_pathPhotons[i - 1].length;
            }
        }

        _beams.reset(new PhotonBeam[tail]);
        for (uint32 i = 0; i < tail; ++i)
            _beams[i].valid = false;

        if (_settings.volumePhotonType == VOLUME_BEAMS) {
            if (_settings.useGrid)
                buildBeamGrid(tail, volumeRadiusScale);
            else
                buildBeamBvh(tail, volumeRadiusScale);
        } else if (_settings.volumePhotonType == VOLUME_PLANES || _settings.volumePhotonType == VOLUME_PLANES_1D) {
            if (_settings.volumePhotonType == VOLUME_PLANES) {
                 _planes0D.reset(new PhotonPlane0D[tail]);
                for (uint32 i = 0; i < tail; ++i)
                    _planes0D[i].valid = false;
            }
            if (_settings.volumePhotonType == VOLUME_PLANES_1D) {
                _planes1D.reset(new PhotonPlane1D[tail]);
                for (uint32 i = 0; i < tail; ++i)
                    _planes1D[i].valid = false;
            }

            if (_settings.useGrid)
                buildPlaneGrid(tail, volumeRadiusScale);
            else
                buildPlaneBvh(tail, volumeRadiusScale);
        } else if (_settings.volumePhotonType == VOLUME_VOLUMES) {
            _volumes.reset(new PhotonVolume[tail]);
            for (uint32 i = 0; i < tail; ++i)
                _volumes[i].valid = false;

            if (_settings.useGrid)
                buildVolumeGrid(tail, volumeRadiusScale);
            else
                buildVolumeBvh(tail, volumeRadiusScale);
        } else if (_settings.volumePhotonType == VOLUME_VOLUMES_BALLS || _settings.volumePhotonType == VOLUME_MIS_VOLUMES_BALLS) {
            _volumes.reset(new PhotonVolume[tail]);
            for (uint32 i = 0; i < tail; ++i)
                _volumes[i].valid = false;

            if (_settings.lowOrderScattering)
            {
                _balls.reset(new PhotonBall[tail]);
                for (uint32 i = 0; i < tail; ++i)
                    _balls[i].valid = false;

                _beams.reset(new PhotonBeam[tail]);
                for (uint32 i = 0; i < tail; ++i)
                    _beams[i].valid = false;
            }

            bool enableMIS = _settings.volumePhotonType == VOLUME_MIS_VOLUMES_BALLS;
            if (_settings.useGrid)
                FAIL("VOLUME_BALL does not support grid now");
            else
                buildVolumeBallBvh(tail, volumeRadiusScale, enableMIS);
        } else if (_settings.volumePhotonType == VOLUME_HYPERVOLUMES) {
            ASSERT(isTransient == true, "does not support hyper volumes in non-transient mode");

            _hyperVolumes.reset(new PhotonHyperVolume[tail]);
            for (uint32 i = 0; i < tail; ++i)
            {
                _hyperVolumes[i].valid = false;
            }

            if (_settings.useGrid)
                buildHyperVolumeGrid(tail, volumeRadiusScale);
            else
                buildHyperVolumeBvh(tail, volumeRadiusScale);
        } else if (_settings.volumePhotonType == VOLUME_BALLS) {
            ASSERT(isTransient == true, "does not support photon balls in non-transient mode");

            _balls.reset(new PhotonBall[tail]);
            for (uint32 i = 0; i < tail; ++i)
                _balls[i].valid = false;

            buildBallBvh(tail, volumeRadiusScale);
        }

        _pathPhotonCount = tail;
    }
}

template <bool isTransient>
void PhotonMapIntegrator<isTransient>::fromJson(JsonPtr value, const Scene & /*scene*/)
{
    _settings.fromJson(value);
}

template <bool isTransient>
rapidjson::Value PhotonMapIntegrator<isTransient>::toJson(Allocator &allocator) const
{
    return _settings.toJson(allocator);
}

template <bool isTransient>
void PhotonMapIntegrator<isTransient>::prepareForRender(TraceableScene &scene, uint32 seed)
{
    _sampler = UniformSampler(MathUtil::hash32(seed));
    _currentSpp = 0;
    _totalTracedSurfacePaths = 0;
    _totalTracedVolumePaths  = 0;
    _totalTracedPaths        = 0;
    _pathPhotonCount         = 0;
    _scene = &scene;
    advanceSpp();
    scene.cam().requestColorBuffer();
    scene.cam().requestSplatBuffer();

    _useFrustumGrid = _settings.useFrustumGrid;
    if (_useFrustumGrid && !dynamic_cast<const PinholeCamera *>(&scene.cam())) {
        std::cout << "Warning: Frustum grid acceleration structure is only supported for a pinhole camera. "
                "Frustum grid will be disabled for this render." << std::endl;
        _useFrustumGrid = false;
    }

    if (_settings.includeSurfaces)
        _surfacePhotons.resize(_settings.photonCount);
    if (!_scene->media().empty()) {
        if (_settings.volumePhotonType == VOLUME_POINTS)
            _volumePhotons.resize(_settings.volumePhotonCount);
        else
            _pathPhotons.resize(_settings.volumePhotonCount);
    }

    _sampling_info = nullptr;
    if (_settings.enable_elliptical || _settings.enable_guiding) {
        Vec3f cam_pos = scene.cam().pos();
        _sampling_info.reset(new GuidedSamplingInfo(cam_pos, 0.f));

        if (_scene->lights().size() > 1)
            printf("Warning: Support for multiple light sources is not tested for DARTS.\n");
        // choose the first light
        const Primitive *light =  _scene->lights()[0].get();

        PositionSample point;
        UniformPathSampler dummy_sampler(MathUtil::hash32(_sampler.nextI()));
        if (!light->samplePosition(dummy_sampler, point)) {
            std::cerr << "Error: failed to sample a point on the light source." << std::endl;
            throw 1;
        }

        // If interested time window is narrower than the distance gap
        _sampling_info->glob_min_time = fmaxf((cam_pos - point.p).length(), _settings.transientTimeBeg);
        _sampling_info->time_range = _settings.transientTimeEnd - _sampling_info->glob_min_time;
        if (_sampling_info->time_range <= 0) {
            std::cerr << "Guided Sampling Error: Max time is no greater than shortest path time "<< _sampling_info->glob_min_time <<". Exitting..." << std::endl;
            throw 1;
        }

        printf("Photon tracer DARTS enabled.\n");
    }

    int numThreads = ThreadUtils::pool->threadCount();
    for (int i = 0; i < numThreads; ++i) {
        uint32 surfaceRangeStart = intLerp(0, uint32(     _surfacePhotons.size()), i + 0, numThreads);
        uint32 surfaceRangeEnd   = intLerp(0, uint32(     _surfacePhotons.size()), i + 1, numThreads);
        uint32  volumeRangeStart = intLerp(0, uint32(_settings.volumePhotonCount), i + 0, numThreads);
        uint32  volumeRangeEnd   = intLerp(0, uint32(_settings.volumePhotonCount), i + 1, numThreads);
        _taskData.emplace_back(SubTaskData{
            SurfacePhotonRange(_surfacePhotons.empty() ? nullptr : &_surfacePhotons[0], surfaceRangeStart, surfaceRangeEnd),
            VolumePhotonRange(  _volumePhotons.empty() ? nullptr : & _volumePhotons[0],  volumeRangeStart,  volumeRangeEnd),
              PathPhotonRange(    _pathPhotons.empty() ? nullptr : &   _pathPhotons[0],  volumeRangeStart,  volumeRangeEnd)
        });
        _samplers.emplace_back(_scene->rendererSettings().useSobol() ?
            std::unique_ptr<PathSampleGenerator>(new SobolPathSampler(MathUtil::hash32(_sampler.nextI()))) :
            std::unique_ptr<PathSampleGenerator>(new UniformPathSampler(MathUtil::hash32(_sampler.nextI())))
        );

        _tracers.emplace_back(new PhotonTracer<isTransient>(&scene, _settings, i));
    }

    Vec2u res = _scene->cam().resolution();
    _w = res.x();
    _h = res.y();

    if (_useFrustumGrid)
        _depthBuffer.reset(new Ray[_w*_h]);

    diceTiles();
}

template <bool isTransient>
void PhotonMapIntegrator<isTransient>::teardownAfterRender()
{
    _group.reset();
    _depthBuffer.reset();

    _beams.reset();
    _planes0D.reset();
    _planes1D.reset();
    _volumes.reset();
    _hyperVolumes.reset();

    _surfacePhotons.clear();
     _volumePhotons.clear();
       _pathPhotons.clear();
          _taskData.clear();
          _samplers.clear();
           _tracers.clear();

    _surfacePhotons.shrink_to_fit();
     _volumePhotons.shrink_to_fit();
       _pathPhotons.shrink_to_fit();
          _taskData.shrink_to_fit();
          _samplers.shrink_to_fit();
           _tracers.shrink_to_fit();

    _surfaceTree.reset();
    _volumeTree.reset();
    _volumeGrid.reset();
    _volumeBvh.reset();
    printf("Tearing down...\n");
    printf("PhotonMapIntegrator photon path trace call cnt: %d, avg running time: %f ms\n", (int)photon_pass_call_cnt, float(photon_pass_time) / float(photon_pass_call_cnt) / 1000.f);
    printf("PhotonMapIntegrator sensor path trace call cnt: %d, avg running time: %f ms\n", (int)sensor_pass_call_cnt, float(sensor_pass_time) / float(sensor_pass_call_cnt) / 1000.f);
}

template <bool isTransient>
void PhotonMapIntegrator<isTransient>::renderSegment(std::function<void()> completionCallback)
{
    using namespace std::placeholders;

    _scene->cam().setSplatWeight(1.0/_nextSpp);

    if (!_surfaceTree) {
        ThreadUtils::pool->yield(*ThreadUtils::pool->enqueue(
            std::bind(&PhotonMapIntegrator::tracePhotons, this, _1, _2, _3, 0),
            _tracers.size(), [](){}
        ));

        buildPhotonDataStructures(1.0f);
    }
    if (_settings.frame_num > 1) {
        _scene->cam().transientBuffer()->initTransientBuffer(_settings.frame_num);
    }
    ThreadUtils::pool->yield(*ThreadUtils::pool->enqueue(
        std::bind(&PhotonMapIntegrator::tracePixels, this, _1, _3, _settings.gatherRadius, _settings.volumeGatherRadius, _settings.transientTimeWidth),
        _tiles.size(), [](){}
    ));

    if (_useFrustumGrid) {
        ThreadUtils::pool->yield(*ThreadUtils::pool->enqueue(
            [&](uint32 tracerId, uint32 numTracers, uint32) {
                uint32 start = intLerp(0, _pathPhotonCount, tracerId,     numTracers);
                uint32 end   = intLerp(0, _pathPhotonCount, tracerId + 1, numTracers);
                _tracers[tracerId]->evalPrimaryRays(_beams.get(), _planes0D.get(), _planes1D.get(),
                        start, end, _settings.volumeGatherRadius, _depthBuffer.get(), *_samplers[tracerId],
                        _nextSpp - _currentSpp);
            }, _tracers.size(), [](){}
        ));
    }

    _currentSpp = _nextSpp;
    advanceSpp();

    completionCallback();
}

template <bool isTransient>
void PhotonMapIntegrator<isTransient>::startRender(std::function<void()> completionCallback)
{
    if (done()) {
        completionCallback();
        return;
    }

    _group = ThreadUtils::pool->enqueue([&, completionCallback](uint32, uint32, uint32) {
        renderSegment(completionCallback);
    }, 1, [](){});
}

template <bool isTransient>
void PhotonMapIntegrator<isTransient>::waitForCompletion()
{
    if (_group) {
        _group->wait();
        _group.reset();
    }
}

template <bool isTransient>
void PhotonMapIntegrator<isTransient>::abortRender()
{
    if (_group) {
        _group->abort();
        _group->wait();
        _group.reset();
    }
}

template <bool isTransient>
void PhotonMapIntegrator<isTransient>::setTimeCenter(float timeCenter)
{
    _settings.setTimeCenter(timeCenter);
}

// explicit instantiations, required for separating implementations to cpp file.
template class PhotonMapIntegrator<true>;
template class PhotonMapIntegrator<false>;

}
