#include "PhotonTracer.hpp"
#include "GridAccel.hpp"

#include "math/FastMath.hpp"

#include "bvh/BinaryBvh.hpp"

#include "Timer.hpp"

namespace Tungsten {

static constexpr bool QUERY_PROFILE   = false;
static constexpr bool KD_TREE_PROFILE = false;

#define STATIC_INITIALIZE(name, type, value) \
    template<> \
    std::atomic<type> PhotonTracer<false>::name = 0; \
    template<> \
    std::atomic<type> PhotonTracer<true>::name = 0;

STATIC_INITIALIZE(valid_cnt, long, 0)
STATIC_INITIALIZE(all_cnt, long, 0)
STATIC_INITIALIZE(profile_output_flag, bool, false)
STATIC_INITIALIZE(kd_tree_time, uint64_t, 0)
STATIC_INITIALIZE(kd_tree_cnt, uint64_t, 0)
STATIC_INITIALIZE(query_func_time, uint64_t, 0)
STATIC_INITIALIZE(query_func_cnt, uint64_t, 0)

#undef STATIC_INITIALIZE

constexpr bool camera_warped = true;
constexpr bool enable_darts  = true;           // skip non-darts part

template<bool isTransient>
PhotonTracer<isTransient>::PhotonTracer(TraceableScene *scene, const PhotonMapSettings &settings, uint32 threadId)
: TraceBase(scene, settings, threadId),
  _settings(settings),
  _mailIdx(0),
  _photonQuery(new const Photon *[settings.gatherCount]),
  _distanceQuery(new float[settings.gatherCount]),
  _mailboxes(zeroAlloc<uint32>(settings.volumePhotonCount)),
  _indirectCache(settings.volumePhotonCount*100),
  _frustumGrid(scene->cam()) {}

template <bool isTransient>
void PhotonTracer<isTransient>::clearCache()
{
    _directCache.clear();
    _indirectCache.clear();
}

static inline Vec3f exponentialIntegral(Vec3f a, Vec3f b, float t0, float t1)
{
    return (FastMath::exp(-a - b*t0) - FastMath::exp(-a - b*t1))/b;
}

static inline bool intersectBeam1D(const SteadyPhotonBeam &beam, const Ray &ray, const Vec3pf *bounds,
        float tMin, float tMax, float radius, float &invSinTheta, float &t, float &s)
{
    Vec3f l = beam.p0 - ray.pos();
    Vec3f u = l.cross(beam.dir).normalized();

    Vec3f n = beam.dir.cross(u);
    t = n.dot(l)/n.dot(ray.dir());
    Vec3f hitPoint = ray.pos() + ray.dir()*t;

    invSinTheta = 1.0f/std::sqrt(max(0.0f, 1.0f - sqr(ray.dir().dot(beam.dir))));
    if (std::abs(u.dot(hitPoint - beam.p0)) > radius)
        return false;

    if (bounds) {
        int majorAxis = std::abs(beam.dir).maxDim();
        float intervalMin = min((*bounds)[majorAxis][0], (*bounds)[majorAxis][1]);
        float intervalMax = max((*bounds)[majorAxis][2], (*bounds)[majorAxis][3]);

        if (hitPoint[majorAxis] < intervalMin || hitPoint[majorAxis] > intervalMax)
            return false;
    }

    if (t < tMin || t > tMax)
        return false;

    s = beam.dir.dot(hitPoint - beam.p0);
    if (s < 0.0f || s > beam.length)
        return false;

    return true;
}
static inline bool intersectPlane0D(const Ray &ray, float tMin, float tMax, Vec3f p0, Vec3f p1, Vec3f p2,
        float &invDet, float &farT, Vec2f &uv)
{
    Vec3f e1 = p1 - p0;
    Vec3f e2 = p2 - p0;
    Vec3f P = ray.dir().cross(e2);
    float det = e1.dot(P);
    if (std::abs(det) < 1e-5f)
        return false;

    invDet = 1.0f/det;
    Vec3f T = ray.pos() - p0;
    float u = T.dot(P)*invDet;
    if (u < 0.0f || u > 1.0f)
        return false;

    Vec3f Q = T.cross(e1);
    float v = ray.dir().dot(Q)*invDet;
    if (v < 0.0f || v > 1.0f)
        return false;

    float maxT = e2.dot(Q)*invDet;
    if (maxT <= tMin || maxT >= tMax)
        return false;

    farT = maxT;
    uv = Vec2f(u, v);
    return true;
}
static inline bool intersectPlane1D(const Ray &ray, float minT, float maxT, Vec3f p0, Vec3f u, Vec3f v, Vec3f w,
        Vec3f &o, Vec3f &d, float &tMin, float &tMax)
{
    o = ray.pos() - p0;
    d = ray.dir();

    o = Vec3f(u.dot(o), v.dot(o), w.dot(o));
    d = Vec3f(u.dot(d), v.dot(d), w.dot(d));
    Vec3f invD = 1.0f/d;

    Vec3f t0 = -o*invD;
    Vec3f t1 = t0 + invD;

    float ttMin = max(min(t0, t1).max(), minT);
    float ttMax = min(max(t0, t1).min(), maxT);

    if (ttMin <= ttMax) {
        tMin = ttMin;
        tMax = ttMax;
        return true;
    }
    return false;
}

static inline Vec3f clampToUnitCube(const Vec3f &uvw)
{
    static constexpr float CLAMP_TOL = 0.1f;

    Vec3f output(uvw);

    if (uvw.min() < 0.0f || uvw.max() > 1.0f)
    {
        if (uvw.min() < -CLAMP_TOL || uvw.max() > 1.0f + CLAMP_TOL)
        {
            printf("min: %f, max: %f\n", uvw.min(), uvw.max());
            printf("warning: deviates too much\n");
        }
        output = clamp(uvw, Vec3f(0.0f, 0.0f, 0.0f), Vec3f(1.0f, 1.0f, 1.0f));
    }

    return output;
}

static inline bool intersectUnitCube(const Vec3f &o, const Vec3f &d, float &tMin, float &tMax)
{
    // transcribed from volume-frag.shader
    bool isMinSet = false;
    bool isMaxSet = false;
    // int numHitDims = 0;

    // float t1s[3];
    // float t2s[3];

    for (uint32_t dim = 0; dim < 3; ++dim)
    {
        float p0 = 0.f;
        float p1 = 1.f;
        Vec3f n = Vec3f(0.f, 0.f, 0.f);
        n[dim] = 1.f;

        float f = d.dot(n);

        if (std::abs(f) > 1e-5f)
        {
            // numHitDims++;
            float t1 = (p0 - o[dim]) / f;
            float t2 = (p1 - o[dim]) / f;

            if (t1 > t2)
            {
                std::swap(t1, t2);
            }
            // t1s[dim] = t1;
            // t2s[dim] = t2;

            if (!isMinSet || t1 > tMin)
            {
                tMin = t1;
                isMinSet = true;
            }
            if (!isMaxSet || t2 < tMax)
            {
                tMax = t2;
                isMaxSet = true;
            }

            if (tMin > tMax || tMax < 0.0f)
            {
                return false;
            }
        }
        else if (o[dim] < 0.0 || o[dim] > 1.0)
        {
            return false;
        }
    }

    if (tMin < 0.0)
    {
        // ray origin inside cube?
        tMin = 0.0;
    }

    if (tMin == tMax)
    {
        return false;
    }
    return true;
}

static inline bool intersectTriangle(const Ray &ray, float minT, float maxT, Vec3f p0, Vec3f p1, Vec3f p2,
                                     float &u, float &v, float &t)
{
    constexpr float EPSILON = 1e-5f;

    Vec3f o = ray.pos();
    Vec3f d = ray.dir();

    Vec3f e1 = p1 - p0;
    Vec3f e2 = p2 - p0;

    Vec3f q = d.cross(e2);
    float a = e1.dot(q);
    if (a > -EPSILON && a < EPSILON)
    {
        return false;
    }

    float f = 1.f / a;
    Vec3f s = o - p0;
    u = f * s.dot(q);
    if (u < 0.f)
    {
        return false;
    }

    Vec3f r = s.cross(e1);
    v = f * d.dot(r);
    if (v < 0.f || u + v > 1.f)
    {
        return false;
    }

    t = f * e2.dot(r);
    if (t < minT || t > maxT)
    {
        return false;
    }

    return true;
}

// input: ray, minT, maxT is in world space
// output: o, d, tMin, tMax is in unit cube space to simplify future calculations
//         NOTE: The length of `d` does not have to be 1.
static inline bool intersectVolume(const Ray &ray, float minT, float maxT, Vec3f p0, Vec3f a, Vec3f b, Vec3f c,
                                   Vec3f &o, Vec3f &d, float &tMin, float &tMax)
{
    Mat4f unitCubeToWorld(
        a.x(), b.x(), c.x(), p0.x(),
        a.y(), b.y(), c.y(), p0.y(),
        a.z(), b.z(), c.z(), p0.z(),
        0.f, 0.f, 0.f, 1.f);
    Mat4f worldToUnitCube = unitCubeToWorld.invert();

    // *operator assumes rhs is a position
    Vec3f oUnit = worldToUnitCube * ray.pos();
    Vec3f dUnit = worldToUnitCube.transformVector(ray.dir());

    float tMinUnit = -1.0f;
    float tMaxUnit = -1.0f;

    if (intersectUnitCube(oUnit, dUnit, tMinUnit, tMaxUnit))
    {
        // NOTE: Since dUnit's has the length of a world space unit d vector transformed into unit cube space,
        //       there is no need to scale world space d vector length `minT` and `maxT`.
        float ttMin = max(tMinUnit, minT);
        float ttMax = min(tMaxUnit, maxT);

        if (ttMin <= ttMax)
        {
            o = oUnit;
            d = dUnit;
            tMin = ttMin;
            tMax = ttMax;
            // ASSERT(tMin != tMax, "tMin != tMax");
            return true;
        }
    }

    return false;
}

static inline bool intersectInfinitePlane(const Ray& ray, Vec3f p0, Vec3f n, float &outT)
{
    float a = n.dot(ray.dir());
    if (std::abs(a) < 1e-5f)
        return false;

    outT = n.dot(p0 - ray.pos()) / a;
    return true;
}

// does not test against minT maxT
static inline bool intersectSphere(const Ray& ray, Vec3f c, float r, float &t1, float &t2)
{
    Vec3f o = ray.pos();
    Vec3f d = ray.dir();

    Vec3f l = c - o;
    float s = l.dot(d);
    float lsqr = l.dot(l);
    float rsqr = r * r;

    if (s < 0 && lsqr > rsqr)
    {
        return false;
    }

    float msqr = lsqr - s * s;
    if (msqr > rsqr)
    {
        return false;
    }

    float q = std::sqrt(rsqr - msqr);
    t1 = s - q;
    t2 = s + q;
    return true;
}

static inline bool intersectBall(const Ray& ray, float minT, float maxT, Vec3f p0, float r, float &tMin, float &tMax)
{
    Vec3f o = ray.pos() - p0;
    Vec3f d = ray.dir();

    float A = d.dot(d);
    float B = 2.f * d.dot(o);
    float C = o.dot(o) - r * r;

    float delta = B * B - 4.f * A * C;
    if (delta < 0 || A == 0.f)
    {
        return false;
    }

    delta = std::sqrt(delta);
    float nomi = 2.f * A;
    float t1 = (-B + delta) / nomi;
    float t2 = (-B - delta) / nomi;
    if (t1 > t2)
    {
        std::swap(t1, t2);
    }

    tMin = std::max(t1, minT);
    tMax = std::min(t2, maxT);
    if (tMin > tMax)
    {
        return false;
    }

    return true;
}

static inline bool getSlabBoundary(const Ray &ray, const Vec3f *origins, const size_t numOrigins, Vec3f e1, Vec3f e2, float &tMin, float &tMax)
{
    float posInf = std::numeric_limits<float>::infinity();
    float negInf = -posInf;
    Vec3f n = e1.cross(e2);

    tMin = posInf;
    tMax = negInf;
    for(size_t i = 0; i < numOrigins; ++i)
    {
        Vec3f origin = origins[i];

        float planeT;
        if (intersectInfinitePlane(ray, origin, n, planeT))
        {
            tMin = std::min(tMin, planeT);
            tMax = std::max(tMax, planeT);
        }
    }

    // NOTE: always return true because if ray does not touch slab, will return tMin = posInf and tMax = negInf
    //       and cause tMin > tMax. This will cause caller intersectHyperVolume() to return false.
    return true;
}

static inline bool intersectHyperVolume(const Ray &ray, float minT, float maxT,
                                        Vec3f p0, Vec3f a, Vec3f b, Vec3f c, Vec3f d,
                                        float &outTMin, float &outTMax)
{
    constexpr size_t NUM_SLABS = 6;
    constexpr size_t NUM_PLANES_PER_SLAB = 4;

    Vec3f originTable[NUM_SLABS][NUM_PLANES_PER_SLAB] = {
        {p0, p0 + a, p0 + d, p0 + a + d},
        {p0, p0 + b, p0 + d, p0 + b + d},
        {p0, p0 + c, p0 + d, p0 + c + d},

        {p0, p0 + b, p0 + c, p0 + b + c},
        {p0, p0 + a, p0 + c, p0 + a + c},
        {p0, p0 + a, p0 + b, p0 + a + b},
    };

    Vec3f eTable[NUM_SLABS][2] = {
        {b, c},
        {a, c},
        {a, b},

        {a, d},
        {b, d},
        {c, d},
    };

    float tMin = minT;
    float tMax = maxT;
    for (size_t i = 0; i < NUM_SLABS; ++i)
    {
        float slabTMin, slabTMax;
        if (getSlabBoundary(ray, &originTable[i][0], NUM_PLANES_PER_SLAB, eTable[i][0], eTable[i][1], slabTMin, slabTMax))
        {
            tMin = std::max(tMin, slabTMin);
            tMax = std::min(tMax, slabTMax);
        }
    }

    if (tMin > tMax)
    {
        return false;
    }

    ASSERT(!std::isinf(tMin) && !std::isnan(tMin), "tMin is inf or nan");
    ASSERT(!std::isinf(tMax) && !std::isnan(tMin), "tMax is inf or nan");

    outTMin = tMin;
    outTMax = tMax;
    return true;
}


static inline bool intersectPlane(const Ray& ray, Vec3f p0, Vec3f n, float& t)
{
    const Vec3f o = ray.pos();
    const Vec3f d = ray.dir();

    float a = d.dot(n);
    if(std::abs(a) < 1e-9)
    {
        return false;
    }
    t = 1.0f / a * n.dot(p0 - o);
    return true;
}

static bool tryGetTransientTimeFactor(float pathTime, const PhotonMapSettings &settings, float &timeFactor)
{
    if (pathTime < settings.transientTimeBeg || pathTime > settings.transientTimeEnd)
    {
        timeFactor = 0.0f;
        return false;
    }
    else
    {
        timeFactor = settings.invTransientTimeWidth;
        return true;
    }
}

static int getTransientBin(float pathTime, const PhotonMapSettings &settings)
{
    if (pathTime < settings.transientTimeBeg || pathTime >= settings.transientTimeEnd) return -1;
    return static_cast<int>(std::floor((pathTime - settings.transientTimeBeg) * settings.invTransientTimeWidth));
}

struct PathInfo
{
    // store the photon path vertices in reverse order
    // i.e. vertices[0] will be the intersection point.
    Vec3f vertices[4];
    Vec3f estimate;
    uint32_t numVertices = 0;
    float camIsectT = 0.f;
};

struct SurfaceEval
{
    PathInfo pInfos[2];
    uint32_t numPaths = 0;
};

static void addToEstimate(const SurfaceEval& eval, Vec3f &estimate)
{
    for(uint32_t i = 0; i < eval.numPaths; ++i)
    {
        estimate += eval.pInfos[i].estimate;
    }
}

static inline bool occuluded(const TraceableScene *scene, Vec3f origin, Vec3f dir, const float len, float nearT = 1e-4f)
{
    Ray shadowRay = Ray(origin, dir, nearT, len);
    shadowRay.setNearT(nearT);
    return scene->occluded(shadowRay);
}

static bool evalBeam1D(const SteadyPhotonBeam &beam, PathSampleGenerator &sampler, const Ray &ray, const Medium *medium,
        const Vec3pf *bounds, float tMin, float tMax, float radius, Vec3f &beamEstimate)
{
    float invSinTheta, t, s;
    if (intersectBeam1D(beam, ray, bounds, tMin, tMax, radius, invSinTheta, t, s)) {
        Vec3f hitPoint = ray.pos() + ray.dir()*t;

        Ray mediumQuery(ray);
        mediumQuery.setFarT(t);
        beamEstimate += medium->sigmaT(hitPoint)*invSinTheta/(2.0f*radius)
                *medium->phaseFunction(hitPoint)->eval(beam.dir, -ray.dir())
                *medium->transmittance(sampler, mediumQuery, true, false)*beam.power;

        return true;
    }

    return false;
}

static bool evalTransientBeam1D(
    const Transient::TransientPhotonBeam &beam, PathSampleGenerator &sampler, 
    const Ray &ray, const Medium *medium, const Vec3pf *bounds, const PhotonMapSettings& settings, 
    float tMin, float tMax, float radius, float timeWidth, float camTimeTraveled, Vec3f &beamEstimate, Vec3f* transients = nullptr, Vec3f* throughput = nullptr
) {
    float invSinTheta, t, s;
    if (intersectBeam1D(beam, ray, bounds, tMin, tMax, radius, invSinTheta, t, s)) {
        Vec3f hitPoint = ray.pos() + ray.dir() * t;
        if (camTimeTraveled > 1e-5) {
            // camera warped setting, account for ray elapsed time ray-beam intersection time
            camTimeTraveled += medium->timeTraveled(t);
        }

        // s is the distance of beam start to hitPoint, projected onto the direction of the photon beam
        float beamSegmentTimeTraveled = medium->timeTraveled(s);
        float pathTime = camTimeTraveled + beam.timeTraveled + beamSegmentTimeTraveled;

        if (pathTime < settings.transientTimeBeg || pathTime >= settings.transientTimeEnd)
            return false;

        Ray mediumQuery(ray);
        mediumQuery.setFarT(t);
        Vec3f estimate =  medium->sigmaT(hitPoint)*invSinTheta/(2.0f*radius)
                *medium->phaseFunction(hitPoint)->eval(beam.dir, -ray.dir())
                *medium->transmittance(sampler, mediumQuery, true, false) * beam.power;
        beamEstimate += estimate / timeWidth;

        if (transients != nullptr && throughput != nullptr) {
            int frame_idx = getTransientBin(pathTime, settings);
            if (frame_idx >= 0)
                transients[frame_idx] += estimate * (*throughput);
        }

        return true;
    }

    return false;
}

static bool evalPlane0D(const SteadyPhotonPlane0D &p, PathSampleGenerator &sampler, const Ray &ray, const Medium *medium,
        const TraceableScene *scene, float tMin, float tMax, float &isectT, Vec3f &beamEstimate)
{
    Vec2f uv;
    float invDet, t;
    if (intersectPlane0D(ray, tMin, tMax, p.p0, p.p1, p.p3, invDet, t, uv)) {
        Vec3f hitPoint = ray.pos() + ray.dir()*t;

        Ray shadowRay = Ray(hitPoint, -p.d1, 0.0f, p.l1*uv.y());
        if (!scene->occluded(shadowRay)) {
            Ray mediumQuery(ray);
            mediumQuery.setFarT(t);
            beamEstimate += sqr(medium->sigmaT(hitPoint))*std::abs(invDet)
                    *medium->phaseFunction(hitPoint)->eval(p.d1, -ray.dir())
                    *medium->transmittance(sampler, mediumQuery, true, false)*p.power;

            isectT = t;
            return true;
        }
    }
    return false;
}
static bool evalPlane0D(const SteadyPhotonPlane0D &p, PathSampleGenerator &sampler, const Ray &ray, const Medium *medium,
        const TraceableScene *scene, float tMin, float tMax, Vec3f &beamEstimate)
{
    float isectT;
    return evalPlane0D(p, sampler, ray, medium, scene, tMin, tMax, isectT, beamEstimate);
}

static bool evalTransientPlane0D(const Transient::TransientPhotonPlane0D &p, PathSampleGenerator &sampler, const Ray &ray, const Medium *medium,
        const TraceableScene *scene, const PhotonMapSettings& settings, float tMin, float tMax, float camTimeTraveled, Vec3f &beamEstimate)
{
    Vec2f uv;
    float invDet, t;
    // two extra ray-intersection (expensive)
    if (intersectPlane0D(ray, tMin, tMax, p.p0, p.p1, p.p3, invDet, t, uv)) {
        Vec3f hitPoint = ray.pos() + ray.dir()*t;
        if (camTimeTraveled > 1e-5) {
            // camera warped setting, account for ray elapsed time ray-beam intersection time
            camTimeTraveled += medium->timeTraveled(t);
        }

        Vec3f midPoint = lerp(p.p0, p.p1, uv.x());
        // NOTE: first travel in u, then travel in v. order might matter for heterogeneous media.
        float uTimeTraveled = medium->timeTraveled(p.p0, midPoint);
        float vTimeTraveled = medium->timeTraveled(midPoint, hitPoint);
        float pathTime = camTimeTraveled + p.timeTraveled + uTimeTraveled + vTimeTraveled;

        float timeFactor = 0.0f;
        if (!tryGetTransientTimeFactor(pathTime, settings, timeFactor))
        {
            return false;
        }

        Ray shadowRay = Ray(hitPoint, -p.d1, 0.0f, p.l1*uv.y());
        if (!scene->occluded(shadowRay)) {
            Ray mediumQuery(ray);
            mediumQuery.setFarT(t);
            beamEstimate += sqr(medium->sigmaT(hitPoint))*std::abs(invDet)
                    *medium->phaseFunction(hitPoint)->eval(p.d1, -ray.dir())
                    *medium->transmittance(sampler, mediumQuery, true, false)*p.power*timeFactor;

            return true;
        }
    }
    return false;
}
template<typename ShadowCache>
bool evalPlane1D(const PhotonPlane1D &p, PathSampleGenerator &sampler, const Ray &ray, const Medium *medium,
        const TraceableScene *_scene, float tMin, float tMax, uint32 photonIdx,
        ShadowCache &shadowCache, Vec3f &beamEstimate)
{
    Vec3f o, d;
    float minT, maxT;
    if (intersectPlane1D(ray, tMin, tMax, p.p, p.invU, p.invV, p.invW, o, d, minT, maxT)) {
        float t = lerp(minT, maxT, sampler.next1D());
        Vec3f uvw = o + d*t;
        if (uvw.min() < 0.0f || uvw.max() > 1.0f)
            return false;

        Vec3f d0 = p.a*2.0f;
        Vec3f d1 = p.b*2.0f;
        Vec3f v0 = p.p + p.c;
        Vec3f v1 = v0 + uvw.x()*d0;
        Vec3f v2 = v1 + uvw.y()*d1;

        Vec3f sigmaT = medium->sigmaT(v2);
        Vec3f controlVariate = exponentialIntegral(Vec3f(0.0f), sigmaT, minT, maxT);

        float dist = shadowCache.hitDistance(photonIdx, uint32(p.binCount*uvw.x()), [&]() {
            Ray shadowRay = Ray(v1, p.d1, 0.0f, p.l1);
            return _scene->hitDistance(shadowRay);
        });

        if (dist < uvw.y()*p.l1*0.99f) {
            Ray mediumQuery(ray);
            mediumQuery.setFarT(t);

            controlVariate -= medium->transmittance(sampler, mediumQuery, true, false)*(maxT - minT);
        }

        beamEstimate += sqr(medium->sigmaT(v2))*medium->phaseFunction(v2)->eval(p.d1, -ray.dir())*p.power*controlVariate;
        return true;
    }
    return false;
}

static bool evalVolume(const SteadyPhotonVolume &p, PathSampleGenerator &sampler, const Ray &ray, const Medium *medium,
        const TraceableScene *scene, float tMin, float tMax, bool integrateAlongCam, bool importanceSampleCam, float &isectT, Vec3f &beamEstimate)
{
    Vec3f o, d;
    float minT, maxT;
    if (intersectVolume(ray, tMin, tMax, p.p, p.a, p.b, p.c, o, d, minT, maxT))
    {
        float t = 0.f;
        float pdf = 0.f;
        if (importanceSampleCam)
        {
            t = lerp(minT, maxT, sampler.next1D());
            pdf = 1.f / (maxT - minT);
        }
        else
        {
            Medium::MediumState state;
            state.reset();
            MediumSample sample;
            Ray sampleRay = ray;
            sampleRay.setFarT(std::numeric_limits<float>::infinity());
            if (!medium->sampleDistance(sampler, sampleRay, state, sample))
                return false;

            t = sample.t;
            if (t > maxT || t < minT)
            {
                return false;
            }
            pdf = sample.pdf;
        }

        Vec3f uvw = o + d * t;
        uvw = clampToUnitCube(uvw);

        Vec3f p0 = p.p;
        Vec3f p1 = p0 + p.a * uvw.x();
        Vec3f p2 = p1 + p.b * uvw.y();
        Vec3f p3 = p2 + p.c * uvw.z();

        Vec3f sigmaT = medium->sigmaT(p3);
        Vec3f controlVariate;
        if (integrateAlongCam)
        {
            controlVariate = exponentialIntegral(Vec3f(0.0f), sigmaT, minT, maxT);
        }
        else
        {
            Ray mediumQuery(ray);
            mediumQuery.setFarT(t);
            controlVariate = medium->transmittance(sampler, mediumQuery, false, false) / pdf;
        }

        if (occuluded(scene, p1, p.bDir, p.bLen * uvw.y()) || occuluded(scene, p2, p.cDir, p.cLen * uvw.z()))
        {
            if (integrateAlongCam)
            {
                Ray mediumQuery(ray);
                mediumQuery.setFarT(t);
                Vec3f trans = medium->transmittance(sampler, mediumQuery, true, false);
                controlVariate -= trans / pdf;
            }
            else
            {
                controlVariate = Vec3f(0.f);
            }

        }
        Vec3f estTransmittance = controlVariate;

        // NOTE: no need to multiply by inv determinant since it is already multiplied in power
        isectT = t;
        beamEstimate += cube(sigmaT)
            *medium->phaseFunction(p3)->eval(p.cDir, -ray.dir())
            *p.power*p.invDet*estTransmittance;
        return true;
    }
    return false;
}

static bool evalVolume(const SteadyPhotonVolume &p, PathSampleGenerator &sampler, const Ray &ray, const Medium *medium,
        const TraceableScene *scene, float tMin, float tMax, Vec3f &beamEstimate)
{
    float isectT;
    return evalVolume(p, sampler, ray, medium, scene, tMin, tMax, false, false, isectT, beamEstimate);
}

static inline Vec3f scoreBasedWeight(Vec3f selfEstimate, Vec3f otherEstimate)
{
    Vec3f selfScore = 1.f / selfEstimate;
    Vec3f otherScore = 1.f / otherEstimate;
    Vec3f selfWeight = selfScore / (selfScore + otherScore);
    return selfWeight;
}

static float timeInVolume(const Medium *medium, const Vec3f &p, const Vec3f &a, const Vec3f &b, const Vec3f &c, const Vec3f& uvw, Vec3f* last_p = nullptr)
{
    Vec3f p0 = p;
    Vec3f p1 = p0 + a * uvw.x();
    Vec3f p2 = p1 + b * uvw.y();
    Vec3f p3 = p2 + c * uvw.z();
    float t = medium->timeTraveled(p0, p1)
            + medium->timeTraveled(p1, p2)
            + medium->timeTraveled(p2, p3)
            ;
    if (last_p != nullptr) *last_p = p3;
    return t;
}

static bool evalDeltaSlicedVolume(const Transient::TransientPhotonVolume &p, PathSampleGenerator & /* sampler */, const Ray &ray, const Medium *medium,
                                const TraceableScene *scene, const PhotonMapSettings & /* settings */, float tMin, float tMax, float camTimeTraveled,
                                SurfaceEval &eval)
{
    ASSERT(camTimeTraveled == 0.f, "supports camera unwarped only");
    ASSERT(p.isDeltaSliced, "not delta sliced");

    float u = 0.f, v = 0.f, t = 0.f;
    if (!intersectTriangle(ray, tMin, tMax, p.p0, p.p1, p.p2, u, v, t))
    {
        return false;
    }
    float w = 1.f - u - v;
    float speedOfLight = medium->speedOfLight(p.p0);

    float t_a = w * speedOfLight * p.sampledDeltaTime;
    float t_b = u * speedOfLight * p.sampledDeltaTime;
    float t_c = v * speedOfLight * p.sampledDeltaTime;

    if (t_a > p.aLen || t_b > p.bLen || t_c > p.cLen)
    {
        return false;
    }

    Vec3f p0 = p.p;
    Vec3f p1 = p0 + p.aDir * t_a;
    Vec3f p2 = p1 + p.bDir * t_b;
    if (occuluded(scene, p1, p.bDir, t_b) || occuluded(scene, p2, p.cDir, t_c))
    {
        return false;
    }
    Vec3f p3 = p2 + p.cDir * t_c;

    Vec3f sigmaT = medium->sigmaT(p0);
    const PhaseFunction *pf = medium->phaseFunction(p0);
    Vec3f camTransmittance = FastMath::exp(-sigmaT * t);
    float invJacobian = 1.0f / std::abs(ray.dir().dot(p.n));
    if (invJacobian < 1e-5f)
    {
        return false;
    }

    Vec3f estimate = cube(sigmaT) * p.power * pf->eval(p.cDir, -ray.dir()) * camTransmittance * invJacobian;

    eval.numPaths = 1;
    PathInfo &pInfo = eval.pInfos[0];
    pInfo.camIsectT = t;
    pInfo.numVertices = 4;
    pInfo.vertices[0] = p3;
    pInfo.vertices[1] = p2;
    pInfo.vertices[2] = p1;
    pInfo.vertices[3] = p0;
    pInfo.estimate = estimate;
    return true;
}

static inline Vec3f estimateTransientVolume(const PhotonMapSettings &settings, Vec3f sigmaT, const PhaseFunction *pf, const Ray &ray, Vec3f camTransmittance, Vec3f lastVolumeDir, Vec3f power, float invDet, Vec3f n)
{
    Vec3f estimate = cube(sigmaT) * pf->eval(lastVolumeDir, -ray.dir()) * power * camTransmittance * settings.invTransientTimeWidth;

    float invJacobian;
    if (!settings.deltaTimeGate)
    {
        invJacobian = invDet;
    }
    else
    {
        invJacobian = 1.0f / std::abs(ray.dir().normalized().dot(n));
    }
    estimate *= invJacobian;
    return estimate;
}

static bool evalTransientVolume(const Transient::TransientPhotonVolume &p, PathSampleGenerator &sampler, const Ray &ray, const Medium *medium,
                                const TraceableScene *scene, const PhotonMapSettings &settings, float tMin, float tMax, float camTimeTraveled,
                                PathInfo &pInfo, Vec3f &totalEstimate)
{
    Vec3f o, d;
    float minT, maxT;
    if (intersectVolume(ray, tMin, tMax, p.p, p.a, p.b, p.c, o, d, minT, maxT))
    {
        ASSERT(minT <= maxT, "minT <= maxT");
        Vec3f minUvw = o + d * minT;
        Vec3f maxUvw = o + d * maxT;
        minUvw = clampToUnitCube(minUvw);
        maxUvw = clampToUnitCube(maxUvw);

        // rescale to world space?
        Vec3f min_p, max_p;
        float minTimeTraveled = timeInVolume(medium, p.p, p.a, p.b, p.c, minUvw, &min_p);
        float maxTimeTraveled = timeInVolume(medium, p.p, p.a, p.b, p.c, maxUvw, &max_p);
        if (minTimeTraveled == maxTimeTraveled)
        {
            // will still happen even if minT != maxT
            // maybe because of numerical precision
            return false;
        }

        if (minTimeTraveled > maxTimeTraveled)
        {
            std::swap(minTimeTraveled, maxTimeTraveled);
            std::swap(minT, maxT);
        }

        float minRatio = 0.0, maxRatio = 1.0;
        if (camTimeTraveled < 1e-4) {
            // camera warped setting, account for ray elapsed time ray-beam intersection time
            // transient minT, maxT
            float transMinTime = std::max(minTimeTraveled, settings.transientTimeBeg - p.timeTraveled - camTimeTraveled);
            float transMaxTime = std::min(maxTimeTraveled, settings.transientTimeEnd - p.timeTraveled - camTimeTraveled);
            if (transMinTime > transMaxTime)
            {
                return false;
            }

            // assume light travels at the same speed
            minRatio = (transMinTime - minTimeTraveled) / (maxTimeTraveled - minTimeTraveled);
            maxRatio = (transMaxTime - minTimeTraveled) / (maxTimeTraveled - minTimeTraveled);
        }


        ASSERT(minRatio >= 0.0f && minRatio <= 1.0f, "minRatio should be within 0.0 and 1.0");
        ASSERT(maxRatio >= 0.0f && maxRatio <= 1.0f, "minRatio should be within 0.0 and 1.0");

        float transMinT = minT + (maxT - minT) * minRatio;
        float transMaxT = minT + (maxT - minT) * maxRatio;
        minT = transMinT;
        maxT = transMaxT;

        if (minT > maxT)
        {
            std::swap(minT, maxT);
        }

        {
            float t = settings.deltaTimeGate ? minT : lerp(minT, maxT, sampler.next1D());
            Vec3f uvw = o + d * t;
            Vec3f p0 = p.p;
            Vec3f p1 = p0 + p.a * uvw.x();
            Vec3f p2 = p1 + p.b * uvw.y();
            Vec3f p3 = p2 + p.c * uvw.z();
            // I think that the final result being erroneous is caused by Jacobian.
            // Plane and beams actually won't have to consider Jacobian problem
            if (camTimeTraveled > 1e-4) {         // camera warped exits
                float time_traveled = medium->timeTraveled(p0, p1)
                                    + medium->timeTraveled(p1, p2)
                                    + medium->timeTraveled(p2, p3);
                // printf("Camera time: %f, time_traveled: %f\n", camTimeTraveled, time_traveled);
                time_traveled += medium->timeTraveled(ray.pos(), p3) + p.timeTraveled + camTimeTraveled;
                if (time_traveled < settings.transientTimeBeg || time_traveled > settings.transientTimeEnd)
                    return false;
            }

            if (uvw.min() == 0.f)
            {
                return false;
            }

            // update pInfo
            pInfo.camIsectT = t;
            pInfo.numVertices = 4;
            pInfo.vertices[0] = p3;
            pInfo.vertices[1] = p2;
            pInfo.vertices[2] = p1;
            pInfo.vertices[3] = p0;

            Vec3f sigmaT = medium->sigmaT(p3);
            Vec3f controlVariate;
            if (settings.deltaTimeGate)
            {
                // exp(-u_SigmaT * camToHitPointLength)
                controlVariate = FastMath::exp(-sigmaT * t);
            }
            else
            {
                controlVariate = exponentialIntegral(Vec3f(0.0f), sigmaT, minT, maxT);
            }

            if (occuluded(scene, p1, p.bDir, p.bLen * uvw.y()) || occuluded(scene, p2, p.cDir, p.cLen * uvw.z()))
            {
                if (!settings.deltaTimeGate)
                {
                    Ray mediumQuery(ray);

                    mediumQuery.setFarT(t);
                    Vec3f trans = medium->transmittance(sampler, mediumQuery, true, false);
                    controlVariate -= trans * (maxT - minT);
                }
                else
                {
                    controlVariate = Vec3f(0.f);
                }
            }
            Vec3f estTransmittance = controlVariate;

            // NOTE: no need to multiply by inv determinant since it is already multiplied in power
            Vec3f estimate = estimateTransientVolume(
                settings,
                sigmaT,
                medium->phaseFunction(p3),
                ray,
                estTransmittance,
                p.cDir,
                p.power,
                p.invDet,
                p.n
            );

            totalEstimate += estimate;
            return true;
        }
    }
    return false;
}

static inline void sampleClampedFreePathDistance(PathSampleGenerator &sampler, Vec3f sigmaT, float t1, float t2, float &outT, float &outPdf)
{
    float sigmaTc = sigmaT[sampler.nextDiscrete(3)];
    float r = FastMath::exp(-sigmaTc * t1);
    float q = r - FastMath::exp(-sigmaTc * t2);
    float m = sigmaTc / q;

    float xi = sampler.next1D();
    float t = -FastMath::log(r - q * xi) / sigmaTc;
    outT = t;
    outPdf = m * FastMath::exp(-sigmaTc * t);
    return;
}

static inline void sampleUniform(PathSampleGenerator &sampler, float t1, float t2, float &outT, float &outPdf)
{
    outT = lerp(t1, t2, sampler.next1D());
    outPdf = 1.f / (t2 - t1);
}

static bool evalTransientHyperVolume(const Transient::TransientPhotonHyperVolume &p, PathSampleGenerator &sampler, const Ray &ray, const Medium *medium,
                                     const TraceableScene *scene, const PhotonMapSettings &settings, float tMin, float tMax, float /* camTimeTraveled */, float &isectT, Vec3f &totalEstimate)
{
    float hyperVolumeTMin, hyperVolumeTMax;
    if (!intersectHyperVolume(ray, tMin, tMax, p.p, p.a, p.b, p.c, p.d, hyperVolumeTMin, hyperVolumeTMax))
    {
        return false;
    }
    Vec3f sigmaT = medium->sigmaT(ray.pos() + ray.dir() * hyperVolumeTMin);

    float sampleT, samplePdf;
    if (hyperVolumeTMin == hyperVolumeTMax)
    {
        return false;
    }

    // // sample according to Tr pdf
    // sampleClampedFreePathDistance(sampler, sigmaT, hyperVolumeTMin, hyperVolumeTMax, sampleT, samplePdf);

    // // uniform sample
    // sampleUniform(sampler, hyperVolumeTMin, hyperVolumeTMax, sampleT, samplePdf);

    // NOTE: using the slower camera ray sample for now because the other two above do not seem to work with MIS
    // camera ray sample
    Medium::MediumState state;
    state.reset();
    MediumSample sample;
    if (!medium->sampleDistance(sampler, ray, state, sample))
        return false;
    if (sample.exited)
        return false;
    if (sample.t < hyperVolumeTMin || sample.t > hyperVolumeTMax)
        return false;
    sampleT = sample.t;
    samplePdf = sample.pdf;

    Vec3f blurRayStart = ray.pos() + ray.dir() * sampleT;
    Ray blurRay(blurRayStart, -p.dDir, 1e-4f, p.dLen);
    Vec3f o, d;
    float minT, maxT;
    if (intersectVolume(blurRay, 1e-4f, p.dLen, p.p, p.a, p.b, p.c, o, d, minT, maxT))
    {
        Vec3f minUvw = o + d * minT;
        Vec3f maxUvw = o + d * maxT;
        minUvw = clampToUnitCube(minUvw);
        maxUvw = clampToUnitCube(maxUvw);

        float T1 = timeInVolume(medium, p.p, p.a, p.b, p.c, minUvw);
        float T2 = timeInVolume(medium, p.p, p.a, p.b, p.c, maxUvw);

        // TODO: move timeCenter sampling to photon precomputation stage
        float timeCenter = settings.deltaTimeGate ? settings.transientTimeCenter : lerp(settings.transientTimeBeg, settings.transientTimeEnd, sampler.next1D());
        float T = timeCenter - p.timeTraveled;
        if (T < 0)
        {
            return false;
        }

        float Td1 = medium->timeTraveled(minT);
        float Td2 = medium->timeTraveled(maxT);

        float left = T2 - T1 + Td2 - Td1;
        float right = T - Td1 - T1;
        if (left * right < 0.f)
        {
            return false;
        }
        if (abs(left) < abs(right))
        {
            return false;
        }

        float q = right / left;

        Vec3f uvw = lerp(minUvw, maxUvw, q);
        Vec3f p0 = p.p;
        Vec3f p1 = p0 + p.a * uvw.x();
        Vec3f p2 = p1 + p.b * uvw.y();
        Vec3f p3 = p2 + p.c * uvw.z();
        Vec3f p4 = blurRayStart;
        float lastDist = (p4 - p3).length();

        bool pathOcculuded = occuluded(scene, p1, p.bDir, p.bLen * uvw.y())
                          || occuluded(scene, p2, p.cDir, p.cLen * uvw.z())
                          || occuluded(scene, p3, p.dDir, lastDist);
        if (pathOcculuded)
        {
            return false;
        }

        Ray mediumQuery(ray);
        mediumQuery.setFarT(sampleT);
        Vec3f transmittance = medium->transmittance(sampler, mediumQuery, true, false);
        Vec3f estTransmittance = transmittance / samplePdf;

        Vec3f hvEstimate = cube(sigmaT) * sigmaT * medium->phaseFunction(p.p)->eval(p.dDir, -ray.dir()) * p.power * p.invJacobian * estTransmittance;
        totalEstimate += hvEstimate;
        isectT = sampleT;
        return true;
    }
    return false;
}

static bool evalDeltaSlicedBall(const Transient::TransientPhotonBall &p, PathSampleGenerator &sampler, const Ray &ray, const Medium *medium,
                              const TraceableScene *scene, const PhotonMapSettings & /*settings*/, float tMin, float tMax, float /* camTimeTraveled */,
                              SurfaceEval &eval)
{
    ASSERT(p.isDeltaSliced, "not delta sliced");
    float isectT1 = 0.f, isectT2 = 0.f;
    // tMin and tMax is ray.nearT (small, fixed value, 0.0001) and ray.farT (to closed surface)
    if (!intersectSphere(ray, p.p, p.temporalRadius, isectT1, isectT2))
    {
        return false;
    }

    float validTs[2];
    uint32_t numValidT = 0;
    if (isectT1 >= tMin && isectT1 <= tMax)
    {
        validTs[numValidT++] = isectT1;
    }
    if (isectT2 >= tMin && isectT2 <= tMax)
    {
        validTs[numValidT++] = isectT2;
    }

    if (numValidT == 0)
    {
        return false;
    }

    uint32_t curIdx = 0;
    for (uint32_t i = 0; i < numValidT; ++i)
    {
        float validT = validTs[i];
        Vec3f hitPoint = ray.pos() + ray.dir() * validT;
        Vec3f offset = hitPoint - p.p;
        Vec3f w1 = offset.normalized();
        float t1 = p.temporalRadius;
        if (w1.dot(w1) == 0.f || t1 == 0.f)
        {
            continue;
        }

        if (occuluded(scene, p.p, w1, t1, p.scatter.epsilon()))
        {
            continue;
        }
        Vec3f n = t1 * t1 * w1;

        // NOTE: assume speed of light = 1
        float jacobian = std::abs(n.dot(ray.dir()));
        if (std::abs(jacobian) < 1e-5f)
        {
            continue;
        }
        float invJacobian = 1.f / jacobian;

        Ray mediumQuery(ray);
        mediumQuery.setFarT(validT);
        Vec3f camTransmittance = medium->transmittance(sampler, mediumQuery, false, false);

        Vec3f lastScatter = p.scatter.eval(w1, sampler);
        Vec3f fp1 = medium->phaseFunction(hitPoint)->eval(w1, -ray.dir());
        Vec3f sigmaS = medium->sigmaS(hitPoint);

        Vec3f estimate = lastScatter * fp1 * medium->sigmaT(hitPoint) * p.power * invJacobian * camTransmittance;

        PathInfo &pInfo = eval.pInfos[curIdx];
        pInfo.camIsectT = validT;
        pInfo.numVertices = 2;
        pInfo.vertices[0] = hitPoint;
        pInfo.vertices[1] = p.p;
        pInfo.estimate = estimate;

        curIdx++;
    }
    eval.numPaths = curIdx;

    return curIdx > 0;
}

static bool evalSlicedVolumeBallMIS(const Transient::TransientPhotonVolume &volume, const Transient::TransientPhotonBall &ball, bool isVolume,
                                       PathSampleGenerator &sampler, const Ray &ray, const Medium *medium,
                                       const TraceableScene *scene, const PhotonMapSettings &settings, float tMin, float tMax, float camTimeTraveled, Vec3f &totalEstimate)
{
    if (isVolume)
    {
        SurfaceEval eval;
        if (!evalDeltaSlicedVolume(volume, sampler, ray, medium, scene, settings, tMin, tMax, camTimeTraveled, eval))
        {
            return false;
        }
        ASSERT(eval.numPaths == 1, "numPath wrong!");
        PathInfo &pInfo = eval.pInfos[0];
        Vec3f volEstimate = pInfo.estimate;

        ASSERT(pInfo.numVertices >= 3, "invalid pInfo!");
        // calculate ball center
        Vec3f ballCenter = pInfo.vertices[1];
        Vec3f hitPoint = pInfo.vertices[0];
        Vec3f lastDir = pInfo.vertices[1] - pInfo.vertices[2];
        lastDir.normalize();

        Vec3f offset = hitPoint - ballCenter;
        Vec3f w1 = offset.normalized();
        float t1 = offset.length();
        Vec3f n = t1 * t1 * w1;

        // NOTE: assume speed of light = 1
        float jacobian = std::abs(n.dot(ray.dir()));
        if (jacobian < 1e-5f)
        {
            totalEstimate += volEstimate;
            return true;
        }

        float invJacobian = 1.f / jacobian;

        Ray mediumQuery(ray);
        mediumQuery.setFarT(pInfo.camIsectT);
        Vec3f camTransmittance = medium->transmittance(sampler, mediumQuery, false, false);

        Vec3f lastScatter = ball.scatter.eval(w1, sampler);
        Vec3f fp1 = medium->phaseFunction(hitPoint)->eval(w1, -ray.dir());
        Vec3f sigmaS = medium->sigmaS(hitPoint);

        Vec3f ballEstimate = lastScatter * fp1 * medium->sigmaT(hitPoint) * ball.power * invJacobian * camTransmittance;

        // TODO: debug the rare case where ballEstimate is infinite
        // ASSERT(std::isfinite(ballEstimate), "{}, {}, {}, {}, {}", pInfo.vertices[0], pInfo.vertices[1], pInfo.vertices[2], pInfo.vertices[3]);
        if (!std::isfinite(ballEstimate))
        {
            totalEstimate += volEstimate;
            return true;
        }

        Vec3f volWeight = scoreBasedWeight(volEstimate, ballEstimate);
        totalEstimate += volWeight * volEstimate * 2.f;
        return true;
    }
    else
    {
        SurfaceEval eval;
        if (!evalDeltaSlicedBall(ball, sampler, ray, medium, scene, settings, tMin, tMax, camTimeTraveled, eval))
        {
            return false;
        }

        for (uint32 i = 0; i < eval.numPaths; ++i)
        {
            PathInfo &pInfo = eval.pInfos[i];
            Vec3f ballEstimate = pInfo.estimate;

            Vec3f x0 = pInfo.vertices[0];
            Vec3f x1 = pInfo.vertices[1];
            Vec3f w1 = volume.aDir;
            Vec3f w2 = volume.bDir;
            Vec3f w3 = (x0 - x1).normalized();

            Vec3f n = w1.cross(w2) + w2.cross(w3) + w3.cross(w1);
            float jacobian = std::abs(n.dot(ray.dir()));
            if (jacobian < 1e-5f)
            {
                totalEstimate += ballEstimate;
                continue;
            }
            float invJacobian = 1.0f / jacobian;

            Ray mediumQuery(ray);
            mediumQuery.setFarT(pInfo.camIsectT);
            Vec3f camTransmittance = medium->transmittance(sampler, mediumQuery, false, false);
            Vec3f sigmaT = medium->sigmaT(x0);

            Vec3f volEstimate = cube(sigmaT) * medium->phaseFunction(x0)->eval(w1, -ray.dir()) * volume.power * invJacobian * camTransmittance;

            Vec3f ballWeight = scoreBasedWeight(ballEstimate, volEstimate);
            totalEstimate += ballWeight * ballEstimate * 2.f;
        }

        return true;
    }
}

// TODO: check this out
template <bool isTransient>
void PhotonTracer<isTransient>::evalPrimaryRays(const PhotonBeam *beams, const PhotonPlane0D *planes0D, const PhotonPlane1D *planes1D,
                                                uint32 start, uint32 end, float radius, const Ray *depthBuffer, PathSampleGenerator &sampler, float scale)
{
    const Medium *medium = _scene->cam().medium().get();
    auto splatBuffer = _scene->cam().splatBuffer();
    Vec3f pos = _scene->cam().pos();

    int minBounce = _settings.minBounces - 1;
    int maxBounce = _settings.maxBounces - 1;

    for (uint32 i = start; i < end; ++i) {
        if (beams[i].valid && beams[i].bounce >= minBounce && beams[i].bounce < maxBounce) {
            const PhotonBeam &b = beams[i];
            Vec3f u = (b.p0 - pos).cross(b.dir).normalized();

            _frustumGrid.binBeam(b.p0, b.p1, u, radius, [&](uint32 x, uint32 y, uint32 idx) {
                const Ray &ray = depthBuffer[idx];
                Vec3f value(0.0f);
                // WARNING: NOTE: does not support frustum binning for transient for now
                if (evalBeam1D(b, sampler, ray, medium, nullptr, ray.nearT(), ray.farT(), radius, value))
                    splatBuffer->splat(Vec2u(x, y), value*scale);
            });
        }

        if (planes0D && planes0D[i].valid && planes0D[i].bounce >= minBounce && planes0D[i].bounce < maxBounce) {
            const PhotonPlane0D &p = planes0D[i];

            _frustumGrid.binPlane(p.p0, p.p1, p.p2, p.p3, [&](uint32 x, uint32 y, uint32 idx) {
                const Ray &ray = depthBuffer[idx];
                Vec3f value(0.0f);
                // WARNING: NOTE: does not support frustum binning for transient for now
                if (evalPlane0D(p, sampler, ray, medium, _scene, ray.nearT(), ray.farT(), value))
                    splatBuffer->splat(Vec2u(x, y), value*scale);
            });
        }

        if (planes1D && planes1D[i].valid && planes1D[i].bounce >= minBounce && planes1D[i].bounce < maxBounce) {
            const PhotonPlane1D &p = planes1D[i];

            _frustumGrid.binPlane1D(p.center, p.a, p.b, p.c, [&](uint32 x, uint32 y, uint32 idx) {
                const Ray &ray = depthBuffer[idx];
                Vec3f value(0.0f);
                if (evalPlane1D(p, sampler, ray, medium, _scene, ray.nearT(), ray.farT(), i, _directCache, value))
                    splatBuffer->splat(Vec2u(x, y), value*scale);
            });
        }
    }
}

template <bool isTransient>
Vec3f PhotonTracer<isTransient>::traceSensorPath(Vec2u pixel, const KdTree<Photon> &surfaceTree,
        const KdTree<VolumePhoton> *mediumTree, const Bvh::BinaryBvh *mediumBvh, const GridAccel *mediumGrid,
        const PhotonBeam *beams, const PhotonPlane0D *planes0D, const PhotonPlane1D *planes1D,
        const PhotonVolume *volumes, const PhotonHyperVolume *hyperVolumes, const PhotonBall *balls,
        PathSampleGenerator &sampler,
        float gatherRadius, float volumeGatherRadius, float beamTimeWidth,
        PhotonMapSettings::VolumePhotonType photonType, Ray &depthRay, bool useFrustumGrid, Vec3f* transients)
{
    PhotonPrimitives<isTransient> prims;
    prims.beams = beams;
    prims.planes0D = planes0D;
    prims.planes1D = planes1D;
    prims.volumes = volumes;
    prims.hyperVolumes = hyperVolumes;
    prims.balls = balls;

    _mailIdx++;

    PositionSample point;
    if (!_scene->cam().samplePosition(sampler, point, pixel))
        return Vec3f(0.0f);
    DirectionSample direction;
    if (!_scene->cam().sampleDirection(sampler, point, pixel, direction))
        return Vec3f(0.0f);

    Vec3f throughput = point.weight*direction.weight;
    Ray ray(point.p, direction.d);
    ray.setPrimaryRay(true);

    IntersectionTemporary data;
    IntersectionInfo info;
    const Medium *medium = _scene->cam().medium().get();
    const bool includeSurfaces = _settings.includeSurfaces;

    Vec3f result(0.0f);
    int bounce = 0;
    bool didHit = _scene->intersect(ray, data, info);

    depthRay = ray;

    float timeTraveled = 0.0f, total_camera_time = 0.0f;

    /**
     * We can actually derive transient rendering easily by simply substitute `Vec3f estimate` by a list of Vec3f
     * For this purpose, we might need to add another field called transients (a pointer) in the function param list
     * since we can not return the whole transient
    */

    while ((medium || didHit) && bounce < _settings.maxBounces) {
        bounce++;

        if (medium) {
            if (bounce > 1 || !useFrustumGrid) {
                // transient is not difficult to derive, since the contribution function is evaluated per photon
                // question is the temporal domain density estimation... it is prone to have bugs since we do not know
                // the specific implementation detail: we can only refer to histogram based methods?
                Vec3f estimate(0.0f);

                // A lambda function, captures timeTraveled and uses volumePhoton's travelling time
                auto pointContribution = [&](const VolumePhoton &p, float t, float distSq) {
                    std::chrono::steady_clock::time_point start_time;
                    if constexpr (QUERY_PROFILE)
                        start_time = std::chrono::steady_clock::now();
                    int fullPathBounce = bounce + p.bounce - 1;
                    if (fullPathBounce < _settings.minBounces || fullPathBounce >= _settings.maxBounces) {
                        if constexpr (QUERY_PROFILE) {
                            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start_time).count();
                            query_func_time += duration;
                            query_func_cnt  += 1;
                        }
                        return;
                    }

                    Ray mediumQuery(ray);
                    mediumQuery.setFarT(t);
                    // There should be yet another segment of time
                    Vec3f pointEstimate = (3.0f*INV_PI*sqr(1.0f - distSq/p.radiusSq))/p.radiusSq        // p.radiusSq is the sqr of volume gathering radius
                            *medium->phaseFunction(p.pos)->eval(p.dir, -ray.dir())
                            *medium->transmittance(sampler, mediumQuery, true, false) * p.power;

                    if constexpr (isTransient)
                    {
                        float pathTime = timeTraveled + p.timeTraveled;
                        if constexpr (camera_warped)
                            pathTime += medium->timeTraveled(ray.pos(), p.pos);
                        float timeFactor = 0.0f;
                        if (tryGetTransientTimeFactor(pathTime, _settings, timeFactor))
                            estimate += pointEstimate * timeFactor;
                        if (transients != nullptr) {
                            int frame_idx = getTransientBin(pathTime, _settings);
                            if (frame_idx >= 0)
                                transients[frame_idx] += pointEstimate * throughput;
                        }
                    }
                    else
                    {
                        estimate += pointEstimate;
                    }
                    if constexpr (QUERY_PROFILE) {
                        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start_time).count();
                        query_func_time += duration;
                        query_func_cnt  += 1;
                    }
                };

                // point contribution will account for photon time, beam contribution won't?
                auto beamContribution = [&](uint32 photonIndex, const Vec3pf *bounds, float tMin, float tMax) {
                    const PhotonBeam &beam = beams[photonIndex];
                    int fullPathBounce = bounce + beam.bounce;
                    if (fullPathBounce < _settings.minBounces || fullPathBounce >= _settings.maxBounces)
                        return;

                    if constexpr (isTransient) {
                        if constexpr (camera_warped) timeTraveled = std::max(2e-5f, timeTraveled);
                        evalTransientBeam1D(beam, sampler, ray, medium, bounds, _settings, tMin, tMax, volumeGatherRadius, beamTimeWidth, timeTraveled, estimate, transients, &throughput);
                    } else
                        evalBeam1D(beam, sampler, ray, medium, bounds, tMin, tMax, volumeGatherRadius, estimate);

                };
                auto planeContribution = [&](uint32 photon, const Vec3pf *bounds, float tMin, float tMax) {
                    int photonBounce = beams[photon].valid ? beams[photon].bounce : (planes0D ? planes0D[photon].bounce : planes1D[photon].bounce);
                    int fullPathBounce = bounce + photonBounce;
                    if (fullPathBounce < _settings.minBounces || fullPathBounce >= _settings.maxBounces)
                        return;

                    if (beams[photon].valid)
                    {
                        if constexpr (isTransient) {
                            if constexpr (camera_warped) timeTraveled = std::max(2e-5f, timeTraveled);
                            evalTransientBeam1D(beams[photon], sampler, ray, medium, bounds, _settings, tMin, tMax, volumeGatherRadius, beamTimeWidth, timeTraveled, estimate);
                        } else
                            evalBeam1D(beams[photon], sampler, ray, medium, bounds, tMin, tMax, volumeGatherRadius, estimate);
                    }
                    else if (photonType == VOLUME_PLANES_1D)
                        evalPlane1D(planes1D[photon], sampler, ray, medium, _scene, tMin, tMax, photon, _indirectCache, estimate);
                    else
                    {
                        if constexpr (isTransient) {
                            if constexpr (camera_warped) timeTraveled = std::max(2e-5f, timeTraveled);
                            evalTransientPlane0D(planes0D[photon], sampler, ray, medium, _scene, _settings, tMin, tMax, timeTraveled, estimate);
                        } else
                            evalPlane0D(planes0D[photon], sampler, ray, medium, _scene, tMin, tMax, estimate);
                    }

                };
                auto volumeContribution = [&](uint32 photon, const Vec3pf* /* bound */, float tMin, float tMax)
                {
                    if (!volumes[photon].valid)
                        return;
                    int photonBounce = volumes[photon].bounce;
                    int fullPathBounce = bounce + photonBounce;
                    if (fullPathBounce < _settings.minBounces || fullPathBounce >= _settings.maxBounces)
                        return;

                    PathInfo pInfo;
                    if constexpr (isTransient) {
                        if constexpr (camera_warped) timeTraveled = std::max(2e-5f, timeTraveled);
                        evalTransientVolume(volumes[photon], sampler, ray, medium, _scene, _settings, tMin, tMax, timeTraveled, pInfo, estimate);
                    } else
                        evalVolume(volumes[photon], sampler, ray, medium, _scene, tMin, tMax, estimate);
                };
                auto volumeBallContribution = [&](uint32 rawIdx, const Vec3pf *bounds, float tMin, float tMax, bool enableMIS)
                {
                    TypedPhotonIdx typedIdx(rawIdx);
                    uint32_t photon = typedIdx.idx;

                    if constexpr (isTransient)
                    {
                        int photonBounce = 0;
                        if (typedIdx.type == VOLUME_BEAMS)
                        {
                            photonBounce = beams[photon].bounce;
                        }
                        else if (typedIdx.type == VOLUME_BALLS)
                        {
                            photonBounce = balls[photon].bounce;
                        }
                        else if (typedIdx.type == VOLUME_VOLUMES)
                        {
                            photonBounce = volumes[photon].bounce;
                        }
                        else
                        {
                            FAIL("unknown typedIdx type");
                        }

                        int fullPathBounce = bounce + photonBounce;
                        if (fullPathBounce < _settings.minBounces || fullPathBounce >= _settings.maxBounces)
                            return;

                        if (typedIdx.type == VOLUME_BEAMS)
                        {
                            evalTransientBeam1D(beams[photon], sampler, ray, medium, bounds, _settings, tMin, tMax, volumeGatherRadius, beamTimeWidth, timeTraveled, estimate);
                        }
                        else if (typedIdx.type == VOLUME_BALLS)
                        {
                            SurfaceEval eval{};
                            ASSERT(balls[photon].isDeltaSliced, "expected delta sliced ball!");
                            if (!enableMIS || !volumes[photon].valid)
                            {
                                if (evalDeltaSlicedBall(balls[photon], sampler, ray, medium, _scene, _settings, tMin, tMax, timeTraveled, eval))
                                {
                                    addToEstimate(eval, estimate);
                                }
                            }
                            else
                            {
                                evalSlicedVolumeBallMIS(volumes[photon], balls[photon], false, sampler, ray, medium, _scene, _settings, tMin, tMax, timeTraveled, estimate);
                            }

                        }
                        else
                        {
                            SurfaceEval eval{};
                            ASSERT(volumes[photon].isDeltaSliced, "expected delta sliced volume!");
                            if (!enableMIS || !balls[photon].valid)
                            {
                                if (evalDeltaSlicedVolume(volumes[photon], sampler, ray, medium, _scene, _settings, tMin, tMax, timeTraveled, eval))
                                {
                                    addToEstimate(eval, estimate);
                                }
                            }
                            else
                            {
                                evalSlicedVolumeBallMIS(volumes[photon], balls[photon], true, sampler, ray, medium, _scene, _settings, tMin, tMax, timeTraveled, estimate);
                            }
                        }
                    }
                    else
                    {
                        FAIL("volume ball does not support non-transient");
                    }
                };
                auto hyperVolumeContribution = [&](uint32 rawIdx, const Vec3pf * /* bound */, float tMin, float tMax) {
                    TypedPhotonIdx typedIdx(rawIdx);
                    uint32_t photon = typedIdx.idx;

                    if (!hyperVolumes[photon].valid)
                    {
                        return;
                    }
                    int photonBounce = hyperVolumes[photon].bounce;
                    int fullPathBounce = bounce + photonBounce;
                    if (fullPathBounce < _settings.minBounces || fullPathBounce >= _settings.maxBounces)
                        return;

                    if constexpr (isTransient)
                    {
                        float isectT;
                        evalTransientHyperVolume(hyperVolumes[photon], sampler, ray, medium, _scene, _settings, tMin, tMax, timeTraveled, isectT, estimate);
                    }
                    else
                    {
                        FAIL("evaluating non-transient hypervolumes is not supported!");
                    }
                };
                auto ballContribution = [&](uint32 photon, const Vec3pf * /*bound*/, float tMin, float tMax) {
                    if (!balls[photon].valid)
                        return;
                    int photonBounce = balls[photon].bounce;
                    int fullPathBounce = bounce + photonBounce;
                    if (fullPathBounce < _settings.minBounces || fullPathBounce >= _settings.maxBounces)
                        return;

                    if constexpr (isTransient)
                    {
                        SurfaceEval eval;
                        ASSERT(balls[photon].isDeltaSliced, "expected delta sliced ball!");
                        if (evalDeltaSlicedBall(balls[photon], sampler, ray, medium, _scene, _settings, tMin, tMax, timeTraveled, eval))
                        {
                            addToEstimate(eval, estimate);
                        }
                    }
                    else
                    {
                        FAIL("ball does not support non transient mode");
                    }
                };

                if (photonType == VOLUME_POINTS) {
                    std::chrono::steady_clock::time_point stime;
                    if constexpr (KD_TREE_PROFILE)
                        stime = std::chrono::steady_clock::now();
                    mediumTree->beamQuery(ray.pos(), ray.dir(), ray.farT(), pointContribution);
                    if constexpr (KD_TREE_PROFILE) {
                        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - stime).count();
                        kd_tree_time += duration;
                        kd_tree_cnt  += 1;
                    }
                } else if (photonType == VOLUME_BEAMS) {
                    if (mediumBvh) {
                        mediumBvh->trace(ray, [&](Ray &ray, uint32 photonIndex, float /*tMin*/, const Vec3pf &bounds) {
                            beamContribution(photonIndex, &bounds, ray.nearT(), ray.farT());
                        });
                    } else {
                        mediumGrid->trace(ray, [&](uint32 photonIndex, float tMin, float tMax) {
                            beamContribution(photonIndex, nullptr, tMin, tMax);
                        });
                    }
                } else if (photonType == VOLUME_PLANES || photonType == VOLUME_PLANES_1D) {
                    if (mediumBvh) {
                        mediumBvh->trace(ray, [&](Ray &ray, uint32 photonIndex, float /*tMin*/, const Vec3pf &bounds) {
                            planeContribution(photonIndex, &bounds, ray.nearT(), ray.farT());
                        });
                    } else {
                        mediumGrid->trace(ray, [&](uint32 photonIndex, float /*tMin*/, float /*tMax*/) {
                            if (_mailboxes[photonIndex] == _mailIdx)
                                return;
                            _mailboxes[photonIndex] = _mailIdx;
                            planeContribution(photonIndex, nullptr, ray.nearT(), ray.farT());
                        });
                    }
                } else if (photonType == VOLUME_VOLUMES) {
                    if (mediumBvh) {
                        mediumBvh->trace(ray, [&](Ray &ray, uint32 photonIndex, float /*tMin*/, const Vec3pf &bounds) {
                            volumeContribution(photonIndex, &bounds, ray.nearT(), ray.farT());
                        });
                    } else {
                        mediumGrid->trace(ray, [&](uint32 photonIndex, float /*tMin*/, float /*tMax*/) {
                            if (_mailboxes[photonIndex] == _mailIdx)
                                return;
                            _mailboxes[photonIndex] = _mailIdx;
                            volumeContribution(photonIndex, nullptr, ray.nearT(), ray.farT());
                        });
                    }
                } else if (photonType == VOLUME_VOLUMES_BALLS || photonType == VOLUME_MIS_VOLUMES_BALLS) {
                    bool enableMIS = photonType == VOLUME_MIS_VOLUMES_BALLS;
                    if (mediumBvh) {
                        mediumBvh->trace(ray, [&](Ray &ray, uint32 photonIndex, float /*tMin*/, const Vec3pf &bounds) {
                            volumeBallContribution(photonIndex, &bounds, ray.nearT(), ray.farT(), enableMIS);
                        });
                    } else {
                        FAIL("does not support grid with volume+ball mixed mode");
                    }
                } else if (photonType == VOLUME_HYPERVOLUMES) {
                    if (mediumBvh) {
                        mediumBvh->trace(ray, [&](Ray &ray, uint32 photonIndex, float /*tMin*/, const Vec3pf &bounds) {
                            hyperVolumeContribution(photonIndex, &bounds, ray.nearT(), ray.farT());
                        });
                    } else {
                        FAIL("does not support grid with hypervolume");
                    }
                } else if (photonType == VOLUME_BALLS) {
                    if (mediumBvh) {
                        mediumBvh->trace(ray, [&](Ray &ray, uint32 photonIndex, float /*tMin*/, const Vec3pf &bounds) {
                            ballContribution(photonIndex, &bounds, ray.nearT(), ray.farT());
                        });
                    } else {
                        FAIL("does not support grid with photon balls");
                    }
                }
                result += throughput*estimate;
            }
            throughput *= medium->transmittance(sampler, ray, true, true);
        }
        if (!didHit)
            break;

        const Bsdf &bsdf = *info.bsdf;
        if (!includeSurfaces && !bsdf.lobes().isForward()) {        // forward can not be removed so easily
            // actually, we won't break here if includeSurface == true
            break;
        }

        SurfaceScatterEvent event = makeLocalScatterEvent(data, info, ray, &sampler);

        Vec3f transparency = bsdf.eval(event.makeForwardEvent(), false);
        float transparencyScalar = transparency.avg();

        Vec3f wo;
        if (sampler.nextBoolean(transparencyScalar)) {
            wo = ray.dir();
            throughput *= transparency/transparencyScalar;
        } else {
            event.requestedLobe = BsdfLobes(BsdfLobes::SpecularLobe, BsdfLobes::ForwardLobe);
            if (!bsdf.sample(event, false)) {
                // printf("Break from here, bounce = %d\n", bounce);
                break;
            }
            // Here is where specular lobe is coped with

            wo = event.frame.toGlobal(event.wo);

            throughput *= event.weight;
        }
        if constexpr (isTransient && camera_warped) {
            float local_travel_time = (info.p - ray.pos()).length();
            total_camera_time += local_travel_time;
            timeTraveled = total_camera_time;
        }

        bool geometricBackside = (wo.dot(info.Ng) < 0.0f);
        medium = info.primitive->selectMedium(medium, geometricBackside);

        ray = ray.scatter(ray.hitpoint(), wo, info.epsilon);

        if (std::isnan(ray.dir().sum() + ray.pos().sum()))
            break;
        if (std::isnan(throughput.sum()))
            break;

        if (bounce < _settings.maxBounces)
            didHit = _scene->intersect(ray, data, info);

    }

    // only path contribution is returned, which is not good. We might need to return time as well?
    if (!includeSurfaces)
        return result;

    if (!didHit) {
        if (!medium && bounce > _settings.minBounces && _scene->intersectInfinites(ray, data, info))
            result += throughput*info.primitive->evalDirect(data, info);
        return result;
    }
    if (info.primitive->isEmissive() && bounce > _settings.minBounces)
        result += throughput*info.primitive->evalDirect(data, info);

    int count = surfaceTree.nearestNeighbours(ray.hitpoint(), _photonQuery.get(), _distanceQuery.get(),
            _settings.gatherCount, gatherRadius);
    if (count == 0)
        return result;

    const Bsdf &bsdf = *info.bsdf;
    SurfaceScatterEvent event = makeLocalScatterEvent(data, info, ray, &sampler);

    Vec3f surfaceEstimate(0.0f);
    float camera_time = 0.0, time_factor = 1.0;
    if constexpr (isTransient && camera_warped) {
        if (medium) camera_time = medium->timeTraveled(ray.pos(), ray.hitpoint()) + total_camera_time;
    }
    float radiusSq = count == int(_settings.gatherCount) ? _distanceQuery[0] : gatherRadius*gatherRadius;
    for (int i = 0; i < count; ++i) {
        int fullPathBounce = bounce + _photonQuery[i]->bounce - 1;
        if (fullPathBounce < _settings.minBounces || fullPathBounce >= _settings.maxBounces)
            continue;
        float path_time = 0.f;
        if constexpr (isTransient) {
            path_time = _photonQuery[i]->time;
            if (medium)
                path_time += camera_time;
            if (path_time <= _settings.transientTimeBeg || path_time >= _settings.transientTimeEnd)
                continue;
            time_factor = _settings.invTransientTimeWidth;
        }

        event.wo = event.frame.toLocal(-_photonQuery[i]->dir);
        // Asymmetry due to shading normals already compensated for when storing the photon,
        // so we don't use the adjoint BSDF here
        auto estimate = _photonQuery[i]->power*bsdf.eval(event, false)/std::abs(event.wo.z());
        surfaceEstimate += estimate * time_factor;
        float gather_r2 = gatherRadius * gatherRadius;
        if (transients != nullptr) {
            int frame_idx = getTransientBin(path_time, _settings);
            if (frame_idx >= 0)
                transients[frame_idx] += estimate * throughput * INV_PI / radiusSq;
        }
    }
    result += throughput*surfaceEstimate*(INV_PI/radiusSq);
    return result;
}

template<bool isTransient>
PhotonTracer<isTransient>::~PhotonTracer() {
    bool expected = false;
    if (profile_output_flag.compare_exchange_strong(expected, true)) {
        float ratio = all_cnt == 0 ? -1 : float(valid_cnt) / float(all_cnt) * 100.f;
        printf("Valid count: %d, all count: %d, ratio: %f %%\n", (int)valid_cnt, (int)all_cnt, ratio);
        if constexpr (KD_TREE_PROFILE)
            printf("KD-tree call cnt: %lu, KD-tree avg time: %lf ms\n", (uint64_t)kd_tree_cnt, double(kd_tree_time) / double(kd_tree_cnt) / 1000.0);
        if constexpr (QUERY_PROFILE)
            printf("query func call cnt: %lu, query func avg time: %lf us\n", (uint64_t)query_func_cnt, double(query_func_time) / double(query_func_cnt) / 1000.0);
    }
}

template<bool isTransient>
bool PhotonTracer<isTransient>::frustumCulling(const Vec3f& photon_p) const {
    bool result = _scene->cam().canSplatOnScreen(photon_p);
    all_cnt += 1;
    valid_cnt += int(result);
    return result;
}

template<bool isTransient>
inline bool PhotonTracer<isTransient>::isInTimeRange(
    const Vec3f& src_p, const Vec3f& dst_p, const Medium* medium, 
    float speedOfLight, float time_travelled, float frame_start, bool& lt_max
) const {
    bool in_time_range = true;
    if constexpr (enable_darts && isTransient) {
        // actually, since we are unable to find photons that are on the surface and in the meantime, satisfy the time constraints
        // we can only discard those that are outside of the time range
        if (_settings.enable_elliptical || _settings.enable_guiding) {
            const float distance = (src_p - dst_p).length();
            time_travelled += medium ? medium->timeTraveled(distance) : (distance / speedOfLight);
            lt_max = time_travelled < (frame_start + _settings.transientTimeWidth);
            in_time_range &= lt_max;
            // strict_time_width will not account for **specular effect**, since the target vertex in this situation can not be known
            if (_settings.strict_time_width)
                in_time_range &= time_travelled >= frame_start;
        }
    }
    return in_time_range;
}   

template<bool isTransient>
void PhotonTracer<isTransient>::tracePhotonPath(SurfacePhotonRange &surfaceRange, VolumePhotonRange &volumeRange,
        PathPhotonRange &pathRange, PathSampleGenerator &sampler, GuideInfoConstPtr sampling_info, float bin_pdf, float min_remaining_t)
{
    // For DARTS to be integrated, photon path generation should be modified
    float lightPdf;
    const Primitive *light = chooseLightAdjoint(sampler, lightPdf);
    const Medium *medium = light->extMedium().get();

    PositionSample point;
    if (!light->samplePosition(sampler, point))
        return;
    DirectionSample direction;
    if (!light->sampleDirection(sampler, point, direction))
        return;

    Ray ray(point.p, direction.d);
    Vec3f throughput(point.weight*direction.weight/lightPdf);

    if (!pathRange.full()) {
        PathPhoton &p = pathRange.addPhoton();
        p.pos = point.p;
        p.power = throughput;
        p.surfPower = point.weight/lightPdf;
        p.setPathInfo(0, true);
        p.scatter().setLightScatter(light, point);
        if constexpr (isTransient)
        {
            p.timeTraveled = 0.0f;
        }
    }

    SurfaceScatterEvent event;
    IntersectionTemporary data;
    IntersectionInfo info;
    Medium::MediumState state;
    state.reset();
    Vec3f emission(0.0f);

    // # of additional medium bounces needed after hitting a surface
    int numGhostBounces = 0;
    switch (_settings.volumePhotonType)
    {
        case VOLUME_BALLS:
        case VOLUME_PLANES:
        case VOLUME_PLANES_1D:
            numGhostBounces = 1;
            break;
        case VOLUME_VOLUMES:
        case VOLUME_VOLUMES_BALLS:
        case VOLUME_MIS_VOLUMES_BALLS:
            numGhostBounces = 2;
            break;
        case VOLUME_HYPERVOLUMES:
            numGhostBounces = 3;
            break;
    }
    bool traceHigherOrder = numGhostBounces > 0, is_beam = _settings.volumePhotonType == VOLUME_BEAMS;

    // therefore, it is always low order if we are using mis volume ball
    bool useLowOrder = _settings.lowOrderScattering || _settings.volumePhotonType != VOLUME_POINTS;
    int bounce = 0;
    int bounceSinceSurface = 0;
    bool wasSpecular = true, exceed_time_range = false;
    bool didHit = _scene->intersect(ray, data, info);
 
    // NOTE: only needed for transient, but if put in "if constexpr (isTransient)" compiler will complain
    float timeTraveled = 0.0f, remaining_time = 0.f;
    float speedOfLight = medium ? medium->speedOfLight(ray.pos()) : _settings.vaccumSpeedOfLight;
    EllipseInfo ell_info(sampling_info, ray.pos(), min_remaining_t, timeTraveled, _settings.transientTimeWidth, sampler);
    Vec3f ell_direction = ray.dir(), ell_throughput = throughput;

    while ((didHit || medium) && bounce < _settings.maxBounces - 1) {
        bool hitSurface = didHit;
        bounce++;
        bounceSinceSurface++;
        Vec3f continuedThroughput = throughput;

        if constexpr (enable_darts && isTransient) {
            remaining_time = ell_info.target_t;
            if (_settings.enable_elliptical && medium) {            // elliptical sampling logic (might only work with photon points)
                // this will account for 3 vertex path (originally we only have vertex >= 4 path)
                if (!ell_info.valid_target()) {
                    // break if the current target can't be reached 
                    break;
                }
                // This is not for avoiding numerical singularity: we don't want to account for surface photons twice
                if (bounceSinceSurface > 1 && ell_info.direct_connect()) {          // this could be extremely rare
                    // we should be aware whether we are on the surface or 
                    if (!volumeRange.full()) {
                        VolumePhoton &p = volumeRange.addPhoton();
                        p.pos = ray.pos();
                        p.dir = ray.dir();
                        p.power = ell_throughput;
                        p.bounce = bounce;
                        p.timeTraveled = timeTraveled;
                    }
                } else if (!volumeRange.full()) {
                    MediumSample mit;
                    mit.t = -1;
                    Ray ell_ray = ray;
                    if (!_settings.reuse_path_dir && bounceSinceSurface > 1) {
                        ell_ray.setDir(ell_direction);
                        ell_ray.setFarT(Ray::infinity());
                        IntersectionTemporary _temp_data;
                        IntersectionInfo _temp_info;
                        _scene->intersect(ell_ray, _temp_data, _temp_info);
                    }
                    medium->elliptical_sample(ell_ray, sampler, mit, &ell_info);
                    // if we do not use frustum_culling, or when we use + the photon is actually in the frustum:
                    bool in_cam_frustum = (!_settings.frustum_culling) || (mit.t > 0 && frustumCulling(mit.p));
                    // The volume photons sampled via ellipitcal sampling will fall in the time range (with high probability)
                    if (mit.t > 0 && in_cam_frustum) {
                        VolumePhoton &p = volumeRange.addPhoton();
                        p.pos = mit.p;
                        p.dir = ell_ray.dir();
                        p.power = ell_throughput * mit.weight / bin_pdf;                                    // extra elliptical weight
                        p.bounce = bounce + 1;                                                          // elliptical vertex actually adds one more bounce
                        p.timeTraveled = timeTraveled + medium->timeTraveled(mit.t);                    // account for elliptical time
                    }
                }
            }
        }   // this closing-bracket tail is toooo ugly

        if (medium) {
            MediumSample mediumSample;
            // Here, DARTS shall be used
            if (!medium->sampleDistance(sampler, ray, state, mediumSample, remaining_time, sampling_info, _settings.enable_guiding))
                break;
            // difference between continuedWeight and weight?
            continuedThroughput *= mediumSample.continuedWeight;
            throughput *= mediumSample.weight;
            // only if medium->sample returns surface interaction
            hitSurface = mediumSample.exited;
            if constexpr (isTransient) {
                timeTraveled += medium->timeTraveled(mediumSample.t);
                speedOfLight = medium->speedOfLight(ray.pos());
            }

            // Elliptical sampling does not allow the existance of origin-method sampled photons
            if (!hitSurface && (useLowOrder || bounceSinceSurface > 1) && !volumeRange.full() && !_settings.enable_elliptical) {
                VolumePhoton &p = volumeRange.addPhoton();
                p.pos = mediumSample.p;
                p.dir = ray.dir();
                p.power = throughput;                                   // throughput is only used in volumePhotons
                p.bounce = bounce;
                if constexpr (isTransient) {
                    p.timeTraveled = timeTraveled;
                }
            }

            // there is no restriction on pathRange, since pathRange will not be used in photon points
            // current DARTS method can not be used on photon beam method (except for DA-based distance sampling)
            if ((!hitSurface || traceHigherOrder) && !pathRange.full()) {
                bool lt_max = true;
                isInTimeRange(mediumSample.p, sampling_info->vertex_p, medium, speedOfLight, timeTraveled, min_remaining_t, lt_max);
                if (is_beam && lt_max == false) {
                    // The first time a path photon exceeds path range, we will not break
                    // but when the second time comes, we break from path construction, since the photon
                    // beams will be useless (the start and end will neither be in time range)
                    if (exceed_time_range) break;
                    exceed_time_range = true;
                }
                pathRange.nextPtr()[-1].sampledLength = mediumSample.continuedT;
                PathPhoton &p = pathRange.addPhoton();
                p.pos = mediumSample.p;
                p.power = continuedThroughput;
                p.inMedium = medium;
                p.setPathInfo(bounce, false, hitSurface);
                p.scatter().setMediumScatter(medium, mediumSample.p, ray.dir());
                if constexpr (isTransient)
                {
                    p.timeTraveled = timeTraveled;
                }
            }

            Ray continuedRay;
            PhaseSample phaseSample;

            // update elliptical info for IDA phase function

            if (!hitSurface || traceHigherOrder) {
                ell_info = EllipseInfo(sampling_info, mediumSample.p, min_remaining_t, timeTraveled, _settings.transientTimeWidth, sampler);
                if (!mediumSample.phase->sample(sampler, ray.dir(), phaseSample, &ell_info))
                    break;

                if (!_settings.reuse_path_dir) {
                    PhaseSample temp_smp;
                    mediumSample.phase->sample(sampler, ray.dir(), temp_smp);
                    ell_direction = temp_smp.w;
                }

                continuedRay = ray.scatter(mediumSample.p, phaseSample.w, 0.0f);
                continuedRay.setPrimaryRay(false);
            }

            if (!hitSurface) {      // current vertex is not surface
                ray = continuedRay;
                // Originally we have erroneous logic here. For no path reuse case
                // elliptical sampling should be multiplied by throughput without phaseSample.weight
                if (!_settings.reuse_path_dir) {            // no reuse
                    ell_throughput = throughput;
                    throughput *= phaseSample.weight;
                } else {                                    // reuse
                    ell_throughput = throughput * phaseSample.weight;
                    throughput = ell_throughput;
                }
            } else if (traceHigherOrder) {  // hit surface, but we will continue tracing (medium bounces?)
                Medium::MediumState continuedState = state;
                float ghostTimeTraveled = timeTraveled;
                // Photon beam method will not trace `HigherOrder`, since num of ghost bounces = 0
                // Here, only DA-based distance sampling will be used. Elliptical sampling will not be helpful
                // FIXME: efficiency can be improved here, to skip this part for DARTS PP
                EllipseInfo tmp_ell_info = ell_info;
                for (int ghostBounceIdx = 0; ghostBounceIdx < numGhostBounces; ++ghostBounceIdx) {
                    if (!medium->sampleDistance(sampler, continuedRay, continuedState, mediumSample))
                        break;
                    if constexpr (isTransient) {
                        ghostTimeTraveled += medium->timeTraveled(mediumSample.continuedT);
                    }

                    if (!pathRange.full()) {
                        pathRange.nextPtr()[-1].sampledLength = mediumSample.continuedT;
                        PathPhoton &p = pathRange.addPhoton();
                        p.pos = mediumSample.p;
                        p.power = throughput * mediumSample.weight * phaseSample.weight;
                        p.inMedium = medium;
                        p.setPathInfo(bounce + 1 + ghostBounceIdx, ghostBounceIdx == numGhostBounces - 1, true);
                        p.scatter().setMediumScatter(medium, mediumSample.p, continuedRay.dir());
                        if constexpr (isTransient) {
                            p.timeTraveled = ghostTimeTraveled;
                        }
                    }

                    // using IDA sampling (possibly) during higher-dimensional photon primitives
                    tmp_ell_info = EllipseInfo(sampling_info, mediumSample.p, min_remaining_t, 
                                        ghostTimeTraveled, _settings.transientTimeWidth, sampler);
                    if (!mediumSample.phase->sample(sampler, ray.dir(), phaseSample, &tmp_ell_info))
                        break;
                    continuedRay = ray.scatter(mediumSample.p, phaseSample.w, 0.0f);
                    continuedRay.setPrimaryRay(false);
                }
            }
        }

        if constexpr (isTransient) {
            if (!medium) {
                Vec3f offset = info.p - ray.pos();
                timeTraveled += offset.length() / speedOfLight;
            }
        }

        if (hitSurface) {
            Vec3f surfPower = throughput * std::abs(info.Ns.dot(ray.dir()) / info.Ng.dot(ray.dir()));
            event = makeLocalScatterEvent(data, info, ray, &sampler);

            if (!info.bsdf->lobes().isPureSpecular() && !surfaceRange.full()) {
                // time selection for photons might not be a good idea in scenes with specular lobe
                bool lt_max = true, in_time_range = isInTimeRange(
                    info.p, sampling_info->vertex_p, medium, speedOfLight, timeTraveled, min_remaining_t, lt_max);
                if (in_time_range) {
                    Photon &p = surfaceRange.addPhoton();
                    p.pos = info.p;
                    p.dir = ray.dir();
                    p.power = surfPower / bin_pdf;
                    p.bounce = bounce;
                    if constexpr (isTransient) {            // now path photons has time information!
                        p.time = timeTraveled;
                    }
                }
            }
            if (!pathRange.full()) {
                bool lt_max = true;
                isInTimeRange(info.p, sampling_info->vertex_p, medium, speedOfLight, timeTraveled, min_remaining_t, lt_max);
                if (is_beam && lt_max == false) {
                    // The first time a path photon exceeds path range, we will not break
                    // but when the second time comes, we break from path construction, since the photon
                    // beams will be useless (the start and end will neither be in time range)
                    if (exceed_time_range) break;
                    exceed_time_range = true;
                }
                PathPhoton &p = pathRange.addPhoton();
                p.pos = info.p;
                p.power = continuedThroughput;
                p.surfPower = surfPower / bin_pdf;
                p.inMedium = medium;
                p.setPathInfo(bounce, true);
                p.scatter().setSurfaceScatter(&event);

                if constexpr (isTransient) {
                    p.timeTraveled = timeTraveled;
                }
            }
        }

        if (volumeRange.full() && surfaceRange.full() && pathRange.full())
            break;

        if (hitSurface) {
            bool succeed = false;
            // throughput changes induced by surfaces are accounted for here. 
            if constexpr (isTransient) {
                succeed = handleSurface(event, data, info, medium, speedOfLight,
                                        bounce, true, false, ray, throughput, emission, wasSpecular, state);
            } else {
                succeed = handleSurface(event, data, info, medium,
                                        bounce, true, false, ray, throughput, emission, wasSpecular, state);
            }
            ell_throughput = throughput;

            // we should of course update the elliptical info struct in surface pass
            ell_info = EllipseInfo(sampling_info, ray.pos(), min_remaining_t, timeTraveled, _settings.transientTimeWidth, sampler);

            if (!succeed)
                break;
            bounceSinceSurface = 0;
        }

        if (throughput.max() == 0.0f)
            break;

        if (std::isnan(ray.dir().sum() + ray.pos().sum()))
            break;
        if (std::isnan(throughput.sum()))
            break;

        if (bounce < _settings.maxBounces)
            didHit = _scene->intersect(ray, data, info);
    }
}



template class PhotonTracer<true>;
template class PhotonTracer<false>;

}
