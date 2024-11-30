#ifndef PHOTON_HPP_
#define PHOTON_HPP_

#include "math/Vec.hpp"
#include "math/Box.hpp"
#include "Debug.hpp"
#include "media/Medium.hpp"
#include "samplerecords/SurfaceScatterEvent.hpp"
#include "primitives/IntersectionInfo.hpp"
#include "bsdfs/Bsdf.hpp"
#include "samplerecords/PositionSample.hpp"
#include "primitives/Primitive.hpp"

namespace Tungsten {

struct Photon
{
    uint32 splitData;
    uint32 bounce;
    Vec3f pos;
    Vec3f dir;
    Vec3f power;
    float time;         // do not use the same name as the derived class (timeTravelled)

    void setSplitInfo(uint32 childIdx, uint32 splitDim, uint32 childCount)
    {
        uint32 childMask = childCount == 0 ? 0 : (childCount == 1 ? 1 : 3);
        splitData = (splitDim << 30u) | (childMask << 28u) | childIdx;
    }

    bool hasLeftChild() const
    {
        return (splitData & (1u << 28u)) != 0;
    }

    bool hasRightChild() const
    {
        return (splitData & (1u << 29u)) != 0;
    }

    uint32 splitDim() const
    {
        return splitData >> 30u;
    }

    uint32 childIdx() const
    {
        return splitData & 0x0FFFFFFFu;
    }
};

struct SteadyVolumePhoton : public Photon
{
    Vec3f minBounds;
    Vec3f maxBounds;
    float radiusSq;
};

enum PhotonScatterType : uint8
{
    PHOTON_SCATTER_INVALID = 0,
    PHOTON_SCATTER_LIGHT,
    PHOTON_SCATTER_MEDIUM,
    PHOTON_SCATTER_SURFACE,
    PHOTON_SCATTER_COUNT,
};

struct PhotonScatterEvent
{
    PhotonScatterType _scatterType = PHOTON_SCATTER_INVALID;
    const void* _scatterPtr;

    Vec3f _pos;
    Vec3f _inDir;
    SurfaceScatterEvent _event;
    IntersectionInfo _isectInfo;
    PositionSample _posSample;

    void setLightScatter(const Primitive *light, const PositionSample& posSample)
    {
        _scatterType = PHOTON_SCATTER_LIGHT;
        _posSample = posSample;
        _scatterPtr = static_cast<const void *>(light);
    }
    void setMediumScatter(const Medium *medium, Vec3f pos, Vec3f inDir)
    {
        _scatterType = PHOTON_SCATTER_MEDIUM;
        _pos = pos;
        _inDir = inDir;
        _scatterPtr = static_cast<const void *>(medium);
    }
    void setSurfaceScatter(const SurfaceScatterEvent *event)
    {
        _scatterType = PHOTON_SCATTER_SURFACE;
        // copy the whole data
        _isectInfo = *(event->info);
        _event = *event;
        _event.info = &_isectInfo;
        _event.sampler = nullptr;
    }

    PhotonScatterType scatterType() const
    {
        return _scatterType;
    }
    bool isLightScatter() const
    {
        return _scatterType == PHOTON_SCATTER_LIGHT;
    }
    bool isSurfaceScatter() const
    {
        return _scatterType == PHOTON_SCATTER_SURFACE;
    }
    bool isMediumScatter() const
    {
        return _scatterType == PHOTON_SCATTER_MEDIUM;
    }

    bool isSpecular() const
    {
        switch(_scatterType)
        {
            case PHOTON_SCATTER_LIGHT:
                return light()->isDirectionDirac();
            case PHOTON_SCATTER_MEDIUM:
                return false;
            case PHOTON_SCATTER_SURFACE:
                return _isectInfo.bsdf->lobes().isPureSpecular() || _isectInfo.bsdf->lobes().isForward();
            default:
                FAIL("unsupported scatter type");
                return true;
        }
    }

    Vec3f eval(Vec3f outDir, PathSampleGenerator &sampler) const
    {
        Vec3f ret;
        switch(_scatterType)
        {
            case PHOTON_SCATTER_LIGHT:
            {
                DirectionSample dirSample(outDir);
                ret = light()->evalDirectionalEmission(_posSample, dirSample);
                break;
            }
            case PHOTON_SCATTER_MEDIUM:
            {
                const Medium *medium = reinterpret_cast<const Medium *>(_scatterPtr);
                ret = medium->phaseFunction(_pos)->eval(_inDir, outDir);
                break;
            }
            case PHOTON_SCATTER_SURFACE:
            {
                SurfaceScatterEvent event = _event;
                event.sampler = &sampler;
                event.wo = event.frame.toLocal(outDir);
                // Asymmetry due to shading normals already compensated for when storing the photon,
                // so we don't use the adjoint BSDF here
                ret = _isectInfo.bsdf->eval(event, false);
                break;
            }
            default:
            {
                FAIL("unsupported scatter type");
                break;
            }
        }
        return ret;
    }

    float epsilon() const
    {
        switch(_scatterType)
        {
            case PHOTON_SCATTER_LIGHT:
                return 1e-4f;
            case PHOTON_SCATTER_SURFACE:
                return _isectInfo.epsilon;
            case PHOTON_SCATTER_MEDIUM:
                return 0.f;
            default:
                FAIL("unsupported scatter type");
                return 1e-4f;
        }
    }

    const Medium *medium() const
    {
        ASSERT(_scatterType == PHOTON_SCATTER_MEDIUM, "wrong scatter type!");
        return reinterpret_cast<const Medium *>(_scatterPtr);
    }

    const Primitive *light() const
    {
        ASSERT(_scatterType == PHOTON_SCATTER_LIGHT, "wrong scatter type!");
        return reinterpret_cast<const Primitive *>(_scatterPtr);
    }
};

struct SteadyPathPhoton
{
    Vec3f pos;
    Vec3f power;
    Vec3f dir;
    float length;
    float sampledLength;
    uint32 data;
    // TODO: incorporate following two fields into data as bit type
    bool _isGhostBounce;
    const Medium *inMedium = nullptr;
    PhotonScatterEvent _scatter;
    Vec3f surfPower;

    void setPathInfo(uint32 bounce, bool onSurface, bool isGhostBounce = false)
    {
        data = bounce;
        if (onSurface)
            data |= (1u << 31u);
        _isGhostBounce = isGhostBounce;
    }

    bool onSurface() const
    {
        return (data & (1u << 31u)) != 0;
    }
    uint32 bounce() const
    {
        return data & ~(1u << 31u);
    }
    bool isGhostBounce() const
    {
        return _isGhostBounce;
    }

    PhotonScatterEvent& scatter()
    {
        return _scatter;
    }
    const PhotonScatterEvent& scatter() const
    {
        return _scatter;
    }

};
struct SteadyPhotonBeam
{
    Vec3f p0, p1;
    Vec3f dir;
    float length;
    Vec3f power;
    int bounce;
    bool valid;
};
struct SteadyPhotonPlane0D
{
    Vec3f p0, p1, p2, p3;
    Vec3f power;
    Vec3f d1;
    float l1;
    int bounce;
    bool valid;

    Box3f bounds() const
    {
        Box3f box;
        box.grow(p0);
        box.grow(p1);
        box.grow(p2);
        box.grow(p3);
        return box;
    }
};
struct PhotonPlane1D
{
    Vec3f p;
    Vec3f invU, invV, invW;
    Vec3f center, a, b, c;
    Vec3f power;
    Vec3f d1;
    float l1;
    float invDet;
    float binCount;
    int bounce;
    bool valid;

    Box3f bounds() const
    {
        Box3f box;
        box.grow(center + a + b + c);
        box.grow(center - a + b + c);
        box.grow(center + a - b + c);
        box.grow(center - a - b + c);
        box.grow(center + a + b - c);
        box.grow(center - a + b - c);
        box.grow(center + a - b - c);
        box.grow(center - a - b - c);
        return box;
    }
};

struct SteadyPhotonVolume
{
    // p: starting point of the volume
    // a, b, c: three edges of the photon volume connecting with p
    //         _____
    //      v /    /|
    //     p /____/
    //       | w  | |
    //     u |____|/
    //

    Vec3f p;
    Vec3f a, b, c;

    Vec3f aDir, bDir, cDir;
    float aLen, bLen, cLen;

    Vec3f power;
    float invDet;
    int bounce;
    bool valid;

    virtual Box3f bounds() const
    {
        Box3f box;
        box.grow(p            );
        box.grow(p + a        );
        box.grow(p     + b    );
        box.grow(p         + c);
        box.grow(p + a + b    );
        box.grow(p + a     + c);
        box.grow(p +     b + c);
        box.grow(p + a + b + c);
        return box;
    }
};

struct SteadyPhotonHyperVolume
{
    Vec3f p;
    Vec3f a, b, c, d;

    Vec3f aDir, bDir, cDir, dDir;
    float aLen, bLen, cLen, dLen;

    Vec3f power;
    int bounce;
    bool valid;

    Box3f bounds() const
    {
        Box3f box;
        for(uint32_t i = 0; i <= 0b1111; ++i)
        {
            Vec3f x = p;
            if (i & 0b0001)  x += a;
            if (i & 0b0010)  x += b;
            if (i & 0b0100)  x += c;
            if (i & 0b1000)  x += d;
            box.grow(x);
        }
        return box;
    };
};

struct SteadyPhotonBall
{
    Vec3f p;
    float r;
    PhotonScatterEvent scatter;

    Vec3f power;
    int bounce;
    bool valid;

    virtual Box3f bounds() const
    {
        Box3f box;
        box.grow(p + Vec3f(r, 0.f, 0.f));
        box.grow(p - Vec3f(r, 0.f, 0.f));
        box.grow(p + Vec3f(0.f, r, 0.f));
        box.grow(p - Vec3f(0.f, r, 0.f));
        box.grow(p + Vec3f(0.f, 0.f, r));
        box.grow(p - Vec3f(0.f, 0.f, r));
        return box;
    };
};

enum VolumePhotonEnum : uint32
{
    VOLUME_POINTS = 0,
    VOLUME_BEAMS,
    VOLUME_PLANES,
    VOLUME_PLANES_1D,
    VOLUME_VOLUMES,
    VOLUME_HYPERVOLUMES,
    VOLUME_BALLS,
    VOLUME_VOLUMES_BALLS,
    VOLUME_MIS_VOLUMES_BALLS,
    VOLUME_COUNT,
};

struct TypedPhotonIdx
{
    constexpr static uint32 NUM_TYPE_BITS = 4;
    constexpr static uint32 NUM_IDX_BITS = sizeof(uint32) * 8 - NUM_TYPE_BITS;
    constexpr static uint32 IDX_MASK =  (1 << NUM_IDX_BITS) - 1;

    uint32 type : NUM_TYPE_BITS;
    uint32 idx  : NUM_IDX_BITS;

    TypedPhotonIdx() = default;
    TypedPhotonIdx(VolumePhotonEnum _type, uint32 _idx)
        :type(_type), idx(_idx)
    {
    }
    TypedPhotonIdx(uint32 _typedIdx)
    {
        type = _typedIdx >> NUM_IDX_BITS;
        idx = _typedIdx & IDX_MASK;
    }

    explicit operator uint32() const
    {
        return (idx & IDX_MASK) | (type << NUM_IDX_BITS);
    }
};
static_assert(VOLUME_COUNT <= (1 << TypedPhotonIdx::NUM_TYPE_BITS));
static_assert(sizeof(TypedPhotonIdx) == sizeof(uint32));

namespace Transient
{
    struct TransientVolumePhoton : public SteadyVolumePhoton
    {
        float timeTraveled;
    };

    struct TransientPathPhoton : public SteadyPathPhoton
    {
        float timeTraveled;
    };

    struct TransientPhotonBeam : public SteadyPhotonBeam
    {
        float timeTraveled;
    };

    struct TransientPhotonPlane0D : public SteadyPhotonPlane0D
    {
        float timeTraveled;
    };

    struct TransientPhotonVolume : public SteadyPhotonVolume
    {
        float timeTraveled;
        Vec3f n;

        bool isDeltaSliced;
        // sampled time elapsed within photon volume
        float sampledDeltaTime;

        Vec3f p0, p1, p2;

        virtual Box3f bounds() const override
        {
            Box3f volumeBox = SteadyPhotonVolume::bounds();
            if (!isDeltaSliced)
            {
                return volumeBox;
            }

            Box3f box;
            box.grow(p0);
            box.grow(p1);
            box.grow(p2);
            box.intersect(volumeBox);
            return box;
        }
    };

    struct TransientPhotonHyperVolume : public SteadyPhotonHyperVolume
    {
        float timeTraveled;
        float invJacobian;
    };

    struct TransientPhotonBall : public SteadyPhotonBall
    {
        float timeTraveled;

        bool isDeltaSliced;
        float temporalRadius;

        // sampled time elapsed within photon volume
        float sampledDeltaTime;

        virtual Box3f bounds() const override
        {
            if (!isDeltaSliced)
            {
                return SteadyPhotonBall::bounds();
            }

            // TODO: deduplicate with parent class
            Box3f box;
            box.grow(p + Vec3f(temporalRadius, 0.f, 0.f));
            box.grow(p - Vec3f(temporalRadius, 0.f, 0.f));
            box.grow(p + Vec3f(0.f, temporalRadius, 0.f));
            box.grow(p - Vec3f(0.f, temporalRadius, 0.f));
            box.grow(p + Vec3f(0.f, 0.f, temporalRadius));
            box.grow(p - Vec3f(0.f, 0.f, temporalRadius));
            return box;
        }
    };
}

template <bool isTransient>
struct PhotonPrimitives
{
    static CONSTEXPR bool isSteadyState = !isTransient;
    typedef typename std::conditional<isSteadyState, SteadyPhotonBeam, Transient::TransientPhotonBeam>::type PhotonBeam;
    typedef typename std::conditional<isSteadyState, SteadyPhotonPlane0D, Transient::TransientPhotonPlane0D>::type PhotonPlane0D;
    typedef typename std::conditional<isSteadyState, SteadyPhotonVolume, Transient::TransientPhotonVolume>::type PhotonVolume;
    typedef typename std::conditional<isSteadyState, SteadyPhotonHyperVolume, Transient::TransientPhotonHyperVolume>::type PhotonHyperVolume;
    typedef typename std::conditional<isSteadyState, SteadyPhotonBall, Transient::TransientPhotonBall>::type PhotonBall;

    const PhotonBeam *beams;
    const PhotonPlane0D *planes0D;
    const PhotonPlane1D *planes1D;
    const PhotonVolume *volumes;
    const PhotonHyperVolume *hyperVolumes;
    const PhotonBall *balls;
};

}

#endif /* PHOTON_HPP_ */
