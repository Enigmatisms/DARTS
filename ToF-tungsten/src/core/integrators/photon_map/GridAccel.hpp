#ifndef GRIDACCEL_HPP_
#define GRIDACCEL_HPP_

#include "thread/ThreadUtils.hpp"
#include "thread/ThreadPool.hpp"
#include "Timer.hpp"

#include <tribox/tribox.hpp>
#include <atomic>
#include <cmath>
#include "Photon.hpp"

namespace Tungsten {

class GridAccel
{
public:
    struct Primitive
    {
        Vec3f p0, p1, p2, p3;
        float r;
        TypedPhotonIdx typedIdx;

        Primitive() = default;
        Primitive(uint32 idx_, Vec3f p0_, Vec3f p1_, Vec3f p2_, Vec3f p3_, float r_, VolumePhotonEnum type_)
        : p0(p0_), p1(p1_), p2(p2_), p3(p3_), r(r_), typedIdx()
        {
            typedIdx.type = type_;
            typedIdx.idx = idx_;
        }

        inline VolumePhotonEnum type() const
        {
            return static_cast<VolumePhotonEnum>(typedIdx.type);
        }

        inline uint32 idx() const
        {
            return typedIdx.idx;
        }
    };

private:
    using TriVerts = float[3][3];
    std::unique_ptr<std::atomic<uint32>[]> _atomicListOffsets;
    const uint32 *_listOffsets;
    std::unique_ptr<uint32[]> _lists;
    Vec3f _offset;
    Vec3f _scale;
    Vec3f _invScale;
    Vec3i _sizes;
    Vec3f _fSizes;

    int64 _yStride;
    int64 _zStride;
    uint64 _cellCount;

    uint64 idx(int x, int y, int z) const
    {
        return x + y*_yStride + z*_zStride;
    }

    template<typename LoopBody>
    void iterateBounds(Box3f bounds, LoopBody body)
    {
        Vec3i minI = max(Vec3i(bounds.min()), Vec3i(0));
        Vec3i maxI = min(Vec3i(bounds.max()), _sizes - 1);

        for (int z = minI.z(); z <= maxI.z(); ++z)
            for (int y = minI.y(); y <= maxI.y(); ++y)
                for (int x = minI.x(); x <= maxI.x(); ++x)
                    body(x, y, z);
    }

    inline Box3f volumeBounds(Vec3f p, Vec3f a, Vec3f b, Vec3f c, float r, Vec3f scale, Vec3f offset)
    {
        Vec3f points[8];
        points[0] = (p            );
        points[1] = (p + a        );
        points[2] = (p     + b    );
        points[3] = (p         + c);
        points[4] = (p + a + b    );
        points[5] = (p + a     + c);
        points[6] = (p +     b + c);
        points[7] = (p + a + b + c);

        Vec3f radius = r * scale;
        for (uint32 i = 0; i < 8; ++i)
        {
            points[i] = (points[i] - offset) * scale;
        }

        Box3f bounds;
        for (uint32 i = 0; i < 8; ++i)
        {
            bounds.grow(points[i] + radius);
            bounds.grow(points[i] - radius);
        }
        return bounds;
    }

    inline void getPlaneTriangles(Vec3f p0, Vec3f p1, Vec3f p2, Vec3f p3, TriVerts& t0, TriVerts& t1)
    {
        memcpy(&t0[0][0], p0.data(), sizeof(float) * 3);
        memcpy(&t0[1][0], p1.data(), sizeof(float) * 3);
        memcpy(&t0[2][0], p2.data(), sizeof(float) * 3);

        memcpy(&t1[0][0], p0.data(), sizeof(float) * 3);
        memcpy(&t1[1][0], p2.data(), sizeof(float) * 3);
        memcpy(&t1[2][0], p3.data(), sizeof(float) * 3);
    }

    template<typename LoopBody>
    void iterateVolume(Vec3f p, Vec3f a, Vec3f b, Vec3f c, float r, LoopBody body)
    {
        Box3f bounds = volumeBounds(p, a, b, c, r, _scale, _offset);

        Vec3f points[8];
        points[0] = (p            );
        points[1] = (p + a        );
        points[2] = (p     + b    );
        points[3] = (p         + c);
        points[4] = (p + a + b    );
        points[5] = (p + a     + c);
        points[6] = (p +     b + c);
        points[7] = (p + a + b + c);

        Vec3f radius = r * _scale;
        for (uint32 i = 0; i < 8; ++i)
        {
            points[i] = (points[i] - _offset) * _scale;
        }

        static const uint32 quadFaceIndices [6][4] =
        {
            {0, 1, 4, 2},
            {3, 5, 7, 6},
            {0, 1, 5, 3},
            {1, 4, 7, 5},
            {2, 4, 7, 6},
            {0, 2, 6, 3},
        };

        TriVerts quadTris[12];
        for (uint32 iface = 0; iface < 6; ++iface)
        {
            Vec3f p0 = points[quadFaceIndices[iface][0]];
            Vec3f p1 = points[quadFaceIndices[iface][1]];
            Vec3f p2 = points[quadFaceIndices[iface][2]];
            Vec3f p3 = points[quadFaceIndices[iface][3]];

            getPlaneTriangles(p0, p1, p2, p3, quadTris[2 * iface], quadTris[2 * iface + 1]);
        }

        iterateBounds(bounds, [&](int x, int y, int z) {
            Vec3f boxCenter = Vec3f(Vec3i(x, y, z)) + 0.5f;
            Vec3f boxHalfSize(0.5f + radius);

            bool overlap = false;
            for (uint32 i = 0; i < 12; ++i)
            {
                if (triBoxOverlap(boxCenter.data(), boxHalfSize.data(), quadTris[i]))
                {
                    overlap = true;
                    break;
                }
            }

            if (overlap)
            {
                body(x, y, z);
            }
        });
    }

    inline Box3f trapezoidBounds(Vec3f p0, Vec3f p1, Vec3f p2, Vec3f p3, float r, Vec3f scale, Vec3f offset)
    {
        Vec3f radius = r * scale;
        p0 = (p0 - offset) * scale;
        p1 = (p1 - offset) * scale;
        p2 = (p2 - offset) * scale;
        p3 = (p3 - offset) * scale;
        Box3f bounds;
        bounds.grow(p0 + radius); bounds.grow(p0 - radius);
        bounds.grow(p1 + radius); bounds.grow(p1 - radius);
        bounds.grow(p2 + radius); bounds.grow(p2 - radius);
        bounds.grow(p3 + radius); bounds.grow(p3 - radius);
        return bounds;
    }

    template<typename LoopBody>
    void iterateTrapezoid(Vec3f p0, Vec3f p1, Vec3f p2, Vec3f p3, float r, LoopBody body)
    {
        Box3f bounds = trapezoidBounds(p0, p1, p2, p3, r, _scale, _offset);
        Vec3f radius = r*_scale;
        p0 = (p0 - _offset)*_scale;
        p1 = (p1 - _offset)*_scale;
        p2 = (p2 - _offset)*_scale;
        p3 = (p3 - _offset)*_scale;
        TriVerts vertsA, vertsB;
        getPlaneTriangles(p0, p1, p2, p3, vertsA, vertsB);

        iterateBounds(bounds, [&](int x, int y, int z) {
            Vec3f boxCenter = Vec3f(Vec3i(x, y, z)) + 0.5f;
            Vec3f boxHalfSize(0.5f + radius);
            if (triBoxOverlap(boxCenter.data(), boxHalfSize.data(), vertsA) || triBoxOverlap(boxCenter.data(), boxHalfSize.data(), vertsB))
                body(x, y, z);
        });
    }

    inline Box3f beamBounds(Vec3f p0, Vec3f p1, float r, Vec3f scale, Vec3f offset)
    {
        Vec3f radius = r * scale;
        Box3f bounds;
        bounds.grow((p0 - offset) * scale + radius);
        bounds.grow((p0 - offset) * scale - radius);
        bounds.grow((p1 - offset) * scale + radius);
        bounds.grow((p1 - offset) * scale - radius);
        return bounds;
    }

    template<typename LoopBody>
    void iterateBeam(Vec3f p0, Vec3f p1, float r, LoopBody body)
    {
        Box3f bounds = beamBounds(p0, p1, r, _scale, _offset);

        Vec3f d = p1 - p0;
        Vec3f invD = 1.0f/d;
        Vec3f coordScale = _invScale*invD;

        Vec3f relMin(-r + _offset - p0);
        Vec3f relMax((_invScale + r) + _offset - p0);
        Vec3f tMins, tMaxs;
        for (int i = 0; i < 3; ++i) {
            if (invD[i] >= 0.0f) {
                tMins[i] = relMin[i]*invD[i];
                tMaxs[i] = relMax[i]*invD[i];
            } else {
                tMins[i] = relMax[i]*invD[i];
                tMaxs[i] = relMin[i]*invD[i];
            }
        }

        iterateBounds(bounds, [&](int x, int y, int z) {
            Vec3f boxTs = Vec3f(Vec3i(x, y, z))*coordScale;
            float tMin = max((tMins + boxTs).max(), 0.0f);
            float tMax = min((tMaxs + boxTs).min(), 1.0f);
            if (tMin <= tMax)
                body(x, y, z);
        });
    }


    Box3f getPrimsBounds(const std::vector<Primitive>& prims)
    {
        Box3f bounds;
        for(size_t i = 0; i < prims.size(); ++i)
        {
            switch (prims[i].type())
            {
                case VOLUME_BEAMS:
                    bounds.grow(beamBounds(prims[i].p0, prims[i].p1, prims[i].r, Vec3f(1.f), Vec3f(0.f)));
                    break;
                case VOLUME_PLANES:
                    bounds.grow(trapezoidBounds(prims[i].p0, prims[i].p1, prims[i].p2, prims[i].p3, prims[i].r, Vec3f(1.f), Vec3f(0.f)));
                    break;
                case VOLUME_VOLUMES:
                    bounds.grow(volumeBounds(prims[i].p0, prims[i].p1, prims[i].p2, prims[i].p3, prims[i].r, Vec3f(1.f), Vec3f(0.f)));
                    break;
                default:
                    FAIL("unknown PrimType: %u", prims[i].type());
            }
        }
        return bounds;
    }

    void buildAccel(std::vector<Primitive> prims)
    {
        _atomicListOffsets = zeroAlloc<std::atomic<uint32>>(_cellCount + 1);

        ThreadUtils::parallelFor(0, prims.size(), ThreadUtils::pool->threadCount() + 1, [&](uint32 i) {
            switch (prims[i].type())
            {
                case VOLUME_BEAMS:
                    iterateBeam(prims[i].p0, prims[i].p1, prims[i].r, [&](int x, int y, int z) {
                        _atomicListOffsets[idx(x, y, z)]++;
                    });
                    break;
                case VOLUME_PLANES:
                    iterateTrapezoid(prims[i].p0, prims[i].p1, prims[i].p2, prims[i].p3, prims[i].r, [&](int x, int y, int z) {
                        _atomicListOffsets[idx(x, y, z)]++;
                    });
                    break;
                case VOLUME_VOLUMES:
                    iterateVolume(prims[i].p0, prims[i].p1, prims[i].p2, prims[i].p3, prims[i].r, [&](int x, int y, int z) {
                        _atomicListOffsets[idx(x, y, z)]++;
                    });
                    break;
                default:
                    FAIL("unknown PrimType!");
            }
        });

        uint32 prefixSum = 0;
        for (uint64 i = 0; i <= _cellCount; ++i) {
            prefixSum += _atomicListOffsets[i];
            _atomicListOffsets[i] = prefixSum;
        }

        _lists.reset(new uint32[prefixSum]);

        ThreadUtils::parallelFor(0, prims.size(), ThreadUtils::pool->threadCount() + 1, [&](uint32 i) {
            switch (prims[i].type())
            {
                case VOLUME_BEAMS:
                    iterateBeam(prims[i].p0, prims[i].p1, prims[i].r, [&](int x, int y, int z) {
                        _lists[--_atomicListOffsets[idx(x, y, z)]] = prims[i].typedIdx.idx;
                    });
                    break;
                case VOLUME_PLANES:
                    iterateTrapezoid(prims[i].p0, prims[i].p1, prims[i].p2, prims[i].p3, prims[i].r, [&](int x, int y, int z) {
                        _lists[--_atomicListOffsets[idx(x, y, z)]] = prims[i].typedIdx.idx;
                    });
                    break;
                case VOLUME_VOLUMES:
                    iterateVolume(prims[i].p0, prims[i].p1, prims[i].p2, prims[i].p3, prims[i].r, [&](int x, int y, int z) {
                        _lists[--_atomicListOffsets[idx(x, y, z)]] = prims[i].typedIdx.idx;
                    });
                    break;
                default:
                    FAIL("unknown PrimType!");
            }
        });

        _listOffsets = reinterpret_cast<const uint32 *>(&_atomicListOffsets[0]);
    }

public:
    GridAccel(Box3f bounds, int memBudgetKb, std::vector<Primitive> prims)
    {
        Timer timer;
        if (bounds.empty())
        {
            bounds = getPrimsBounds(prims);
        }


        Vec3f diag = bounds.diagonal();
        Vec3f relDiag = diag/diag.max();
        float maxCells = std::cbrt(double(int64(memBudgetKb) << 10)/(4.0*relDiag.product()));
        _sizes = max(Vec3i(relDiag*maxCells), Vec3i(1));
        _offset = bounds.min();
        _scale = Vec3f(_sizes)/diag;
        _invScale = 1.0f/_scale;
        _fSizes = Vec3f(_sizes);

        _yStride = _sizes.x();
        _zStride = _sizes.x()*_sizes.y();

        _cellCount = _zStride*_sizes.z();

        std::cout << "Building grid accelerator with bounds " << bounds << " and size " << _sizes << " (" << (_sizes.product()*4)/(1024*1024) << "mb)" << std::endl;

        timer.bench("Initialization");

        buildAccel(std::move(prims));
    }

    template<typename Iterator>
    void trace(Ray ray, Iterator iterator) const
    {
        Vec3f o = (ray.pos() - _offset)*_scale;
        Vec3f d = ray.dir()*_scale;

        Vec3f relMin = -o;
        Vec3f relMax = _fSizes - o;

        float tMin = ray.nearT(), tMax = ray.farT();
        Vec3f tMins;
        Vec3f tStep = 1.0f/d;
        for (int i = 0; i < 3; ++i) {
            if (d[i] >= 0.0f) {
                tMins[i] = relMin[i]*tStep[i];
                tMin = max(tMin, tMins[i]);
                tMax = min(tMax, relMax[i]*tStep[i]);
            } else {
                tMins[i] = relMax[i]*tStep[i];
                tMin = max(tMin, tMins[i]);
                tMax = min(tMax, relMin[i]*tStep[i]);
            }
        }
        if (tMin >= tMax)
            return;

        tStep = std::abs(tStep);

        Vec3f p = o + d*tMin;
        Vec3f nextT;
        Vec3i iStep, iP;
        for (int i = 0; i < 3; ++i) {
            if (d[i] >= 0.0f) {
                iP[i] = max(int(p[i]), 0);
                nextT[i] = tMin + (float(iP[i] + 1) - p[i])*tStep[i];
                iStep[i] = 1;
            } else {
                iP[i] = min(int(p[i]), _sizes[i] - 1);
                nextT[i] = tMin + (p[i] - float(iP[i]))*tStep[i];
                iStep[i] = -1;
            }
        }

        while (tMin < tMax) {
            int minIdx = nextT.minDim();
            float cellTmax = nextT[minIdx];

            uint64 i = idx(iP.x(), iP.y(), iP.z());
            uint32 listStart = _listOffsets[i];
            uint32 count = _listOffsets[i + 1] - listStart;
            if (count)
                for (uint32 t = 0; t < count; ++t)
                    iterator(_lists[listStart + t], tMin, min(cellTmax, tMax));

            tMin = cellTmax;
            nextT[minIdx] += tStep[minIdx];
            iP   [minIdx] += iStep[minIdx];

            if (iP[minIdx] < 0 || iP[minIdx] >= _sizes[minIdx])
                return;
        }
    }
};

}

#endif /* GRIDACCEL_HPP_ */
