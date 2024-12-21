#ifndef TRANSIENTBUFFER_HPP_
#define TRANSIENTBUFFER_HPP_

#include "OutputBufferSettings.hpp"

#include "math/Vec.hpp"
#include "math/Ray.hpp"

#include "io/JsonSerializable.hpp"
#include "io/FileUtils.hpp"
#include "io/ImageIO.hpp"

#include "Memory.hpp"

#include <memory>

namespace Tungsten {

template<typename T>
class TransientBuffer
{
    Vec2u _res;
    uint32 num_frames;

    // add transient sample buffer
    // transients are arrange in [][][][] frame order (but not pixel order)
    std::unique_ptr<T[]> _transients;
    std::unique_ptr<uint32[]> _sampleCount;

    const OutputBufferSettings &_settings;

    inline float average(float x) const
    {
        return x;
    }
    inline float average(const Vec3f &x) const {return x.avg();}

    const float *elementPointer(const float *p) const {return p;}
    const float *elementPointer(const Vec3f *p) const {return p->data();}

    int elementCount(float /*x*/) const {return 1;}
    int elementCount(Vec3f /*x*/) const {return 3;}

    template<typename Texel>
    void saveLdr(const Texel *hdr, const Path &path, bool /*useless*/) const;
public:
    TransientBuffer(Vec2u res, const OutputBufferSettings &settings)
    : _res(res),
      _settings(settings)
    {
        size_t numPixels = res.product();
        _sampleCount = zeroAlloc<uint32>(numPixels);
    }

    void initTransientBuffer(int nframes);

    /// @brief  Temporal path reused transient recording (originally not supported, by Qianyue He)
    void addTransientSample(T* pixel_transient, Vec2u pixel);

    inline T operator[](uint32 idx) const
    {
        return _transients[idx];
    }

    void save() const;

    void deserialize(InputStreamHandle &) const
    {
        throw "Not implemented in transient buffer";
    }

    void serialize(OutputStreamHandle &) const
    {
        throw "Not implemented in transient buffer";
    }

    inline T variance(int, int) const
    {
        printf("Warning: variance recording is not implemented in transient buffer, please do not call this function.\n");
        return _transients[0];
    }
};

using TransientBufferF = TransientBuffer<float>;
using TransientBuffer3f = TransientBuffer<Vec3f>;

template class TransientBuffer<float>;
template class TransientBuffer<Vec3f>;
}

#endif /* TRANSIENTBUFFER_HPP_ */
