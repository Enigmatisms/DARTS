#ifndef COLLIMATED_HPP_
#define COLLIMATED_HPP_

/**
 * @brief simple collimated light source that models an ideal laser emitter:
 * an illustration:
 *     ------------------------------------
 *                  | radius
 * _pos------------------------------------> (_dir)
 * 
 *     ------------------------------------
 * 
 * 
*/

#include "Point.hpp"

namespace Tungsten {

// Ideal collimated light source with beam radius
class Collimated : public Point
{
protected:
    Vec3f _dir;             // direction
    Vec3f _up;              // up direction (can be set)
    Vec3f _lat;             // lateral direction (computed)
    float radius;           // beam radius
    float _area;            // area of the beam cross section
    bool  is_delta;         // whether the source is delta (if radius is 0, then true)
public:
    Collimated() = default;
    Collimated(Mat4f &transform);

    virtual void fromJson(JsonPtr value, const Scene &scene) override;
    virtual rapidjson::Value toJson(Allocator &allocator) const override;

    virtual bool samplePosition(PathSampleGenerator &sampler, PositionSample &sample) const override;

    virtual bool sampleDirection(PathSampleGenerator &sampler, const PositionSample &point, DirectionSample &sample) const override;
    virtual bool sampleDirect(uint32 threadIndex, const Vec3f &p, PathSampleGenerator &sampler, LightSample &sample) const override;
    virtual bool invertDirection(WritablePathSampleGenerator &sampler, const PositionSample & /*point*/, const DirectionSample &direction) const override;
    virtual bool invertPosition(WritablePathSampleGenerator &sampler, const PositionSample &point) const override;

    virtual float directionalPdf(const PositionSample &point, const DirectionSample &sample) const override;
    virtual float directPdf(uint32 threadIndex, const IntersectionTemporary &data,
                            const IntersectionInfo &info, const Vec3f &p) const override;

    virtual Vec3f evalDirectionalEmission(const PositionSample &point, const DirectionSample &sample) const override;
    virtual Vec3f evalDirect(const IntersectionTemporary &data, const IntersectionInfo &info) const override;

    virtual bool isDirac() const override;
    virtual bool isDirectionDirac() const override;

    virtual void prepareForRender() override;
};

}

#endif /* COLLIMATED_HPP_ */