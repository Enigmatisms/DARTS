#ifndef DIRECTIONALPOINT_HPP_
#define DIRECTIONALPOINT_HPP_

#include "Point.hpp"

namespace Tungsten {

class DirectionalPoint : public Point
{
protected:
    Vec3f _dir;

public:
    DirectionalPoint() = default;
    DirectionalPoint(Mat4f &transform);

    virtual void fromJson(JsonPtr value, const Scene &scene) override;
    virtual rapidjson::Value toJson(Allocator &allocator) const override;

    virtual bool sampleDirection(PathSampleGenerator &sampler, const PositionSample &point, DirectionSample &sample) const override;
    virtual bool sampleDirect(uint32 threadIndex, const Vec3f &p, PathSampleGenerator &sampler, LightSample &sample) const override;
    bool invertDirection(WritablePathSampleGenerator &sampler, const PositionSample & /*point*/, const DirectionSample &direction) const;

    virtual float directionalPdf(const PositionSample &point, const DirectionSample &sample) const override;
    virtual float directPdf(uint32 threadIndex, const IntersectionTemporary &data,
                            const IntersectionInfo &info, const Vec3f &p) const override;

    virtual Vec3f evalDirectionalEmission(const PositionSample &point, const DirectionSample &sample) const override;
    virtual Vec3f evalDirect(const IntersectionTemporary &data, const IntersectionInfo &info) const override;

    virtual bool isDirectionDirac() const override;

    virtual void prepareForRender() override;
};

}

#endif /* DIRECTIONALPOINT_HPP_ */