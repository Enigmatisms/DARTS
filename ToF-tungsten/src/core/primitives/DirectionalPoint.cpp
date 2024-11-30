#include "DirectionalPoint.hpp"
#include "TriangleMesh.hpp"

#include "sampling/PathSampleGenerator.hpp"
#include "sampling/SampleWarp.hpp"

#include "io/JsonObject.hpp"
#include "io/Scene.hpp"

#include "Debug.hpp"

namespace Tungsten {

DirectionalPoint::DirectionalPoint(Mat4f &transform)
    : Point(transform),
      _dir(0.0f, 0.0f, 1.0f)
{
}

void DirectionalPoint::fromJson(JsonPtr value, const Scene &scene)
{
    Primitive::fromJson(value, scene);
}

rapidjson::Value DirectionalPoint::toJson(Allocator &allocator) const
{
    return JsonObject{Primitive::toJson(allocator), allocator,
                      "type", "directional_point"};
}

bool DirectionalPoint::sampleDirection(PathSampleGenerator &/* sampler */, const PositionSample &/* point */, DirectionSample &sample) const
{
    sample.d = _dir;
    sample.weight = Vec3f(1.0f);
    sample.pdf = 1.0f;

    return true;
}

bool DirectionalPoint::sampleDirect(uint32 threadIndex, const Vec3f &p, PathSampleGenerator &sampler, LightSample &sample) const
{
    // possibly of the light source pos and input position aligns with the direction
    // of the light is zero
    return false;
}

bool DirectionalPoint::invertDirection(WritablePathSampleGenerator &sampler, const PositionSample & /*point*/, const DirectionSample &direction) const
{
    // WARNING: not sure if this is correct.
    return true;
}

float DirectionalPoint::directionalPdf(const PositionSample &point, const DirectionSample &sample) const
{
    Vec3f lightToPoint = point.p - _pos;
    Vec3f cross = lightToPoint.cross(_dir);
    float dot = lightToPoint.dot(_dir);
    if(dot > 0.0f && cross.lengthSq() == 0.0f)
    {
        return 1.0f;
    }

    return 0.0f;
}

float DirectionalPoint::directPdf(uint32 threadIndex, const IntersectionTemporary &data,
                                  const IntersectionInfo &info, const Vec3f &p) const
{
    // WARNING: not sure if this is correct.
    //          just copying implementation from Point.cpp
    return (p - _pos).lengthSq();
}

Vec3f DirectionalPoint::evalDirectionalEmission(const PositionSample &point, const DirectionSample &sample) const
{
    if(sample.d.dot(_dir) == 1.0)
    {
        return Vec3f(INV_FOUR_PI);
    }
    else
    {
        return Vec3f(0.0f);
    }
}

Vec3f DirectionalPoint::evalDirect(const IntersectionTemporary &data, const IntersectionInfo &info) const
{
    if(info.Ng.dot(_dir) == 1.0)
    {
        return (*_emission)[Vec2f(0.0f)];
    }
    else
    {
        return Vec3f(0.0f);
    }
}

bool DirectionalPoint::isDirectionDirac() const
{
    return true;
}

void DirectionalPoint::prepareForRender()
{
    _dir = _transform.fwd();

    Point::prepareForRender();
}

}