#include "Collimated.hpp"
#include "TriangleMesh.hpp"

#include "sampling/PathSampleGenerator.hpp"
#include "sampling/SampleWarp.hpp"

#include "io/JsonObject.hpp"
#include "io/Scene.hpp"

#include "Debug.hpp"

namespace Tungsten {

constexpr float COLLIMATED_THS = 1.f - 1e-5f;

Collimated::Collimated(Mat4f &transform)
    : Point(transform),
      _dir(0.0f, 0.0f, 1.0f),       // lat x up
      _up(0.0f, 1.0f, 0.f),         // dir x lat
      _lat(1.0f, 0.0f, 0.0f),        // up x dir
      radius(0),
      _area(1),                      // delta beam is 1
      is_delta(true)
{}

void Collimated::fromJson(JsonPtr value, const Scene &scene)
{
    Primitive::fromJson(value, scene);
    bool ret = value.getField("radius", radius);
    if (!ret) {
        printf("Collimated source radius not set, set to zero (ideal beam) by default.\n");
        radius = 0.f;
    }
    // ret = value.getField("up", _up);
    // if (!ret) {
    //     printf("Up not specified, calculate with (0, 1, 0) by default.\n");
    //     printf("Please make sure that dir is neither (0, 1, 0) nor (0, -1, 0).\n");
    //     _lat = Vec3f(0, 1, 0).cross(_dir).normalized();
    //     _up = _dir.cross(_lat).normalized();
    // }
    is_delta = radius == 0.f;
    _area = is_delta ? 1.f : (M_PI * radius * radius);
}

rapidjson::Value Collimated::toJson(Allocator &allocator) const
{
    return JsonObject{Primitive::toJson(allocator), allocator,
                      "type", "collimated",
                      "radius", radius
    };
}

bool Collimated::samplePosition(PathSampleGenerator &sampler, PositionSample &sample) const
{
    Vec2f xi = sampler.next2D();
    Vec2f lQ = SampleWarp::uniformDisk(xi).xy()* radius;
    sample.uv = Vec2f(xi.x() + 0.5f, std::sqrt(xi.y()));
    if (sample.uv.x() > 1.0f)
        sample.uv.x() -= 1.0f;

    sample.p = _pos + lQ.x() * _lat + lQ.y() * _up;
    sample.Ng = _dir;
    sample.weight = (*_emission)[Vec2f(0.0f)] * _area;
    sample.pdf = 1.f / _area;
    return true;
}

// Direction sample for collimated source is always the direction
bool Collimated::sampleDirection(PathSampleGenerator &/* sampler */, const PositionSample &/* point */, DirectionSample &sample) const
{
    sample.d = _dir;
    sample.weight = Vec3f(1.0f);
    sample.pdf = 1.0f;
    return true;
}

bool Collimated::sampleDirect(uint32 threadIndex, const Vec3f &p, PathSampleGenerator &sampler, LightSample &sample) const
{
    if (is_delta) return false;     // can not sample direct illumination when radius is 0
    Vec3f sample_d = p - _pos;
    float dist2 = sample_d.lengthSq(), proj_len = sample_d.dot(_dir);
    if (proj_len <= 0) return false;
    if (dist2 - proj_len * proj_len > radius * radius) return false;        // out of range

    sample.d = _dir;

    sample.dist = proj_len;
    sample.pdf = 1;
    return false;
}

bool Collimated::invertPosition(WritablePathSampleGenerator &sampler, const PositionSample &point) const {
    if (is_delta) return true;
    Vec3f p = point.p - _pos;
    Vec3f lQ = Vec3f(_lat.dot(p) / radius, _up.dot(p) / radius, 0.0f);
    sampler.put2D(SampleWarp::invertUniformDisk(lQ, sampler.untracked1D()));
    return true;
}

bool Collimated::invertDirection(WritablePathSampleGenerator &sampler, const PositionSample & /*point*/, const DirectionSample &direction) const
{
    // direction does not need to be inverted
    return true;
}

float Collimated::directionalPdf(const PositionSample &point, const DirectionSample &sample) const
{
    Vec3f lightToPoint = point.p - _pos;
    Vec3f cross = lightToPoint.cross(_dir);
    float dot = lightToPoint.dot(_dir);
    if(dot > 0.0f && cross.lengthSq() < 1.f - COLLIMATED_THS)
    {
        return 1.0f;
    }

    return 0.0f;
}

float Collimated::directPdf(uint32 threadIndex, const IntersectionTemporary &data,
                                  const IntersectionInfo &info, const Vec3f &p) const
{
    return 1.f;
}

Vec3f Collimated::evalDirectionalEmission(const PositionSample &point, const DirectionSample &sample) const
{
    if(sample.d.dot(_dir) >= COLLIMATED_THS)
    {
        return Vec3f(1.0);
    }
    else
    {
        return Vec3f(0.0f);
    }
}

Vec3f Collimated::evalDirect(const IntersectionTemporary &data, const IntersectionInfo &info) const
{
    if(info.Ng.dot(_dir) >= COLLIMATED_THS)
    {
        return (*_emission)[Vec2f(0.0f)];
    }
    else
    {
        return Vec3f(0.0f);
    }
}

bool Collimated::isDirac() const {
    return is_delta;
}

bool Collimated::isDirectionDirac() const
{
    return true;
}

void Collimated::prepareForRender()
{
    _dir = _transform.fwd();
    _up  = _transform.up();
    _lat = _up.cross(_dir).normalized();

    Point::prepareForRender();
}

}