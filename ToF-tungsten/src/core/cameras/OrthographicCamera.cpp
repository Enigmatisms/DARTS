#include "OrthographicCamera.hpp"

#include "sampling/PathSampleGenerator.hpp"

#include "math/Angle.hpp"
#include "math/Ray.hpp"

#include "io/JsonObject.hpp"

#include <cmath>

namespace Tungsten
{

    OrthographicCamera::OrthographicCamera()
        : Camera(),
          _hOrthoSize(0.0f), _vOrthoSize(1.0f)
    {
        precompute();
    }

    OrthographicCamera::OrthographicCamera(const Mat4f &transform, const Vec2u &res, float vOrthoSize)
        : Camera(transform, res),
          _hOrthoSize(0.0f), _vOrthoSize(vOrthoSize)
    {
        precompute();
    }

    void OrthographicCamera::precompute()
    {
        _hOrthoSize = _vOrthoSize * (1.0f / _ratio); // ratio is y divided by x
        float planeArea = (2.0f * _hOrthoSize) * (2.0f * _vOrthoSize);
        _invPlaneArea = 1.0f / planeArea;
    }

    void OrthographicCamera::fromJson(JsonPtr value, const Scene &scene)
    {
        Camera::fromJson(value, scene);
        value.getField("orthographic_size", _vOrthoSize);

        precompute();
    }

    rapidjson::Value OrthographicCamera::toJson(Allocator &allocator) const
    {
        return JsonObject{Camera::toJson(allocator), allocator,
                          "type", "orthographic",
                          "orthographic_size", _vOrthoSize};
    }

    bool OrthographicCamera::samplePosition(PathSampleGenerator & sampler, PositionSample &sample) const
    {
        Vec2f ndc = (sampler.next2D() - Vec2f(0.5f, 0.5f)) * 2.0f;
        Vec2f eyePosXy = ndc * Vec2f(_hOrthoSize, _vOrthoSize);
        Vec3f eyePos = Vec3f(eyePosXy.x(), eyePosXy.y(), 0.0f);

        sample.p = _transform * eyePos;
        sample.weight = Vec3f(1.0f);
        sample.pdf = _invPlaneArea;
        sample.Ng = _transform.fwd();

        return true;
    }

    bool OrthographicCamera::samplePosition(PathSampleGenerator &sampler, PositionSample &sample,
        const Vec2u &pixel) const
    {
        float pdf;
        // uv is in [-0.5, 0.5) after using the reconstruction filter
        Vec2f uv = _filter.sample(sampler.next2D(), pdf);
        float pixelWidth = _vOrthoSize * 2.0f * _pixelSize.y();
        Vec3f localP = Vec3f(
            -_hOrthoSize + (float(pixel.x()) + 0.5f + uv.x()) * pixelWidth,
            _vOrthoSize - (float(pixel.y()) + 0.5f + uv.y()) * pixelWidth,  // pixel.y() starts from top to bottom
            0.0f
        );

        sample.p = _transform * localP;
        sample.weight = Vec3f(1.0f);
        sample.pdf = pixelWidth * pixelWidth;
        sample.Ng = _transform.fwd();

        return true;
    }

    bool OrthographicCamera::sampleDirectionAndPixel(PathSampleGenerator &sampler, const PositionSample &point,
                                                Vec2u &pixel, DirectionSample &sample) const
    {
        pixel = Vec2u(sampler.next2D() * Vec2f(_res));
        return sampleDirection(sampler, point, pixel, sample);
    }

    bool OrthographicCamera::sampleDirection(PathSampleGenerator &sampler, const PositionSample & /*point*/, Vec2u pixel,
                                        DirectionSample &sample) const
    {
        sample.d = _transform.fwd();
        sample.weight = Vec3f(1.0f);
        sample.pdf = 1.0f;

        return true;
    }

    bool OrthographicCamera::invertDirection(WritablePathSampleGenerator &sampler, const PositionSample & /*point*/,
                                        const DirectionSample &direction) const
    {
        return false;
    }

    bool OrthographicCamera::sampleDirect(const Vec3f &p, PathSampleGenerator &sampler, LensSample &sample) const
    {
        sample.d = _pos - p;

        if (!evalDirection(sampler, PositionSample(), DirectionSample(-sample.d), sample.weight, sample.pixel))
            return false;

        float rSq = sample.d.lengthSq();
        sample.dist = std::sqrt(rSq);
        sample.d /= sample.dist;
        sample.weight /= rSq;
        return true;
    }

    // bool OrthographicCamera::invertPosition(WritablePathSampleGenerator & /*sampler*/, const PositionSample & /*point*/) const
    // {
    //     return true;
    // }

    bool OrthographicCamera::evalDirection(PathSampleGenerator & /*sampler*/, const PositionSample & /*point*/,
                                      const DirectionSample &direction, Vec3f &weight, Vec2f &pixel) const
    {
        return false;
    }

    float OrthographicCamera::directionPdf(const PositionSample & /*point*/, const DirectionSample &direction) const
    {
        return 1.0f;
    }

    bool OrthographicCamera::isDirac() const
    {
        return false;
    }

    float OrthographicCamera::approximateFov() const
    {
        // NOTE: Don't know what fov to return of an orthographic camera.
        //       Returning 90 degrees for now.
        return Angle::degToRad(90.0f);
    }
}
