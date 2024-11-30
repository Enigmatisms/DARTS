#ifndef PATHTRACER_HPP_
#define PATHTRACER_HPP_

#include "PathTracerSettings.hpp"

#include "integrators/TraceBase.hpp"

namespace Tungsten {

class PathTracer : public TraceBase
{
    PathTracerSettings _settings;
    bool _trackOutputValues;

public:
    PathTracer(TraceableScene *scene, const PathTracerSettings &settings, uint32 threadId);

    Vec3f traceSample(Vec2u pixel, PathSampleGenerator &sampler, GuideInfoConstPtr sampling_info = nullptr, 
            Vec3f* const transients = nullptr, float bin_pdf = 0.0, float min_remaining_t = 0.0);
};

}

#endif /* PATHTRACER_HPP_ */
