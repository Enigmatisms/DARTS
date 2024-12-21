#ifndef PATHTRACERSETTINGS_HPP_
#define PATHTRACERSETTINGS_HPP_

#include "integrators/TraceSettings.hpp"

#include "io/JsonObject.hpp"

namespace Tungsten {

struct PathTracerSettings : public TraceSettings
{
    bool enableLightSampling;
    bool enableVolumeLightSampling;
    bool lowOrderScattering;
    bool includeSurfaces;
    // DARTS related configurations
    bool enable_guiding;        // enable DA-based guiding
    bool enable_elliptical;     // enable elliptical sampling (might only works for point method)
    bool enable_rr;             // enable russian roullete

    float transientTimeCenter;
    float transientTimeWidth;
    float transientTimeBeg;
    float transientTimeEnd;
    float invTransientTimeWidth;
    float vaccumSpeedOfLight;

    int frame_num;

    PathTracerSettings()
    : enableLightSampling(true),
      enableVolumeLightSampling(true),
      lowOrderScattering(true),
      includeSurfaces(true),
      enable_guiding(false),
      enable_elliptical(false),
      enable_rr(false),
      frame_num(0)
    {
    }

    void fromJson(JsonPtr value)
    {
        TraceSettings::fromJson(value);
        value.getField("enable_light_sampling", enableLightSampling);
        value.getField("enable_volume_light_sampling", enableVolumeLightSampling);
        value.getField("low_order_scattering", lowOrderScattering);
        value.getField("include_surfaces", includeSurfaces);

        if (!value.getField("transient_time_center", transientTimeCenter))
            transientTimeCenter = -1;
        if (!value.getField("transient_time_width", transientTimeWidth))
            transientTimeCenter = 0;
        if (!value.getField("vaccum_speed_of_light", vaccumSpeedOfLight))
            vaccumSpeedOfLight = 1.0f;

        value.getField("frame_num", frame_num);
        if (frame_num <= 0) {
            if (transientTimeCenter >= 0 && transientTimeWidth > 0) {
                transientTimeBeg      = transientTimeCenter - 0.5f * transientTimeWidth;
                transientTimeEnd      = transientTimeCenter + 0.5f * transientTimeWidth;
                invTransientTimeWidth = 1.f / transientTimeWidth;
                printf("Time-gated output enabled for path tracing: [%f, %f)\n", transientTimeBeg, transientTimeEnd);
            } else {
                printf("Time-gated output disabled for path tracing since time center or time width is set to non-positive.\n");
            }
        } else {
            float start_time = 0.0;
            if (value.getField("start_time", start_time))
                transientTimeBeg = start_time;
            else
                printf("Warning: transient <start_time> not specified, using transientTimeBeg: %.4f\n", transientTimeBeg);
            transientTimeEnd = transientTimeBeg + float(frame_num) * transientTimeWidth;
            invTransientTimeWidth = 1.f / transientTimeWidth;
            printf("Transient rendering is enabled, start t = %.4f, end t = %.4f, frame count: %d, width: %.4f\n", 
                        transientTimeBeg, transientTimeEnd, frame_num, transientTimeWidth);
        }

        value.getField("enable_guiding", enable_guiding);
        value.getField("enable_elliptical", enable_elliptical);
        if (enable_guiding || enable_elliptical)
            enable_rr = false;
        else
            value.getField("enable_rr", enable_rr);
        printf("DARTS support: DA-sampling: (%d) | elliptical sampling (%d) | enable rr (%d)\n", int(enable_guiding), int(enable_elliptical), int(enable_rr));
    }

    rapidjson::Value toJson(rapidjson::Document::AllocatorType &allocator) const
    {
        return JsonObject{TraceSettings::toJson(allocator), allocator,
            "type", "path_tracer",
            "enable_light_sampling", enableLightSampling,
            "enable_volume_light_sampling", enableVolumeLightSampling,
            "low_order_scattering", lowOrderScattering,
            "include_surfaces", includeSurfaces,
            "transient_time_center", transientTimeCenter,
            "transient_time_width", transientTimeWidth,
            "vaccum_speed_of_light", vaccumSpeedOfLight,
            "enable_guiding", enable_guiding,
            "enable_elliptical", enable_elliptical,
            "enable_rr", enable_rr
        };
    }
};

}

#endif /* PATHTRACERSETTINGS_HPP_ */
