#ifndef PHOTONMAPSETTINGS_HPP_
#define PHOTONMAPSETTINGS_HPP_

#include "Photon.hpp"

#include "integrators/TraceSettings.hpp"

#include "io/JsonObject.hpp"

#include "StringableEnum.hpp"

#include <tinyformat/tinyformat.hpp>

namespace Tungsten {

struct PhotonMapSettings : public TraceSettings
{
    typedef StringableEnum<VolumePhotonEnum> VolumePhotonType;
    friend VolumePhotonType;

    uint32 photonCount;
    uint32 volumePhotonCount;
    uint32 gatherCount;
    float gatherRadius;
    float volumeGatherRadius;
    VolumePhotonType volumePhotonType;
    bool includeSurfaces;
    bool lowOrderScattering;
    bool fixedVolumeRadius;
    bool useGrid;
    bool useFrustumGrid;
    int gridMemBudgetKb;
    int frame_num;
    bool deltaTimeGate;
    float transientTimeCenter;
    float transientTimeWidth;
    float transientTimeBeg;
    float transientTimeEnd;
    float invTransientTimeWidth;
    float vaccumSpeedOfLight;
    bool excludeNonMIS;         // exclude light transport we cannot MIS with
    bool enable_guiding;        // enable DA-based guiding
    bool enable_elliptical;     // enable elliptical sampling (might only works for point method)
    bool frustum_culling;       // Camera frustum culling
    bool strict_time_width;     // enable time selection (in scenes with negligible specular effect)
    bool reuse_path_dir;        // Whether to reuse path direction for elliptical sampling direction

    PhotonMapSettings()
    : photonCount(1000000),
      volumePhotonCount(1000000),
      gatherCount(20),
      gatherRadius(1e30f),
      volumeGatherRadius(gatherRadius),
      volumePhotonType("points"),
      includeSurfaces(true),
      lowOrderScattering(true),
      fixedVolumeRadius(false),
      useGrid(false),
      useFrustumGrid(false),
      gridMemBudgetKb(32*1024),
      frame_num(0),
      deltaTimeGate(false),
      transientTimeCenter(0.0f),
      transientTimeWidth(0.1f),
      transientTimeBeg(0.0f),
      transientTimeEnd(0.0f),
      invTransientTimeWidth(1.0f/ transientTimeWidth),
      vaccumSpeedOfLight(1.f),
      excludeNonMIS(false),
      enable_guiding(false),
      enable_elliptical(false),
      frustum_culling(false),
      strict_time_width(false),
      reuse_path_dir(true)
    {
    }

    void setTimeCenter(float timeCenter)
    {
        transientTimeCenter = timeCenter;
        transientTimeBeg = deltaTimeGate ? transientTimeCenter : transientTimeCenter - 0.5f * transientTimeWidth;
        transientTimeEnd = deltaTimeGate ? transientTimeCenter : transientTimeCenter + 0.5f * transientTimeWidth;
    }

    void fromJson(JsonPtr value)
    {
        TraceSettings::fromJson(value);
        value.getField("photon_count", photonCount);
        value.getField("volume_photon_count", volumePhotonCount);
        value.getField("gather_photon_count", gatherCount);
        if (auto type = value["volume_photon_type"])
            volumePhotonType = type;
        bool gatherRadiusSet = value.getField("gather_radius", gatherRadius);
        if (!value.getField("volume_gather_radius", volumeGatherRadius) && gatherRadiusSet)
            volumeGatherRadius = gatherRadius;
        value.getField("low_order_scattering", lowOrderScattering);
        value.getField("include_surfaces", includeSurfaces);
        value.getField("fixed_volume_radius", fixedVolumeRadius);
        value.getField("use_grid", useGrid);
        value.getField("use_frustum_grid", useFrustumGrid);
        value.getField("grid_memory", gridMemBudgetKb);

        if (useFrustumGrid && volumePhotonType == VOLUME_POINTS)
            value.parseError("Photon points cannot be used with a frustum aligned grid");

        value.getField("transient_time_center", transientTimeCenter);

        value.getField("transient_time_width", transientTimeWidth);
        value.getField("delta_time_gate", deltaTimeGate);
        invTransientTimeWidth = deltaTimeGate ? 1.f : 1.0f / transientTimeWidth;
        value.getField("frame_num", frame_num);
        if (frame_num <= 0) {
            setTimeCenter(transientTimeCenter);
        } else {
            float start_time = 0.0;
            if (value.getField("start_time", start_time))
                transientTimeBeg = start_time;
            transientTimeEnd = transientTimeBeg + float(frame_num) * transientTimeWidth;
            printf("Transient rendering is enabled, start t = %.4f, end t = %.4f, frame count: %d, width: %.4f\n", 
                        transientTimeBeg, transientTimeEnd, frame_num, transientTimeWidth);
        }

        value.getField("vaccum_speed_of_light", vaccumSpeedOfLight);
        value.getField("exclude_non_mis", excludeNonMIS);
        value.getField("enable_guiding", enable_guiding);
        value.getField("enable_elliptical", enable_elliptical);
        value.getField("strict_time_width", strict_time_width);
        value.getField("frustum_culling", frustum_culling);
        value.getField("reuse_path_dir", reuse_path_dir);
        printf("DARTS support: DA-sampling: (%d) | elliptical sampling (%d) | strict mode (%d) | frustum culling (%d) | dir reuse (%d)\n", 
                                int(enable_guiding), int(enable_elliptical), int(strict_time_width), int(frustum_culling), int(reuse_path_dir));

        static const VolumePhotonType deltaTimeGateSupportedTypes [] = {
            VOLUME_BEAMS,
            VOLUME_VOLUMES,
            VOLUME_HYPERVOLUMES,
            VOLUME_BALLS,
            VOLUME_VOLUMES_BALLS,
            VOLUME_MIS_VOLUMES_BALLS
        };

        if (deltaTimeGate)
        {
            auto iter = std::find(std::begin(deltaTimeGateSupportedTypes), std::end(deltaTimeGateSupportedTypes), volumePhotonType);
            if (iter == std::end(deltaTimeGateSupportedTypes))
            {
                std::string supportedTypesString {};
                for (auto iterSupported = std::begin(deltaTimeGateSupportedTypes); iterSupported != std::end(deltaTimeGateSupportedTypes); ++iterSupported)
                {
                    supportedTypesString += (*iterSupported).toString();
                    supportedTypesString += " ";
                }
                std::string errorString = tfm::format("Delta time gate mode is not supported by volume photon type \"%s\". ", volumePhotonType.toString());
                errorString += tfm::format("Supported types are: %s", supportedTypesString);
                value.parseError(errorString);
            }
        }
    }

    rapidjson::Value toJson(rapidjson::Document::AllocatorType &allocator) const
    {
        return JsonObject{TraceSettings::toJson(allocator), allocator,
            "type", "photon_map",
            "photon_count", photonCount,
            "volume_photon_count", volumePhotonCount,
            "gather_photon_count", gatherCount,
            "gather_radius", gatherRadius,
            "volume_gather_radius", volumeGatherRadius,
            "volume_photon_type", volumePhotonType.toString(),
            "low_order_scattering", lowOrderScattering,
            "include_surfaces", includeSurfaces,
            "fixed_volume_radius", fixedVolumeRadius,
            "use_grid", useGrid,
            "use_frustum_grid", useFrustumGrid,
            "grid_memory", gridMemBudgetKb,
            "transient_time_center", transientTimeCenter,
            "transient_time_width", transientTimeWidth,
            "vaccum_speed_of_light", vaccumSpeedOfLight,
            "enable_guiding", enable_guiding,
            "enable_elliptical", enable_elliptical,
            "strict_time_width", strict_time_width,
            "frustum_culling", frustum_culling,
            "reuse_path_dir", reuse_path_dir,
        };
    }
};

}

#endif /* PHOTONMAPSETTINGS_HPP_ */
