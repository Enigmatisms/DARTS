#include "PhotonMapSettings.hpp"

namespace Tungsten {

DEFINE_STRINGABLE_ENUM(PhotonMapSettings::VolumePhotonType, "volume_photon_type", ({
    {"points", VOLUME_POINTS},
    {"beams", VOLUME_BEAMS},
    {"planes", VOLUME_PLANES},
    {"planes_1d", VOLUME_PLANES_1D},
    {"volumes", VOLUME_VOLUMES},
    {"hypervolumes", VOLUME_HYPERVOLUMES},
    {"balls", VOLUME_BALLS},
    {"volumes_balls", VOLUME_VOLUMES_BALLS},
    {"mis_volumes_balls", VOLUME_MIS_VOLUMES_BALLS},
}))

}
