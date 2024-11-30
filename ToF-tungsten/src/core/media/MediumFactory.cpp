#include "MediumFactory.hpp"

#include "HomogeneousMedium.hpp"
#include "AtmosphericMedium.hpp"
#include "ExponentialMedium.hpp"
#include "GuidedHomoMedium.hpp"
#include "VoxelMedium.hpp"

namespace Tungsten {

DEFINE_STRINGABLE_ENUM(MediumFactory, "medium", ({
    {"homogeneous", std::make_shared<HomogeneousMedium>},
    {"guided_homo", std::make_shared<GuidedHomoMedium>},
    {"atmosphere", std::make_shared<AtmosphericMedium>},
    {"exponential", std::make_shared<ExponentialMedium>},
    {"voxel", std::make_shared<VoxelMedium>},
}))

}
