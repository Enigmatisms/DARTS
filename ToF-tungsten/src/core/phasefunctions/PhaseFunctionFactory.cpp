#include "PhaseFunctionFactory.hpp"

#include "HenyeyGreensteinPhaseFunction.hpp"
#include "IsotropicPhaseFunction.hpp"
#include "RayleighPhaseFunction.hpp"
#include "IntegratedDAPhaseFunction.hpp"

namespace Tungsten {

DEFINE_STRINGABLE_ENUM(PhaseFunctionFactory, "phase function", ({
    {"isotropic", std::make_shared<IsotropicPhaseFunction>},
    {"henyey_greenstein", std::make_shared<HenyeyGreensteinPhaseFunction>},
    {"rayleigh", std::make_shared<RayleighPhaseFunction>},
    {"ida", std::make_shared<IntegratedDAPhaseFunction>},
}))

}
