#ifndef HENYEYGREENSTEINPHASEFUNCTION_HPP_
#define HENYEYGREENSTEINPHASEFUNCTION_HPP_

#include "PhaseFunction.hpp"
#include "math/Angle.hpp"

namespace Tungsten {

class HenyeyGreensteinPhaseFunction : public PhaseFunction {
protected:
    float _g;
    float henyeyGreenstein(float cosTheta) const {
        float term = 1.0f + _g*_g - 2.0f*_g*cosTheta;
        return INV_FOUR_PI*(1.0f - _g*_g)/(term*std::sqrt(term));
    }
public:
    HenyeyGreensteinPhaseFunction();

    virtual void fromJson(JsonPtr value, const Scene &scene) override;
    virtual rapidjson::Value toJson(Allocator &allocator) const override;

    virtual Vec3f eval(const Vec3f &wi, const Vec3f &wo) const override;
    virtual bool sample(PathSampleGenerator &sampler, const Vec3f &wi, PhaseSample &sample, EllipseConstPtr _ell = nullptr) const override;
    virtual bool invert(WritablePathSampleGenerator &sampler, const Vec3f &wi, const Vec3f &wo) const;
    virtual float pdf(const Vec3f &wi, const Vec3f &wo, EllipseConstPtr ell_info = nullptr) const override;

    float g() const
    {
        return _g;
    }
};

}

#endif /* HENYEYGREENSTEINPHASEFUNCTION_HPP_ */
