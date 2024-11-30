/**
 * In pbrt, plastic BSDF is actually implemented via the 'mixing'
 * of Microfacet specular model and the Lambertian
 * For diffuse part, we don't care about the Fresnel effect, which will be considered in RoughPlastic
 * Date:  2023.11.15
*/

#ifndef PBRTPLASTICBSDF_HPP_
#define PBRTPLASTICBSDF_HPP_

#include "Bsdf.hpp"
#include "Microfacet.hpp"

namespace Tungsten {

class Scene;

class PbrtPlasticBsdf : public Bsdf
{
    // PbrtPlasticBsdf will sample different reflection component uniformly (namely, 50/50)
    float _ior;
    Microfacet::Distribution _distribution;
    std::shared_ptr<Texture> _roughness;
    Vec3f _Ks;                      

public:
    PbrtPlasticBsdf();

    virtual void fromJson(JsonPtr value, const Scene &scene) override;
    virtual rapidjson::Value toJson(Allocator &allocator) const override;

    virtual bool sample(SurfaceScatterEvent &event) const override;
    virtual bool invert(WritablePathSampleGenerator &sampler, const SurfaceScatterEvent &event) const override;
    virtual Vec3f eval(const SurfaceScatterEvent &event) const override;
    virtual float pdf(const SurfaceScatterEvent &event) const override;

    virtual void prepareForRender() override;

    const char *distributionName() const
    {
        return _distribution.toString();
    }

    float ior() const
    {
        return _ior;
    }

    const std::shared_ptr<Texture> &roughness() const
    {
        return _roughness;
    }

    Vec3f getKs() const
    {
        return _Ks;
    }

    void setDistributionName(const std::string &distributionName)
    {
        _distribution = distributionName;
    }

    void setIor(float ior)
    {
        _ior = ior;
    }

    void setRoughness(const std::shared_ptr<Texture> &roughness)
    {
        _roughness = roughness;
    }

    void setKs(Vec3f Ks)
    {
        _Ks = Ks;
    }
};

}


#endif /* PBRTPLASTICBSDF_HPP_ */
