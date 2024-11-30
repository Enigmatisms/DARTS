#ifndef HOMOGENEOUSMEDIUM_HPP_
#define HOMOGENEOUSMEDIUM_HPP_

#include "Medium.hpp"

namespace Tungsten {

class HomogeneousMedium : public Medium
{
protected:
    Vec3f _materialSigmaA, _materialSigmaS;
    float _density;

    Vec3f _sigmaA, _sigmaS;
    Vec3f _sigmaT;
    bool _absorptionOnly;

    // For transient rendering:
    float _speedOfLight;
    float _invSpeedOfLight;

public:
    HomogeneousMedium();

    virtual void fromJson(JsonPtr value, const Scene &scene) override;
    virtual rapidjson::Value toJson(Allocator &allocator) const override;

    virtual bool isHomogeneous() const override;

    virtual void prepareForRender() override;

    virtual Vec3f sigmaA(Vec3f p) const override;
    virtual Vec3f sigmaS(Vec3f p) const override;
    virtual Vec3f sigmaT(Vec3f p) const override;

    virtual bool sampleDistance(
        PathSampleGenerator &sampler, const Ray &ray, MediumState &state, 
		MediumSample &sample, float remaining_time = 0.0,
		GuideInfoConstPtr guide_info = nullptr, bool guiding = false
    ) const override;
    virtual Vec3f transmittance(PathSampleGenerator &sampler, const Ray &ray, bool startOnSurface, bool endOnSurface) const override;
    virtual float pdf(PathSampleGenerator &sampler, const Ray &ray, bool startOnSurface, bool endOnSurface) const override;
    virtual float timeTraveled(float distance) const override;
    virtual float timeTraveled(const Vec3f& pStart, const Vec3f& pEnd) const override;
    virtual Vec3f travel(const Vec3f& o, const Vec3f &d, float time) const override;
    virtual float speedOfLight(const Vec3f& p) const override;

    Vec3f sigmaA() const { return _sigmaA; }
    Vec3f sigmaS() const { return _sigmaS; }
};

}

#endif /* HOMOGENEOUSMEDIUM_HPP_ */
