#ifndef GUIDEDHOMOMEDIUM_HPP_
#define GUIDEDHOMOMEDIUM_HPP_

#include "sse/vectorclass.h"

#define RIS_V8
#ifdef RIS_V8
using VecType = Vec8f;
# define CASCADE_NUM 8
#else 
using VecType = Vec4f;
# define CASCADE_NUM 4
#endif
#include "HomogeneousMedium.hpp"

namespace Tungsten {

class GuidedHomoMedium : public HomogeneousMedium
{
protected:
    using HomogeneousMedium::_materialSigmaA;
    using HomogeneousMedium::_materialSigmaS;
    using HomogeneousMedium::_density;
    using HomogeneousMedium::_sigmaA;
    using HomogeneousMedium::_sigmaS;
    using HomogeneousMedium::_sigmaT;
    using HomogeneousMedium::_absorptionOnly;
    using HomogeneousMedium::_speedOfLight;
    using HomogeneousMedium::_invSpeedOfLight;

    Vec3f diffusion_d;
	Vec3f inv_sigma_t;
	Vec3f inv_diffusion;

	static constexpr int cascade_num = CASCADE_NUM;
	static constexpr float inv_cascasde = 1.f / CASCADE_NUM;
public:
    GuidedHomoMedium();
    void prepareForRender() override;

    bool sampleDistance(
        PathSampleGenerator &sampler, const Ray &ray, MediumState &state, 
		MediumSample &sample, float remaining_time = 0.0,
		GuideInfoConstPtr guide_info = nullptr, bool guiding = false
    ) const override;

    float pdf(PathSampleGenerator &sampler, const Ray &ray, bool startOnSurface, bool endOnSurface) const override;
    
    float elliptical_sample(
        const Ray& ray, PathSampleGenerator &sampler, MediumSample& deter_mi, EllipseConstPtr ell_info
    ) const override;

    /**
     * Account for medium sampling via DA-based RIS
     * note that ris PDF is not a PDF value with valid measure
    */
    float importance_resample_sse(
		const Ray& ray, PathSampleGenerator &sampler,
		float& ris_pdf, float diff_len2, float dot_val, float target_t, int channel
	) const;

    VecType full_diffusion_sse(const VecType& length2, const VecType& res_ts, float inv_d, float sigma_a) const;
};

}

#endif /* GUIDEDHOMOMEDIUM_HPP_ */
