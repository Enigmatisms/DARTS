#include "GuidedHomoMedium.hpp"
#include "sampling/PathSampleGenerator.hpp"

#include "media/low_discrep.h"
#include "math/TangentFrame.hpp"
#include "math/Ray.hpp"
#include "sse/vectormath_exp.h" 
#include "sse/vectormath_trig.h" 

#include "io/JsonObject.hpp"
#include <numeric>

namespace Tungsten {

GuidedHomoMedium::GuidedHomoMedium()
: HomogeneousMedium() {}

void GuidedHomoMedium::prepareForRender()
{
    _sigmaA = _materialSigmaA*_density;
    _sigmaS = _materialSigmaS*_density;
    _sigmaT = _sigmaA + _sigmaS;
    _absorptionOnly = _sigmaS == 0.0f;
    if (_absorptionOnly) {
        std::cerr << "Error: DA-guided homogeneous medium can not be absorption only." << std::endl;
        throw 0;
    }
    diffusion_d = 1.f / 3.f / _sigmaT;
    inv_sigma_t = 1.f / _sigmaT;
    inv_diffusion = 1.f / diffusion_d;
}

VecType GuidedHomoMedium::full_diffusion_sse(const VecType& length2, const VecType& res_ts, float inv_d, float sigma_a) const {
	// SSE acceleration
	VecType denom_inv = 0.25f * inv_d / res_ts;
	VecType coeff = denom_inv * sqrt(denom_inv);
	return coeff * exp(-length2 * denom_inv - sigma_a * res_ts);
}

float GuidedHomoMedium::importance_resample_sse(
    const Ray& ray, PathSampleGenerator &sampler,
    float& ris_pdf, float diff_len2, float dot_val, float target_t, int channel
) const {
    // in this function, only medium events are accounted for, so we need not to worry about PDF or anything
	const float proj_len = dot_val, maxT = ray.farT();
	const float sqr_dist = diff_len2 - proj_len * proj_len;
	float proposals[cascade_num], da_vals[cascade_num], weights[cascade_num];

	float sig_t = _sigmaT[channel], tr_alpha = 1.f - expf(-sig_t * maxT),
		inv_sig_t = inv_sigma_t[channel], prop_coeff = sig_t / tr_alpha;

    float inv_d = inv_diffusion[channel], sig_a = _sigmaA[channel];
    auto rands = sampler.next2D();
    int arr_idx = std::min((int)(rands[0] * 32.f), 31);
    
    VecType samples;
    samples.load(low_discrep[arr_idx]);
    samples += rands[1];
    samples = select(samples >= 1, samples - 1, samples);			// float mod

    const VecType dists = -log(1 - samples * tr_alpha) * inv_sig_t;
    const VecType trans = exp(-sig_t * dists), prop_pdf = prop_coeff * trans;
    const VecType dlens = proj_len - dists, length2 = sqr_dist + dlens * dlens;
    const VecType res_ts = target_t - dists;
    const auto valid_flags = (res_ts > 0) && ((res_ts * res_ts) > length2);
    const VecType das   = select(valid_flags, full_diffusion_sse(length2, res_ts, inv_d, sig_a) * trans, VecType(1));
    const VecType ws    = select(valid_flags, das / prop_pdf, VecType(0));
    ws.store(weights);
    das.store(da_vals);
    dists.store(proposals);
	
	// if all samples have zero proba (almost impossible), we will use the first sample
	std::partial_sum(weights, weights + cascade_num, weights);
	float weight_sum = weights[cascade_num - 1];
	int selected_id = std::lower_bound(weights, weights + cascade_num, sampler.next1D() * weight_sum) - weights;
    ris_pdf = da_vals[selected_id] / (inv_cascasde * weight_sum);
	return proposals[selected_id];          // returns sampled distance (truncated to ray.nearT() and ray.farT())
}

bool GuidedHomoMedium::sampleDistance(
    PathSampleGenerator &sampler, const Ray &ray,
    MediumState &state, MediumSample &sample, float remaining_time,
    GuideInfoConstPtr guide_info, bool guiding
) const {
    // note that, GuidedHomoMedium currently only accept expoential medium as its transmittance model
    sample.emission = Vec3f(0.0f);
    if (state.bounce > _maxBounce)
        return false;

    if (!guiding || guide_info ==  nullptr)
        return HomogeneousMedium::sampleDistance(sampler, ray, state, sample);
    
    int component = sampler.nextDiscrete(3);
    float sigmaTc = _sigmaT[component], maxT = ray.farT(), 
            t = -std::log(1 - sampler.next1D()) / sigmaTc;
    sample.exited = (t >= maxT);
    if (sample.exited) {			// surface interaction
        sample.t = min(t, maxT);
        sample.continuedT = t;
        Vec3f tau = sample.t * _sigmaT;
        Vec3f continuedTau = sample.continuedT * _sigmaT;
        sample.weight = _transmittance->eval(tau, state.firstScatter, true);
        sample.continuedWeight = _transmittance->eval(continuedTau, state.firstScatter, true);
        sample.pdf = _transmittance->surfaceProbability(tau, state.firstScatter).avg();

        sample.weight /= sample.pdf;
        sample.continuedWeight = _sigmaS * sample.continuedWeight / (_sigmaT * _transmittance->mediumPdf(continuedTau, state.firstScatter)).avg();
    } else {
        Vec3f diff_vec = guide_info->vertex_p - ray.pos(); 
		float dot_val  = diff_vec.dot(ray.dir()), ris_pdf = 1.f;
		// Here during proposal phase we can use equiangular sampling or MIS equiangular sampling (actually, equiangular sampling is useless)
		float t = importance_resample_sse(ray, sampler, ris_pdf, diff_vec.lengthSq(), dot_val, remaining_time, component);
        sample.t = t;
        sample.continuedT = t;
        Vec3f Tau = sample.t * _sigmaT;
        sample.weight = _sigmaS * _transmittance->eval(Tau, state.firstScatter, false);
        sample.pdf = ris_pdf * (1.f - expf(-sigmaTc * maxT));
        sample.weight /= sample.pdf;
        sample.continuedWeight = sample.weight;
    }
    state.advance();

    sample.p = ray.pos() + sample.t * ray.dir();
    sample.phase = _phaseFunction.get();

    return true;
}

float GuidedHomoMedium::elliptical_sample(
	const Ray& ray, PathSampleGenerator &sampler, MediumSample& deter_mi, EllipseConstPtr ell_info
) const {
	// this might be improved
	// Since elliptical sample will utilize UniformSampleOneLight
	// So we don't need to calculate phase function and the transmittance for the second segment
	const float dot_val = ray.dir().dot(ell_info->to_emitter),
		  	    dot_foci = dot_val * ell_info->foci_dist;
	float target_time = ell_info->target_t, maxT = ray.farT();
	float t_m_dotf = target_time - dot_foci;
	float sample_dist = ell_info->half_power_diff / t_m_dotf;

    deter_mi.emission = Vec3f(0.0f);
	if (sample_dist >= maxT) {		// the vertex on the equi-time ellipse is not in the scattering medium
		// unable to sample target time in the medium, therefore we can opt for surface interaction
		return 0;
	}
	float max_dist = ell_info->max_power_diff / (ell_info->max_t - dot_foci);	// max time ellipse

    deter_mi.phase = _phaseFunction.get();
    deter_mi.p = ray.pos() + sample_dist * ray.dir();
    deter_mi.t = sample_dist;
    deter_mi.continuedT = sample_dist;
    Vec3f Tau = deter_mi.t * _sigmaT;
    // we can input any startOnSurface / endOnSurface to eval for exponential transmittance
    deter_mi.weight = _transmittance->eval(_sigmaT * sample_dist, true, true) * _sigmaS;
	// this is measure conversion Jacobian
    deter_mi.pdf = t_m_dotf / ((target_time - sample_dist) * ell_info->inv_sampling_pdf);
    deter_mi.weight /= deter_mi.pdf;
	return (max_dist >= maxT) ? 0 : max_dist;
}

float GuidedHomoMedium::pdf(PathSampleGenerator &/*sampler*/, const Ray &ray, bool startOnSurface, bool endOnSurface) const
{
    // Since in this medium, we sample distance via RIS and RIS does not produce PDF with meaningful measure
    std::cerr << "Error: DA-guided homogeneous medium can not evaluate distance sampling PDF." << std::endl;
    return 0;
}

}
