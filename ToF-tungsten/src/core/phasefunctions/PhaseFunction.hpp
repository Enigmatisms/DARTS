#ifndef PHASEFUNCTION_HPP_
#define PHASEFUNCTION_HPP_

#include "samplerecords/PhaseSample.hpp"

#include "sampling/WritablePathSampleGenerator.hpp"

#include "io/JsonSerializable.hpp"

namespace Tungsten {

class PathSampleGenerator;
class Scene;

struct GuidedSamplingInfo {
	/**
	 * We store vertex pointer and the arrival time of the vertex
	 * The vertex is by the fault the first vertex from the emitter that enters the medium
	 */
	GuidedSamplingInfo(): time(float(-1)) {}
	GuidedSamplingInfo(Vec3f vp, float time, float glob_min_time = -1): 
			time(time), glob_min_time(glob_min_time), vertex_p(vp) {}

	float sample_time_point(PathSampleGenerator &sampler, float& bin_pdf, float min_time, float film_interv) const {
		float remaining_time = sampler.next1D() * time_range + this->glob_min_time + 1e-6;
		int time_bin_index = (int)std::floor((remaining_time - min_time) / film_interv);
		remaining_time = float(time_bin_index) * film_interv + min_time;
		float delta_glob_min = remaining_time + film_interv - this->glob_min_time;
		// the sampled bin is the very first bin that has non-zero radiance
		// the very first bin have different bin PDF (if film->min_time is shorter than the time travelling from camera to emitter)
		if (delta_glob_min + 1e-5 < film_interv)
			bin_pdf = delta_glob_min / time_range;
		return remaining_time;
	}
	
	// Using pointer could be too dangerous, since the pointer is pointing to a vector.
	float time;           // vertex time of arrival
	float glob_min_time;  // global shortest path time
	float time_range;	  // max_time - global_min_time
	Vec3f vertex_p;       // vertex position
};

using GuideInfoConstPtr = const GuidedSamplingInfo* const;


class EllipseInfo {
public:
	/**
	 * @brief Elliptical sample is the extension for the method 
	 * ellipsoidal path connection in 2019 MitsubaToF
	 * @param full_init: sometimes, the EllipseInfo is used for EPS and does not need to calculate all terms
	 */
	EllipseInfo (
		GuideInfoConstPtr sampling_info, const Vec3f& p, float min_time, 
		float p_time, float interval, PathSampleGenerator &sampler, bool full_init = true);
	
	float half_power_diff;						// precomputed 0.5 * (T^2 - d^2)
	float max_power_diff;						// To calculate max_t
	float min_t, max_t, target_t, foci_dist;   	// max_T for this time bin, T and d
	float inv_sampling_pdf;						// PDF for sampling the current T value (reciprocal)
	float T_div_D;								// target_t / foci_dist
	Vec3f to_emitter;						    // direction to the emitter (normalized), length is foci_dist

	bool valid_target() const {
		return target_t > 0;
	}

	bool direct_connect() const {
		// being extremely close to the target time
		return fabsf(foci_dist - target_t) < 1e-5;
	}

	bool in_time_range() const {
		return false;
	}

	/// @brief  whether ell_info is useful for EPS
	bool useful() const {
		return valid_target() && !direct_connect();
	}

	bool ill_defined() const {		// in case we get some ill-defined values
		return fabsf(T_div_D - 1.f) < 1e-3;
	}
};

using EllipseConstPtr = const EllipseInfo* const;

class PhaseFunction : public JsonSerializable
{
public:
    virtual void fromJson(JsonPtr value, const Scene &scene) override;
    virtual rapidjson::Value toJson(Allocator &allocator) const override;

    virtual Vec3f eval(const Vec3f &wi, const Vec3f &wo) const = 0;
    virtual bool sample(PathSampleGenerator &sampler, const Vec3f &wi, PhaseSample &sample, EllipseConstPtr _ell = nullptr) const = 0;
    virtual bool invert(WritablePathSampleGenerator &sampler, const Vec3f &wi, const Vec3f &wo) const;
    virtual float pdf(const Vec3f &wi, const Vec3f &wo, EllipseConstPtr ell_info = nullptr) const = 0;
};

}

#endif /* PHASEFUNCTION_HPP_ */
