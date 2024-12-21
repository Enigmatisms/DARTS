#pragma once
#include "HenyeyGreensteinPhaseFunction.hpp"

constexpr int ANGULAR_RES = 256;

namespace Tungsten {

class IntegratedDAPhaseFunction : public HenyeyGreensteinPhaseFunction {
private:
    float alpha;		// One sample model control factor
	float dT_s;		    // d/T start value
	float inv_dT_div; // resolution / (end - start) value
	float inv_max_T;  // reciprocal to maximum T tabulation range (multiplied by 128)

    // A naked ptr pointed to tabulation. Should be safe
	// since I guarantee safe-access and there is no need for memory management

    float (*table)[128][ANGULAR_RES];

    std::string table_path;
public:
    IntegratedDAPhaseFunction();
    ~IntegratedDAPhaseFunction();

    void fromJson(JsonPtr value, const Scene &scene) override;
    rapidjson::Value toJson(Allocator &allocator) const override;
    bool invert(WritablePathSampleGenerator &, const Vec3f &, const Vec3f &) const {
        throw std::runtime_error("Integrated DA phase function does not support sample inversion.\n");
    }

    bool sample(PathSampleGenerator &sampler, const Vec3f &wi, PhaseSample &sample, EllipseConstPtr ell_info = nullptr) const override;

    float pdf(const Vec3f &wi, const Vec3f &wo, EllipseConstPtr ell_info = nullptr) const override;
};

}
