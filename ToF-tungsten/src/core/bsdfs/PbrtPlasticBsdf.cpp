#include "PbrtPlasticBsdf.hpp"
#include "RoughDielectricBsdf.hpp"

#include "samplerecords/SurfaceScatterEvent.hpp"

#include "textures/ConstantTexture.hpp"

#include "sampling/PathSampleGenerator.hpp"
#include "sampling/SampleWarp.hpp"

#include "math/MathUtil.hpp"
#include "math/Angle.hpp"
#include "math/Vec.hpp"

#include "io/JsonObject.hpp"
#include "io/Scene.hpp"

namespace Tungsten {

PbrtPlasticBsdf::PbrtPlasticBsdf()
: _ior(1.5f),
  _distribution("pbrt-ggx"),
  _roughness(std::make_shared<ConstantTexture>(0.1f)),
  _Ks(0.25f)
{
    _lobes = BsdfLobes(BsdfLobes::GlossyReflectionLobe | BsdfLobes::DiffuseReflectionLobe);
}

void PbrtPlasticBsdf::fromJson(JsonPtr value, const Scene &scene)
{
    Bsdf::fromJson(value, scene);
    value.getField("ior", _ior);
    _ior = 1. / _ior;           // PBRT requires inversion
    _distribution = value["distribution"];
    if (auto roughness = value["roughness"])
        _roughness = scene.fetchTexture(roughness, TexelConversion::REQUEST_AVERAGE);
     value.getField("Ks", _Ks);
}

rapidjson::Value PbrtPlasticBsdf::toJson(Allocator &allocator) const
{
    return JsonObject{Bsdf::toJson(allocator), allocator,
        "type", "rough_plastic",
        "ior", 1.f / _ior,
        "distribution", _distribution.toString(),
        "roughness", *_roughness,
        "Ks", _Ks
    };
}

bool PbrtPlasticBsdf::sample(SurfaceScatterEvent &event) const
{
    if (event.wi.z() <= 0.0f)
        return false;

    bool sampleR = event.requestedLobe.test(BsdfLobes::GlossyReflectionLobe);
    bool sampleT = event.requestedLobe.test(BsdfLobes::DiffuseReflectionLobe);

    if (!sampleR && !sampleT)
        return false;

    if (sampleR && (event.sampler->nextBoolean(0.5) || !sampleT)) {     // sample specular component (if sampleT, both specular and diffuse will be sampled)
        float roughness = (*_roughness)[*event.info].x();
        if (!RoughDielectricBsdf::sampleBase(event, true, false, roughness, _ior, _distribution, true))
            return false;
        if (sampleT) {
            Vec3f diffuseAlbedo = albedo(event.info);
            // these two are not brdf value, they are already cosine ATTENUATED!
            // TODO: specular weight seems to have no spectrum scaling
            // Also, in pbrt-v3, we don't need to scale the diffuse part by diffuseFresnel
            float pdfSubstrate = SampleWarp::cosineHemispherePdf(event.wo);
            Vec3f brdfSubstrate = diffuseAlbedo * pdfSubstrate;
            float pdfSpecular = event.pdf;
            Vec3f brdfSpecular = event.weight * event.pdf * _Ks;

            event.pdf = (pdfSpecular + pdfSubstrate) / 2.f;     //
            event.weight = (brdfSpecular + brdfSubstrate) / event.pdf;
        } else {
            event.weight *= _Ks;
        }
        return true;
    } else {
        Vec3f wo(SampleWarp::cosineHemisphere(event.sampler->next2D()));
        Vec3f diffuseAlbedo = albedo(event.info);

        event.wo = wo;
        if (sampleR) {
            float pdfSubstrate = SampleWarp::cosineHemispherePdf(event.wo);
            Vec3f brdfSubstrate = diffuseAlbedo * pdfSubstrate;
            Vec3f brdfSpecular = RoughDielectricBsdf::evalBase(event, true, false, (*_roughness)[*event.info].x(), _ior, _distribution) * _Ks;
            float pdfSpecular  = RoughDielectricBsdf::pdfBase(event, true, false, (*_roughness)[*event.info].x(), _ior, _distribution, true);
            event.pdf = (pdfSpecular + pdfSubstrate) / 2.f;
            event.weight = (brdfSpecular + brdfSubstrate) / event.pdf;
        }
        event.sampledLobe = BsdfLobes::DiffuseReflectionLobe;
    }
    return true;
}

Vec3f PbrtPlasticBsdf::eval(const SurfaceScatterEvent &event) const
{
    bool sampleR = event.requestedLobe.test(BsdfLobes::GlossyReflectionLobe);
    bool sampleT = event.requestedLobe.test(BsdfLobes::DiffuseReflectionLobe);
    if (!sampleR && !sampleT)
        return Vec3f(0.0f);
    if (event.wi.z() <= 0.0f || event.wo.z() <= 0.0f)
        return Vec3f(0.0f);

    Vec3f diffuseR(0.0f);
    if (sampleT) {
        Vec3f diffuseAlbedo = albedo(event.info);

        diffuseR = event.wo.z() * INV_PI * diffuseAlbedo;
    }
    Vec3f glossyR(0.0f);
    if (sampleR)
        glossyR = RoughDielectricBsdf::evalBase(event, true, false, (*_roughness)[*event.info].x(), _ior, _distribution) * _Ks;


    return glossyR + diffuseR;
}

bool PbrtPlasticBsdf::invert(WritablePathSampleGenerator &sampler, const SurfaceScatterEvent &event) const
{
    // This implementation might be erroneous, since I don't fully understand what `invert` does here.
    if (event.wi.z() <= 0.0f || event.wo.z() <= 0.0f)
        return false;

    bool sampleR = event.requestedLobe.test(BsdfLobes::GlossyReflectionLobe);
    bool sampleT = event.requestedLobe.test(BsdfLobes::DiffuseReflectionLobe);

    if (!sampleR && !sampleT)
        return false;

    if (sampleR)
        RoughDielectricBsdf::pdfBase(event, true, false, (*_roughness)[*event.info].x(), _ior, _distribution, true);

    if (sampleT)
        SampleWarp::cosineHemispherePdf(event.wo);

    if (sampler.untrackedBoolean(0.5)) {
        sampler.putBoolean(0.5, true);
        float roughness = (*_roughness)[*event.info].x();
        return RoughDielectricBsdf::invertBase(sampler, event, true, false, roughness, _ior, _distribution);
    } else {
        if (sampleR)
            sampler.putBoolean(0.5, false);
        sampler.put2D(SampleWarp::invertCosineHemisphere(event.wo, sampler.untracked1D()));
        return true;
    }
    return false;
}

float PbrtPlasticBsdf::pdf(const SurfaceScatterEvent &event) const
{
    bool sampleR = event.requestedLobe.test(BsdfLobes::GlossyReflectionLobe);
    bool sampleT = event.requestedLobe.test(BsdfLobes::DiffuseReflectionLobe);
    if (!sampleR && !sampleT)
        return 0.0f;
    if (event.wi.z() <= 0.0f || event.wo.z() <= 0.0f)
        return 0.0f;

    float glossyPdf = 0.0f;
    if (sampleR) {
        glossyPdf = RoughDielectricBsdf::pdfBase(event, true, false, (*_roughness)[*event.info].x(), _ior, _distribution, true);
    }

    float diffusePdf = 0.0f;
    if (sampleT) {
        diffusePdf = SampleWarp::cosineHemispherePdf(event.wo);
    }

    return (glossyPdf + diffusePdf) / float(int(sampleR) + int(sampleT));
}

void PbrtPlasticBsdf::prepareForRender() {}

}
