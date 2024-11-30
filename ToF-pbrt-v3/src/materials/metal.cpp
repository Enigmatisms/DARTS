
/*
    pbrt source code is Copyright(c) 1998-2016
                        Matt Pharr, Greg Humphreys, and Wenzel Jakob.

    This file is part of pbrt.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
    IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
    TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

 */


// materials/metal.cpp*
#include "materials/metal.h"
#include "reflection.h"
#include "paramset.h"
#include "texture.h"
#include "interaction.h"

namespace pbrt {

// MetalMaterial Method Definitions
MetalMaterial::MetalMaterial(const std::shared_ptr<Texture<Spectrum>> &eta,
                             const std::shared_ptr<Texture<Spectrum>> &k,
                             const std::shared_ptr<Texture<Float>> &roughness,
                             const std::shared_ptr<Texture<Float>> &uRoughness,
                             const std::shared_ptr<Texture<Float>> &vRoughness,
                             const std::shared_ptr<Texture<Float>> &bumpMap,
                             bool remapRoughness)
    : eta(eta),
      k(k),
      roughness(roughness),
      uRoughness(uRoughness),
      vRoughness(vRoughness),
      bumpMap(bumpMap),
      remapRoughness(remapRoughness) {}

void MetalMaterial::ComputeScatteringFunctions(SurfaceInteraction *si,
                                               MemoryArena &arena,
                                               TransportMode mode,
                                               bool allowMultipleLobes) const {
    // Perform bump mapping with _bumpMap_, if present
    if (bumpMap) Bump(bumpMap, si);
    si->bsdf = ARENA_ALLOC(arena, BSDF)(*si);

    Float uRough =
        uRoughness ? uRoughness->Evaluate(*si) : roughness->Evaluate(*si);
    Float vRough =
        vRoughness ? vRoughness->Evaluate(*si) : roughness->Evaluate(*si);
    if (remapRoughness) {
        uRough = TrowbridgeReitzDistribution::RoughnessToAlpha(uRough);
        vRough = TrowbridgeReitzDistribution::RoughnessToAlpha(vRough);
    }
    Fresnel *frMf = ARENA_ALLOC(arena, FresnelConductor)(1., eta->Evaluate(*si),
                                                         k->Evaluate(*si));
    MicrofacetDistribution *distrib =
        ARENA_ALLOC(arena, TrowbridgeReitzDistribution)(uRough, vRough);
    si->bsdf->Add(ARENA_ALLOC(arena, MicrofacetReflection)(1., distrib, frMf));
}

const int CopperSamples = 56;
const Float CopperWavelengths[CopperSamples] = {
    298.7570554, 302.4004341, 306.1337728, 309.960445,  313.8839949,
    317.9081487, 322.036826,  326.2741526, 330.6244747, 335.092373,
    339.6826795, 344.4004944, 349.2512056, 354.2405086, 359.374429,
    364.6593471, 370.1020239, 375.7096303, 381.4897785, 387.4505563,
    393.6005651, 399.9489613, 406.5055016, 413.2805933, 420.2853492,
    427.5316483, 435.0322035, 442.8006357, 450.8515564, 459.2006593,
    467.8648226, 476.8622231, 486.2124627, 495.936712,  506.0578694,
    516.6007417, 527.5922468, 539.0616435, 551.0407911, 563.5644455,
    576.6705953, 590.4008476, 604.8008683, 619.92089,   635.8162974,
    652.5483053, 670.1847459, 688.8009889, 708.4810171, 729.3186941,
    751.4192606, 774.9011125, 799.8979226, 826.5611867, 855.0632966,
    885.6012714};

const Float CopperN[CopperSamples] = {
    1.400313, 1.38,  1.358438, 1.34,  1.329063, 1.325, 1.3325,   1.34,
    1.334375, 1.325, 1.317812, 1.31,  1.300313, 1.29,  1.281563, 1.27,
    1.249062, 1.225, 1.2,      1.18,  1.174375, 1.175, 1.1775,   1.18,
    1.178125, 1.175, 1.172812, 1.17,  1.165312, 1.16,  1.155312, 1.15,
    1.142812, 1.135, 1.131562, 1.12,  1.092437, 1.04,  0.950375, 0.826,
    0.645875, 0.468, 0.35125,  0.272, 0.230813, 0.214, 0.20925,  0.213,
    0.21625,  0.223, 0.2365,   0.25,  0.254188, 0.26,  0.28,     0.3};

const Float CopperK[CopperSamples] = {
    1.662125, 1.687, 1.703313, 1.72,  1.744563, 1.77,  1.791625, 1.81,
    1.822125, 1.834, 1.85175,  1.872, 1.89425,  1.916, 1.931688, 1.95,
    1.972438, 2.015, 2.121562, 2.21,  2.177188, 2.13,  2.160063, 2.21,
    2.249938, 2.289, 2.326,    2.362, 2.397625, 2.433, 2.469187, 2.504,
    2.535875, 2.564, 2.589625, 2.605, 2.595562, 2.583, 2.5765,   2.599,
    2.678062, 2.809, 3.01075,  3.24,  3.458187, 3.67,  3.863125, 4.05,
    4.239563, 4.43,  4.619563, 4.817, 5.034125, 5.26,  5.485625, 5.717};

// Added some other metals from PBRT-v4 (by Qianyue He)

const int SilverSamples = 56;
const Float SilverWavelengths[SilverSamples] = {
    298.757050, 302.400421, 306.133759, 309.960449, 313.884003, 317.908142, 322.036835,
    326.274139, 330.624481, 335.092377, 339.682678, 344.400482, 349.251221, 354.240509,
    359.374420, 364.659332, 370.102020, 375.709625, 381.489777, 387.450562, 393.600555,
    399.948975, 406.505493, 413.280579, 420.285339, 427.531647, 435.032196, 442.800629,
    450.851562, 459.200653, 467.864838, 476.862213, 486.212463, 495.936707, 506.057861,
    516.600769, 527.592224, 539.061646, 551.040771, 563.564453, 576.670593, 590.400818,
    604.800842, 619.920898, 635.816284, 652.548279, 670.184753, 688.800964, 708.481018,
    729.318665, 751.419250, 774.901123, 799.897949, 826.561157, 855.063293, 885.601257};

const Float SilverN[SilverSamples] = {
    1.519000, 1.496000, 1.432500, 1.323000, 1.142062, 0.932000, 0.719062,
    0.526000, 0.388125, 0.294000, 0.253313, 0.238000, 0.221438, 0.209000,
    0.194813, 0.186000, 0.192063, 0.200000, 0.198063, 0.192000, 0.182000,
    0.173000, 0.172625, 0.173000, 0.166688, 0.160000, 0.158500, 0.157000,
    0.151063, 0.144000, 0.137313, 0.132000, 0.130250, 0.130000, 0.129938,
    0.130000, 0.130063, 0.129000, 0.124375, 0.120000, 0.119313, 0.121000,
    0.125500, 0.131000, 0.136125, 0.140000, 0.140063, 0.140000, 0.144313,
    0.148000, 0.145875, 0.143000, 0.142563, 0.145000, 0.151938, 0.163000};

const Float SilverK[SilverSamples] = {
    1.080000, 0.882000, 0.761063, 0.647000, 0.550875, 0.504000, 0.554375,
    0.663000, 0.818563, 0.986000, 1.120687, 1.240000, 1.345250, 1.440000,
    1.533750, 1.610000, 1.641875, 1.670000, 1.735000, 1.810000, 1.878750,
    1.950000, 2.029375, 2.110000, 2.186250, 2.260000, 2.329375, 2.400000,
    2.478750, 2.560000, 2.640000, 2.720000, 2.798125, 2.880000, 2.973750,
    3.070000, 3.159375, 3.250000, 3.348125, 3.450000, 3.553750, 3.660000,
    3.766250, 3.880000, 4.010625, 4.150000, 4.293125, 4.440000, 4.586250,
    4.740000, 4.908125, 5.090000, 5.288750, 5.500000, 5.720624, 5.950000};

const int GoldSamples = 56;
const Float GoldWavelengths[GoldSamples] = {
    298.757050, 302.400421, 306.133759, 309.960449, 313.884003, 317.908142, 322.036835,
    326.274139, 330.624481, 335.092377, 339.682678, 344.400482, 349.251221, 354.240509,
    359.374420, 364.659332, 370.102020, 375.709625, 381.489777, 387.450562, 393.600555,
    399.948975, 406.505493, 413.280579, 420.285339, 427.531647, 435.032196, 442.800629,
    450.851562, 459.200653, 467.864838, 476.862213, 486.212463, 495.936707, 506.057861,
    516.600769, 527.592224, 539.061646, 551.040771, 563.564453, 576.670593, 590.400818,
    604.800842, 619.920898, 635.816284, 652.548279, 670.184753, 688.800964, 708.481018,
    729.318665, 751.419250, 774.901123, 799.897949, 826.561157, 855.063293, 885.601257};

const Float GoldN[GoldSamples] = {
    1.795000, 1.812000, 1.822625, 1.830000, 1.837125, 1.840000, 1.834250,
    1.824000, 1.812000, 1.798000, 1.782000, 1.766000, 1.752500, 1.740000,
    1.727625, 1.716000, 1.705875, 1.696000, 1.684750, 1.674000, 1.666000,
    1.658000, 1.647250, 1.636000, 1.628000, 1.616000, 1.596250, 1.562000,
    1.502125, 1.426000, 1.345875, 1.242000, 1.086750, 0.916000, 0.754500,
    0.608000, 0.491750, 0.402000, 0.345500, 0.306000, 0.267625, 0.236000,
    0.212375, 0.194000, 0.177750, 0.166000, 0.161000, 0.160000, 0.160875,
    0.164000, 0.169500, 0.176000, 0.181375, 0.188000, 0.198125, 0.210000};

const Float GoldK[GoldSamples] = {
    1.920375, 1.920000, 1.918875, 1.916000, 1.911375, 1.904000, 1.891375,
    1.878000, 1.868250, 1.860000, 1.851750, 1.846000, 1.845250, 1.848000,
    1.852375, 1.862000, 1.883000, 1.906000, 1.922500, 1.936000, 1.947750,
    1.956000, 1.959375, 1.958000, 1.951375, 1.940000, 1.924500, 1.904000,
    1.875875, 1.846000, 1.814625, 1.796000, 1.797375, 1.840000, 1.956500,
    2.120000, 2.326250, 2.540000, 2.730625, 2.880000, 2.940625, 2.970000,
    3.015000, 3.060000, 3.070000, 3.150000, 3.445812, 3.800000, 4.087687,
    4.357000, 4.610188, 4.860000, 5.125813, 5.390000, 5.631250, 5.880000};

MetalMaterial *CreateMetalMaterial(const TextureParams &mp) {
    std::string metal_type = mp.FindString("metal_type", "copper");
    static Spectrum copperN, copperK;
    if (metal_type == "copper") {
        copperN = Spectrum::FromSampled(CopperWavelengths, CopperN, CopperSamples);
        copperK = Spectrum::FromSampled(CopperWavelengths, CopperK, CopperSamples);
    } else if (metal_type == "silver") {
        copperN = Spectrum::FromSampled(SilverWavelengths, SilverN, SilverSamples);
        copperK = Spectrum::FromSampled(SilverWavelengths, SilverK, SilverSamples);
    } else if (metal_type == "gold") {
        copperN = Spectrum::FromSampled(GoldWavelengths, GoldN, GoldSamples);
        copperK = Spectrum::FromSampled(GoldWavelengths, GoldK, GoldSamples);
    } else {
        copperN = Spectrum::FromSampled(CopperWavelengths, CopperN, CopperSamples);
        copperK = Spectrum::FromSampled(CopperWavelengths, CopperK, CopperSamples);
        std::cout << "Warning: metal type '" << metal_type << "' not supproted. Fall back to 'copper'\n";
    }
    std::shared_ptr<Texture<Spectrum>> eta = mp.GetSpectrumTexture("eta", copperN);
    std::shared_ptr<Texture<Spectrum>> k = mp.GetSpectrumTexture("k", copperK);
    std::shared_ptr<Texture<Float>> roughness =
        mp.GetFloatTexture("roughness", .01f);
    std::shared_ptr<Texture<Float>> uRoughness =
        mp.GetFloatTextureOrNull("uroughness");
    std::shared_ptr<Texture<Float>> vRoughness =
        mp.GetFloatTextureOrNull("vroughness");
    std::shared_ptr<Texture<Float>> bumpMap =
        mp.GetFloatTextureOrNull("bumpmap");
    bool remapRoughness = mp.FindBool("remaproughness", true);
    return new MetalMaterial(eta, k, roughness, uRoughness, vRoughness, bumpMap,
                             remapRoughness);
}

}  // namespace pbrt
