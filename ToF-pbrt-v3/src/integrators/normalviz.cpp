
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

// integrators/path.cpp*
#include "integrators/normalviz.h"
#include "bssrdf.h"
#include "camera.h"
#include "film.h"
#include "interaction.h"
#include "paramset.h"
#include "scene.h"
#include "stats.h"

namespace pbrt {

STAT_PERCENT("Integrator/Zero-radiance paths", zeroRadiancePaths, totalPaths);
STAT_INT_DISTRIBUTION("Integrator/Path length", pathLength);

// PathIntegrator Method Definitions
NormalIntegrator::NormalIntegrator(std::shared_ptr<const Camera> camera,
                               std::shared_ptr<Sampler> sampler,
                               const Bounds2i &pixelBounds)
    : SamplerIntegrator(camera, sampler, pixelBounds) {
        Point3f camera_pos = camera->CameraToWorld(0, Point3f(0, 0, 0));
        Vector3f camera_dir = camera->CameraToWorld(0, Vector3f(0, 0, 1));
        auto lookat_pt = camera_pos + camera_dir;
        printf("Camera pos: [%f %f %f], camera dir: %f, %f, %f\n", camera_pos[0], camera_pos[1], camera_pos[2], camera_dir[0], camera_dir[1], camera_dir[2]);
        printf("Look at: [%f %f %f]\n", lookat_pt[0], lookat_pt[1], lookat_pt[2]);
    }

Spectrum NormalIntegrator::Li(const RayDifferential &r, const Scene &scene,
                            Sampler &sampler, MemoryArena &arena,
                            int depth, Float origin_remain_t, Float, FilmInfoConstPtr film_info, FilmTilePixel* const tfilm, bool) const {
    ProfilePhase p(Prof::SamplerIntegratorLi);
    Spectrum L(0.f);
    RayDifferential ray(r);
    int bounces;

    // Find next path vertex and accumulate contribution
    for (bounces = 0 ;; bounces ++) {
        // Intersect _ray_ with scene and store intersection in _isect_
        SurfaceInteraction isect;
        bool foundIntersection = scene.Intersect(ray, &isect);

        // Terminate path if ray escaped or _maxDepth_ was reached
        if (!foundIntersection) break;

        // Compute scattering functions and skip over medium boundaries
        isect.ComputeScatteringFunctions(ray, arena, true);
        if (!isect.bsdf) {
            VLOG(2) << "Skipping intersection due to null bsdf";
            ray = isect.SpawnRay(ray.d);
            bounces--;
            continue;
        }

        // Sample illumination from lights to find path contribution.
        // (But skip this for perfectly specular BSDFs.)
        Float dot_val = Dot(ray.d, isect.shading.n);
        Spectrum color(0);
        if (dot_val > 0) {          // dot_val > 0 means we are penetrating into (or through) the object
            Float tmp = (1.f - dot_val) * (1.f - dot_val);
            color[0] = 1.f;
            color[1] = tmp;
            color[2] = tmp;
        } else {
            Float tmp = (1.f + dot_val) * (1.f + dot_val);
            color[0] = tmp;
            color[1] = tmp;
            color[2] = 1.f;
        }
        L = color;
        break;
    }
    return L;
}

NormalIntegrator *CreateNormalIntegrator(const ParamSet &params,
                                     std::shared_ptr<Sampler> sampler,
                                     std::shared_ptr<const Camera> camera) {
    int np;
    const int *pb = params.FindInt("pixelbounds", &np);
    Bounds2i pixelBounds = camera->film->GetSampleBounds();
    if (pb) {
        if (np != 4)
            Error("Expected four values for \"pixelbounds\" parameter. Got %d.",
                  np);
        else {
            pixelBounds = Intersect(pixelBounds,
                                    Bounds2i{{pb[0], pb[2]}, {pb[1], pb[3]}});
            if (pixelBounds.Area() == 0)
                Error("Degenerate \"pixelbounds\" specified.");
        }
    }
    return new NormalIntegrator(camera, sampler, pixelBounds);
}

}  // namespace pbrt
