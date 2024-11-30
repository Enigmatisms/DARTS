
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

// integrators/bdpt.cpp*
#include <chrono>
#include "integrators/bdpt.h"
#include "film.h"
#include "filters/box.h"
#include "integrator.h"
#include "lightdistrib.h"
#include "paramset.h"
#include "progressreporter.h"
#include "sampler.h"
#include "stats.h"

namespace pbrt {

STAT_PERCENT("Integrator/Zero-radiance paths", zeroRadiancePaths, totalPaths);
STAT_INT_DISTRIBUTION("Integrator/Path length", pathLength);

/**
 * @brief Random walk forward declaration
 * when called in light path sampling without interlaced guiding
 */
int RandomWalk(const Scene &scene, RayDifferential ray, Sampler &sampler,
               MemoryArena &arena, Spectrum beta, Float pdf, int maxDepth,
               TransportMode mode, Vertex *path, Float remain_time, bool use_guide,
               GuidedSamplingInfo* const guide_info = nullptr, FilmInfoConstPtr film_info = nullptr);

// BDPT Utility Functions
Float CorrectShadingNormal(const SurfaceInteraction &isect, const Vector3f &wo,
                           const Vector3f &wi, TransportMode mode) {
    if (mode == TransportMode::Importance) {
        Float num = AbsDot(wo, isect.shading.n) * AbsDot(wi, isect.n);
        Float denom = AbsDot(wo, isect.n) * AbsDot(wi, isect.shading.n);
        // wi is occasionally perpendicular to isect.shading.n; this is
        // fine, but we don't want to return an infinite or NaN value in
        // that case.
        if (denom == 0) return 0;
        return num / denom;
    } else
        return 1;
}

int GenerateCameraSubpath(
    const Scene &scene, Sampler &sampler,
    MemoryArena &arena, int maxDepth,
    const Camera &camera, const Point2f &pFilm,
    Vertex *path, Float remain_time, bool use_guide,
    GuidedSamplingInfo* const guide_info, FilmInfoConstPtr film_info
) {
    if (maxDepth == 0) return 0;
    ProfilePhase _(Prof::BDPTGenerateSubpath);
    // Sample initial ray for camera subpath
    CameraSample cameraSample;
    cameraSample.pFilm = pFilm;
    cameraSample.time = sampler.Get1D();
    cameraSample.pLens = sampler.Get2D();
    RayDifferential ray;
    Spectrum beta = camera.GenerateRayDifferential(cameraSample, &ray);
    ray.ScaleDifferentials(1 / std::sqrt(sampler.samplesPerPixel));

    // Generate first vertex on camera subpath and start random walk
    Float pdfPos, pdfDir;
    path[0] = Vertex::CreateCamera(&camera, ray, beta);
    camera.Pdf_We(ray, &pdfPos, &pdfDir);
    VLOG(2) << "Starting camera subpath. Ray: " << ray << ", beta " << beta
            << ", pdfPos " << pdfPos << ", pdfDir " << pdfDir;
    return RandomWalk(scene, ray, sampler, arena, beta, pdfDir, maxDepth - 1,
                      TransportMode::Radiance, path + 1, remain_time, use_guide, guide_info, film_info) +
           1;
}

// TODO: this is to be modified, we need to construct a guided sampling info struct here
int GenerateLightSubpath(
    const Scene &scene, Sampler &sampler, MemoryArena &arena, int maxDepth,
    Float time, const Distribution1D &lightDistr,
    const std::unordered_map<const Light *, size_t> &lightToIndex,
    Vertex *path, Float remain_time, bool use_guide,
    GuidedSamplingInfo* const guide_info, FilmInfoConstPtr film_info
) {

    // TODO: there are more to consider, like what if we are spawned inside of the medium?
    // We can not use the emitter vertex as our goal (since this might not be a point source)

    if (maxDepth == 0) return 0;
    ProfilePhase _(Prof::BDPTGenerateSubpath);
    // Sample initial ray for light subpath
    Float lightPdf;
    int lightNum = lightDistr.SampleDiscrete(sampler.Get1D(), &lightPdf);
    const std::shared_ptr<Light> &light = scene.lights[lightNum];
    RayDifferential ray;
    Normal3f nLight;
    Float pdfPos, pdfDir;
    Spectrum Le = light->Sample_Le(sampler.Get2D(), sampler.Get2D(), time, &ray,
                                   &nLight, &pdfPos, &pdfDir);
    if (pdfPos == 0 || pdfDir == 0 || Le.IsBlack()) return 0;

    // Generate first vertex on light subpath and start random walk
    path[0] =
        Vertex::CreateLight(light.get(), ray, nLight, Le, pdfPos * lightPdf);
    Spectrum beta = Le * AbsDot(nLight, ray.d) / (lightPdf * pdfPos * pdfDir);
    VLOG(2) << "Starting light subpath. Ray: " << ray << ", Le " << Le <<
        ", beta " << beta << ", pdfPos " << pdfPos << ", pdfDir " << pdfDir;

    // therefore light path can not be guided
    int nVertices =
        RandomWalk(scene, ray, sampler, arena, beta, pdfDir, maxDepth - 1,
                   TransportMode::Importance, path + 1, remain_time, use_guide, guide_info, film_info);

    // Correct subpath sampling densities for infinite area lights
    if (path[0].IsInfiniteLight()) {
        // Set spatial density of _path[1]_ for infinite area light
        if (nVertices > 0) {
            path[1].pdfFwd = pdfPos;
            if (path[1].IsOnSurface())
                path[1].pdfFwd *= AbsDot(ray.d, path[1].ng());
        }

        // Set spatial density of _path[0]_ for infinite area light
        path[0].pdfFwd =
            InfiniteLightDensity(scene, lightDistr, lightToIndex, ray.d);
    }
    return nVertices + 1;
}

int RandomWalk(const Scene &scene, RayDifferential ray, Sampler &sampler,
               MemoryArena &arena, Spectrum beta, Float pdf, int maxDepth,
               TransportMode mode, Vertex *path, Float remain_time, bool use_guide,
               GuidedSamplingInfo* const guide_info, FilmInfoConstPtr film_info) {
    // if guide_info->v_ptr is not null_ptr, this would mean that we need to fill in the fields
    if (maxDepth == 0) return 0;
    int bounces = 0;
    // Declare variables for forward and reverse probability densities
    Float pdfFwd = pdf, pdfRev = 0;
    // guide_info is not empty and guide_info field is not filled, then we should fill the information
    const bool fill_field = (guide_info) && (guide_info->time < 0);
    bool transient_valid = (film_info != nullptr);
    Float origin_remaining_t = transient_valid ? remain_time : 0;
    // path entered the scattering medium for the first time
    bool scattered = false;         

    // TODO: calculate (sampling) and pass-in the target time

    while (true) {
        // Attempt to create the next subpath vertex in _path_
        MediumInteraction mi;

        VLOG(2) << "Random walk. Bounces " << bounces << ", beta " << beta <<
            ", pdfFwd " << pdfFwd << ", pdfRev " << pdfRev;
        // Trace a ray and sample the medium, if any
        SurfaceInteraction isect;
        // After intersection, isect carries ray.time, which does not account for the travelling time from the last intersection (tMax?)
        bool foundIntersection = scene.Intersect(ray, &isect);
        if (ray.medium) {
            Float remaining_time = 0.;
            if (guide_info != nullptr) {
                Float shortest_time = (ray.o - guide_info->vertex_p).Length();
                remaining_time = fmaxf(origin_remaining_t - ray.time - guide_info->time, shortest_time + 1e-4);
            }
            beta *= ray.medium->Sample(ray, sampler, arena, &mi, remaining_time, guide_info, use_guide);
        }
        if (beta.IsBlack()) break;
        Vertex &vertex = path[bounces], &prev = path[bounces - 1];
        
        if (mi.IsValid()) {
            // Record medium interaction in _path_ and compute forward density
            // The first medium vertex is recorded
            vertex = Vertex::CreateMedium(mi, beta, pdfFwd, prev);
            if (fill_field && scattered == false) {
                scattered = true;
                guide_info->time = vertex.time();
                guide_info->vertex_p = vertex.p();
            }
            if (++bounces >= maxDepth) break;

            // Sample direction and compute reverse density at preceding vertex
            Vector3f wi;
            // FIXME: now g = 0 (we do not account for other situations)
            pdfFwd = pdfRev = mi.phase->Sample_p(-ray.d, &wi, sampler.Get2D());
            if (mi.phase->symmetric == false) {
                pdfRev = mi.phase->p(wi, -ray.d);
            }
            // If we are using other sampling method other than phase function, pdfRev is not pdfFwd
            // Also, throughput should be modified (can not be 1.)

            // We don't need to calculate PDF here since we do not do directional sampling
            ray = mi.SpawnRay(wi);
        } else {
            // Handle surface interaction for path generation
            if (!foundIntersection) {
                // Capture escaped rays when tracing from the camera -- for inifinity light source
                if (mode == TransportMode::Radiance) {
                    vertex = Vertex::CreateLight(EndpointInteraction(ray), beta,
                                                 pdfFwd);
                    ++bounces;
                }
                // This does not influence the result, we can skip the infinite light case
                break;
            }
            isect.time += ray.tMax;
                
            // Compute scattering functions for _mode_ and skip over medium
            // boundaries
            isect.ComputeScatteringFunctions(ray, arena, true, mode);
            if (!isect.bsdf) {
                ray = isect.SpawnRay(ray.d);
                continue;
            }

            // Initialize _vertex_ with surface intersection information
            vertex = Vertex::CreateSurface(isect, beta, pdfFwd, prev);
            
            if (++bounces >= maxDepth) break;

            // Sample BSDF at current vertex and compute reverse probability
            Vector3f wi, wo = isect.wo;
            BxDFType type;
            Spectrum f = isect.bsdf->Sample_f(wo, &wi, sampler.Get2D(), &pdfFwd,
                                              BSDF_ALL, &type);
            VLOG(2) << "Random walk sampled dir " << wi << " f: " << f <<
                ", pdfFwd: " << pdfFwd;
            if (f.IsBlack() || pdfFwd == 0.f) break;
            beta *= f * AbsDot(wi, isect.shading.n) / pdfFwd;

            // We actually need to record radiance here, if the medium has a non-null boundary
            VLOG(2) << "Random walk beta now " << beta;
            pdfRev = isect.bsdf->Pdf(wi, wo, BSDF_ALL);
            if (type & BSDF_SPECULAR) {
                vertex.delta = true;
                pdfRev = pdfFwd = 0;
            }
            beta *= CorrectShadingNormal(isect, wo, wi, mode);
            VLOG(2) << "Random walk beta after shading normal correction " << beta;
            ray = isect.SpawnRay(wi);
        }
        if (transient_valid && ray.time > film_info->max_time) break;

        // Compute reverse area density at preceding vertex
        prev.pdfRev = vertex.ConvertDensity(pdfRev, prev);
    }
    return bounces;
}

Spectrum G(const Scene &scene, Sampler &sampler, const Vertex &v0,
           const Vertex &v1) {
    Vector3f d = v0.p() - v1.p();
    Float g = 1 / d.LengthSquared();
    d *= std::sqrt(g);
    if (v0.IsOnSurface()) g *= AbsDot(v0.ns(), d);
    if (v1.IsOnSurface()) g *= AbsDot(v1.ns(), d);
    VisibilityTester vis(v0.GetInteraction(), v1.GetInteraction());
    return g * vis.Tr(scene, sampler);
}

Float MISWeight(const Scene &scene, Vertex *lightVertices,
                Vertex *cameraVertices, Vertex &sampled, int s, int t,
                const Distribution1D &lightPdf,
                const std::unordered_map<const Light *, size_t> &lightToIndex) {
    if (s + t == 2) return 1;
    Float sumRi = 0;
    // Define helper function _remap0_ that deals with Dirac delta functions
    auto remap0 = [](Float f) -> Float { return f != 0 ? f : 1; };

    // Temporarily update vertex properties for current strategy

    // Look up connection vertices and their predecessors
    Vertex *qs = s > 0 ? &lightVertices[s - 1] : nullptr,
           *pt = t > 0 ? &cameraVertices[t - 1] : nullptr,
           *qsMinus = s > 1 ? &lightVertices[s - 2] : nullptr,
           *ptMinus = t > 1 ? &cameraVertices[t - 2] : nullptr;

    // Update sampled vertex for $s=1$ or $t=1$ strategy
    ScopedAssignment<Vertex> a1;
    if (s == 1)
        a1 = {qs, sampled};
    else if (t == 1)
        a1 = {pt, sampled};

    // Mark connection vertices as non-degenerate
    ScopedAssignment<bool> a2, a3;
    if (pt) a2 = {&pt->delta, false};
    if (qs) a3 = {&qs->delta, false};

    // Update reverse density of vertex $\pt{}_{t-1}$
    ScopedAssignment<Float> a4;
    if (pt)
        a4 = {&pt->pdfRev, s > 0 ? qs->Pdf(scene, qsMinus, *pt)
                                 : pt->PdfLightOrigin(scene, *ptMinus, lightPdf,
                                                      lightToIndex)};

    // Update reverse density of vertex $\pt{}_{t-2}$
    ScopedAssignment<Float> a5;
    if (ptMinus)
        a5 = {&ptMinus->pdfRev, s > 0 ? pt->Pdf(scene, qs, *ptMinus)
                                      : pt->PdfLight(scene, *ptMinus)};

    // Update reverse density of vertices $\pq{}_{s-1}$ and $\pq{}_{s-2}$
    ScopedAssignment<Float> a6;
    if (qs) a6 = {&qs->pdfRev, pt->Pdf(scene, ptMinus, *qs)};
    ScopedAssignment<Float> a7;
    if (qsMinus) a7 = {&qsMinus->pdfRev, qs->Pdf(scene, pt, *qsMinus)};

    // Consider hypothetical connection strategies along the camera subpath
    Float ri = 1;
    for (int i = t - 1; i > 0; --i) {
        ri *=
            remap0(cameraVertices[i].pdfRev) / remap0(cameraVertices[i].pdfFwd);
        if (!cameraVertices[i].delta && !cameraVertices[i - 1].delta)
            sumRi += ri;
    }

    // Consider hypothetical connection strategies along the light subpath
    ri = 1;
    for (int i = s - 1; i >= 0; --i) {
        ri *= remap0(lightVertices[i].pdfRev) / remap0(lightVertices[i].pdfFwd);
        bool deltaLightvertex = i > 0 ? lightVertices[i - 1].delta
                                      : lightVertices[0].IsDeltaLight();
        if (!lightVertices[i].delta && !deltaLightvertex) sumRi += ri;
    }
    return 1 / (1 + sumRi);
}

// BDPT Method Definitions
inline int BufferIndex(int s, int t) {
    int above = s + t - 2;
    return s + above * (5 + above) / 2;
}

void BDPTIntegrator::Render(const Scene &scene) {
    std::unique_ptr<LightDistribution> lightDistribution =
        CreateLightSampleDistribution(lightSampleStrategy, scene);

    // Compute a reverse mapping from light pointers to offsets into the
    // scene lights vector (and, equivalently, offsets into
    // lightDistr). Added after book text was finalized; this is critical
    // to reasonable performance with 100s+ of light sources.
    std::unordered_map<const Light *, size_t> lightToIndex;
    for (size_t i = 0; i < scene.lights.size(); ++i)
        lightToIndex[scene.lights[i].get()] = i;

    // Partition the image into tiles
    Film *film = camera->film;
    const Bounds2i sampleBounds = film->GetSampleBounds();
    const Vector2i sampleExtent = sampleBounds.Diagonal();
    // TODO: adjustable tile size should be implemented
    const int tileSize = 2;
    const int nXTiles = (sampleExtent.x + tileSize - 1) / tileSize;
    const int nYTiles = (sampleExtent.y + tileSize - 1) / tileSize;
    
    ProgressReporter reporter(nXTiles * nYTiles, "Rendering", log_time);

    // Allocate buffers for debug visualization
    const int bufferCount = (1 + maxDepth) * (6 + maxDepth) / 2;
    std::vector<std::unique_ptr<Film>> weightFilms(bufferCount);
    if (visualizeStrategies || visualizeWeights) {
        for (int depth = 0; depth <= maxDepth; ++depth) {
            for (int s = 0; s <= depth + 2; ++s) {
                int t = depth + 2 - s;
                if (t == 0 || (s == 1 && t == 1)) continue;

                std::string filename =
                    StringPrintf("bdpt_d%02i_s%02i_t%02i.exr", depth, s, t);

                weightFilms[BufferIndex(s, t)] = std::unique_ptr<Film>(new Film(
                    film->fullResolution,
                    Bounds2f(Point2f(0, 0), Point2f(1, 1)),
                    std::unique_ptr<Filter>(CreateBoxFilter(ParamSet())),
                    film->diagonal * 1000, filename, 1.f));
            }
        }
    }
    std::unique_ptr<FilmInfo> film_info(nullptr);
    if (film->sample_cnt > 0) {             // only when we are doing transient rendering
        film_info = std::make_unique<FilmInfo>(camera->film);
    }

    // Render and write the output image to disk
    if (scene.lights.size() > 0) {
        ParallelFor2D([&](const Point2i tile) {
            // Render a single tile using BDPT
            MemoryArena arena;
            uint64_t seed = rseed_time ? std::chrono::system_clock().now().time_since_epoch().count() :
                    uint64_t(tile.y * nXTiles + tile.x);
            std::unique_ptr<Sampler> tileSampler = sampler->Clone(seed);
            int x0 = sampleBounds.pMin.x + tile.x * tileSize;
            int x1 = std::min(x0 + tileSize, sampleBounds.pMax.x);
            int y0 = sampleBounds.pMin.y + tile.y * tileSize;
            int y1 = std::min(y0 + tileSize, sampleBounds.pMax.y);
            Bounds2i tileBounds(Point2i(x0, y0), Point2i(x1, y1));
            LOG(INFO) << "Starting image tile " << tileBounds;

            std::unique_ptr<FilmTile> filmTile =
                camera->film->GetFilmTile(tileBounds);
            for (Point2i pPixel : tileBounds) {
                tileSampler->StartPixel(pPixel);
                if (!InsideExclusive(pPixel, pixelBounds))
                    continue;
                do {
                    // Generate a single sample using BDPT
                    Point2f pFilm = (Point2f)pPixel + tileSampler->Get2D();

                    // Trace the camera subpath
                    Vertex *cameraVertices = arena.Alloc<Vertex>(maxDepth + 2);
                    Vertex *lightVertices = arena.Alloc<Vertex>(maxDepth + 1);

                    std::unique_ptr<GuidedSamplingInfo> sampling_info(nullptr);
                    if (da_guide) {
                        sampling_info.reset(new GuidedSamplingInfo());
                    }

                    /**
                     * Actually, I got an idea, that if we do not generate the whole path each time
                     * Instead we generate camera/emitter path interlacedly. Then both sides can be guided
                    */
#ifdef ANALYTICAL_GUIDING
                    Point3f persp_cam_pos = camera->CameraToWorld(0.0, Point3f(0, 0, 0));
                    const Distribution1D *lightDistr =
                        lightDistribution->Lookup(persp_cam_pos);
                    // Here we will start light subpath sampling first, since later on
                    // We will need some information to guide the camera sub path

                    int nLight = GenerateLightSubpath(
                        scene, *tileSampler, arena, maxDepth + 1,
                        cameraVertices[0].time(), *lightDistr, lightToIndex,
                        lightVertices, 0.0, da_guide, sampling_info.get(), film_info.get());

                    // After generating sampling info (a vertex and incident radiance), passing to camera side

                    Float time_interval = -1.f;
                    if (valid_guide) {
                        // FIXME: the ROI is not considered. This will be a little bit more complicated
                        // if ROI is to be considered. Now I assume the ROI is matched.
                        // This is the minimum possible ToF. We can not achieve time that is less than this.
                        Point3f camera_pos = camera->CameraToWorld(0, Point3f(0, 0, 0));
                        sampling_info->glob_min_time = (camera_pos - sampling_info->vertex_p).Length();
                        time_interval = (camera->film->max_time - sampling_info->glob_min_time) / Float(sampler->samplesPerPixel);
                    }
                    Float remaining_time = time_interval * tileSampler->CurrentSampleNumber() + sampling_info->glob_min_time + 1e-3;

                    int nCamera = GenerateCameraSubpath(
                        scene, *tileSampler, arena, maxDepth + 2, *camera,
                        pFilm, cameraVertices, remaining_time, da_guide, sampling_info.get(), film_info.get());

#else   // ANALYTICAL_GUIDING
                    // When analytical guiding is not used
                    
                    int nCamera = GenerateCameraSubpath(
                        scene, *tileSampler, arena, maxDepth + 2, *camera,
                        pFilm, cameraVertices);
                    // Get a distribution for sampling the light at the
                    // start of the light subpath. Because the light path
                    // follows multiple bounces, basing the sampling
                    // distribution on any of the vertices of the camera
                    // path is unlikely to be a good strategy. We use the
                    // PowerLightDistribution by default here, which
                    // doesn't use the point passed to it.
                    const Distribution1D *lightDistr =
                        lightDistribution->Lookup(cameraVertices[0].p());
                    // Now trace the light subpath
                    int nLight = GenerateLightSubpath(
                        scene, *tileSampler, arena, maxDepth + 1,
                        cameraVertices[0].time(), *lightDistr, lightToIndex,
                        lightVertices);
#endif  // ANALYTICAL_GUIDING
                    // Execute all BDPT connection strategies
                    Spectrum L(0.f);
                    std::unique_ptr<FilmTilePixel[]> local_storage = nullptr;
                    if (film->sample_cnt > 0) {
                        local_storage = std::make_unique<FilmTilePixel[]>(film->sample_cnt);
                    }
                    for (int t = 1; t <= nCamera; ++t) {
                        for (int s = 0; s <= nLight; ++s) {
                            int depth = t + s - 2;
                            if ((s == 1 && t == 1) || depth < 0 ||
                                depth > maxDepth)
                                continue;
                            // Execute the $(s, t)$ connection strategy and
                            // update _L_
                            Point2f pFilmNew = pFilm;
                            Float misWeight = 0.f, sum_time = 0.0;

                            // Yet the computation could get bulky if we use BDPT
                            // But if this is not a vertex connection strategy, the meaning of this algorithm can be degraded
                            // Check mitsubaToF (since ellipse intersection is also computationally intensive)

                            Spectrum Lpath = ConnectBDPT(
                                scene, lightVertices, cameraVertices, s, t,
                                *lightDistr, lightToIndex, *camera, *tileSampler, sum_time, 
                                useMISWeight, &pFilmNew, &misWeight);
                            VLOG(2) << "Connect bdpt s: " << s <<", t: " << t <<
                                ", Lpath: " << Lpath << ", misWeight: " << misWeight;
                            if (visualizeStrategies || visualizeWeights) {
                                Spectrum value;
                                if (visualizeStrategies)
                                    value =
                                        misWeight == 0 ? 0 : Lpath / misWeight;
                                if (visualizeWeights) value = Lpath;
                                weightFilms[BufferIndex(s, t)]->AddSplat(
                                    pFilmNew, value);
                            }
                            if (t != 1) {
                                L += Lpath;
                                if (!Lpath.IsBlack() && sum_time > film->min_time && sum_time < film->max_time) {
                                    int local_idx = int(std::floor((sum_time - film->min_time) / film->interval));
                                    local_storage[local_idx].contribSum += Lpath;
                                } 
                            } else {
                                film->AddSplat(pFilmNew, Lpath, sum_time);
                            }
                        }
                    }
                    VLOG(2) << "Add film sample pFilm: " << pFilm << ", L: " << L <<
                        ", (y: " << L.y() << ")";
                    filmTile->AddSample(pFilm, L, 1.0, std::move(local_storage));
                    arena.Reset();
                } while (tileSampler->StartNextSample());
            }
            film->MergeFilmTile(std::move(filmTile));
            reporter.Update();
            LOG(INFO) << "Finished image tile " << tileBounds;
        }, Point2i(nXTiles, nYTiles));
        reporter.Done();
    }
    film->WriteImage(1.0f / sampler->samplesPerPixel);
    film->WriteTransient(sampler->samplesPerPixel);

    // Write buffers for debug visualization
    if (visualizeStrategies || visualizeWeights) {
        const Float invSampleCount = 1.0f / sampler->samplesPerPixel;
        for (size_t i = 0; i < weightFilms.size(); ++i)
            if (weightFilms[i]) weightFilms[i]->WriteImage(invSampleCount);
    }
}

Spectrum ConnectBDPT(
    const Scene &scene, Vertex *lightVertices, Vertex *cameraVertices, int s,
    int t, const Distribution1D &lightDistr,
    const std::unordered_map<const Light *, size_t> &lightToIndex,
    const Camera &camera, Sampler &sampler, Float& sum_time, bool use_mis, Point2f *pRaster,
    Float *misWeightPtr) {
    ProfilePhase _(Prof::BDPTConnectSubpaths);
    Spectrum L(0.f);
    // Ignore invalid connections related to infinite area lights
    if (t > 1 && s != 0 && cameraVertices[t - 1].type == VertexType::Light)
        return Spectrum(0.f);

    // Perform connection and write contribution to _L_
    Vertex sampled;
    if (s == 0) {
        // Interpret the camera subpath as a complete path
        const Vertex &pt = cameraVertices[t - 1];
        if (pt.IsLight()) L = pt.Le(scene, cameraVertices[t - 2]) * pt.beta;
        DCHECK(!L.HasNaNs());
        if (!L.IsBlack()) sum_time = pt.time();
    } else if (t == 1) {
        // Sample a point on the camera and connect it to the light subpath
        const Vertex &qs = lightVertices[s - 1];
        if (qs.IsConnectible()) {
            VisibilityTester vis;
            Vector3f wi;
            Float pdf;
            Spectrum Wi = camera.Sample_Wi(qs.GetInteraction(), sampler.Get2D(),
                                           &wi, &pdf, pRaster, &vis);
            if (pdf > 0 && !Wi.IsBlack()) {
                // Initialize dynamically sampled vertex and _L_ for $t=1$ case
                sampled = Vertex::CreateCamera(&camera, vis.P1(), Wi / pdf);
                L = qs.beta * qs.f(sampled, TransportMode::Importance) * sampled.beta;
                if (qs.IsOnSurface()) L *= AbsDot(wi, qs.ns());
                DCHECK(!L.HasNaNs());
                // Only check visibility after we know that the path would
                // make a non-zero contribution.
                if (!L.IsBlack()) L *= vis.Tr(scene, sampler);
                if (!L.IsBlack()) {
                    sum_time = qs.time() + (vis.P0().p - vis.P1().p).Length();
                }
            }
        }
    } else if (s == 1) {
        // Sample a point on a light and connect it to the camera subpath
        const Vertex &pt = cameraVertices[t - 1];
        if (pt.IsConnectible()) {
            Float lightPdf;
            VisibilityTester vis;
            Vector3f wi;
            Float pdf;
            int lightNum =
                lightDistr.SampleDiscrete(sampler.Get1D(), &lightPdf);
            const std::shared_ptr<Light> &light = scene.lights[lightNum];
            Spectrum lightWeight = light->Sample_Li(
                pt.GetInteraction(), sampler.Get2D(), &wi, &pdf, &vis);
            if (pdf > 0 && !lightWeight.IsBlack()) {
                EndpointInteraction ei(vis.P1(), light.get());
                sampled =
                    Vertex::CreateLight(ei, lightWeight / (pdf * lightPdf), 0);
                sampled.pdfFwd =
                    sampled.PdfLightOrigin(scene, pt, lightDistr, lightToIndex);
                L = pt.beta * pt.f(sampled, TransportMode::Radiance) * sampled.beta;
                if (pt.IsOnSurface()) L *= AbsDot(wi, pt.ns());
                // Only check visibility if the path would carry radiance.
                if (!L.IsBlack()) L *= vis.Tr(scene, sampler);
                if (!L.IsBlack()) {
                    sum_time = pt.time() + (vis.P0().p - vis.P1().p).Length();
                }
            }
        }
    } else {
        // Handle all other bidirectional connection cases
        const Vertex &qs = lightVertices[s - 1], &pt = cameraVertices[t - 1];
        if (qs.IsConnectible() && pt.IsConnectible()) {
            L = qs.beta * qs.f(pt, TransportMode::Importance) * pt.f(qs, TransportMode::Radiance) * pt.beta;
            VLOG(2) << "General connect s: " << s << ", t: " << t <<
                " qs: " << qs << ", pt: " << pt << ", qs.f(pt): " << qs.f(pt, TransportMode::Importance) <<
                ", pt.f(qs): " << pt.f(qs, TransportMode::Radiance) << ", G: " << G(scene, sampler, qs, pt) <<
                ", dist^2: " << DistanceSquared(qs.p(), pt.p());
            if (!L.IsBlack()) L *= G(scene, sampler, qs, pt);
            if (!L.IsBlack()) {
                sum_time = pt.time() + qs.time() + (qs.p() - pt.p()).Length();
            }
        }
    }

    ++totalPaths;
    if (L.IsBlack()) ++zeroRadiancePaths;
    ReportValue(pathLength, s + t - 2);

    // Compute MIS weight for connection strategy
    Float misWeight = 1.;
    if (use_mis) {
        misWeight = L.IsBlack() ? 0.f : MISWeight(scene, lightVertices, cameraVertices,
                                      sampled, s, t, lightDistr, lightToIndex);
    }
    VLOG(2) << "MIS weight for (s,t) = (" << s << ", " << t << ") connection: "
            << misWeight;
    DCHECK(!std::isnan(misWeight));
    L *= misWeight;
    if (misWeightPtr) *misWeightPtr = misWeight;
    if (L.IsBlack()) sum_time = 0.0;
    return L;
}

BDPTIntegrator *CreateBDPTIntegrator(const ParamSet &params,
                                     std::shared_ptr<Sampler> sampler,
                                     std::shared_ptr<const Camera> camera) {
    int maxDepth    = params.FindOneInt("maxdepth", 5);
    int tileSize    = params.FindOneInt("tileSize", 16);
    bool useMIS     = params.FindOneInt("useMIS", 1) != 0;
    bool rseed_time = params.FindOneInt("rseed_time", 0) != 0;
    bool log_time   = params.FindOneInt("logTime", 1) != 0;
    bool visualizeStrategies = params.FindOneBool("visualizestrategies", false);
    bool visualizeWeights = params.FindOneBool("visualizeweights", false);
    bool use_guiding = params.FindOneInt("guiding", false);

    float guiding_p = -1;
    if (use_guiding) {
        guiding_p = fminf(params.FindOneFloat("guiding_p", -1), 1.);
    }

    if ((visualizeStrategies || visualizeWeights) && maxDepth > 5) {
        Warning(
            "visualizestrategies/visualizeweights was enabled, limiting "
            "maxdepth to 5");
        maxDepth = 5;
    }
    if (tileSize <= 0) {
        Warning("Tile size must be positive, setting to 16");
        tileSize = 16;
    }
    printf("BDPT integrator: MIS weight = %d, max depth = %d, tile size = %d\n", int(useMIS), maxDepth, tileSize);
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

    std::string lightStrategy = params.FindOneString("lightsamplestrategy",
                                                     "power");
    // TODO: light sampling strategy might be modified with Meng 2016
    return new BDPTIntegrator(sampler, camera, maxDepth, tileSize, useMIS, rseed_time, log_time,
                visualizeStrategies, visualizeWeights, pixelBounds, lightStrategy, guiding_p);
}

}  // namespace pbrt
