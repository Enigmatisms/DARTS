#include "PathTraceIntegrator.hpp"

#include "sampling/UniformPathSampler.hpp"
#include "sampling/SobolPathSampler.hpp"

#include "cameras/Camera.hpp"

#include "thread/ThreadUtils.hpp"
#include "thread/ThreadPool.hpp"

namespace Tungsten {

CONSTEXPR uint32 PathTraceIntegrator::TileSize;
CONSTEXPR uint32 PathTraceIntegrator::VarianceTileSize;
CONSTEXPR uint32 PathTraceIntegrator::AdaptiveThreshold;

PathTraceIntegrator::PathTraceIntegrator()
: Integrator(),
  _w(0),
  _h(0),
  _varianceW(0),
  _varianceH(0),
  _sampler(0xBA5EBA11)
{
}

void PathTraceIntegrator::diceTiles()
{
    for (uint32 y = 0; y < _h; y += TileSize) {
        for (uint32 x = 0; x < _w; x += TileSize) {
            _tiles.emplace_back(
                x,
                y,
                min(TileSize, _w - x),
                min(TileSize, _h - y),
                _scene->rendererSettings().useSobol() ?
                    std::unique_ptr<PathSampleGenerator>(new SobolPathSampler(MathUtil::hash32(_sampler.nextI()))) :
                    std::unique_ptr<PathSampleGenerator>(new UniformPathSampler(MathUtil::hash32(_sampler.nextI())))
            );
        }
    }
}

float PathTraceIntegrator::errorPercentile95()
{
    std::vector<float> errors;
    errors.reserve(_samples.size());

    for (size_t i = 0; i < _samples.size(); ++i) {
        _samples[i].adaptiveWeight = _samples[i].errorEstimate();
        if (_samples[i].adaptiveWeight > 0.0f)
            errors.push_back(_samples[i].adaptiveWeight);
    }
    if (errors.empty())
        return 0.0f;
    std::sort(errors.begin(), errors.end());

    return errors[(errors.size()*95)/100];
}

void PathTraceIntegrator::dilateAdaptiveWeights()
{
    for (uint32 y = 0; y < _varianceH; ++y) {
        for (uint32 x = 0; x < _varianceW; ++x) {
            int idx = x + y*_varianceW;
            if (y < _varianceH - 1)
                _samples[idx].adaptiveWeight = max(_samples[idx].adaptiveWeight,
                        _samples[idx + _varianceW].adaptiveWeight);
            if (x < _varianceW - 1)
                _samples[idx].adaptiveWeight = max(_samples[idx].adaptiveWeight,
                        _samples[idx + 1].adaptiveWeight);
        }
    }
    for (int y = _varianceH - 1; y >= 0; --y) {
        for (int x = _varianceW - 1; x >= 0; --x) {
            int idx = x + y*_varianceW;
            if (y > 0)
                _samples[idx].adaptiveWeight = max(_samples[idx].adaptiveWeight,
                        _samples[idx - _varianceW].adaptiveWeight);
            if (x > 0)
                _samples[idx].adaptiveWeight = max(_samples[idx].adaptiveWeight,
                        _samples[idx - 1].adaptiveWeight);
        }
    }
}

void PathTraceIntegrator::distributeAdaptiveSamples(int spp)
{
    double totalWeight = 0.0;
    for (SampleRecord &record : _samples)
        totalWeight += record.adaptiveWeight;

    int adaptiveBudget = (spp - 1)*_w*_h;
    int budgetPerTile = adaptiveBudget/(VarianceTileSize*VarianceTileSize);
    float weightToSampleFactor = double(budgetPerTile)/totalWeight;

    float pixelPdf = 0.0f;
    for (SampleRecord &record : _samples) {
        float fractionalSamples = record.adaptiveWeight*weightToSampleFactor;
        int adaptiveSamples = int(fractionalSamples);
        pixelPdf += fractionalSamples - float(adaptiveSamples);
        if (_sampler.next1D() < pixelPdf) {
            adaptiveSamples++;
            pixelPdf -= 1.0f;
        }
        record.nextSampleCount = adaptiveSamples + 1;
    }
}

bool PathTraceIntegrator::generateWork()
{
    for (SampleRecord &record : _samples)
        record.sampleIndex += record.nextSampleCount;

    int sppCount = _nextSpp - _currentSpp;
    bool enableAdaptive = _scene->rendererSettings().useAdaptiveSampling();

    if (enableAdaptive && _currentSpp >= AdaptiveThreshold) {
        float maxError = errorPercentile95();
        if (maxError == 0.0f)
            return false;

        for (SampleRecord &record : _samples)
            record.adaptiveWeight = min(record.adaptiveWeight, maxError);

        dilateAdaptiveWeights();
        distributeAdaptiveSamples(sppCount);
    } else {
        for (SampleRecord &record : _samples)
            record.nextSampleCount = sppCount;
    }

    return true;
}

void PathTraceIntegrator::renderTile(uint32 id, uint32 tileId)
{
    ImageTile &tile = _tiles[tileId];
    auto sampling_info = _sampling_info.get();
    for (uint32 y = 0; y < tile.h; ++y) {
        for (uint32 x = 0; x < tile.w; ++x) {
            Vec2u pixel(tile.x + x, tile.y + y);
            uint32 pixelIndex = pixel.x() + pixel.y()*_w;
            uint32 variancePixelIndex = pixel.x()/VarianceTileSize + pixel.y()/VarianceTileSize*_varianceW;

            SampleRecord &record = _samples[variancePixelIndex];
            int spp = record.nextSampleCount;
            float bin_pdf = 1.f, remaining_time = 0.0;
            if (_sampling_info) {
                bin_pdf = _settings.transientTimeWidth / _sampling_info->time_range;        // normally, 1
                remaining_time = _sampling_info->sample_time_point(*tile.sampler, bin_pdf, _settings.transientTimeBeg, _settings.transientTimeWidth);
            }

            for (int i = 0; i < spp; ++i) {
                std::unique_ptr<Vec3f[]> transients;
                if (_settings.frame_num > 1) {
                    transients.reset(new Vec3f[_settings.frame_num]);
                    memset(transients.get(), 0, sizeof(Vec3f) * _settings.frame_num);
                }
                tile.sampler->startPath(pixelIndex, record.sampleIndex + i);
                Vec3f c = _tracers[id]->traceSample(pixel, *tile.sampler, sampling_info, transients.get(), bin_pdf, remaining_time);

                record.addSample(c);
                _scene->cam().colorBuffer()->addSample(pixel, c);
                if (transients)
                    _scene->cam().tryAddTransient(transients.get(), pixel);
            }
        }
    }
}

void PathTraceIntegrator::saveState(OutputStreamHandle &out)
{
    for (SampleRecord &s : _samples)
        s.saveState(out);
    for (ImageTile &i : _tiles)
        i.sampler->saveState(out);
}

void PathTraceIntegrator::loadState(InputStreamHandle &in)
{
    for (SampleRecord &s : _samples)
        s.loadState(in);
    for (ImageTile &i : _tiles)
        i.sampler->loadState(in);
}

void PathTraceIntegrator::fromJson(JsonPtr value, const Scene &/*scene*/)
{
    _settings.fromJson(value);
}

rapidjson::Value PathTraceIntegrator::toJson(Allocator &allocator) const
{
    return _settings.toJson(allocator);
}

void PathTraceIntegrator::prepareForRender(TraceableScene &scene, uint32 seed)
{
    _currentSpp = 0;
    _sampler = UniformSampler(MathUtil::hash32(seed));
    _scene = &scene;
    advanceSpp();
    scene.cam().requestColorBuffer();

    _sampling_info = nullptr;
    if (_settings.enable_elliptical || _settings.enable_guiding) {
        const Primitive *light =  _scene->lights()[0].get();
        PositionSample point;
        UniformPathSampler dummy_sampler(MathUtil::hash32(_sampler.nextI()));
        if (!light->samplePosition(dummy_sampler, point)) {
            std::cerr << "Error: failed to sample a point on the light source." << std::endl;
            throw 1;
        }

        _sampling_info.reset(new GuidedSamplingInfo(point.p, 0.f));

        Vec3f cam_pos = scene.cam().pos();
        if (_scene->lights().size() > 1)
            printf("Warning: Support for multiple light sources is not tested for DARTS.\n");
        // If interested time window is narrower than the distance gap
        _sampling_info->glob_min_time = fmaxf((cam_pos - point.p).length(), _settings.transientTimeBeg);
        _sampling_info->time_range = _settings.transientTimeEnd - _sampling_info->glob_min_time;
        if (_sampling_info->time_range <= 0) {
            std::cerr << "Guided Sampling Error: Max time is no greater than shortest path time "<< _sampling_info->glob_min_time <<". Exitting..." << std::endl;
            throw 1;
        }

        printf("Path tracer DARTS enabled.\n");
    }

    for (uint32 i = 0; i < ThreadUtils::pool->threadCount(); ++i)
        _tracers.emplace_back(new PathTracer(&scene, _settings, i));

    _w = scene.cam().resolution().x();
    _h = scene.cam().resolution().y();
    _varianceW = (_w + VarianceTileSize - 1)/VarianceTileSize;
    _varianceH = (_h + VarianceTileSize - 1)/VarianceTileSize;
    diceTiles();
    _samples.resize(_varianceW*_varianceH);
}

void PathTraceIntegrator::teardownAfterRender()
{
    _group.reset();

    _tracers.clear();
    _samples.clear();
    _tiles  .clear();
    _tracers.shrink_to_fit();
    _samples.shrink_to_fit();
    _tiles  .shrink_to_fit();
}

bool PathTraceIntegrator::supportsResumeRender() const
{
    return true;
}

void PathTraceIntegrator::startRender(std::function<void()> completionCallback)
{
    if (done() || !generateWork()) {
        _currentSpp = _nextSpp;
        advanceSpp();
        completionCallback();
        return;
    }

    using namespace std::placeholders;
    _group = ThreadUtils::pool->enqueue(
        std::bind(&PathTraceIntegrator::renderTile, this, _3, _1),
        _tiles.size(),
        [&, completionCallback]() {
            _currentSpp = _nextSpp;
            advanceSpp();
            completionCallback();
        }
    );
}

void PathTraceIntegrator::waitForCompletion()
{
    if (_group) {
        _group->wait();
        _group.reset();
    }
}

void PathTraceIntegrator::abortRender()
{
    if (_group) {
        _group->abort();
        _group->wait();
        _group.reset();
    }
}

}
