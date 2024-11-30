#include "TransientBuffer.hpp"

namespace Tungsten {

constexpr size_t MAX_TRANSIENT_BYTES = 5033164800; // 1024 * 1024 resolution, 400 frames, Vec3f (12 bytes each)

template<typename T>
template<typename Texel>
void TransientBuffer<T>::saveLdr(const Texel *hdr, const Path &path, bool /*useless*/) const {
    uint32 pixelCount = _res.product();
    std::unique_ptr<Vec3c[]> ldr(new Vec3c[pixelCount]);
    for (uint32 i = 0; i < pixelCount; ++i) {
        Texel f = hdr[i];
        
        if (std::isnan(average(f)) || std::isinf(average(f)))
            ldr[i] = Vec3c(255);
        else
            ldr[i] = Vec3c(clamp(Vec3i(Vec3f(f*255.0f)), Vec3i(0), Vec3i(255)));
    }

    ImageIO::saveLdr(path, &ldr[0].x(), _res.x(), _res.y(), 3);
}

template<typename T>
void TransientBuffer<T>::initTransientBuffer(int nframes) {
    if (_transients) return;
    size_t pixel_per_frame = _res.y() *_res.x();
    num_frames = nframes;
    if (num_frames > 0) {
        size_t est_size = pixel_per_frame * size_t(num_frames) * sizeof(T);
        float est_size_gb = float(est_size) / std::pow(1024.f, 3.f);
        if (est_size > MAX_TRANSIENT_BYTES) {
            constexpr float max_size_gb = float(MAX_TRANSIENT_BYTES) / std::pow(1024.f, 3.f);
            printf("Warning: estimated memory consumption is higher than allowed %.3f / (allowed %.3f GB). Not creating transient buffer.\n", est_size_gb, max_size_gb);
        } else {
            _transients = std::unique_ptr<T[]>(new T[pixel_per_frame * num_frames]);
            memset(_transients.get(), 0, pixel_per_frame * num_frames * sizeof(T));
            printf("Transient buffer initialized, memory consumption: %.3f GB\n", est_size_gb);
        }
    }
}

/// @brief  Temporal path reused transient recording (originally not supported, by authors of DARTS paper)
template<typename T>
void TransientBuffer<T>::addTransientSample(T* pixel_transient, Vec2u pixel) {
    if (!_transients || pixel_transient == nullptr) return;
    size_t frame_base = _res.y() *_res.x();
    // variance will not be recorded here, since we have more than one frame
    int pix_idx = pixel.x() + pixel.y()*_res.x();       // starting (index) address of the transient samples
    _sampleCount[pix_idx]++;
    for (size_t i = 0; i < num_frames; i++, pix_idx += frame_base) {
        auto c = pixel_transient[i];
        if (std::isnan(c) || std::isinf(c)) continue;
        _transients[pix_idx] += c;
    }
}
    
template<typename T>
void TransientBuffer<T>::save() const
{
    Path ldrFile(_settings.ldrOutputFile());
    Path hdrFile(_settings.hdrOutputFile());
    uint32 pixel_num = _res.y() *_res.x();
    // output frame by frame
    for (uint32 i = 0; i < _res.y(); i++) {
        for (uint32 j = 0; j < _res.x(); j++) {
            uint32 pixel = j + _res.x() * i;
            float sample_num = static_cast<float>(_sampleCount[pixel]);
            for (uint32 k = 0; k < num_frames; k++, pixel += pixel_num) {
                _transients[pixel] /= sample_num;                               // normalized by number of samples
            }
        }
    }
    if (!hdrFile.empty() && _transients) {
        auto hdr_stem = hdrFile.baseName();
        for (uint32 i = 0; i < num_frames; i++) {
            auto parent_path = hdrFile.parent();
            parent_path /= (hdr_stem.asString() + "_" + std::to_string(i) + hdrFile.extension().asString());
            ImageIO::saveHdr(parent_path, elementPointer(_transients.get() + pixel_num * i), _res.x(), _res.y(), elementCount(_transients[0]));
        }
    }
    if (!ldrFile.empty() && _transients) {
        auto hdr_stem = ldrFile.baseName();
        for (uint32 i = 0; i < num_frames; i++) {
            auto parent_path = ldrFile.parent();
            parent_path /= (hdr_stem.asString() + "_" + std::to_string(i) + ldrFile.extension().asString());
            saveLdr(_transients.get() + pixel_num * i, parent_path, true);
        }
    }
}

}
