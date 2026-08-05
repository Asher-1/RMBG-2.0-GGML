#include "rmbg_preprocess.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace rmbg {
namespace {

float cubic(float x) {
    x = std::fabs(x);
    if (x < 1.f) return ((1.5f * x - 2.5f) * x) * x + 1.f;
    if (x < 2.f) return ((-0.5f * x + 2.5f) * x - 4.f) * x + 2.f;
    return 0.f;
}

template<typename Sample>
float resize_bicubic(Sample sample, int sw, int sh, int dw, int dh, int x, int y) {
    const float sx = ((float) x + 0.5f) * sw / dw - 0.5f;
    const float sy = ((float) y + 0.5f) * sh / dh - 0.5f;
    const int ix = (int) std::floor(sx), iy = (int) std::floor(sy);
    float sum = 0.f, norm = 0.f;
    for (int ky = -1; ky <= 2; ++ky) {
        const float wy = cubic(sy - (iy + ky));
        const int py = std::clamp(iy + ky, 0, sh - 1);
        for (int kx = -1; kx <= 2; ++kx) {
            const float w = wy * cubic(sx - (ix + kx));
            const int px = std::clamp(ix + kx, 0, sw - 1);
            sum += sample(px, py) * w;
            norm += w;
        }
    }
    return norm != 0.f ? sum / norm : 0.f;
}

template<typename Sample>
float resize_bilinear(Sample sample, int sw, int sh, int dw, int dh, int x, int y) {
    const float sx = ((float) x + 0.5f) * sw / dw - 0.5f;
    const float sy = ((float) y + 0.5f) * sh / dh - 0.5f;
    const int x0 = std::clamp((int) std::floor(sx), 0, sw - 1);
    const int y0 = std::clamp((int) std::floor(sy), 0, sh - 1);
    const int x1 = std::min(x0 + 1, sw - 1);
    const int y1 = std::min(y0 + 1, sh - 1);
    const float dx = std::clamp(sx, 0.f, (float) sw - 1.f) - x0;
    const float dy = std::clamp(sy, 0.f, (float) sh - 1.f) - y0;
    return (1.f - dy) * ((1.f - dx) * sample(x0, y0) + dx * sample(x1, y0)) +
           dy * ((1.f - dx) * sample(x0, y1) + dx * sample(x1, y1));
}

void png_write(void * context, void * data, int size) {
    auto & out = *static_cast<std::vector<uint8_t> *>(context);
    const auto * begin = static_cast<const uint8_t *>(data);
    out.insert(out.end(), begin, begin + size);
}

} // namespace

bool decode_preprocess(const void * bytes, int length, int size,
                       const float mean[3], const float std[3],
                       std::vector<uint8_t> & original_rgba, int & width, int & height,
                       std::vector<float> & input_nchw, std::string & err) {
    if (!bytes || length <= 0 || size <= 0) { err = "invalid image input"; return false; }
    int channels = 0;
    stbi_uc * decoded = stbi_load_from_memory(static_cast<const stbi_uc *>(bytes), length,
                                              &width, &height, &channels, 4);
    if (!decoded) { err = std::string("image decode failed: ") + stbi_failure_reason(); return false; }
    original_rgba.assign(decoded, decoded + (size_t) width * height * 4);
    input_nchw.resize((size_t) 3 * size * size);
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            for (int c = 0; c < 3; ++c) {
                const float interpolated = resize_bilinear(
                    [&](int px, int py) { return decoded[((size_t) py * width + px) * 4 + c] / 255.f; },
                    width, height, size, size, x, y);
                const float value = std::lround(std::clamp(interpolated, 0.f, 1.f) * 255.f) / 255.f;
                input_nchw[((size_t) c * size + y) * size + x] = (value - mean[c]) / std[c];
            }
        }
    }
    stbi_image_free(decoded);
    return true;
}

bool encode_result_png(const std::vector<uint8_t> & original_rgba, int width, int height,
                       const std::vector<float> & alpha, int alpha_width, int alpha_height,
                       std::vector<uint8_t> & png, std::string & err) {
    if (width <= 0 || height <= 0 || alpha_width <= 0 || alpha_height <= 0 ||
        original_rgba.size() != (size_t) width * height * 4 ||
        alpha.size() != (size_t) alpha_width * alpha_height) {
        err = "invalid result image shape";
        return false;
    }
    std::vector<uint8_t> rgba = original_rgba;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const float value = resize_bicubic(
                [&](int px, int py) { return alpha[(size_t) py * alpha_width + px]; },
                alpha_width, alpha_height, width, height, x, y);
            rgba[((size_t) y * width + x) * 4 + 3] =
                (uint8_t) std::lround(std::clamp(value, 0.f, 1.f) * 255.f);
        }
    }
    png.clear();
    if (!stbi_write_png_to_func(png_write, &png, width, height, 4, rgba.data(), width * 4)) {
        err = "PNG encode failed";
        return false;
    }
    return true;
}

} // namespace rmbg
