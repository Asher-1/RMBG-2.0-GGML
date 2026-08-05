#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace rmbg {

bool decode_preprocess(const void * bytes, int length, int size,
                       const float mean[3], const float std[3],
                       std::vector<uint8_t> & original_rgba, int & width, int & height,
                       std::vector<float> & input_nchw, std::string & err);

bool encode_result_png(const std::vector<uint8_t> & original_rgba, int width, int height,
                       const std::vector<float> & alpha, int alpha_width, int alpha_height,
                       std::vector<uint8_t> & png, std::string & err);

} // namespace rmbg
