#pragma once

#include <string>
#include <vector>
#include "ggml-backend.h"

namespace rmbg {

struct Config {
    int input_size = 1024;
    float mean[3] = {0.485f, 0.456f, 0.406f};
    float std[3]  = {0.229f, 0.224f, 0.225f};
    std::string backbone = "swin_v1_l";
};

class RmbgDeviceGraph;

struct Model {
    Model() = default;
    Model(const Model &) = delete;
    Model & operator=(const Model &) = delete;

    Config cfg;
    ggml_backend_t backend = nullptr;
    RmbgDeviceGraph * graph = nullptr;
    std::string backend_name;
    bool graph_ready = false;
};

bool load_gguf(const char * path, const char * device, Model & out, std::string & err);
inline bool load_gguf(const char * path, Model & out, std::string & err) {
    return load_gguf(path, "auto", out, err);
}
void free_model(Model & m);
bool remove_background(Model & m,
                       const void * image_bytes, int image_len,
                       std::vector<uint8_t> & out_png,
                       std::string & err);

} // namespace rmbg
