#include "rmbg_capi.h"
#include "rmbg.hpp"
#include <cstring>
#include <new>
#include <string>

struct rmbg_model {
    rmbg::Model m;
};

extern "C" {

int rmbg_abi_version(void) { return RMBG_CAPI_ABI_VERSION; }

rmbg_model * rmbg_load(const char * gguf_path, const char * device,
                       char * err, int err_len) {
    auto * h = new (std::nothrow) rmbg_model();
    if (!h) return nullptr;
    std::string e;
    if (!rmbg::load_gguf(gguf_path, device, h->m, e)) {
        if (err && err_len > 0) std::snprintf(err, (size_t) err_len, "%s", e.c_str());
        delete h;
        return nullptr;
    }
    return h;
}

void rmbg_free(rmbg_model * m) {
    if (!m) return;
    rmbg::free_model(m->m);
    delete m;
}

int rmbg_remove_background(rmbg_model * m,
                           const void * image_bytes, int image_len,
                           uint8_t ** out_png, int * out_len,
                           char * err, int err_len) {
    if (!m || !out_png || !out_len) {
        if (err && err_len > 0) std::snprintf(err, (size_t) err_len, "invalid argument");
        return 1;
    }
    *out_png = nullptr;
    *out_len = 0;
    std::vector<uint8_t> png;
    std::string e;
    if (!rmbg::remove_background(m->m, image_bytes, image_len, png, e)) {
        if (err && err_len > 0) std::snprintf(err, (size_t) err_len, "%s", e.c_str());
        return 1;
    }
    *out_png = (uint8_t *) std::malloc(png.size());
    if (!*out_png) {
        if (err && err_len > 0) std::snprintf(err, (size_t) err_len, "output allocation failed");
        return 1;
    }
    *out_len = (int) png.size();
    std::memcpy(*out_png, png.data(), png.size());
    return 0;
}

void rmbg_free_buffer(uint8_t * buf) { std::free(buf); }

} // extern "C"
