#include "rmbg_capi.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static void usage(const char * prog) {
    std::fprintf(stderr,
        "usage: %s remove --model MODEL.gguf --input IN [--output OUT.png] [--device auto|cpu|cuda|vulkan]\n"
        "       %s info --model MODEL.gguf\n", prog, prog);
}

int main(int argc, char ** argv) {
    if (argc < 2) { usage(argv[0]); return 2; }
    const char * cmd = argv[1];
    if (std::strcmp(cmd, "info") == 0) {
        const char * model = nullptr;
        for (int i = 2; i < argc; ++i) {
            if (std::strcmp(argv[i], "--model") == 0 && i + 1 < argc) model = argv[++i];
        }
        if (!model) { usage(argv[0]); return 2; }
        char err[256] = {};
        rmbg_model * m = rmbg_load(model, "auto", err, sizeof err);
        if (!m) { std::fprintf(stderr, "load failed: %s\n", err); return 1; }
        std::printf("rmbg abi=%d model=%s (end-to-end graph ready)\n",
                    rmbg_abi_version(), model);
        rmbg_free(m);
        return 0;
    }
    if (std::strcmp(cmd, "remove") != 0) { usage(argv[0]); return 2; }

    const char * model = nullptr, * input = nullptr, * output = nullptr, * device = "auto";
    for (int i = 2; i < argc; ++i) {
        if (std::strcmp(argv[i], "--model") == 0 && i + 1 < argc) model = argv[++i];
        else if (std::strcmp(argv[i], "--input") == 0 && i + 1 < argc) input = argv[++i];
        else if (std::strcmp(argv[i], "--output") == 0 && i + 1 < argc) output = argv[++i];
        else if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc) device = argv[++i];
    }
    if (!model || !input) { usage(argv[0]); return 2; }

    FILE * f = std::fopen(input, "rb");
    if (!f) { std::perror(input); return 1; }
    std::fseek(f, 0, SEEK_END);
    long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> bytes((size_t) n);
    if (std::fread(bytes.data(), 1, bytes.size(), f) != bytes.size()) {
        std::fclose(f);
        std::fprintf(stderr, "read failed\n");
        return 1;
    }
    std::fclose(f);

    char err[512] = {};
    rmbg_model * m = rmbg_load(model, device, err, sizeof err);
    if (!m) { std::fprintf(stderr, "load: %s\n", err); return 1; }

    uint8_t * png = nullptr;
    int plen = 0;
    if (rmbg_remove_background(m, bytes.data(), (int) bytes.size(), &png, &plen, err, sizeof err)) {
        std::fprintf(stderr, "remove failed: %s\n", err);
        rmbg_free(m);
        return 1;
    }
    rmbg_free(m);
    if (output && png) {
        FILE * out = std::fopen(output, "wb");
        if (!out || std::fwrite(png, 1, (size_t) plen, out) != (size_t) plen) {
            std::fprintf(stderr, "write %s failed\n", output);
            rmbg_free_buffer(png);
            return 1;
        }
        std::fclose(out);
        std::fprintf(stderr, "wrote %s (%d bytes)\n", output, plen);
    }
    rmbg_free_buffer(png);
    return 0;
}
