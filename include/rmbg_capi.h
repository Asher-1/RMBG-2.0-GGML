#pragma once

#include <stdint.h>

#ifdef RMBG_SHARED
#  if defined(_WIN32) && !defined(__MINGW32__)
#    ifdef RMBG_BUILD
#      define RMBG_API __declspec(dllexport)
#    else
#      define RMBG_API __declspec(dllimport)
#    endif
#  else
#    define RMBG_API __attribute__((visibility("default")))
#  endif
#else
#  define RMBG_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define RMBG_CAPI_ABI_VERSION 1

RMBG_API int rmbg_abi_version(void);

typedef struct rmbg_model rmbg_model;

/* Load BiRefNet/RMBG-2.0 weights from GGUF. device: "auto", "cpu", "cuda", "vulkan". */
RMBG_API rmbg_model * rmbg_load(const char * gguf_path, const char * device,
                                char * err, int err_len);
RMBG_API void rmbg_free(rmbg_model * m);

/* Remove background: RGB/RGBA image bytes -> RGBA PNG bytes (caller frees with rmbg_free_buffer).
** Returns 0 on success. */
RMBG_API int rmbg_remove_background(rmbg_model * m,
                                    const void * image_bytes, int image_len,
                                    uint8_t ** out_png, int * out_len,
                                    char * err, int err_len);

RMBG_API void rmbg_free_buffer(uint8_t * buf);

#ifdef __cplusplus
}
#endif
