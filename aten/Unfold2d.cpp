#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/native/Unfold2d.h>
#include <ATen/native/cpu/Loops.h>
#include <c10/util/irange.h>
#include <ATen/native/cpu/utils.h>
#include <cmath>

namespace at::native {

namespace {

extern "C" void im2col_pack_rvv(const float* input_data, float* finput_data,  
		                int64_t kH, int64_t kW, int64_t dH, int64_t dW, int64_t padH, int64_t padW,
                                int64_t n_input_plane, int64_t input_height, int64_t input_width, 
				int64_t output_height, int64_t output_width);


template <typename scalar_t>
inline void cadd(
    scalar_t* z,
    const scalar_t* x,
    const scalar_t* y,
    int64_t n) {
  using Vec = vec::Vectorized<scalar_t>;
  // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
  char* ptrs[] = {reinterpret_cast<char*>(z),
                  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
                  reinterpret_cast<char*>(const_cast<scalar_t*>(x)),
                  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
                  reinterpret_cast<char*>(const_cast<scalar_t*>(y))};
  vectorized_loop(
      ptrs,
      n,
      -1,
      [](scalar_t x, scalar_t y) -> scalar_t { return x + y; },
      [](Vec x, Vec y) -> Vec { return x + y; });
}

template <typename scalar_t>
void unfolded2d_acc(
    scalar_t* finput_data,
    scalar_t* input_data,
    int64_t kH,
    int64_t kW,
    int64_t dH,
    int64_t dW,
    int64_t padH,
    int64_t padW,
    int64_t n_input_plane,
    int64_t input_height,
    int64_t input_width,
    int64_t output_height,
    int64_t output_width) {
    printf("unfolded2d_acc\n");
  at::parallel_for(0, n_input_plane, 0, [&](int64_t start, int64_t end) {
    for (const auto nip : c10::irange(start, end)) {
      for (int64_t kh = 0; kh < kH; kh++) {
        for (int64_t kw = 0; kw < kW; kw++) {
          scalar_t* src = finput_data +
              nip * ((size_t)kH * kW * output_height * output_width) +
              kh * ((size_t)kW * output_height * output_width) +
              kw * ((size_t)output_height * output_width);
          scalar_t* dst =
              input_data + nip * ((size_t)input_height * input_width);
          if (padW > 0 || padH > 0) {
            for (int64_t y = 0; y < output_height; y++) {
              auto iy = y * dH - padH + kh;
              if (iy < 0 || iy >= input_height) {
              } else {
                if (dW == 1) {
                  auto ix = 0 - padW + kw;
                  auto lpad = std::max<int64_t>(0, padW - kw);
                  auto rpad = std::max<int64_t>(0, padW - (kW - kw - 1));
                  scalar_t* dst_slice =
                      dst + (size_t)iy * input_width + ix + lpad;
                  cadd(
                      dst_slice,
                      dst_slice,
                      src + (size_t)y * output_width + lpad,
                      output_width - lpad - rpad);
                } else {
                  for (int64_t x = 0; x < output_width; x++) {
                    auto ix = x * dW - padW + kw;
                    if (ix < 0 || ix >= input_width) {
                    } else {
                      scalar_t* dst_slice = dst + (size_t)iy * input_width + ix;
                      *dst_slice = *dst_slice + src[(size_t)y * output_width + x];
                    }
                  }
                }
              }
            }
          } else {
            for (int64_t y = 0; y < output_height; y++) {
              auto iy = y * dH + kh;
              auto ix = 0 + kw;
              if (dW == 1) {
                scalar_t* dst_slice = dst + (size_t)iy * input_width + ix;
                cadd(
                    dst_slice,
                    dst_slice,
                    src + (size_t)y * output_width,
                    output_width);
              } else {
                for (int64_t x = 0; x < output_width; x++) {
                  scalar_t* dst_slice =
                      dst + (size_t)iy * input_width + ix + x * dW;
                  *dst_slice = *dst_slice + src[(size_t)y * output_width + x];
                }
              }
            }
          }
        }
      }
    }
  });
}

template <typename scalar_t>
void unfolded2d_acc_channels_last(
    scalar_t* finput_data,
    scalar_t* input_data,
    int64_t kH,
    int64_t kW,
    int64_t dH,
    int64_t dW,
    int64_t padH,
    int64_t padW,
    int64_t n_input_plane,
    int64_t input_height,
    int64_t input_width,
    int64_t output_height,
    int64_t output_width) {
    printf("unfolded2d_acc channels last\n");

  for (int64_t y = 0; y < output_height; y++) {
    for (int64_t x = 0; x < output_width; x++) {
      scalar_t* src = finput_data + y * output_width * kH * kW * n_input_plane + x * kH * kW * n_input_plane;
      scalar_t* dst = input_data;

      if (padW > 0 || padH > 0) {
        for (int64_t kh = 0; kh < kH; kh++) {
          for (int64_t kw = 0; kw < kW; kw++) {
            int64_t iy = y * dH - padH + kh;
            int64_t ix = x * dW - padW + kw;
            if (iy < 0 || iy >= input_height || ix < 0 || ix >= input_width) {
            } else {
              scalar_t* dst_slice = dst + iy * input_width * n_input_plane + ix * n_input_plane;
              scalar_t* src_slice = src + kh * kW * n_input_plane + kw * n_input_plane;
              cadd(dst_slice,
                   dst_slice,
                   src_slice,
                   n_input_plane);
            }
          }
        }
      } else {
        for (int64_t kh = 0; kh < kH; kh++) {
          for (int64_t kw = 0; kw < kW; kw++) {
            int64_t iy = y * dH + kh;
            int64_t ix = x * dW + kw;
            scalar_t* dst_slice = dst + iy * input_width * n_input_plane + ix * n_input_plane;
            scalar_t* src_slice = src + kh * kW * n_input_plane + kw * n_input_plane;
            cadd(dst_slice,
                 dst_slice,
                 src_slice,
                 n_input_plane);
          }
        }
      }
    }
  }
}

/* note: due to write issues, this one cannot be parallelized as well as
 * unfolded2d_copy */
#if defined(__GNUC__) && __GNUC__ == 14 && defined(__ARM_FEATURE_SVE)
// Workaround for gcc-14.2.0 ICE during RTL pass: vregs when compiling for SVE
// NS: With or without BF16, see https://github.com/pytorch/pytorch/issues/172630
__attribute__((optimize("no-tree-vectorize")))
#endif
void unfolded2d_acc_kernel(
    ScalarType dtype,
    void *finput_data,
    void *input_data,
    int64_t kH,
    int64_t kW,
    int64_t dH,
    int64_t dW,
    int64_t padH,
    int64_t padW,
    int64_t n_input_plane,
    int64_t input_height,
    int64_t input_width,
    int64_t output_height,
    int64_t output_width,
    bool is_channels_last) {
  // This function assumes that
  // output_height*dH does not overflow a int64_t
  // output_width*dW does not overflow a int64_t

  if (is_channels_last) {
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::BFloat16, at::ScalarType::Half, dtype, "unfolded2d_acc_channels_last", [&] {
      unfolded2d_acc_channels_last(
          static_cast<scalar_t*>(finput_data),
          static_cast<scalar_t*>(input_data),
          kH, kW,
          dH, dW,
          padH, padW,
          n_input_plane,
          input_height,
          input_width,
          output_height,
          output_width);
     });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::BFloat16, at::ScalarType::Half, dtype, "unfolded2d_acc", [&] {
      unfolded2d_acc(
          static_cast<scalar_t*>(finput_data),
          static_cast<scalar_t*>(input_data),
          kH, kW,
          dH, dW,
          padH, padW,
          n_input_plane,
          input_height,
          input_width,
          output_height,
          output_width);
      });
  }
}

//----------------------------------------------------------------------
//----------------------------------------------------------------------

namespace {
constexpr int64_t MR = 16;   // 16
constexpr int64_t KC = 512;   // 512 (debe coincidir con el GEMM)

inline int64_t ceil_div(int64_t a, int64_t b) { return (a + b - 1) / b; }

// Escribe len ceros en las posiciones virtuales p..p+len-1 de la fila k
template <typename T>
inline void pk_zero(T* base, int64_t ps, int64_t p, int64_t len) {
  while (len > 0) {
    int64_t po = p % MR, n = std::min<int64_t>(len, MR - po);
    memset(base + (p / MR) * ps + po, 0, sizeof(T) * n);
    p += n; len -= n;
  }
}
// Copia contigua (dW == 1)
template <typename T>
inline void pk_copy(T* base, int64_t ps, int64_t p, const T* s, int64_t len) {
  while (len > 0) {
    int64_t po = p % MR, n = std::min<int64_t>(len, MR - po);
    memcpy(base + (p / MR) * ps + po, s, sizeof(T) * n);
    p += n; s += n; len -= n;
  }
}
// Gather con stride (dW > 1)
template <typename T>
inline void pk_gather(T* base, int64_t ps, int64_t p, const T* s,
                      int64_t sstride, int64_t len) {
  for (int64_t j = 0; j < len; ++j, ++p)
    base[(p / MR) * ps + (p % MR)] = s[j * sstride];
}
} // namespace

template <typename scalar_t>
void unfolded2d_copy_pack(
    const scalar_t* input_data,
    scalar_t* apack,               // salida: M_pad * K elementos (alineado a 64B)
    int64_t kH, 
    int64_t kW, 
    int64_t dH, 
    int64_t dW,
    int64_t padH, 
    int64_t padW,
    int64_t n_input_plane, 
    int64_t input_height, 
    int64_t input_width,
    int64_t output_height, 
    int64_t output_width) {

  const int64_t M     = output_height * output_width;
  const int64_t K     = n_input_plane * kH * kW;
  const int64_t M_pad = ceil_div(M, MR) * MR;

  at::parallel_for(0, K, 0, [&](int64_t start, int64_t end) {
    for (const auto k : c10::irange(start, end)) {
      const int64_t nip  = k / (kH * kW);
      const int64_t rest = k % (kH * kW);
      const int64_t kh   = rest / kW;
      const int64_t kw   = rest % kW;

      // Destino empaquetado de esta fila k
      const int64_t pc     = (k / KC) * KC;
      const int64_t kc     = std::min<int64_t>(K - pc, KC);
      scalar_t* base       = apack + pc * M_pad + (k - pc) * MR;
      const int64_t ps     = MR * kc;               // stride entre paneles

      const scalar_t* src = input_data + nip * input_height * input_width;

      // Rango de x con ix = x*dW - padW + kw dentro de [0, input_width)
      int64_t x_lo = (padW > kw) ? ceil_div(padW - kw, dW) : 0;
      const int64_t num = input_width + padW - kw;
      const int64_t x_hi = (num > 0) ? std::min(output_width, ceil_div(num, dW)) : 0;
      x_lo = std::min(x_lo, x_hi);

      for (int64_t y = 0; y < output_height; y++) {
        const int64_t p0 = y * output_width;
        const int64_t iy = y * dH - padH + kh;

        if (iy < 0 || iy >= input_height || x_lo >= x_hi) {
          pk_zero(base, ps, p0, output_width);
          continue;
        }
        const scalar_t* row = src + iy * input_width;
        const int64_t ix0   = x_lo * dW - padW + kw;

        pk_zero(base, ps, p0, x_lo);
        if (dW == 1) pk_copy(base, ps, p0 + x_lo, row + ix0, x_hi - x_lo);
        else         pk_gather(base, ps, p0 + x_lo, row + ix0, dW, x_hi - x_lo);
        pk_zero(base, ps, p0 + x_hi, output_width - x_hi);
      }
      // Relleno del último panel (filas M..M_pad-1)
      if (M_pad > M) pk_zero(base, ps, M, M_pad - M);
    }
  });
}

//----------------------------------------------------------------------
//----------------------------------------------------------------------

template <typename scalar_t>
void unfolded2d_copy(
    const scalar_t* input_data,
    scalar_t* finput_data,
    int64_t kH,
    int64_t kW,
    int64_t dH,
    int64_t dW,
    int64_t padH,
    int64_t padW,
    int64_t n_input_plane,
    int64_t input_height,
    int64_t input_width,
    int64_t output_height,
    int64_t output_width) {

    //TODO: Optimization with packing

  if constexpr (std::is_same_v<scalar_t, float>) {
    im2col_pack_rvv(input_data, finput_data, kH, kW, dH, dW, padH, padW,
                    n_input_plane, input_height, input_width, output_height, output_width);
    return;
  }

  at::parallel_for(
      0, n_input_plane * kH * kW, 0, [&](int64_t start, int64_t end) {
        for (const auto k : c10::irange(start, end)) {
          int64_t nip = k / (kH * kW);
          int64_t rest = k % (kH * kW);
          int64_t kh = rest / kW;
          int64_t kw = rest % kW;
          scalar_t* dst = finput_data +
              nip * ((size_t)kH * kW * output_height * output_width) +
              kh * ((size_t)kW * output_height * output_width) +
              kw * ((size_t)output_height * output_width);
          const scalar_t* src =
              input_data + nip * ((size_t)input_height * input_width);
          if (padW > 0 || padH > 0) {
            for (int64_t y = 0; y < output_height; y++) {
              auto iy = y * dH - padH + kh;
              if (iy < 0 || iy >= input_height) {
                memset(
                    dst + (size_t)y * output_width,
                    0,
                    sizeof(scalar_t) * output_width);
              } else {
                if (dW == 1) {
                  auto ix = 0 - padW + kw;
                  auto lpad = std::max<int64_t>(0, padW - kw);
                  auto rpad = std::max<int64_t>(0, padW - (kW - kw - 1));
                  if (output_width - rpad - lpad <= 0) {
                    memset(
                        dst + (size_t)y * output_width,
                        0,
                        sizeof(scalar_t) * output_width);
                  } else {
                    if (lpad > 0)
                      memset(
                          dst + (size_t)y * output_width,
                          0,
                          sizeof(scalar_t) * lpad);
                    memcpy(
                        dst + (size_t)y * output_width + lpad,
                        src + (size_t)iy * input_width + ix + lpad,
                        sizeof(scalar_t) * (output_width - rpad - lpad));
                    if (rpad > 0)
                      memset(
                          dst + (size_t)y * output_width + output_width - rpad,
                          0,
                          sizeof(scalar_t) * rpad);
                  }
                } else {
                  for (int64_t x = 0; x < output_width; x++) {
                    auto ix = x * dW - padW + kw;
                    if (ix < 0 || ix >= input_width)
                      memset(
                          dst + (size_t)y * output_width + x,
                          0,
                          sizeof(scalar_t) * 1);
                    else
                      memcpy(
                          dst + (size_t)y * output_width + x,
                          src + (size_t)iy * input_width + ix,
                          sizeof(scalar_t) * 1);
                  }
                }
              }
            }
          } else {
            for (int64_t y = 0; y < output_height; y++) {
              auto iy = y * dH + kh;
              auto ix = 0 + kw;
              if (dW == 1)
                memcpy(
                    dst + (size_t)y * output_width,
                    src + (size_t)iy * input_width + ix,
                    sizeof(scalar_t) * output_width);
              else {
                for (int64_t x = 0; x < output_width; x++)
                  memcpy(
                      dst + (size_t)y * output_width + x,
                      src + (size_t)iy * input_width + ix + x * dW,
                      sizeof(scalar_t) * 1);
              }
            }
          }
        }
      });
}

template <typename scalar_t>
void unfolded2d_copy_channels_last(
    const scalar_t* input_data,
    scalar_t* finput_data,
    int64_t kH,
    int64_t kW,
    int64_t dH,
    int64_t dW,
    int64_t padH,
    int64_t padW,
    int64_t n_input_plane,
    int64_t input_height,
    int64_t input_width,
    int64_t output_height,
    int64_t output_width) {
    printf("unfolded2d_copy_channels_last\n");
  at::parallel_for(0, output_height * output_width, 0, [&](int64_t start, int64_t end) {
    int64_t y = 0;
    int64_t x = 0;
    data_index_init(start, y, output_height, x, output_width);

    for (const auto k [[maybe_unused]] : c10::irange(start, end)) {
      scalar_t* dst = finput_data + y * output_width * kH * kW * n_input_plane +
          x * kH * kW * n_input_plane;
      const scalar_t* src = input_data;

      if (padW > 0 || padH > 0) {
        for (int64_t kh = 0; kh < kH; kh++) {
          for (int64_t kw = 0; kw < kW; kw++) {
            int64_t iy = y * dH - padH + kh;
            int64_t ix = x * dW - padW + kw;
            if (iy < 0 || iy >= input_height || ix < 0 || ix >= input_width) {
              memset(dst + kh * kW * n_input_plane + kw * n_input_plane,
                    0,
                    sizeof(scalar_t) * n_input_plane);
            } else {
              memcpy(dst + kh * kW * n_input_plane + kw * n_input_plane,
                     src + iy * input_width * n_input_plane + ix * n_input_plane,
                     sizeof(scalar_t) * n_input_plane);
            }
          }
        }
      } else {
        for (int64_t kh = 0; kh < kH; kh++) {
          for (int64_t kw = 0; kw < kW; kw++) {
            int64_t iy = y * dH + kh;
            int64_t ix = x * dW + kw;
            memcpy(dst + kh * kW * n_input_plane + kw * n_input_plane,
                   src + iy * input_width * n_input_plane + ix * n_input_plane,
                   sizeof(scalar_t) * n_input_plane);
          }
        }
      }
      // move on to next output index
      data_index_step(y, output_height, x, output_width);
    }
  });
}

void unfolded2d_copy_kernel(
    ScalarType dtype,
    void *finput_data,
    const void *input_data,
    int64_t kH,
    int64_t kW,
    int64_t dH,
    int64_t dW,
    int64_t padH,
    int64_t padW,
    int64_t n_input_plane,
    int64_t input_height,
    int64_t input_width,
    int64_t output_height,
    int64_t output_width,
    bool is_channels_last) {
  // This function assumes that
  // kH*kW does not overflow an int
  // n_input_plane*kH*kW does not overflow a int64_t
  // output_height*dH does not overflow a int64_t
  // output_width*dW does not overflow a int64_t

  if (is_channels_last) {
    AT_DISPATCH_ALL_TYPES_AND2(at::ScalarType::BFloat16, at::ScalarType::Half, dtype, "unfolded2d_copy_channels_last", [&] {
      unfolded2d_copy_channels_last(
          static_cast<const scalar_t*>(input_data),
          static_cast<scalar_t*>(finput_data),
            kH, kW,
            dH, dW,
            padH, padW,
            n_input_plane,
            input_height,
            input_width,
            output_height,
            output_width);
    });
  } else {
    AT_DISPATCH_ALL_TYPES_AND2(at::ScalarType::BFloat16, at::ScalarType::Half, dtype, "unfolded2d_copy", [&] {
      unfolded2d_copy(
          static_cast<const scalar_t*>(input_data),
          static_cast<scalar_t*>(finput_data),
            kH, kW,
            dH, dW,
            padH, padW,
            n_input_plane,
            input_height,
            input_width,
            output_height,
            output_width);
    });
  }
}

} // namespace

REGISTER_DISPATCH(unfolded2d_copy_stub, &unfolded2d_copy_kernel)
REGISTER_DISPATCH(unfolded2d_acc_stub, &unfolded2d_acc_kernel)

} // namespace at::native
