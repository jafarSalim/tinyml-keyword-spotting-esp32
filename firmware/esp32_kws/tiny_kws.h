// Log-mel features and integer inference. No dynamic allocation.
#pragma once
#include <math.h>
#include <stdint.h>
#include <string.h>
#include "model_data.h"

#define KWS_CLIP 6400
#define KWS_WIN 256
#define KWS_HOP 128
#define KWS_MELS 20
#define KWS_FRAMES (1 + (KWS_CLIP - KWS_WIN) / KWS_HOP)

static inline float hz2mel(float f) { return 2595.0f * log10f(1.0f + f / 700.0f); }
static inline float mel2hz(float m) { return 700.0f * (powf(10.0f, m / 2595.0f) - 1.0f); }

static void kws_fft(float* re, float* im, int n) {
  for (int i = 1, j = 0; i < n; i++) {
    int bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { float t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; }
  }
  for (int len = 2; len <= n; len <<= 1) {
    double ang = -2 * M_PI / len;
    for (int i = 0; i < n; i += len)
      for (int k = 0; k < len / 2; k++) {
        float wr = (float)cos(ang * k), wi = (float)sin(ang * k);
        float ur = re[i + k], ui = im[i + k];
        float vr = re[i + k + len / 2] * wr - im[i + k + len / 2] * wi;
        float vi = re[i + k + len / 2] * wi + im[i + k + len / 2] * wr;
        re[i + k] = ur + vr; im[i + k] = ui + vi;
        re[i + k + len / 2] = ur - vr; im[i + k + len / 2] = ui - vi;
      }
  }
}

static int16_t kws_bins[KWS_MELS + 2];
static float kws_hann[KWS_WIN];

static void kws_init() {
  float lo = hz2mel(100), hi = hz2mel(4000);
  for (int i = 0; i < KWS_MELS + 2; i++)
    kws_bins[i] = (int16_t)floor((KWS_WIN + 1) * mel2hz(lo + (hi - lo) * i / (KWS_MELS + 1)) / 8000.0);
  for (int i = 0; i < KWS_WIN; i++) kws_hann[i] = 0.5f - 0.5f * cos(2 * M_PI * i / KWS_WIN);
}

// audio: KWS_CLIP samples in [-1, 1]. out: int8 features, frames x mels.
static void kws_features(const float* audio, int8_t* out) {
  float peak = 1e-3f;
  for (int i = 0; i < KWS_CLIP; i++) if (fabsf(audio[i]) > peak) peak = fabsf(audio[i]);
  static float re[KWS_WIN], im[KWS_WIN], pow_[KWS_WIN / 2 + 1];
  for (int f = 0; f < KWS_FRAMES; f++) {
    for (int i = 0; i < KWS_WIN; i++) { re[i] = audio[f * KWS_HOP + i] / peak * kws_hann[i]; im[i] = 0; }
    kws_fft(re, im, KWS_WIN);
    for (int i = 0; i <= KWS_WIN / 2; i++) pow_[i] = re[i] * re[i] + im[i] * im[i];
    for (int m = 0; m < KWS_MELS; m++) {
      int a = kws_bins[m], b = kws_bins[m + 1], c = kws_bins[m + 2];
      float e = 0;
      for (int k = a; k < b; k++) e += pow_[k] * (float)(k - a) / (b > a ? b - a : 1);
      for (int k = b; k < c; k++) e += pow_[k] * (float)(c - k) / (c > b ? c - b : 1);
      float v = (logf(e + 1e-3f) - KWS_MEAN) / KWS_STD / KWS_IN_SCALE;
      v = roundf(v);
      out[f * KWS_MELS + m] = (int8_t)(v > 127 ? 127 : v < -127 ? -127 : v);
    }
  }
}

template <typename T>
static void kws_conv(const T* in, int ch, int h, int w, const int8_t* W, const int32_t* B, int32_t M, int fo, int16_t* out) {
  const int oh = (h + 1) / 2, ow = (w + 1) / 2;
  for (int f = 0; f < fo; f++)
    for (int y = 0; y < oh; y++)
      for (int x = 0; x < ow; x++) {
        int64_t acc = B[f];
        for (int c = 0; c < ch; c++)
          for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++) {
              int yy = y * 2 + i - 1, xx = x * 2 + j - 1;
              if (yy < 0 || yy >= h || xx < 0 || xx >= w) continue;
              acc += (int32_t)W[((f * ch + c) * 3 + i) * 3 + j] * (int32_t)in[(c * h + yy) * w + xx];
            }
        int64_t v = (acc * M + (1 << 14)) >> 15;
        out[(f * oh + y) * ow + x] = (int16_t)(v < 0 ? 0 : v > 127 ? 127 : v);
      }
}

// Returns the predicted digit. Buffers are static, so RAM use is fixed.
static int kws_predict(const int8_t* feat) {
  const int h0 = KWS_FRAMES, w0 = KWS_MELS;
  const int h1 = (h0 + 1) / 2, w1 = (w0 + 1) / 2, h2 = (h1 + 1) / 2, w2 = (w1 + 1) / 2, h3 = (h2 + 1) / 2, w3 = (w2 + 1) / 2;
  static int16_t a1[12 * ((h0 + 1) / 2) * ((w0 + 1) / 2)];
  static int16_t a2[16 * (((h0 + 1) / 2 + 1) / 2) * (((w0 + 1) / 2 + 1) / 2)];
  static int16_t a3[24 * ((((h0 + 1) / 2 + 1) / 2 + 1) / 2) * (((((w0 + 1) / 2 + 1) / 2) + 1) / 2)];
  kws_conv<int8_t>(feat, 1, h0, w0, W0, B0, M0, 12, a1);
  kws_conv<int16_t>(a1, 12, h1, w1, W1, B1, M1, 16, a2);
  kws_conv<int16_t>(a2, 16, h2, w2, W2, B2, M2, 24, a3);
  int32_t pool[24];
  const int n = h3 * w3;
  for (int c = 0; c < 24; c++) {
    int32_t s = 0;
    for (int i = 0; i < n; i++) s += a3[c * n + i];
    pool[c] = (s + n / 2) / n;
  }
  int best = 0;
  int64_t top = INT64_MIN;
  for (int k = 0; k < 10; k++) {
    int64_t s = BF[k];
    for (int c = 0; c < 24; c++) s += (int32_t)WF[k * 24 + c] * pool[c];
    if (s > top) { top = s; best = k; }
  }
  return best;
}

static const uint32_t KWS_RAM_BYTES =
    sizeof(int16_t) * (12 * 25 * 10 + 16 * 13 * 5 + 24 * 7 * 3) + KWS_FRAMES * KWS_MELS;
