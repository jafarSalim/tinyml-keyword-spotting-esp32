// Runs the exact firmware C++ on this computer against the held-out voice.
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <vector>
#include "../firmware/esp32_kws/tiny_kws.h"

int main() {
  FILE* a = fopen("tools/test_audio.bin", "rb");
  FILE* l = fopen("tools/test_labels.bin", "rb");
  if (!a || !l) { puts("run train.py first"); return 1; }
  std::vector<int32_t> labels;
  int32_t v;
  while (fread(&v, 4, 1, l) == 1) labels.push_back(v);
  static float audio[KWS_CLIP];
  static int8_t feats[KWS_FRAMES * KWS_MELS];
  kws_init();
  int ok = 0;
  double tf = 0, ti = 0;
  for (size_t i = 0; i < labels.size(); i++) {
    if (fread(audio, 4, KWS_CLIP, a) != KWS_CLIP) return 1;
    auto t0 = std::chrono::steady_clock::now();
    kws_features(audio, feats);
    auto t1 = std::chrono::steady_clock::now();
    int p = kws_predict(feats);
    auto t2 = std::chrono::steady_clock::now();
    tf += std::chrono::duration<double, std::milli>(t1 - t0).count();
    ti += std::chrono::duration<double, std::milli>(t2 - t1).count();
    ok += p == labels[i];
  }
  printf("{\"clips\": %zu, \"cpp_int8_acc\": %.4f, \"features_ms\": %.3f, \"inference_ms\": %.3f, \"ram_bytes\": %u}\n",
         labels.size(), (double)ok / labels.size(), tf / labels.size(), ti / labels.size(), KWS_RAM_BYTES);
}
