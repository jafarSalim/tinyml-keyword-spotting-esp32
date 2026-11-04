// Spoken digit recognition on an ESP32 with an INMP441 I2S microphone.
// This sketch has not been compiled or tested on a board. See the README.
#include <driver/i2s.h>
#include "tiny_kws.h"

static float audio[KWS_CLIP];
static int8_t feats[KWS_FRAMES * KWS_MELS];

void setup() {
  Serial.begin(115200);
  kws_init();
  i2s_config_t cfg = {};
  cfg.mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX);
  cfg.sample_rate = 8000;
  cfg.bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT;
  cfg.channel_format = I2S_CHANNEL_FMT_ONLY_LEFT;
  cfg.communication_format = I2S_COMM_FORMAT_STAND_I2S;
  cfg.dma_buf_count = 8;
  cfg.dma_buf_len = 256;
  i2s_driver_install(I2S_NUM_0, &cfg, 0, NULL);
  i2s_pin_config_t pins = {};
  pins.bck_io_num = 26;
  pins.ws_io_num = 25;
  pins.data_in_num = 33;
  pins.data_out_num = I2S_PIN_NO_CHANGE;
  i2s_set_pin(I2S_NUM_0, &pins);
}

void loop() {
  int32_t s;
  size_t got;
  for (int i = 0; i < KWS_CLIP; i++) {
    i2s_read(I2S_NUM_0, &s, 4, &got, portMAX_DELAY);
    audio[i] = (s >> 14) / 131072.0f;
  }
  kws_features(audio, feats);
  Serial.printf("digit: %d\n", kws_predict(feats));
}
