#include <Arduino.h>
#include <PDM.h>

#define I2S_SCK_PIN    3
#define I2S_LRCK_PIN   29
#define I2S_SDOUT_PIN  28
#define SAMPLE_COUNT   256

short pdm_buf[SAMPLE_COUNT];
volatile int samplesRead = 0;

__attribute__((aligned(4))) static int32_t i2s_tx_buf[2][SAMPLE_COUNT];
static volatile int active_buf = 0;

void onPDMdata() {
    int bytes = PDM.available();
    if (bytes > 0) {
        PDM.read(pdm_buf, bytes);
        samplesRead = bytes / 2;
    }
}

void setup() {
    // I2S 설정 (기존과 동일)
    NRF_I2S->ENABLE = 0;
    NRF_I2S->PSEL.SCK   = (0 << 5) | I2S_SCK_PIN;
    NRF_I2S->PSEL.LRCK  = (0 << 5) | I2S_LRCK_PIN;
    NRF_I2S->PSEL.SDOUT = (0 << 5) | I2S_SDOUT_PIN;
    NRF_I2S->CONFIG.MODE   = 0;
    NRF_I2S->CONFIG.FORMAT = 0;
    NRF_I2S->CONFIG.SWIDTH = 1;
    NRF_I2S->CONFIG.ALIGN  = 0;
    NRF_I2S->CONFIG.MCKFREQ = I2S_CONFIG_MCKFREQ_MCKFREQ_32MDIV63;
    NRF_I2S->CONFIG.RATIO   = I2S_CONFIG_RATIO_RATIO_32X;
    NRF_I2S->ENABLE = 1;
    NRF_I2S->TXD.PTR      = (uint32_t)i2s_tx_buf[0];
    NRF_I2S->RXTXD.MAXCNT = SAMPLE_COUNT;
    NRF_I2S->TASKS_START  = 1;

    // PDM은 라이브러리로 간단하게
    PDM.onReceive(onPDMdata);
    PDM.begin(1, 16000);
}

void loop() {
    if (samplesRead > 0) {
        int next_buf = 1 - active_buf;
        int count = min(samplesRead, SAMPLE_COUNT);

        for (int i = 0; i < count; i++) {
            int16_t mic = pdm_buf[i] << 4; // 16배 증폭
            i2s_tx_buf[next_buf][i] = ((uint32_t)(uint16_t)mic << 16) | (uint16_t)mic;
        }

        NRF_I2S->TXD.PTR = (uint32_t)i2s_tx_buf[next_buf];
        active_buf = next_buf;
        samplesRead = 0;
    }
}