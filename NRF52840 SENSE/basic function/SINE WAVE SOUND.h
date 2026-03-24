#include <Arduino.h>

// 실제 GPIO 핀 번호 (XIAO nRF52840 핀맵 기준)
#define I2S_SCK_PIN    3   // D1 = P0.03
#define I2S_LRCK_PIN   29  // D3 = P0.29
#define I2S_SDOUT_PIN  28  // D2 = P0.28

#define SAMPLE_COUNT 128

__attribute__((aligned(4))) static int32_t i2s_buf[2][SAMPLE_COUNT];
static volatile int active_buf = 0;

float phase = 0;
float phaseInc = 2.0 * PI * 440.0 / 16000.0;

void fill_buffer(int index) {
    for (int i = 0; i < SAMPLE_COUNT; i++) {
        int16_t val = (int16_t)(sin(phase) * 3000.0);
        i2s_buf[index][i] = ((uint32_t)(uint16_t)val << 16) | (uint16_t)val;
        phase += phaseInc;
        if (phase >= 2.0 * PI) phase -= 2.0 * PI;
    }
}

void setup() {
    pinMode(LED_RED, OUTPUT);

    fill_buffer(0);
    fill_buffer(1);

    NRF_I2S->ENABLE = 0;

    NRF_I2S->PSEL.SCK   = (0 << 5) | I2S_SCK_PIN;
    NRF_I2S->PSEL.LRCK  = (0 << 5) | I2S_LRCK_PIN;
    NRF_I2S->PSEL.SDOUT = (0 << 5) | I2S_SDOUT_PIN;

    NRF_I2S->CONFIG.MODE   = 0; // Master
    NRF_I2S->CONFIG.FORMAT = 0; // I2S Standard
    NRF_I2S->CONFIG.SWIDTH = 1; // 16bit
    NRF_I2S->CONFIG.ALIGN  = 0; // Left (FORMAT=I2S일 때 무시됨)

    // 32MHz / 63 / 32 ≈ 15.873kHz
    NRF_I2S->CONFIG.MCKFREQ = I2S_CONFIG_MCKFREQ_MCKFREQ_32MDIV63;
    NRF_I2S->CONFIG.RATIO   = I2S_CONFIG_RATIO_RATIO_32X;

    NRF_I2S->ENABLE = 1;
    NRF_I2S->TXD.PTR      = (uint32_t)i2s_buf[0];
    NRF_I2S->RXTXD.MAXCNT = SAMPLE_COUNT;
    NRF_I2S->TASKS_START  = 1;

    active_buf = 0;
}

void loop() {
    if (NRF_I2S->EVENTS_TXPTRUPD) {
        NRF_I2S->EVENTS_TXPTRUPD = 0;

        // 1. 다음 버퍼 채우기
        int next_buf = 1 - active_buf;
        fill_buffer(next_buf);

        // 2. 다 채운 후 포인터 등록
        NRF_I2S->TXD.PTR = (uint32_t)i2s_buf[next_buf];

        // 3. 버퍼 전환
        active_buf = next_buf;

        static int blink = 0;
        if (blink++ > 100) {
            digitalWrite(LED_RED, !digitalRead(LED_RED));
            blink = 0;
        }
    }
}