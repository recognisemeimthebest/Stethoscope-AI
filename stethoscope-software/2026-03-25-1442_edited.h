/**
 * @file 2026-03-25-1442_edited.h
 * @brief AI Stethoscope ESP32 Firmware (JSF++ Compliant Edition)
 * @date 2026-03-25
 *
 * Changes from original:
 *   - AV Rule 20/31: #define constants replaced with static constexpr
 *   - AV Rule 27: Magic numbers replaced with named constants
 *   - AV Rule 33: Braces added to all control structures
 *   - AV Rule 67: Button class data members made private
 *   - AV Rule 185: C-style casts replaced with static_cast
 *   - AV Rule 15: NULL replaced with nullptr
 *   - AV Rule 167: sprintf replaced with snprintf
 *   - Return values checked for critical I2S/BLE operations
 *   - Bug fix: setMinPreferred called twice (second should be setMaxPreferred)
 *   - One statement per line for readability and debugging
 *   - Header guard added
 *   - NVS persistent storage for mode, selected patient, BLE state
 *   - Respiratory Rate (RR) detection via energy envelope onset detection
 *   - drawRR() / drawBreathIndicator() for lung mode UI
 *   - Signal quality indicator (3-bar: None/Weak/Good) for heart & lung modes
 *   - BLE disconnect during recording: REC_FAIL notification + UI feedback
 *   - REC_DONE notification on successful recording completion
 *   - Watchdog Timer (esp_task_wdt) for audioTask and loop()
 */

#ifndef STETHO_FIRMWARE_2026_03_25_1442_H
#define STETHO_FIRMWARE_2026_03_25_1442_H

#include <Arduino.h>
#include <driver/i2s.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>
#include <TFT_eSPI.h>
#include <SPI.h>
#include <math.h>
#include <esp_sleep.h>
#include <esp_task_wdt.h>
#include <Preferences.h>

// ══════════════════════════════════════════════════════════
//  Hardware Pin Configuration (AV Rule 20: constexpr over #define)
// ══════════════════════════════════════════════════════════
static constexpr int PIN_BTN_L_MODE   = 35;
static constexpr int PIN_BTN_R_RECORD = 0;
static constexpr int PIN_TFT_BL       = 4;
static constexpr int PIN_I2S_WS       = 25;
static constexpr int PIN_I2S_SD       = 33;
static constexpr int PIN_I2S_SCK      = 26;
static constexpr i2s_port_t I2S_PORT  = I2S_NUM_0;

// ══════════════════════════════════════════════════════════
//  BLE UUIDs
// ══════════════════════════════════════════════════════════
static const char* const SERVICE_UUID    = "19b10000-e8f2-537e-4f6c-d104768a1214";
static const char* const AUDIO_CHAR_UUID = "19b10001-e8f2-537e-4f6c-d104768a1214";
static const char* const CMD_CHAR_UUID   = "19b10002-e8f2-537e-4f6c-d104768a1214";

// ══════════════════════════════════════════════════════════
//  Timing Constants (AV Rule 27: named constants)
// ══════════════════════════════════════════════════════════
static constexpr unsigned long RECORD_DURATION_MS    = 5000UL;
static constexpr unsigned long TFT_WAKEUP_DELAY_MS   = 120UL;
static constexpr unsigned long BPM_UPDATE_INTERVAL_MS = 500UL;
static constexpr unsigned long BEAT_FLASH_DURATION_MS = 100UL;
static constexpr unsigned long LONG_PRESS_THRESHOLD_MS = 1500UL;
static constexpr unsigned long SHORT_PRESS_MIN_MS      = 50UL;
static constexpr unsigned long DEBOUNCE_DELAY_MS       = 200UL;
static constexpr unsigned long PATIENT_SELECT_DELAY_MS = 1000UL;
static constexpr unsigned long REC_HEADER_DELAY_MS     = 100UL;
static constexpr unsigned long REC_FAIL_DISPLAY_MS     = 1500UL;

// ══════════════════════════════════════════════════════════
//  Watchdog Timer Configuration
// ══════════════════════════════════════════════════════════
static constexpr int WDT_TIMEOUT_SEC       = 10;    // 10초 내 feed 없으면 reset
static constexpr int WDT_LOOP_TIMEOUT_SEC  = 30;    // loop() 30초 내 feed 없으면 reset

// ══════════════════════════════════════════════════════════
//  Audio / DSP Constants
// ══════════════════════════════════════════════════════════
static constexpr int SAMPLE_RATE     = 16000;
static constexpr int TARGET_RATE     = 2000;
static constexpr int DECIMATION      = SAMPLE_RATE / TARGET_RATE;
static constexpr int BLE_BUFFER_SIZE = 50;

static constexpr float HEART_LPF_COEFF   = 0.24f;
static constexpr float HEART_LPF_DECAY   = 0.76f;
static constexpr float LUNG_HPF_COEFF    = 0.61f;
static constexpr float SAMPLE_NORMALIZE  = 32768.0f;
static constexpr int   SAMPLE_LEFT_SHIFT = 2;
static constexpr int16_t SAMPLE_MAX      = 32767;
static constexpr int16_t SAMPLE_MIN      = -32768;

static constexpr int HEART_DISPLAY_RANGE = 6000;
static constexpr int LUNG_DISPLAY_RANGE  = 10000;
static constexpr int HEART_SMOOTH_NUMER  = 5;
static constexpr int HEART_SMOOTH_DENOM  = 6;

// ══════════════════════════════════════════════════════════
//  DSP Filter Constants
// ══════════════════════════════════════════════════════════
static constexpr float DSP_NOISE_FLOOR     = 0.0005f;
static constexpr float DSP_THRESHOLD_RATIO = 0.35f;
static constexpr int   DSP_PEAK_REFRACTORY = 600;
static constexpr int   DSP_ENV_WINDOW      = 20;
static constexpr int   DSP_VAR_WINDOW      = 100;
static constexpr float DSP_SIGNAL_VAR_MIN  = 0.00001f;
static constexpr unsigned long DSP_BPM_TIMEOUT_MS = 4000UL;

static constexpr float DSP_HP_B0 =  0.93959f;
static constexpr float DSP_HP_B1 = -1.87919f;
static constexpr float DSP_HP_B2 =  0.93959f;
static constexpr float DSP_HP_A1 = -1.87514f;
static constexpr float DSP_HP_A2 =  0.88324f;

static constexpr float DSP_LP_B0 =  0.06746f;
static constexpr float DSP_LP_B1 =  0.13492f;
static constexpr float DSP_LP_B2 =  0.06746f;
static constexpr float DSP_LP_A1 = -1.14298f;
static constexpr float DSP_LP_A2 =  0.41282f;

static constexpr float DSP_RUNNING_MAX_SLOW_DECAY  = 0.9998f;
static constexpr float DSP_RUNNING_MAX_FAST_DECAY   = 0.999f;
static constexpr float DSP_RUNNING_MAX_TIMEOUT_DECAY = 0.5f;
static constexpr float DSP_BPM_EMA_OLD  = 0.3f;
static constexpr float DSP_BPM_EMA_NEW  = 0.7f;
static constexpr float DSP_RR_MIN_MS    = 273.0f;   // ~220 bpm upper bound
static constexpr float DSP_RR_MAX_MS    = 2000.0f;   // ~30 bpm lower bound
static constexpr float DSP_BPM_MIN_INIT = 1.0f;
static constexpr float BPM_DISPLAY_MIN  = 20.0f;
static constexpr float BPM_DISPLAY_MAX  = 250.0f;
static constexpr float BPM_CHANGE_THRESHOLD = 0.5f;

// ══════════════════════════════════════════════════════════
//  RR (Respiratory Rate) DSP Constants
//  호흡 주기: 에너지 envelope의 상승 에지 간격으로 측정
//  정상 성인 12~20회/min = 3~5초/cycle
// ══════════════════════════════════════════════════════════
static constexpr int   RR_RMS_WINDOW       = 400;    // RMS 윈도우 200ms (2kHz * 0.2)
static constexpr int   RR_SMOOTH_WINDOW    = 1000;   // 이동평균 500ms (2kHz * 0.5)
static constexpr float RR_NOISE_FLOOR      = 0.002f; // 최소 에너지 (무음 판별)
static constexpr float RR_THRESHOLD_RATIO  = 0.30f;  // running_max 대비 threshold 비율
static constexpr int   RR_REFRACTORY       = 3000;   // 최소 1.5초 간격 (2kHz * 1.5) → ~40회/min 상한
static constexpr float RR_RUNNING_MAX_DECAY = 0.9997f;
static constexpr float RR_TIMEOUT_DECAY    = 0.5f;
static constexpr unsigned long RR_TIMEOUT_MS = 10000UL; // 10초 무호흡 시 reset
static constexpr float RR_CYCLE_MIN_MS     = 1500.0f;  // ~40 breaths/min 상한
static constexpr float RR_CYCLE_MAX_MS     = 12000.0f; // ~5 breaths/min 하한
static constexpr float RR_EMA_OLD          = 0.35f;
static constexpr float RR_EMA_NEW          = 0.65f;
static constexpr float RR_DISPLAY_MIN      = 6.0f;
static constexpr float RR_DISPLAY_MAX      = 45.0f;
static constexpr float RR_CHANGE_THRESHOLD = 0.5f;

// ══════════════════════════════════════════════════════════
//  Waveform Display Layout
// ══════════════════════════════════════════════════════════
static constexpr int WAVE_Y_TOP    = 75;
static constexpr int WAVE_Y_BOTTOM = 150;
static constexpr int WAVE_HEIGHT   = WAVE_Y_BOTTOM - WAVE_Y_TOP;
static constexpr int WAVE_ZERO_Y   = WAVE_Y_TOP + (WAVE_HEIGHT / 2);
static constexpr int WAVE_X_START  = 10;
static constexpr int WAVE_X_END   = 125;
static constexpr int DRAW_INTERVAL_MS = 15;

static constexpr int BPM_Y = 165;
static constexpr int BPM_H = 40;

// Signal quality indicator position (waveform 영역 바로 위)
static constexpr int SIG_QUALITY_Y     = 65;
static constexpr int SIG_BAR_WIDTH     = 3;
static constexpr int SIG_BAR_GAP       = 2;
static const int SIG_BAR_HEIGHTS[] = {4, 7, 10};  // 3단 바 높이
static constexpr int SIG_BAR_COUNT = 3;
static constexpr unsigned long SIG_UPDATE_INTERVAL_MS = 300UL;

// ══════════════════════════════════════════════════════════
//  UI Layout Constants (AV Rule 27: no magic numbers)
// ══════════════════════════════════════════════════════════
static constexpr int STATUS_BAR_HEIGHT  = 16;
static constexpr int BEAT_INDICATOR_X   = 12;
static constexpr int BEAT_INDICATOR_Y   = 8;
static constexpr int BEAT_INDICATOR_R   = 4;
static constexpr int BT_ICON_OFFSET_X   = 15;
static constexpr int PATIENT_ROW_HEIGHT = 25;
static constexpr int PATIENT_VISIBLE_ROWS = 3;
static constexpr int PATIENT_START_Y    = 55;
static constexpr int PATIENT_COL_ID     = 10;
static constexpr int PATIENT_COL_NAME   = 55;
static constexpr int PATIENT_COL_WARD   = 100;
static constexpr int MAX_PATIENTS       = 10;

static constexpr int BPM_HEART_OFFSET_X = 40;
static constexpr int BPM_HEART_CIRCLE_R = 4;
static constexpr int BPM_TEXT_OFFSET_X  = 25;
static constexpr int BPM_UNIT_OFFSET_X  = 20;

// RR display (lung mode — same Y position as BPM)
static constexpr int RR_LUNG_ICON_OFFSET_X = 42;
static constexpr int RR_TEXT_OFFSET_X      = 25;
static constexpr int RR_UNIT_OFFSET_X      = 22;

static constexpr uint16_t COLOR_HIGHLIGHT_BG = 0x03E0;

// TFT sleep/wake commands
static constexpr uint8_t TFT_CMD_SLEEP_IN  = 0x10;
static constexpr uint8_t TFT_CMD_SLEEP_OUT = 0x11;
static constexpr int     TFT_CMD_DELAY_MS  = 5;
static constexpr uint16_t GRID_DOT_COLOR   = 0x39E7;
static constexpr int     GRID_DOT_SPACING  = 5;

// BLE advertising parameters
static constexpr uint8_t BLE_ADV_MIN_INTERVAL = 0x06;
static constexpr uint8_t BLE_ADV_MAX_INTERVAL = 0x12;
static constexpr int     BLE_MTU_SIZE         = 512;

// ══════════════════════════════════════════════════════════
//  Task Configuration
// ══════════════════════════════════════════════════════════
static constexpr int AUDIO_TASK_STACK_SIZE = 10000;
static constexpr int AUDIO_TASK_PRIORITY   = 2;
static constexpr int AUDIO_TASK_CORE       = 0;

// ══════════════════════════════════════════════════════════
//  NVS (Non-Volatile Storage) Configuration
// ══════════════════════════════════════════════════════════
static const char* const NVS_NAMESPACE    = "stetho";
static const char* const NVS_KEY_MODE     = "mode";
static const char* const NVS_KEY_PATIENT  = "patient";
static const char* const NVS_KEY_BLE_ON   = "bleOn";

// ══════════════════════════════════════════════════════════
//  Application State Machine
// ══════════════════════════════════════════════════════════
enum AppState {
    ACTIVE,
    SLEEP,
    POPUP,
    PAIRING
};

enum MeasureMode {
    MODE_HEART   = 1,
    MODE_LUNG    = 2,
    MODE_PATIENT = 3,
    MODE_COUNT   = 3
};

enum ButtonEvent {
    BTN_NONE       = 0,
    BTN_SHORT      = 1,
    BTN_LONG       = 2
};

enum SignalLevel {
    SIG_NONE = 0,
    SIG_WEAK = 1,
    SIG_GOOD = 2
};

// ══════════════════════════════════════════════════════════
//  Patient Data Structure
// ══════════════════════════════════════════════════════════
struct PatientInfo {
    String id;
    String name;
    String ward;
};

// ══════════════════════════════════════════════════════════
//  Global State
// ══════════════════════════════════════════════════════════
static TFT_eSPI tft = TFT_eSPI();
static Preferences nvs;

static BLEServer*         pServer    = nullptr;  // AV Rule 15: nullptr over NULL
static BLECharacteristic* pAudioChar = nullptr;
static BLECharacteristic* pCmdChar   = nullptr;

static volatile bool     deviceConnected  = false;
static volatile bool     isBleOn          = false;
static volatile int      currentMode      = MODE_HEART;
static volatile bool     uiNeedsUpdate    = true;
static volatile bool     isRecording5Sec  = false;
static volatile bool     recFailed        = false;   // BLE 끊김으로 녹음 실패
static volatile AppState currentState     = ACTIVE;
static volatile int16_t  sharedAveragedSample = 0;

static unsigned long recordStartTime = 0;
static unsigned long recFailDisplayTime = 0;

static PatientInfo patientList[MAX_PATIENTS];
static int  patientCount    = 0;
static int  currentPatIdx   = 0;
static String selectedPatient = "NONE";

static int16_t bleBuffer[BLE_BUFFER_SIZE];
static int     bufferIndex = 0;

static int     graphX              = WAVE_X_START;
static int16_t lastY_HEART_WAVE    = WAVE_ZERO_Y;
static int16_t lastY_LUNG_WAVE     = WAVE_ZERO_Y;
static unsigned long lastDrawTime  = 0;

static StackType_t  audioTaskStack[AUDIO_TASK_STACK_SIZE];
static StaticTask_t audioTaskBuffer;
static TaskHandle_t audioTaskHandle;

// Forward declarations
void drawUI();
static void clearRR();

// ══════════════════════════════════════════════════════════
//  Button Class (AV Rule 67: private data, public accessors)
// ══════════════════════════════════════════════════════════
class Button {
public:
    explicit Button(int p)
        : m_pin(p)
        , m_lastReading(HIGH)
        , m_pressTime(0)
        , m_longPressFired(false)
    {
    }

    void begin()
    {
        m_lastReading = digitalRead(m_pin);
    }

    ButtonEvent update()
    {
        const bool reading = digitalRead(m_pin);
        ButtonEvent event = BTN_NONE;

        if (reading == LOW && m_lastReading == HIGH) {
            // Rising edge: button just pressed
            m_pressTime = millis();
            m_longPressFired = false;
        }
        else if (reading == LOW && m_lastReading == LOW) {
            // Held down: check for long press
            if (m_pressTime > 0 && !m_longPressFired) {
                if ((millis() - m_pressTime) > LONG_PRESS_THRESHOLD_MS) {
                    m_longPressFired = true;
                    event = BTN_LONG;
                }
            }
        }
        else if (reading == HIGH && m_lastReading == LOW) {
            // Falling edge: button released
            if (m_pressTime > 0 && !m_longPressFired) {
                if ((millis() - m_pressTime) > SHORT_PRESS_MIN_MS) {
                    event = BTN_SHORT;
                }
            }
            m_pressTime = 0;
        }

        m_lastReading = reading;
        return event;
    }

    void reset()
    {
        m_lastReading = digitalRead(m_pin);
        m_pressTime = 0;
        m_longPressFired = false;
    }

private:
    int           m_pin;
    bool          m_lastReading;
    unsigned long m_pressTime;
    bool          m_longPressFired;
};

static Button btnL(PIN_BTN_L_MODE);
static Button btnR(PIN_BTN_R_RECORD);

// ══════════════════════════════════════════════════════════
//  NVS Helper Functions
// ══════════════════════════════════════════════════════════
static void nvsLoad()
{
    nvs.begin(NVS_NAMESPACE, true);  // read-only

    currentMode = nvs.getInt(NVS_KEY_MODE, MODE_HEART);
    if (currentMode < MODE_HEART || currentMode > MODE_COUNT) {
        currentMode = MODE_HEART;
    }

    selectedPatient = nvs.getString(NVS_KEY_PATIENT, "NONE");
    isBleOn = nvs.getBool(NVS_KEY_BLE_ON, false);

    nvs.end();
    Serial.println("[NVS] Settings loaded");
}

static void nvsSaveMode()
{
    nvs.begin(NVS_NAMESPACE, false);  // read-write
    nvs.putInt(NVS_KEY_MODE, currentMode);
    nvs.end();
}

static void nvsSavePatient()
{
    nvs.begin(NVS_NAMESPACE, false);
    nvs.putString(NVS_KEY_PATIENT, selectedPatient);
    nvs.end();
}

static void nvsSaveBle()
{
    nvs.begin(NVS_NAMESPACE, false);
    nvs.putBool(NVS_KEY_BLE_ON, isBleOn);
    nvs.end();
}

// ══════════════════════════════════════════════════════════
//  State Management
// ══════════════════════════════════════════════════════════
static void changeState(AppState newState)
{
    currentState = newState;
    uiNeedsUpdate = true;
    drawUI();
    btnL.reset();
    btnR.reset();
}

// ══════════════════════════════════════════════════════════
//  DSP (Heart Rate Detection)
// ══════════════════════════════════════════════════════════
static volatile float dsp_bpm         = 0.0f;
static volatile float dsp_instant_bpm = 0.0f;
static volatile bool  dsp_beat_flag   = false;
static volatile int   dsp_beat_count  = 0;
static volatile bool  dsp_signal_ok   = false;  // 심음 신호 품질 (DSP에서 갱신)

static float _hp_x1 = 0.0f;
static float _hp_x2 = 0.0f;
static float _hp_y1 = 0.0f;
static float _hp_y2 = 0.0f;
static float _lp_x1 = 0.0f;
static float _lp_x2 = 0.0f;
static float _lp_y1 = 0.0f;
static float _lp_y2 = 0.0f;

static float _env_buf[DSP_ENV_WINDOW] = {0};
static int   _env_idx = 0;
static float _env_sum = 0.0f;

static float _var_buf[DSP_VAR_WINDOW] = {0};
static int   _var_idx     = 0;
static float _var_sum     = 0.0f;
static float _var_sq_sum  = 0.0f;

static float        _running_max       = 0.0f;
static int          _samples_since_peak = 0;
static unsigned long _last_peak_ms      = 0;

static void dsp_reset()
{
    _hp_x1 = 0.0f;
    _hp_x2 = 0.0f;
    _hp_y1 = 0.0f;
    _hp_y2 = 0.0f;
    _lp_x1 = 0.0f;
    _lp_x2 = 0.0f;
    _lp_y1 = 0.0f;
    _lp_y2 = 0.0f;

    memset(_env_buf, 0, sizeof(_env_buf));
    _env_idx = 0;
    _env_sum = 0.0f;

    memset(_var_buf, 0, sizeof(_var_buf));
    _var_idx    = 0;
    _var_sum    = 0.0f;
    _var_sq_sum = 0.0f;

    _running_max       = 0.0f;
    _samples_since_peak = 0;
    _last_peak_ms      = 0;

    dsp_bpm         = 0.0f;
    dsp_instant_bpm = 0.0f;
    dsp_beat_flag   = false;
    dsp_beat_count  = 0;
}

static void dsp_process(int16_t raw16)
{
    const float s = static_cast<float>(raw16) / SAMPLE_NORMALIZE;

    // High-pass filter (2nd order IIR)
    const float hp = DSP_HP_B0 * s
                   + DSP_HP_B1 * _hp_x1
                   + DSP_HP_B2 * _hp_x2
                   - DSP_HP_A1 * _hp_y1
                   - DSP_HP_A2 * _hp_y2;
    _hp_x2 = _hp_x1;
    _hp_x1 = s;
    _hp_y2 = _hp_y1;
    _hp_y1 = hp;

    // Low-pass filter (2nd order IIR)
    const float lp = DSP_LP_B0 * hp
                   + DSP_LP_B1 * _lp_x1
                   + DSP_LP_B2 * _lp_x2
                   - DSP_LP_A1 * _lp_y1
                   - DSP_LP_A2 * _lp_y2;
    _lp_x2 = _lp_x1;
    _lp_x1 = hp;
    _lp_y2 = _lp_y1;
    _lp_y1 = lp;

    // Shannon energy envelope
    const float x2 = lp * lp;
    float se = 0.0f;
    if (x2 > 1e-10f) {
        const float x2n = (x2 > 1.0f) ? 1.0f : x2;
        se = -x2n * log10f(x2n + 1e-10f);
    }

    _env_sum -= _env_buf[_env_idx];
    _env_buf[_env_idx] = se;
    _env_sum += se;
    _env_idx = (_env_idx + 1) % DSP_ENV_WINDOW;
    const float envelope = _env_sum / static_cast<float>(DSP_ENV_WINDOW);

    // Running variance for signal quality check
    const float old_v = _var_buf[_var_idx];
    _var_sum    -= old_v;
    _var_sq_sum -= old_v * old_v;
    _var_buf[_var_idx] = envelope;
    _var_sum    += envelope;
    _var_sq_sum += envelope * envelope;
    _var_idx = (_var_idx + 1) % DSP_VAR_WINDOW;

    const float mean = _var_sum / static_cast<float>(DSP_VAR_WINDOW);
    float var = (_var_sq_sum / static_cast<float>(DSP_VAR_WINDOW)) - (mean * mean);
    if (var < 0.0f) {
        var = 0.0f;
    }

    const bool signal_ok = (var > DSP_SIGNAL_VAR_MIN) && (envelope > DSP_NOISE_FLOOR);
    dsp_signal_ok = signal_ok;
    const unsigned long now = millis();

    // BPM timeout: reset if no peak for too long
    if (_last_peak_ms > 0 && (now - _last_peak_ms) > DSP_BPM_TIMEOUT_MS) {
        dsp_bpm         = 0.0f;
        dsp_instant_bpm = 0.0f;
        dsp_beat_count  = 0;
        _last_peak_ms   = 0;
        _running_max   *= DSP_RUNNING_MAX_TIMEOUT_DECAY;
    }

    if (!signal_ok) {
        _running_max *= DSP_RUNNING_MAX_FAST_DECAY;
        _samples_since_peak++;
        return;
    }

    _running_max *= DSP_RUNNING_MAX_SLOW_DECAY;
    if (envelope > _running_max) {
        _running_max = envelope;
    }

    const float threshold = _running_max * DSP_THRESHOLD_RATIO;
    _samples_since_peak++;

    // Peak detection
    if (envelope > threshold
        && envelope > DSP_NOISE_FLOOR
        && _samples_since_peak > DSP_PEAK_REFRACTORY) {

        dsp_beat_count++;
        dsp_beat_flag = true;

        if (_last_peak_ms > 0) {
            const float rr_ms = static_cast<float>(now - _last_peak_ms);
            if (rr_ms > DSP_RR_MIN_MS && rr_ms < DSP_RR_MAX_MS) {
                dsp_instant_bpm = 60000.0f / rr_ms;
                if (dsp_bpm < DSP_BPM_MIN_INIT) {
                    dsp_bpm = dsp_instant_bpm;
                } else {
                    dsp_bpm = dsp_bpm * DSP_BPM_EMA_OLD
                            + dsp_instant_bpm * DSP_BPM_EMA_NEW;
                }
            }
        }
        _last_peak_ms       = now;
        _samples_since_peak = 0;
    }
}

// ══════════════════════════════════════════════════════════
//  DSP (Respiratory Rate Detection)
//  에너지 envelope의 상승 에지(조용→소리) 간격으로 호흡 주기 측정
// ══════════════════════════════════════════════════════════
static volatile float rr_rate          = 0.0f;   // 현재 호흡수 (breaths/min)
static volatile float rr_instant       = 0.0f;   // 순간 호흡수
static volatile bool  rr_breath_flag   = false;   // UI용 호흡 감지 플래그
static volatile int   rr_breath_count  = 0;
static volatile bool  rr_signal_ok     = false;   // 폐음 신호 품질

// RMS 에너지 계산용 순환 버퍼
static float _rr_rms_buf[RR_RMS_WINDOW]   = {0};
static int   _rr_rms_idx   = 0;
static float _rr_rms_sum   = 0.0f;

// 이동평균 스무딩용 순환 버퍼
static float _rr_smooth_buf[RR_SMOOTH_WINDOW] = {0};
static int   _rr_smooth_idx = 0;
static float _rr_smooth_sum = 0.0f;

// 상태 변수
static float        _rr_running_max       = 0.0f;
static int          _rr_samples_since     = 0;
static bool         _rr_in_breath         = false;  // 현재 호흡 구간인지
static unsigned long _rr_last_onset_ms    = 0;      // 마지막 호흡 시작 시각

static void rr_reset()
{
    memset(_rr_rms_buf, 0, sizeof(_rr_rms_buf));
    _rr_rms_idx = 0;
    _rr_rms_sum = 0.0f;

    memset(_rr_smooth_buf, 0, sizeof(_rr_smooth_buf));
    _rr_smooth_idx = 0;
    _rr_smooth_sum = 0.0f;

    _rr_running_max    = 0.0f;
    _rr_samples_since  = 0;
    _rr_in_breath      = false;
    _rr_last_onset_ms  = 0;

    rr_rate         = 0.0f;
    rr_instant      = 0.0f;
    rr_breath_flag  = false;
    rr_breath_count = 0;
}

static void rr_process(int16_t raw16)
{
    const float s = static_cast<float>(raw16) / SAMPLE_NORMALIZE;

    // ── Step 1: 단기 RMS 에너지 (200ms 윈도우) ──
    const float s2 = s * s;
    _rr_rms_sum -= _rr_rms_buf[_rr_rms_idx];
    _rr_rms_buf[_rr_rms_idx] = s2;
    _rr_rms_sum += s2;
    _rr_rms_idx = (_rr_rms_idx + 1) % RR_RMS_WINDOW;

    const float rms = sqrtf(_rr_rms_sum / static_cast<float>(RR_RMS_WINDOW));

    // ── Step 2: 이동평균 스무딩 (500ms 윈도우) ──
    _rr_smooth_sum -= _rr_smooth_buf[_rr_smooth_idx];
    _rr_smooth_buf[_rr_smooth_idx] = rms;
    _rr_smooth_sum += rms;
    _rr_smooth_idx = (_rr_smooth_idx + 1) % RR_SMOOTH_WINDOW;

    const float envelope = _rr_smooth_sum / static_cast<float>(RR_SMOOTH_WINDOW);

    // 신호 품질 판정
    rr_signal_ok = (envelope > RR_NOISE_FLOOR);

    // ── Step 3: 적응형 threshold ──
    _rr_running_max *= RR_RUNNING_MAX_DECAY;
    if (envelope > _rr_running_max) {
        _rr_running_max = envelope;
    }

    const float threshold = _rr_running_max * RR_THRESHOLD_RATIO;
    const unsigned long now = millis();
    _rr_samples_since++;

    // Timeout: 10초 이상 호흡 미감지 시 reset
    if (_rr_last_onset_ms > 0 && (now - _rr_last_onset_ms) > RR_TIMEOUT_MS) {
        rr_rate    = 0.0f;
        rr_instant = 0.0f;
        rr_breath_count = 0;
        _rr_last_onset_ms = 0;
        _rr_running_max *= RR_TIMEOUT_DECAY;
    }

    // ── Step 4: 상승 에지 감지 (조용 → 호흡 시작) ──
    if (!_rr_in_breath
        && envelope > threshold
        && envelope > RR_NOISE_FLOOR
        && _rr_samples_since > RR_REFRACTORY) {

        // 호흡 시작 감지
        _rr_in_breath = true;
        rr_breath_count++;
        rr_breath_flag = true;

        if (_rr_last_onset_ms > 0) {
            const float cycle_ms = static_cast<float>(now - _rr_last_onset_ms);
            if (cycle_ms > RR_CYCLE_MIN_MS && cycle_ms < RR_CYCLE_MAX_MS) {
                rr_instant = 60000.0f / cycle_ms;
                if (rr_rate < 1.0f) {
                    rr_rate = rr_instant;
                } else {
                    rr_rate = rr_rate * RR_EMA_OLD
                            + rr_instant * RR_EMA_NEW;
                }
            }
        }
        _rr_last_onset_ms  = now;
        _rr_samples_since  = 0;
    }
    else if (_rr_in_breath && envelope < threshold * 0.6f) {
        // 히스테리시스: threshold의 60%로 내려가야 "조용" 판정
        _rr_in_breath = false;
    }
}

// ══════════════════════════════════════════════════════════
//  BLE Callbacks
// ══════════════════════════════════════════════════════════
class MyServerCallbacks : public BLEServerCallbacks {
    void onConnect(BLEServer* pServer) override
    {
        deviceConnected = true;
    }

    void onDisconnect(BLEServer* pServer) override
    {
        deviceConnected = false;
    }
};

class MyCmdCallbacks : public BLECharacteristicCallbacks {
    void onWrite(BLECharacteristic* pCharacteristic) override
    {
        const String rxValue = pCharacteristic->getValue().c_str();

        // Parse patient list: "PAT:P001|Kim|301,P002|Lee|302"
        if (!rxValue.startsWith("PAT:")) {
            return;
        }

        const String data = rxValue.substring(4);
        patientCount  = 0;
        currentPatIdx = 0;

        int start = 0;
        while (start < static_cast<int>(data.length()) && patientCount < MAX_PATIENTS) {
            int end = data.indexOf(',', start);
            if (end == -1) {
                end = static_cast<int>(data.length());
            }

            const String patStr = data.substring(start, end);
            const int p1 = patStr.indexOf('|');
            const int p2 = patStr.indexOf('|', p1 + 1);

            if (p1 != -1 && p2 != -1) {
                patientList[patientCount].id   = patStr.substring(0, p1);
                patientList[patientCount].name = patStr.substring(p1 + 1, p2);
                patientList[patientCount].ward = patStr.substring(p2 + 1);
                patientCount++;
            }
            start = end + 1;
        }
        uiNeedsUpdate = true;
    }
};

static MyServerCallbacks serverCallbacks;
static MyCmdCallbacks    cmdCallbacks;
static BLE2902           audioDescriptor;
static BLE2902           cmdDescriptor;

// ══════════════════════════════════════════════════════════
//  Deep Sleep
// ══════════════════════════════════════════════════════════
static void enterDeepSleep()
{
    tft.writecommand(TFT_CMD_SLEEP_IN);
    delay(TFT_CMD_DELAY_MS);
    digitalWrite(PIN_TFT_BL, LOW);

    // Wait for all buttons to be released before sleeping
    while (digitalRead(PIN_BTN_L_MODE) == LOW || digitalRead(PIN_BTN_R_RECORD) == LOW) {
        delay(10);
    }
    delay(DEBOUNCE_DELAY_MS);

    esp_sleep_enable_ext0_wakeup(GPIO_NUM_35, 0);
    esp_deep_sleep_start();
}

// ══════════════════════════════════════════════════════════
//  TFT Sleep/Wake Helpers
// ══════════════════════════════════════════════════════════
static void tftSleep()
{
    tft.writecommand(TFT_CMD_SLEEP_IN);
    delay(TFT_CMD_DELAY_MS);
    digitalWrite(PIN_TFT_BL, LOW);
}

static void tftWake()
{
    tft.writecommand(TFT_CMD_SLEEP_OUT);
    delay(TFT_WAKEUP_DELAY_MS);
    digitalWrite(PIN_TFT_BL, HIGH);
}

// ══════════════════════════════════════════════════════════
//  Audio Processing Task (Core 0)
// ══════════════════════════════════════════════════════════
static void audioTask(void* pvParameters)
{
    (void)pvParameters;  // AV Rule 142: unused parameter

    // Register this task with WDT
    esp_task_wdt_add(nullptr);

    int   prevMode     = -1;
    float prev_heart_y = 0.0f;
    float prev_lung_x  = 0.0f;
    float prev_lung_y  = 0.0f;

    for (;;) {
        esp_task_wdt_reset();  // Feed watchdog every iteration
        // Decimate from 16kHz to 2kHz
        int32_t sum = 0;
        for (int i = 0; i < DECIMATION; i++) {
            int32_t sample  = 0;
            size_t  bytesIn = 0;
            const esp_err_t err = i2s_read(I2S_PORT, &sample, sizeof(sample),
                                           &bytesIn, portMAX_DELAY);
            if (err != ESP_OK) {
                continue;  // AV Rule 15: check return values
            }
            sum += static_cast<int16_t>(sample >> 16);
        }

        const int32_t avg = sum / DECIMATION;
        const float current_sample = static_cast<float>(avg << SAMPLE_LEFT_SHIFT);
        int16_t final_audio = 0;

        // Mode-specific filtering
        if (currentMode == MODE_HEART) {
            const float heart_filtered = HEART_LPF_COEFF * current_sample
                                       + HEART_LPF_DECAY * prev_heart_y;
            prev_heart_y = heart_filtered;
            final_audio = static_cast<int16_t>(heart_filtered);
        }
        else if (currentMode == MODE_LUNG) {
            const float lung_filtered = LUNG_HPF_COEFF * prev_lung_y
                                      + LUNG_HPF_COEFF * (current_sample - prev_lung_x);
            prev_lung_x = current_sample;
            prev_lung_y = lung_filtered;
            final_audio = static_cast<int16_t>(lung_filtered);
        }
        else {
            final_audio = static_cast<int16_t>(current_sample);
        }

        // Clamp to int16 range
        if (final_audio > SAMPLE_MAX) {
            final_audio = SAMPLE_MAX;
        } else if (final_audio < SAMPLE_MIN) {
            final_audio = SAMPLE_MIN;
        }

        sharedAveragedSample = final_audio;

        // DSP detection (mode-specific)
        if (currentMode != prevMode) {
            if (currentMode == MODE_HEART) {
                dsp_reset();
            } else if (currentMode == MODE_LUNG) {
                rr_reset();
            }
            prevMode = currentMode;
        }
        if (currentMode == MODE_HEART) {
            dsp_process(final_audio);
        } else if (currentMode == MODE_LUNG) {
            rr_process(final_audio);
        }

        // BLE streaming during 5-second recording
        if (currentState == ACTIVE && deviceConnected && isRecording5Sec) {
            bleBuffer[bufferIndex] = final_audio;
            bufferIndex++;
            if (bufferIndex >= BLE_BUFFER_SIZE) {
                pAudioChar->setValue(
                    reinterpret_cast<uint8_t*>(bleBuffer),
                    BLE_BUFFER_SIZE * static_cast<int>(sizeof(int16_t))
                );
                pAudioChar->notify();
                bufferIndex = 0;
            }
        }
    }
}

// ══════════════════════════════════════════════════════════
//  Setup
// ══════════════════════════════════════════════════════════
void setup()
{
    Serial.begin(115200);

    pinMode(PIN_BTN_L_MODE, INPUT);
    pinMode(PIN_BTN_R_RECORD, INPUT_PULLUP);
    pinMode(PIN_TFT_BL, OUTPUT);
    digitalWrite(PIN_TFT_BL, HIGH);

    // Handle wake from deep sleep
    if (esp_sleep_get_wakeup_cause() == ESP_SLEEP_WAKEUP_EXT0) {
        while (digitalRead(PIN_BTN_L_MODE) == LOW) {
            delay(10);
        }
        delay(DEBOUNCE_DELAY_MS);
    }

    // Initialize TFT display
    tft.init();
    tft.writecommand(TFT_CMD_SLEEP_OUT);
    delay(TFT_WAKEUP_DELAY_MS);
    tft.setRotation(2);
    tft.fillScreen(TFT_BLACK);

    btnL.begin();
    btnR.begin();

    // Load saved settings from NVS
    nvsLoad();

    // I2S configuration for ICS43434 microphone
    const i2s_config_t i2s_config = {
        .mode                 = i2s_mode_t(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate          = SAMPLE_RATE,
        .bits_per_sample      = I2S_BITS_PER_SAMPLE_32BIT,
        .channel_format       = I2S_CHANNEL_FMT_ONLY_LEFT,
        .communication_format = i2s_comm_format_t(I2S_COMM_FORMAT_STAND_I2S),
        .intr_alloc_flags     = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count        = 8,
        .dma_buf_len          = 64,
        .use_apll             = false
    };
    const i2s_pin_config_t pin_config = {
        .bck_io_num   = PIN_I2S_SCK,
        .ws_io_num    = PIN_I2S_WS,
        .data_out_num = I2S_PIN_NO_CHANGE,
        .data_in_num  = PIN_I2S_SD
    };

    // AV Rule 15: check return values for hardware init
    esp_err_t err = i2s_driver_install(I2S_PORT, &i2s_config, 0, nullptr);
    if (err != ESP_OK) {
        Serial.printf("I2S driver install failed: %d\n", err);
    }
    err = i2s_set_pin(I2S_PORT, &pin_config);
    if (err != ESP_OK) {
        Serial.printf("I2S pin config failed: %d\n", err);
    }

    // BLE initialization
    BLEDevice::init("AI_Stetho");
    BLEDevice::setMTU(BLE_MTU_SIZE);

    pServer = BLEDevice::createServer();
    pServer->setCallbacks(&serverCallbacks);

    BLEService* pService = pServer->createService(SERVICE_UUID);

    pAudioChar = pService->createCharacteristic(
        AUDIO_CHAR_UUID,
        BLECharacteristic::PROPERTY_NOTIFY
    );
    pAudioChar->addDescriptor(&audioDescriptor);

    pCmdChar = pService->createCharacteristic(
        CMD_CHAR_UUID,
        BLECharacteristic::PROPERTY_WRITE | BLECharacteristic::PROPERTY_NOTIFY
    );
    pCmdChar->setCallbacks(&cmdCallbacks);
    pCmdChar->addDescriptor(&cmdDescriptor);

    pService->start();

    // BLE advertising
    BLEAdvertising* pAdvertising = BLEDevice::getAdvertising();
    pAdvertising->addServiceUUID(SERVICE_UUID);
    pAdvertising->setScanResponse(true);
    pAdvertising->setMinPreferred(BLE_ADV_MIN_INTERVAL);
    pAdvertising->setMaxPreferred(BLE_ADV_MAX_INTERVAL);  // BUG FIX: was setMinPreferred (overwrote previous value)

    // Restore BLE advertising if it was on before reboot/sleep
    if (isBleOn) {
        BLEDevice::startAdvertising();
        Serial.println("[NVS] BLE advertising restored");
    }

    // Launch audio task on Core 0
    audioTaskHandle = xTaskCreateStaticPinnedToCore(
        audioTask,
        "AudioTask",
        AUDIO_TASK_STACK_SIZE,
        nullptr,
        AUDIO_TASK_PRIORITY,
        audioTaskStack,
        &audioTaskBuffer,
        AUDIO_TASK_CORE
    );

    // Initialize Watchdog Timer
    // WDT config for ESP-IDF v5+ (Arduino ESP32 3.x)
    const esp_task_wdt_config_t wdt_config = {
        .timeout_ms      = WDT_TIMEOUT_SEC * 1000,
        .idle_core_mask  = 0,        // 아이들 태스크 감시 안함
        .trigger_panic   = true      // timeout 시 자동 reset
    };
    esp_task_wdt_init(&wdt_config);
    esp_task_wdt_add(nullptr);  // Register loop() task (runs on current task)
    Serial.println("[WDT] Watchdog initialized");
}

// ══════════════════════════════════════════════════════════
//  UI Drawing Functions
// ══════════════════════════════════════════════════════════
static void drawWaveform(int16_t audioSample)
{
    if (currentState != ACTIVE || isRecording5Sec || currentMode == MODE_PATIENT) {
        return;
    }
    if ((millis() - lastDrawTime) < static_cast<unsigned long>(DRAW_INTERVAL_MS)) {
        return;
    }
    lastDrawTime = millis();

    static int16_t smoothedSample = 0;
    int displayRange = 0;
    uint32_t waveColor = 0;
    int16_t* lastYPtr = nullptr;

    if (currentMode == MODE_HEART) {
        smoothedSample = static_cast<int16_t>(
            (smoothedSample * HEART_SMOOTH_NUMER + audioSample) / HEART_SMOOTH_DENOM
        );
        displayRange = HEART_DISPLAY_RANGE;
        waveColor = TFT_RED;
        lastYPtr = &lastY_HEART_WAVE;
    } else {
        smoothedSample = audioSample;
        displayRange = LUNG_DISPLAY_RANGE;
        waveColor = TFT_CYAN;
        lastYPtr = &lastY_LUNG_WAVE;
    }

    const long mappedY = map(smoothedSample, -displayRange, displayRange,
                             WAVE_Y_BOTTOM, WAVE_Y_TOP);
    const int16_t currY = static_cast<int16_t>(
        constrain(static_cast<int>(mappedY), WAVE_Y_TOP, WAVE_Y_BOTTOM)
    );

    // Clear ahead
    tft.drawFastVLine(graphX + 1, WAVE_Y_TOP, WAVE_HEIGHT, TFT_BLACK);
    tft.drawFastVLine(graphX + 2, WAVE_Y_TOP, WAVE_HEIGHT, TFT_BLACK);

    // Grid dot on center line
    if (graphX % GRID_DOT_SPACING == 0) {
        tft.drawPixel(graphX + 1, WAVE_ZERO_Y, GRID_DOT_COLOR);
    }

    // Draw waveform (double thickness)
    tft.drawLine(graphX, *lastYPtr, graphX + 1, currY, waveColor);
    tft.drawLine(graphX, *lastYPtr + 1, graphX + 1, currY + 1, waveColor);

    *lastYPtr = currY;
    graphX++;

    // Wrap around
    if (graphX >= WAVE_X_END) {
        graphX = WAVE_X_START;
        tft.drawFastVLine(WAVE_X_START, WAVE_Y_TOP, WAVE_HEIGHT, TFT_BLACK);
        tft.drawFastVLine(WAVE_X_START + 1, WAVE_Y_TOP, WAVE_HEIGHT, TFT_BLACK);
    }
}

static float lastDisplayedBPM = -1.0f;
static unsigned long lastBpmDrawTime = 0;

static void drawBPM()
{
    if (currentState != ACTIVE || currentMode != MODE_HEART || isRecording5Sec) {
        return;
    }
    if ((millis() - lastBpmDrawTime) < BPM_UPDATE_INTERVAL_MS) {
        return;
    }
    lastBpmDrawTime = millis();

    const float bpm = dsp_bpm;
    if (fabsf(bpm - lastDisplayedBPM) < BPM_CHANGE_THRESHOLD && lastDisplayedBPM >= 0.0f) {
        return;
    }
    lastDisplayedBPM = bpm;

    const int cx = tft.width() / 2;
    tft.fillRect(0, BPM_Y, tft.width(), BPM_H, TFT_BLACK);

    if (bpm > BPM_DISPLAY_MIN && bpm < BPM_DISPLAY_MAX) {
        // Draw heart icon
        const int hx = cx - BPM_HEART_OFFSET_X;
        const int hy = BPM_Y + 12;
        tft.fillCircle(hx - BPM_HEART_CIRCLE_R, hy, BPM_HEART_CIRCLE_R, TFT_RED);
        tft.fillCircle(hx + BPM_HEART_CIRCLE_R, hy, BPM_HEART_CIRCLE_R, TFT_RED);
        tft.fillTriangle(hx - 8, hy + 1, hx + 8, hy + 1, hx, hy + 10, TFT_RED);

        // Draw BPM value
        tft.setTextDatum(ML_DATUM);
        tft.setTextColor(TFT_WHITE, TFT_BLACK);
        tft.setTextPadding(60);
        char buf[8];
        snprintf(buf, sizeof(buf), "%d", static_cast<int>(bpm));  // AV Rule 167: snprintf
        tft.drawString(buf, cx - BPM_TEXT_OFFSET_X, BPM_Y + 15, 4);

        tft.setTextColor(TFT_DARKGREY, TFT_BLACK);
        tft.setTextPadding(0);
        tft.drawString("bpm", cx + BPM_UNIT_OFFSET_X, BPM_Y + 17, 2);
    } else {
        tft.setTextDatum(MC_DATUM);
        tft.setTextColor(TFT_DARKGREY, TFT_BLACK);
        tft.setTextPadding(tft.width());
        tft.drawString("measuring...", cx, BPM_Y + 15, 2);
        tft.setTextPadding(0);
    }
}

static void clearBPM()
{
    tft.fillRect(0, BPM_Y, tft.width(), BPM_H, TFT_BLACK);
    lastDisplayedBPM  = -1.0f;
    lastBpmDrawTime   = 0;
    clearRR();
}

static unsigned long lastBeatFlash = 0;

static void drawBeatIndicator()
{
    if (currentState != ACTIVE || currentMode != MODE_HEART) {
        return;
    }

    if (dsp_beat_flag) {
        dsp_beat_flag = false;
        tft.fillCircle(BEAT_INDICATOR_X, BEAT_INDICATOR_Y, BEAT_INDICATOR_R, TFT_RED);
        lastBeatFlash = millis();
    } else if (lastBeatFlash > 0 && (millis() - lastBeatFlash) > BEAT_FLASH_DURATION_MS) {
        tft.fillCircle(BEAT_INDICATOR_X, BEAT_INDICATOR_Y, BEAT_INDICATOR_R, TFT_BLACK);
        lastBeatFlash = 0;
    }
}

// ── RR Display (Lung mode) ──
static float lastDisplayedRR = -1.0f;
static unsigned long lastRrDrawTime = 0;

static void drawRR()
{
    if (currentState != ACTIVE || currentMode != MODE_LUNG || isRecording5Sec) {
        return;
    }
    if ((millis() - lastRrDrawTime) < BPM_UPDATE_INTERVAL_MS) {
        return;
    }
    lastRrDrawTime = millis();

    const float rr = rr_rate;
    if (fabsf(rr - lastDisplayedRR) < RR_CHANGE_THRESHOLD && lastDisplayedRR >= 0.0f) {
        return;
    }
    lastDisplayedRR = rr;

    const int cx = tft.width() / 2;
    tft.fillRect(0, BPM_Y, tft.width(), BPM_H, TFT_BLACK);

    if (rr > RR_DISPLAY_MIN && rr < RR_DISPLAY_MAX) {
        // 폐 아이콘 (간략화된 두 원호)
        const int lx = cx - RR_LUNG_ICON_OFFSET_X;
        const int ly = BPM_Y + 12;
        tft.drawCircle(lx - 3, ly, 5, TFT_CYAN);
        tft.drawCircle(lx + 3, ly, 5, TFT_CYAN);
        tft.drawFastVLine(lx, ly + 2, 6, TFT_CYAN);

        // RR 값
        tft.setTextDatum(ML_DATUM);
        tft.setTextColor(TFT_WHITE, TFT_BLACK);
        tft.setTextPadding(60);
        char buf[8];
        snprintf(buf, sizeof(buf), "%d", static_cast<int>(rr));
        tft.drawString(buf, cx - RR_TEXT_OFFSET_X, BPM_Y + 15, 4);

        tft.setTextColor(TFT_DARKGREY, TFT_BLACK);
        tft.setTextPadding(0);
        tft.drawString("br/m", cx + RR_UNIT_OFFSET_X, BPM_Y + 17, 2);
    } else {
        tft.setTextDatum(MC_DATUM);
        tft.setTextColor(TFT_DARKGREY, TFT_BLACK);
        tft.setTextPadding(tft.width());
        tft.drawString("measuring...", cx, BPM_Y + 15, 2);
        tft.setTextPadding(0);
    }
}

static void clearRR()
{
    lastDisplayedRR = -1.0f;
    lastRrDrawTime  = 0;
}

// ── Breath Indicator (Lung mode, status bar) ──
static unsigned long lastBreathFlash = 0;

static void drawBreathIndicator()
{
    if (currentState != ACTIVE || currentMode != MODE_LUNG) {
        return;
    }

    if (rr_breath_flag) {
        rr_breath_flag = false;
        tft.fillCircle(BEAT_INDICATOR_X, BEAT_INDICATOR_Y, BEAT_INDICATOR_R, TFT_CYAN);
        lastBreathFlash = millis();
    } else if (lastBreathFlash > 0 && (millis() - lastBreathFlash) > BEAT_FLASH_DURATION_MS) {
        tft.fillCircle(BEAT_INDICATOR_X, BEAT_INDICATOR_Y, BEAT_INDICATOR_R, TFT_BLACK);
        lastBreathFlash = 0;
    }
}

// ── Signal Quality Indicator ──
// 3단 바 표시: ■□□ = No Signal, ■■□ = Weak, ■■■ = Good
static SignalLevel lastSigLevel = SIG_NONE;
static unsigned long lastSigDrawTime = 0;

static SignalLevel getSignalLevel()
{
    if (currentMode == MODE_HEART) {
        if (!dsp_signal_ok) {
            return SIG_NONE;
        }
        // envelope 크기로 Good/Weak 구분: BPM 감지 중이면 Good
        if (dsp_bpm > BPM_DISPLAY_MIN) {
            return SIG_GOOD;
        }
        return SIG_WEAK;
    }
    else if (currentMode == MODE_LUNG) {
        if (!rr_signal_ok) {
            return SIG_NONE;
        }
        if (rr_rate > RR_DISPLAY_MIN) {
            return SIG_GOOD;
        }
        return SIG_WEAK;
    }
    return SIG_NONE;
}

static void drawSignalQuality()
{
    if (currentState != ACTIVE || currentMode == MODE_PATIENT || isRecording5Sec) {
        return;
    }
    if ((millis() - lastSigDrawTime) < SIG_UPDATE_INTERVAL_MS) {
        return;
    }
    lastSigDrawTime = millis();

    const SignalLevel level = getSignalLevel();
    if (level == lastSigLevel) {
        return;
    }
    lastSigLevel = level;

    // 표시 위치: 화면 우측 상단 (BT 아이콘 왼쪽)
    const int baseX = tft.width() - 35;
    const int baseY = SIG_QUALITY_Y;

    // 배경 지우기
    tft.fillRect(baseX - 2, baseY - 12, 20, 14, TFT_BLACK);

    // 모드별 색상
    const uint16_t activeColor = (currentMode == MODE_HEART) ? TFT_RED : TFT_CYAN;

    for (int i = 0; i < SIG_BAR_COUNT; i++) {
        const int x = baseX + i * (SIG_BAR_WIDTH + SIG_BAR_GAP);
        const int h = SIG_BAR_HEIGHTS[i];
        const int y = baseY - h;

        if (i <= static_cast<int>(level)) {
            tft.fillRect(x, y, SIG_BAR_WIDTH, h, activeColor);
        } else {
            tft.fillRect(x, y, SIG_BAR_WIDTH, h, 0x3186);  // 어두운 회색
        }
    }
}

static void drawStatusBar()
{
    tft.fillRect(0, 0, tft.width(), STATUS_BAR_HEIGHT, TFT_BLACK);

    uint16_t btColor = TFT_DARKGREY;
    if (deviceConnected) {
        btColor = TFT_CYAN;
    } else if (isBleOn) {
        btColor = TFT_ORANGE;
    }

    // Bluetooth icon
    const int cx = tft.width() - BT_ICON_OFFSET_X;
    const int cy = 8;
    tft.drawLine(cx, cy - 6, cx, cy + 6, btColor);
    tft.drawLine(cx, cy - 6, cx + 4, cy - 3, btColor);
    tft.drawLine(cx + 4, cy - 3, cx, cy, btColor);
    tft.drawLine(cx, cy, cx + 4, cy + 3, btColor);
    tft.drawLine(cx + 4, cy + 3, cx, cy + 6, btColor);
    tft.drawLine(cx - 4, cy - 2, cx + 4, cy + 4, btColor);
    tft.drawLine(cx - 4, cy + 4, cx + 4, cy - 2, btColor);
}

// ══════════════════════════════════════════════════════════
//  Main UI Draw
// ══════════════════════════════════════════════════════════
static void drawPatientListUI(int cx)
{
    tft.setTextColor(TFT_YELLOW, TFT_BLACK);
    tft.drawString("PATIENT LIST", cx, 20, 2);
    tft.drawFastHLine(10, 35, tft.width() - 20, TFT_DARKGREY);

    if (patientCount == 0) {
        tft.setTextColor(TFT_WHITE, TFT_BLACK);
        tft.drawString("No Patients", cx, 80, 2);
        return;
    }

    int startIdx = 0;
    if (currentPatIdx >= PATIENT_VISIBLE_ROWS) {
        startIdx = currentPatIdx - PATIENT_VISIBLE_ROWS + 1;
    }

    // Table header
    tft.setTextDatum(ML_DATUM);
    tft.setTextColor(TFT_DARKGREY, TFT_BLACK);
    tft.drawString("ID",   PATIENT_COL_ID,   42, 1);
    tft.drawString("NAME", PATIENT_COL_NAME,  42, 1);
    tft.drawString("WARD", PATIENT_COL_WARD,  42, 1);

    // Table rows
    for (int i = 0; i < PATIENT_VISIBLE_ROWS && (startIdx + i) < patientCount; i++) {
        const int pIdx = startIdx + i;
        const int y = PATIENT_START_Y + (i * PATIENT_ROW_HEIGHT);

        if (pIdx == currentPatIdx) {
            tft.fillRect(0, y - 8, tft.width(), PATIENT_ROW_HEIGHT, COLOR_HIGHLIGHT_BG);
            tft.setTextColor(TFT_WHITE, COLOR_HIGHLIGHT_BG);
            tft.drawString(">", 1, y, 2);
        } else {
            tft.fillRect(0, y - 8, tft.width(), PATIENT_ROW_HEIGHT, TFT_BLACK);
            tft.setTextColor(TFT_LIGHTGREY, TFT_BLACK);
        }

        tft.drawString(patientList[pIdx].id,   PATIENT_COL_ID,   y, 2);
        tft.drawString(patientList[pIdx].name,  PATIENT_COL_NAME, y, 2);
        tft.drawString(patientList[pIdx].ward,  PATIENT_COL_WARD, y, 2);
    }
    tft.setTextDatum(MC_DATUM);
}

void drawUI()
{
    if (!uiNeedsUpdate || isRecording5Sec) {
        return;
    }

    const int cx = tft.width() / 2;
    tft.setTextDatum(MC_DATUM);
    tft.setTextPadding(tft.width());

    // Redraw background only when state/mode changes
    static int      lastDrawnMode  = -1;
    static bool     lastDrawnConn  = false;
    static bool     lastDrawnBle   = false;
    static AppState lastDrawnState = ACTIVE;

    if (currentMode != lastDrawnMode
        || deviceConnected != lastDrawnConn
        || isBleOn != lastDrawnBle
        || currentState != lastDrawnState) {

        tft.fillRect(0, STATUS_BAR_HEIGHT, tft.width(), tft.height() - 52, TFT_BLACK);
        graphX = WAVE_X_START;
        lastDrawnMode  = currentMode;
        lastDrawnConn  = deviceConnected;
        lastDrawnBle   = isBleOn;
        lastDrawnState = currentState;
        clearBPM();
        lastSigLevel = SIG_NONE;
        lastSigDrawTime = 0;
        drawStatusBar();
    }

    // State-specific rendering
    if (currentState == POPUP) {
        tft.setTextColor(TFT_ORANGE, TFT_BLACK);
        tft.drawString("! WARNING !", cx, 40, 2);
        tft.setTextColor(TFT_WHITE, TFT_BLACK);
        tft.drawString("Bluetooth", cx, 80, 2);
        tft.drawString("Disconnected", cx, 100, 2);
        tft.setTextColor(TFT_GREEN, TFT_BLACK);
        tft.drawString("Pair now?", cx, 140, 1);
        tft.drawFastHLine(10, tft.height() - 35, tft.width() - 20, TFT_DARKGREY);
        tft.setTextColor(TFT_LIGHTGREY, TFT_BLACK);
        tft.drawString("L: NO | R: YES", cx, tft.height() - 15, 1);
    }
    else if (currentState == PAIRING) {
        tft.setTextColor(TFT_YELLOW, TFT_BLACK);
        tft.drawString("PAIRING", cx, 40, 4);
        tft.setTextColor(TFT_WHITE, TFT_BLACK);
        tft.drawString("WAITING...", cx, 100, 2);
        tft.drawFastHLine(10, tft.height() - 35, tft.width() - 20, TFT_DARKGREY);
        tft.setTextColor(TFT_LIGHTGREY, TFT_BLACK);
        tft.drawString("PRESS R TO CANCEL", cx, tft.height() - 15, 1);
    }
    else if (currentState == ACTIVE) {
        if (currentMode == MODE_HEART) {
            tft.setTextColor(TFT_RED, TFT_BLACK);
            tft.drawString("HEART", cx, 30, 4);
            tft.setTextColor(TFT_WHITE, TFT_BLACK);
            tft.drawString(selectedPatient, cx, 55, 2);
        }
        else if (currentMode == MODE_LUNG) {
            tft.setTextColor(TFT_CYAN, TFT_BLACK);
            tft.drawString("LUNG", cx, 30, 4);
            tft.setTextColor(TFT_WHITE, TFT_BLACK);
            tft.drawString(selectedPatient, cx, 55, 2);
        }
        else if (currentMode == MODE_PATIENT) {
            drawPatientListUI(cx);
        }

        // Bottom bar
        tft.drawFastHLine(10, tft.height() - 35, tft.width() - 20, TFT_DARKGREY);
        tft.setTextColor(TFT_LIGHTGREY, TFT_BLACK);

        if (currentMode == MODE_HEART || currentMode == MODE_LUNG) {
            tft.drawString("L: MODE | R: RECORD", cx, tft.height() - 15, 1);
        } else {
            tft.drawString("L: MODE | R(Hold): SEL", cx, tft.height() - 15, 1);
        }
    }

    tft.setTextPadding(0);
    uiNeedsUpdate = false;
}

// ══════════════════════════════════════════════════════════
//  Recording Helpers
// ══════════════════════════════════════════════════════════
static void startRecording()
{
    const String header = "REC:" + selectedPatient + ":"
                        + (currentMode == MODE_HEART ? "HEART" : "LUNG");
    pCmdChar->setValue(header.c_str());
    pCmdChar->notify();
    delay(REC_HEADER_DELAY_MS);

    recFailed = false;
    isRecording5Sec = true;
    recordStartTime = millis();
    bufferIndex = 0;
    tftSleep();
}

static void stopRecording()
{
    isRecording5Sec = false;

    // 웹 브릿지에 녹음 완료 알림
    if (deviceConnected && pCmdChar != nullptr) {
        pCmdChar->setValue("REC_DONE");
        pCmdChar->notify();
    }

    tftWake();
    uiNeedsUpdate = true;
    btnL.reset();
    btnR.reset();
}

static void stopRecordingWithFail()
{
    isRecording5Sec = false;
    recFailed = true;
    recFailDisplayTime = millis();
    bufferIndex = 0;

    // 웹 브릿지에 녹음 실패 알림 전송
    if (deviceConnected && pCmdChar != nullptr) {
        pCmdChar->setValue("REC_FAIL:DISCONNECTED");
        pCmdChar->notify();
    }

    tftWake();

    // 화면에 실패 표시
    tft.fillScreen(TFT_BLACK);
    tft.setTextDatum(MC_DATUM);
    tft.setTextColor(TFT_RED, TFT_BLACK);
    tft.drawString("REC FAILED", tft.width() / 2, tft.height() / 2 - 10, 4);
    tft.setTextColor(TFT_ORANGE, TFT_BLACK);
    tft.drawString("BLE Disconnected", tft.width() / 2, tft.height() / 2 + 20, 2);

    uiNeedsUpdate = true;
    btnL.reset();
    btnR.reset();
}

// ══════════════════════════════════════════════════════════
//  Main Loop
// ══════════════════════════════════════════════════════════
void loop()
{
    esp_task_wdt_reset();  // Feed watchdog

    // Connection state change detection
    static bool lastConnState = false;
    if (deviceConnected != lastConnState) {
        if (deviceConnected) {
            dsp_reset();
            rr_reset();
            changeState(ACTIVE);
        } else {
            if (isRecording5Sec) {
                stopRecordingWithFail();
            }
            isBleOn = false;
            nvsSaveBle();
            uiNeedsUpdate = true;
            btnL.reset();
            btnR.reset();
        }
        lastConnState = deviceConnected;
    }

    // REC FAIL 표시 timeout → 일정 시간 후 정상 UI 복귀
    if (recFailed && (millis() - recFailDisplayTime) >= REC_FAIL_DISPLAY_MS) {
        recFailed = false;
        uiNeedsUpdate = true;
        drawUI();
    }

    // During recording: only check timeout
    if (isRecording5Sec) {
        if ((millis() - recordStartTime) >= RECORD_DURATION_MS) {
            stopRecording();
        }
        return;
    }

    const ButtonEvent eventL = btnL.update();
    const ButtonEvent eventR = btnR.update();

    // State machine input handling
    if (currentState == POPUP) {
        if (eventL == BTN_SHORT) {
            changeState(ACTIVE);
        } else if (eventR == BTN_SHORT) {
            isBleOn = true;
            nvsSaveBle();
            BLEDevice::startAdvertising();
            changeState(PAIRING);
        }
    }
    else if (currentState == PAIRING) {
        if (eventR == BTN_SHORT) {
            isBleOn = false;
            nvsSaveBle();
            BLEDevice::stopAdvertising();
            changeState(ACTIVE);
        }
    }
    else if (currentState == ACTIVE) {
        // Left button: long = sleep, short = cycle mode
        if (eventL == BTN_LONG) {
            enterDeepSleep();
        } else if (eventL == BTN_SHORT) {
            currentMode++;
            if (currentMode > MODE_COUNT) {
                currentMode = MODE_HEART;
            }
            nvsSaveMode();
            uiNeedsUpdate = true;
            drawUI();
        }

        // Right button: mode-dependent
        if (currentMode == MODE_PATIENT) {
            if (eventR == BTN_SHORT) {
                if (patientCount > 0) {
                    currentPatIdx = (currentPatIdx + 1) % patientCount;
                    uiNeedsUpdate = true;
                    drawUI();
                }
            } else if (eventR == BTN_LONG) {
                if (patientCount > 0) {
                    selectedPatient = patientList[currentPatIdx].id;
                    nvsSavePatient();
                    currentMode = MODE_HEART;
                    nvsSaveMode();
                    uiNeedsUpdate = true;

                    // Show selection confirmation
                    tft.fillScreen(TFT_BLACK);
                    tft.setTextColor(TFT_GREEN, TFT_BLACK);
                    tft.drawString("SELECTED!", tft.width() / 2, tft.height() / 2 - 10, 4);
                    tft.setTextColor(TFT_WHITE, TFT_BLACK);
                    tft.drawString(selectedPatient, tft.width() / 2, tft.height() / 2 + 20, 2);
                    delay(PATIENT_SELECT_DELAY_MS);

                    btnR.reset();
                    drawUI();
                }
            }
        }
        else {
            // Heart/Lung mode: right button triggers recording
            if (eventR == BTN_SHORT) {
                if (deviceConnected) {
                    startRecording();
                } else {
                    changeState(POPUP);
                }
            }
        }
    }

    drawUI();

    // Real-time waveform and BPM display
    if (currentState == ACTIVE
        && (currentMode == MODE_HEART || currentMode == MODE_LUNG)
        && !isRecording5Sec) {

        drawWaveform(sharedAveragedSample);
        drawSignalQuality();
        if (currentMode == MODE_HEART) {
            drawBPM();
            drawBeatIndicator();
        } else if (currentMode == MODE_LUNG) {
            drawRR();
            drawBreathIndicator();
        }
    }
}

#endif // STETHO_FIRMWARE_2026_03_25_1442_H
