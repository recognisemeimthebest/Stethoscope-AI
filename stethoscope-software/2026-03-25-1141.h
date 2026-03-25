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

TFT_eSPI tft = TFT_eSPI(); 

#define BTN_L_MODE 35   
#define BTN_R_RECORD 0  

#define TFT_BL 4
#define I2S_WS 25
#define I2S_SD 33
#define I2S_SCK 26
#define I2S_PORT I2S_NUM_0

#define SERVICE_UUID        "19b10000-e8f2-537e-4f6c-d104768a1214"
#define AUDIO_CHAR_UUID     "19b10001-e8f2-537e-4f6c-d104768a1214" 
#define CMD_CHAR_UUID       "19b10002-e8f2-537e-4f6c-d104768a1214" 

BLEServer* pServer = NULL;
BLECharacteristic* pAudioChar = NULL;
BLECharacteristic* pCmdChar = NULL;

volatile bool deviceConnected = false;
volatile bool isBleOn = false; 
volatile int currentMode = 1;  
volatile bool uiNeedsUpdate = true; 

volatile bool isRecording5Sec = false; 
unsigned long recordStartTime = 0;

volatile int16_t sharedAveragedSample = 0;

unsigned long lPressTime = 0;
bool lLongPressed = false;
unsigned long rPressTime = 0;
bool rLongPressed = false;

const int SAMPLE_RATE = 16000; 
const int TARGET_RATE = 2000;
const int DECIMATION = SAMPLE_RATE / TARGET_RATE;
const int BLE_BUFFER_SIZE = 50; 
int16_t bleBuffer[BLE_BUFFER_SIZE];
int bufferIndex = 0;

#define WAVE_Y_TOP 55
#define WAVE_Y_BOTTOM 135
#define WAVE_HEIGHT (WAVE_Y_BOTTOM - WAVE_Y_TOP)
#define WAVE_ZERO_Y (WAVE_Y_TOP + (WAVE_HEIGHT / 2)) 
#define WAVE_X_START 10
#define WAVE_X_END 125

#define BPM_Y 145
#define BPM_H 40

int graphX = WAVE_X_START; 
int16_t lastY_HEART_WAVE = WAVE_ZERO_Y; 
int16_t lastY_LUNG_WAVE = WAVE_ZERO_Y;  

unsigned long lastDrawTime = 0;
const int drawInterval = 15; 

TaskHandle_t audioTaskHandle;

enum AppState { ACTIVE, SLEEP, POPUP, PAIRING };
volatile AppState currentState = ACTIVE;

void drawUI(); // 미리 선언

// ══════════════════════════════════════════════════════════
//  🛡️ 화면 전환 철벽 방어 함수 (연쇄 폭발 원천 차단)
// ══════════════════════════════════════════════════════════
void changeState(AppState newState) {
    currentState = newState;
    uiNeedsUpdate = true;
    
    // 💡 1. 팝업창이나 페어링 화면을 즉시 번쩍! 하고 띄워줍니다.
    drawUI(); 
    
    // 💡 2. 사용자가 누르고 있던 손가락을 완벽히 뗄 때까지 다음 코드로 안 넘어갑니다.
    while(digitalRead(BTN_L_MODE) == LOW || digitalRead(BTN_R_RECORD) == LOW) {
        delay(10);
    }
    
    // 💡 3. 손을 뗀 후 용수철이 파르르 떨리는 시간(Bounce) 0.15초 무시!
    delay(150); 
    
    // 💡 4. 타이머 초기화 (안전)
    lPressTime = 0; lLongPressed = false;
    rPressTime = 0; rLongPressed = false;
}

// ══════════════════════════════════════════════════════════
//  DSP Heart Rate
// ══════════════════════════════════════════════════════════
#define DSP_NOISE_FLOOR      0.0005f
#define DSP_THRESHOLD_RATIO  0.35f
#define DSP_PEAK_REFRACTORY  600
#define DSP_ENV_WINDOW       20
#define DSP_VAR_WINDOW       100
#define DSP_SIGNAL_VAR_MIN   0.00001f
#define DSP_BPM_TIMEOUT_MS   4000
#define DSP_HP_B0  0.93959f
#define DSP_HP_B1 -1.87919f
#define DSP_HP_B2  0.93959f
#define DSP_HP_A1 -1.87514f
#define DSP_HP_A2  0.88324f
#define DSP_LP_B0  0.06746f
#define DSP_LP_B1  0.13492f
#define DSP_LP_B2  0.06746f
#define DSP_LP_A1 -1.14298f
#define DSP_LP_A2  0.41282f

volatile float dsp_bpm = 0;
volatile float dsp_instant_bpm = 0;
volatile bool  dsp_beat_flag = false;
volatile int   dsp_beat_count = 0;

static float _hp_x1=0,_hp_x2=0,_hp_y1=0,_hp_y2=0;
static float _lp_x1=0,_lp_x2=0,_lp_y1=0,_lp_y2=0;
static float _env_buf[DSP_ENV_WINDOW]={0};
static int   _env_idx=0;
static float _env_sum=0;
static float _var_buf[DSP_VAR_WINDOW]={0};
static int   _var_idx=0;
static float _var_sum=0;
static float _var_sq_sum=0;
static float _running_max=0;
static int   _samples_since_peak=0;
static unsigned long _last_peak_ms=0;

void dsp_reset() {
    _hp_x1=0;_hp_x2=0;_hp_y1=0;_hp_y2=0;
    _lp_x1=0;_lp_x2=0;_lp_y1=0;_lp_y2=0;
    memset(_env_buf,0,sizeof(_env_buf)); _env_idx=0; _env_sum=0;
    memset(_var_buf,0,sizeof(_var_buf)); _var_idx=0; _var_sum=0; _var_sq_sum=0;
    _running_max=0; _samples_since_peak=0; _last_peak_ms=0;
    dsp_bpm=0; dsp_instant_bpm=0; dsp_beat_flag=false; dsp_beat_count=0;
}

void dsp_process(int16_t raw16) {
    float s = (float)raw16 / 32768.0f;
    float hp = DSP_HP_B0*s + DSP_HP_B1*_hp_x1 + DSP_HP_B2*_hp_x2 - DSP_HP_A1*_hp_y1 - DSP_HP_A2*_hp_y2;
    _hp_x2=_hp_x1; _hp_x1=s; _hp_y2=_hp_y1; _hp_y1=hp;
    float lp = DSP_LP_B0*hp + DSP_LP_B1*_lp_x1 + DSP_LP_B2*_lp_x2 - DSP_LP_A1*_lp_y1 - DSP_LP_A2*_lp_y2;
    _lp_x2=_lp_x1; _lp_x1=hp; _lp_y2=_lp_y1; _lp_y1=lp;
    float x2 = lp * lp;
    float se = 0;
    if (x2 > 1e-10f) {
        float x2n = (x2 > 1.0f) ? 1.0f : x2;
        se = -x2n * log10f(x2n + 1e-10f);
    }
    _env_sum -= _env_buf[_env_idx];
    _env_buf[_env_idx] = se;
    _env_sum += se;
    _env_idx = (_env_idx + 1) % DSP_ENV_WINDOW;
    float envelope = _env_sum / DSP_ENV_WINDOW;
    float old_v = _var_buf[_var_idx];
    _var_sum -= old_v;
    _var_sq_sum -= old_v * old_v;
    _var_buf[_var_idx] = envelope;
    _var_sum += envelope;
    _var_sq_sum += envelope * envelope;
    _var_idx = (_var_idx + 1) % DSP_VAR_WINDOW;
    float mean = _var_sum / DSP_VAR_WINDOW;
    float var = (_var_sq_sum / DSP_VAR_WINDOW) - (mean * mean);
    if (var < 0) var = 0;
    bool signal_ok = (var > DSP_SIGNAL_VAR_MIN) && (envelope > DSP_NOISE_FLOOR);
    unsigned long now = millis();
    if (_last_peak_ms > 0 && (now - _last_peak_ms) > DSP_BPM_TIMEOUT_MS) {
        dsp_bpm = 0; dsp_instant_bpm = 0; dsp_beat_count = 0;
        _last_peak_ms = 0; _running_max *= 0.5f;
    }
    if (!signal_ok) {
        _running_max *= 0.999f;
        _samples_since_peak++;
        return;
    }
    _running_max *= 0.9998f;
    if (envelope > _running_max) _running_max = envelope;
    float threshold = _running_max * DSP_THRESHOLD_RATIO;
    _samples_since_peak++;
    if (envelope > threshold && envelope > DSP_NOISE_FLOOR && _samples_since_peak > DSP_PEAK_REFRACTORY) {
        dsp_beat_count++;
        dsp_beat_flag = true;
        if (_last_peak_ms > 0) {
            float rr_ms = (float)(now - _last_peak_ms);
            if (rr_ms > 273.0f && rr_ms < 2000.0f) {
                dsp_instant_bpm = 60000.0f / rr_ms;
                if (dsp_bpm < 1.0f) dsp_bpm = dsp_instant_bpm;
                else dsp_bpm = dsp_bpm * 0.3f + dsp_instant_bpm * 0.7f;
            }
        }
        _last_peak_ms = now;
        _samples_since_peak = 0;
    }
}

// ══════════════════════════════════════════════════════════
//  BLE Callbacks
// ══════════════════════════════════════════════════════════
class MyServerCallbacks: public BLEServerCallbacks {
    void onConnect(BLEServer* pServer) { 
        deviceConnected = true; 
        if (currentState == PAIRING) {
            changeState(ACTIVE); 
        } else {
            uiNeedsUpdate = true;
        }
    }
    void onDisconnect(BLEServer* pServer) { 
        deviceConnected = false; 
        if (isRecording5Sec) {
            isRecording5Sec = false;
            tft.writecommand(0x11); 
            delay(120); 
            digitalWrite(TFT_BL, HIGH); 
        }
        isBleOn = false; 
        uiNeedsUpdate = true;
    }
};

class MyCmdCallbacks: public BLECharacteristicCallbacks {
    void onWrite(BLECharacteristic *pCharacteristic) {}
};

void enterDeepSleep() {
    tft.writecommand(0x10); 
    delay(5);
    digitalWrite(TFT_BL, LOW); 
    while(digitalRead(BTN_L_MODE) == LOW || digitalRead(BTN_R_RECORD) == LOW) { delay(10); }
    delay(200); 
    esp_sleep_enable_ext0_wakeup(GPIO_NUM_35, 0); 
    esp_deep_sleep_start(); 
}

// ══════════════════════════════════════════════════════════
//  🎧 오디오 처리 태스크 (필터 포함)
// ══════════════════════════════════════════════════════════
void audioTask(void *pvParameters) {
    static int prevMode = -1;
    float prev_heart_y = 0; 
    float prev_lung_x = 0;
    float prev_lung_y = 0;

    for(;;) { 
        int32_t sum = 0;
        for(int i=0; i<DECIMATION; i++) {
            int32_t sample = 0;
            size_t bytesIn = 0;
            i2s_read(I2S_PORT, &sample, sizeof(sample), &bytesIn, portMAX_DELAY);
            sum += (int16_t)(sample >> 16);
        }
        
        int32_t avg = sum / DECIMATION;
        float current_sample = (float)(avg << 2); 
        int16_t final_audio = 0;

        if (currentMode == 1) { 
            float heart_filtered = 0.24f * current_sample + 0.76f * prev_heart_y;
            prev_heart_y = heart_filtered;
            final_audio = (int16_t)heart_filtered;
        } 
        else if (currentMode == 2) { 
            float lung_filtered = 0.61f * prev_lung_y + 0.61f * (current_sample - prev_lung_x);
            prev_lung_x = current_sample;
            prev_lung_y = lung_filtered;
            final_audio = (int16_t)lung_filtered;
        } 
        else { 
            final_audio = (int16_t)current_sample;
        }

        if (final_audio > 32767) final_audio = 32767;
        else if (final_audio < -32768) final_audio = -32768;

        sharedAveragedSample = final_audio; 

        if (currentMode != prevMode) {
            if (currentMode == 1) dsp_reset();
            prevMode = currentMode;
        }
        if (currentMode == 1) dsp_process(final_audio);

        if (currentState == ACTIVE && deviceConnected && isRecording5Sec) {
            bleBuffer[bufferIndex++] = final_audio;
            if (bufferIndex >= BLE_BUFFER_SIZE) {
                pAudioChar->setValue((uint8_t*)bleBuffer, BLE_BUFFER_SIZE * sizeof(int16_t));
                pAudioChar->notify();
                bufferIndex = 0;
            }
        }
    }
}

// ══════════════════════════════════════════════════════════
//  Setup
// ══════════════════════════════════════════════════════════
void setup() {
    Serial.begin(115200);
    pinMode(BTN_L_MODE, INPUT);        
    pinMode(BTN_R_RECORD, INPUT_PULLUP);
    pinMode(TFT_BL, OUTPUT);
    digitalWrite(TFT_BL, HIGH); 

    if (esp_sleep_get_wakeup_cause() == ESP_SLEEP_WAKEUP_EXT0) {
        while(digitalRead(BTN_L_MODE) == LOW) { delay(10); }
        delay(200); 
    }

    tft.init();
    tft.writecommand(0x11); 
    delay(120);
    
    tft.setRotation(2); 
    tft.fillScreen(TFT_BLACK); 

    const i2s_config_t i2s_config = {
        .mode = i2s_mode_t(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate = SAMPLE_RATE,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
        .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
        .communication_format = i2s_comm_format_t(I2S_COMM_FORMAT_STAND_I2S),
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count = 8,
        .dma_buf_len = 64,
        .use_apll = false
    };
    const i2s_pin_config_t pin_config = {
        .bck_io_num = I2S_SCK, .ws_io_num = I2S_WS, .data_out_num = I2S_PIN_NO_CHANGE, .data_in_num = I2S_SD
    };
    i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);
    i2s_set_pin(I2S_PORT, &pin_config);

    BLEDevice::init("Stetho_LilyGO"); 
    pServer = BLEDevice::createServer();
    pServer->setCallbacks(new MyServerCallbacks());
    BLEService *pService = pServer->createService(SERVICE_UUID);
    pAudioChar = pService->createCharacteristic(AUDIO_CHAR_UUID, BLECharacteristic::PROPERTY_NOTIFY);
    pAudioChar->addDescriptor(new BLE2902());
    pCmdChar = pService->createCharacteristic(CMD_CHAR_UUID, BLECharacteristic::PROPERTY_WRITE);
    pCmdChar->setCallbacks(new MyCmdCallbacks());
    pService->start();

    xTaskCreatePinnedToCore(audioTask, "AudioTask", 10000, NULL, 2, &audioTaskHandle, 0);
}

// ══════════════════════════════════════════════════════════
//  UI Drawing Functions
// ══════════════════════════════════════════════════════════
void drawWaveform(int16_t audioSample) {
    if (currentState != ACTIVE || isRecording5Sec || currentMode == 3) return;
    if (millis() - lastDrawTime < drawInterval) return;
    lastDrawTime = millis();

    static int16_t smoothedSample = 0;
    int displayRange = 8000; 
    uint32_t waveColor;
    int16_t *lastYPtr;

    if (currentMode == 1) {
        smoothedSample = (smoothedSample * 5 + audioSample) / 6; 
        displayRange = 6000; 
        waveColor = TFT_RED;
        lastYPtr = &lastY_HEART_WAVE;
    } else {
        smoothedSample = audioSample; 
        displayRange = 10000; 
        waveColor = TFT_CYAN;
        lastYPtr = &lastY_LUNG_WAVE; 
    }

    long mappedY = map(smoothedSample, -displayRange, displayRange, WAVE_Y_BOTTOM, WAVE_Y_TOP);
    int16_t currY = constrain((int)mappedY, WAVE_Y_TOP, WAVE_Y_BOTTOM);

    tft.drawFastVLine(graphX + 1, WAVE_Y_TOP, WAVE_HEIGHT, TFT_BLACK);
    tft.drawFastVLine(graphX + 2, WAVE_Y_TOP, WAVE_HEIGHT, TFT_BLACK); 
    if (graphX % 5 == 0) tft.drawPixel(graphX + 1, WAVE_ZERO_Y, 0x39E7); 

    tft.drawLine(graphX, *lastYPtr, graphX + 1, currY, waveColor);
    tft.drawLine(graphX, *lastYPtr + 1, graphX + 1, currY + 1, waveColor); 

    *lastYPtr = currY;
    graphX++;

    if (graphX >= WAVE_X_END) {
        graphX = WAVE_X_START;
        tft.drawFastVLine(WAVE_X_START, WAVE_Y_TOP, WAVE_HEIGHT, TFT_BLACK); 
        tft.drawFastVLine(WAVE_X_START + 1, WAVE_Y_TOP, WAVE_HEIGHT, TFT_BLACK);
    }
}

static float lastDisplayedBPM = -1;
static unsigned long lastBpmDrawTime = 0;

void drawBPM() {
    if (currentState != ACTIVE || currentMode != 1 || isRecording5Sec) return;
    if (millis() - lastBpmDrawTime < 500) return;
    lastBpmDrawTime = millis();

    float bpm = dsp_bpm;
    if (fabsf(bpm - lastDisplayedBPM) < 0.5f && lastDisplayedBPM >= 0) return;
    lastDisplayedBPM = bpm;

    tft.fillRect(0, BPM_Y, tft.width(), BPM_H, TFT_BLACK);
    int cx = tft.width() / 2;

    if (bpm > 20 && bpm < 250) {
        int hx = cx - 40, hy = BPM_Y + 12;
        tft.fillCircle(hx - 4, hy, 4, TFT_RED);
        tft.fillCircle(hx + 4, hy, 4, TFT_RED);
        tft.fillTriangle(hx - 8, hy + 1, hx + 8, hy + 1, hx, hy + 10, TFT_RED);

        tft.setTextDatum(ML_DATUM);
        tft.setTextColor(TFT_WHITE, TFT_BLACK);
        tft.setTextPadding(60);
        char buf[8];
        sprintf(buf, "%d", (int)bpm);
        tft.drawString(buf, cx - 25, BPM_Y + 15, 4);

        tft.setTextColor(TFT_DARKGREY, TFT_BLACK);
        tft.setTextPadding(0);
        tft.drawString("bpm", cx + 20, BPM_Y + 17, 2);
    } else {
        tft.setTextDatum(MC_DATUM);
        tft.setTextColor(TFT_DARKGREY, TFT_BLACK);
        tft.setTextPadding(tft.width());
        tft.drawString("measuring...", cx, BPM_Y + 15, 2);
        tft.setTextPadding(0);
    }
}

void clearBPM() {
    tft.fillRect(0, BPM_Y, tft.width(), BPM_H, TFT_BLACK);
    lastDisplayedBPM = -1;
    lastBpmDrawTime = 0;
}

static unsigned long lastBeatFlash = 0;

void drawBeatIndicator() {
    if (currentState != ACTIVE || currentMode != 1) return;
    if (dsp_beat_flag) {
        dsp_beat_flag = false;
        tft.fillCircle(12, 8, 4, TFT_RED);
        lastBeatFlash = millis();
    } else if (lastBeatFlash > 0 && millis() - lastBeatFlash > 100) {
        tft.fillCircle(12, 8, 4, TFT_BLACK);
        lastBeatFlash = 0;
    }
}

void drawStatusBar() {
    tft.fillRect(0, 0, tft.width(), 16, TFT_BLACK);

    uint16_t btColor = deviceConnected ? TFT_CYAN : TFT_DARKGREY;
    if (isBleOn && !deviceConnected) btColor = TFT_ORANGE; 
    
    int cx = tft.width() - 15; 
    int cy = 8; 

    tft.drawLine(cx, cy-6, cx, cy+6, btColor);     
    tft.drawLine(cx, cy-6, cx+4, cy-3, btColor);   
    tft.drawLine(cx+4, cy-3, cx, cy, btColor);     
    tft.drawLine(cx, cy, cx+4, cy+3, btColor);     
    tft.drawLine(cx+4, cy+3, cx, cy+6, btColor);   
    tft.drawLine(cx-4, cy-2, cx+4, cy+4, btColor); 
    tft.drawLine(cx-4, cy+4, cx+4, cy-2, btColor); 
}

void drawUI() {
    if (!uiNeedsUpdate || isRecording5Sec) return;

    int cx = tft.width() / 2;
    tft.setTextDatum(MC_DATUM); 
    tft.setTextPadding(tft.width()); 

    static int lastDrawnMode = -1;
    static bool lastDrawnConn = false;
    static bool lastDrawnBle = false; 
    static AppState lastDrawnState = ACTIVE;

    if (currentMode != lastDrawnMode || deviceConnected != lastDrawnConn || isBleOn != lastDrawnBle || currentState != lastDrawnState) {
        tft.fillRect(0, 16, tft.width(), tft.height() - 52, TFT_BLACK); 
        graphX = WAVE_X_START; 
        lastDrawnMode = currentMode;
        lastDrawnConn = deviceConnected;
        lastDrawnBle = isBleOn;
        lastDrawnState = currentState;
        clearBPM();
        drawStatusBar(); 
    }

    if (currentState == POPUP) {
        tft.setTextColor(TFT_ORANGE, TFT_BLACK);
        tft.drawString("! WARNING !", cx, 40, 2);
        tft.setTextColor(TFT_WHITE, TFT_BLACK);
        tft.drawString("To analyse,", cx, 70, 2);
        tft.drawString("Bluetooth has to", cx, 90, 2);
        tft.drawString("be connected.", cx, 110, 2);
        
        tft.setTextColor(TFT_GREEN, TFT_BLACK);
        tft.drawString("Would u like to pair?", cx, 140, 1);

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
        if (currentMode == 1) {
            tft.setTextColor(TFT_RED, TFT_BLACK);
            tft.drawString("HEART", cx, 30, 4); 
        } 
        else if (currentMode == 2) {
            tft.setTextColor(TFT_CYAN, TFT_BLACK);
            tft.drawString("LUNG", cx, 30, 4); 
        } 
        else if (currentMode == 3) {
            tft.setTextColor(TFT_YELLOW, TFT_BLACK);
            tft.drawString("PATIENT LIST", cx, 35, 2);       
            tft.setTextColor(TFT_DARKGREY, TFT_BLACK);
            tft.drawString("- COMING SOON -", cx, 90, 2);
        }

        tft.drawFastHLine(10, tft.height() - 35, tft.width() - 20, TFT_DARKGREY);
        tft.setTextColor(TFT_LIGHTGREY, TFT_BLACK);
        
        if (currentMode == 1 || currentMode == 2) {
            tft.drawString("L: MODE | R: RECORD", cx, tft.height() - 15, 1);
        } else {
            tft.drawString("L: MODE | R: SELECT", cx, tft.height() - 15, 1);
        }
    }

    tft.setTextPadding(0); 
    uiNeedsUpdate = false; 
}

// ══════════════════════════════════════════════════════════
//  Main Loop
// ══════════════════════════════════════════════════════════
void loop() {
    if (isRecording5Sec) {
        if (millis() - recordStartTime >= 5000) {
            isRecording5Sec = false; 
            tft.writecommand(0x11); 
            delay(120); 
            digitalWrite(TFT_BL, HIGH); 
            uiNeedsUpdate = true; 
        }
        return; 
    }

    bool currL = digitalRead(BTN_L_MODE);
    bool currR = digitalRead(BTN_R_RECORD);

    // ──────────────────────────────────────────────
    // 1. POPUP 상태 (경고창)
    // ──────────────────────────────────────────────
    if (currentState == POPUP) {
        if (currL == LOW) {
            if (lPressTime == 0) lPressTime = millis();
        } else if (lPressTime != 0) {
            if (millis() - lPressTime > 20) {
                changeState(ACTIVE); 
            }
            lPressTime = 0;
        }
        
        if (currR == LOW) {
            if (rPressTime == 0) rPressTime = millis();
        } else if (rPressTime != 0) {
            if (millis() - rPressTime > 20) {
                isBleOn = true;
                BLEDevice::startAdvertising(); 
                changeState(PAIRING); 
            }
            rPressTime = 0;
        }
    }
    // ──────────────────────────────────────────────
    // 2. PAIRING 상태 (페어링 대기)
    // ──────────────────────────────────────────────
    else if (currentState == PAIRING) {
        if (currR == LOW) {
            if (rPressTime == 0) rPressTime = millis();
        } else if (rPressTime != 0) {
            if (millis() - rPressTime > 20) {
                isBleOn = false;
                BLEDevice::stopAdvertising(); 
                changeState(ACTIVE); 
            }
            rPressTime = 0;
        }
    }
    // ──────────────────────────────────────────────
    // 3. ACTIVE 상태 (일반 모드)
    // ──────────────────────────────────────────────
    else if (currentState == ACTIVE) {
        
        // --- L 버튼 (모드 변경 및 딥슬립) ---
        if (currL == LOW) { 
            if (lPressTime == 0) lPressTime = millis(); 
            if (millis() - lPressTime > 1500 && !lLongPressed) {
                lLongPressed = true;
                enterDeepSleep(); 
            }
        } else if (lPressTime != 0) { 
            if (!lLongPressed && millis() - lPressTime > 20) {
                currentMode++;
                if (currentMode > 3) currentMode = 1;
                uiNeedsUpdate = true;
                drawUI(); // 즉시 반응
            }
            lPressTime = 0; 
            lLongPressed = false;
        }

        // --- R 버튼 (녹음 또는 팝업 띄우기) ---
        if (currR == LOW) {
            if (rPressTime == 0) rPressTime = millis();
        } else if (rPressTime != 0) {
            if (!rLongPressed && millis() - rPressTime > 20) {
                if (currentMode == 1 || currentMode == 2) {
                    if (deviceConnected) {
                        isRecording5Sec = true;
                        recordStartTime = millis();
                        tft.writecommand(0x10); 
                        delay(5);
                        digitalWrite(TFT_BL, LOW); 
                    } else {
                        changeState(POPUP); 
                    }
                } 
                else if (currentMode == 3) {
                    // PATIENT LIST 선택 버튼 (나중에 구현)
                }
            }
            rPressTime = 0; 
            rLongPressed = false;
        }
    }

    drawUI();

    if (currentState == ACTIVE && (currentMode == 1 || currentMode == 2) && !isRecording5Sec) {
        drawWaveform(sharedAveragedSample);
        if (currentMode == 1) {
            drawBPM();
            drawBeatIndicator();
        }
    }
}