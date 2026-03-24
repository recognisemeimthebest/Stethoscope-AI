#include <Arduino.h>
#include <driver/i2s.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>
#include <TFT_eSPI.h> 
#include <SPI.h>
#include <math.h>
#include <esp_sleep.h> // 💡 하드웨어 딥슬립을 위한 공식 라이브러리 추가

TFT_eSPI tft = TFT_eSPI(); 

#define BTN_R 35 // 오른쪽 (전원/모드)
#define BTN_L 0  // 왼쪽
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

// 더 이상 소프트웨어 가짜 SLEEP 상태가 필요 없습니다.
volatile int16_t sharedAveragedSample = 0;

unsigned long rPressTime = 0;
bool rLongPressed = false;
unsigned long lPressTime = 0;
bool lLongPressed = false;

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

    float hp = DSP_HP_B0*s + DSP_HP_B1*_hp_x1 + DSP_HP_B2*_hp_x2
                            - DSP_HP_A1*_hp_y1 - DSP_HP_A2*_hp_y2;
    _hp_x2=_hp_x1; _hp_x1=s; _hp_y2=_hp_y1; _hp_y1=hp;

    float lp = DSP_LP_B0*hp + DSP_LP_B1*_lp_x1 + DSP_LP_B2*_lp_x2
                             - DSP_LP_A1*_lp_y1 - DSP_LP_A2*_lp_y2;
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

    if (envelope > threshold &&
        envelope > DSP_NOISE_FLOOR &&
        _samples_since_peak > DSP_PEAK_REFRACTORY) {

        dsp_beat_count++;
        dsp_beat_flag = true;

        if (_last_peak_ms > 0) {
            float rr_ms = (float)(now - _last_peak_ms);
            if (rr_ms > 273.0f && rr_ms < 2000.0f) {
                dsp_instant_bpm = 60000.0f / rr_ms;
                if (dsp_bpm < 1.0f)
                    dsp_bpm = dsp_instant_bpm;
                else
                    dsp_bpm = dsp_bpm * 0.3f + dsp_instant_bpm * 0.7f;
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
        uiNeedsUpdate = true; 
    }
    void onDisconnect(BLEServer* pServer) { 
        deviceConnected = false; 
        isRecording5Sec = false; 
        uiNeedsUpdate = true;
        if (isBleOn) BLEDevice::startAdvertising(); 
    }
};

class MyCmdCallbacks: public BLECharacteristicCallbacks {
    void onWrite(BLECharacteristic *pCharacteristic) {}
};

// ══════════════════════════════════════════════════════════
//  🛡️ 궁극의 하드웨어 딥슬립 (물리적 전원 차단)
// ══════════════════════════════════════════════════════════

void enterDeepSleep() {
    // 1. 화면 및 백라이트 물리적 차단
    tft.writecommand(0x10); 
    delay(5);
    digitalWrite(TFT_BL, LOW); 
    
    // 2. 버튼에서 손을 완벽히 뗄 때까지 대기
    while(digitalRead(BTN_R) == LOW || digitalRead(BTN_L) == LOW) { delay(10); }
    delay(200); // 떼고 나서 생기는 기계적 진동(노이즈) 무시

    // 3. 🧠 CPU에게 "R버튼(GPIO 35)이 LOW가 되면 그때 깨어나라"고 예약
    esp_sleep_enable_ext0_wakeup(GPIO_NUM_35, 0);

    // 4. 🧠 진짜로 뇌 전원을 뽑아버림 (이후 코드는 영원히 실행되지 않음)
    esp_deep_sleep_start(); 
}

// ══════════════════════════════════════════════════════════
//  Audio Task (Core 0)
// ══════════════════════════════════════════════════════════

void audioTask(void *pvParameters) {
    static int prevMode = -1;

    for(;;) { 
        int32_t sum = 0;
        for(int i=0; i<DECIMATION; i++) {
            int32_t sample = 0;
            size_t bytesIn = 0;
            i2s_read(I2S_PORT, &sample, sizeof(sample), &bytesIn, portMAX_DELAY);
            sum += (int16_t)(sample >> 16);
        }
        int32_t avg = sum / DECIMATION;
        int32_t amplified = avg << 2; 
        if (amplified > 32767) amplified = 32767;
        else if (amplified < -32768) amplified = -32768;

        sharedAveragedSample = (int16_t)amplified; 

        int mode = currentMode;
        if (mode != prevMode) {
            if (mode == 1) dsp_reset();
            prevMode = mode;
        }
        if (mode == 1) dsp_process((int16_t)amplified);

        if (deviceConnected && isRecording5Sec) {
            bleBuffer[bufferIndex++] = (int16_t)amplified;
            if (bufferIndex >= BLE_BUFFER_SIZE) {
                pAudioChar->setValue((uint8_t*)bleBuffer, BLE_BUFFER_SIZE * sizeof(int16_t));
                pAudioChar->notify();
                bufferIndex = 0;
            }
        }
    }
}

// ══════════════════════════════════════════════════════════
//  Setup & UI
// ══════════════════════════════════════════════════════════

void setup() {
    Serial.begin(115200);
    pinMode(BTN_R, INPUT);        
    pinMode(BTN_L, INPUT_PULLUP);
    pinMode(TFT_BL, OUTPUT);
    digitalWrite(TFT_BL, HIGH); 

    // 💡 딥슬립에서 R버튼을 눌러 깨어났다면, 손을 완전히 뗄 때까지 여기서 대기합니다.
    // 안 그러면 켜지자마자 "버튼이 눌려있네? 다음 모드로 가야지!" 하고 오작동합니다.
    if (esp_sleep_get_wakeup_cause() == ESP_SLEEP_WAKEUP_EXT0) {
        while(digitalRead(BTN_R) == LOW) { delay(10); }
        delay(200); 
    }

    tft.init();
    tft.setRotation(0); 
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

void drawWaveform(int16_t audioSample) {
    if (currentMode == 3 || isRecording5Sec) return;
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
    if (currentMode != 1 || isRecording5Sec) return;
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
    if (currentMode != 1) return;
    if (dsp_beat_flag) {
        dsp_beat_flag = false;
        tft.fillCircle(12, 12, 5, TFT_RED);
        lastBeatFlash = millis();
    } else if (lastBeatFlash > 0 && millis() - lastBeatFlash > 100) {
        tft.fillCircle(12, 12, 5, TFT_BLACK);
        lastBeatFlash = 0;
    }
}

void drawUI() {
    if (!uiNeedsUpdate || isRecording5Sec) return;

    int cx = tft.width() / 2;
    tft.setTextDatum(MC_DATUM); 
    tft.setTextPadding(tft.width()); 

    static int lastDrawnMode = -1;
    static bool lastDrawnConn = false;
    if (currentMode != lastDrawnMode || deviceConnected != lastDrawnConn) {
        tft.fillRect(0, 0, tft.width(), tft.height() - 36, TFT_BLACK);
        graphX = WAVE_X_START; 
        lastDrawnMode = currentMode;
        lastDrawnConn = deviceConnected;
        clearBPM();
    }

    if (deviceConnected) {
        tft.fillCircle(tft.width() - 12, 12, 5, TFT_GREEN);
    } else {
        tft.fillCircle(tft.width() - 12, 12, 5, TFT_BLACK);
    }

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
        tft.drawString("AI", cx, 20, 4);       
        tft.drawString("CONNECT", cx, 45, 4);  
        
        if (deviceConnected) {
            tft.setTextColor(TFT_GREEN, TFT_BLACK);
            tft.drawString("CONNECTED", cx, 100, 4); 
        } else if (isBleOn) {
            tft.setTextColor(TFT_WHITE, TFT_BLACK);
            tft.drawString("WAITING...", cx, 100, 2);
        } else {
            tft.setTextColor(TFT_DARKGREY, TFT_BLACK);
            tft.drawString("HOLD L", cx, 95, 2);
            tft.drawString("TO PAIR", cx, 115, 2);
        }
    }

    tft.drawFastHLine(10, tft.height() - 35, tft.width() - 20, TFT_DARKGREY);
    tft.setTextColor(TFT_LIGHTGREY, TFT_BLACK);
    
    if (currentMode == 1 || currentMode == 2) {
        if (deviceConnected) {
            tft.drawString("L: RECORD | R: MODE", cx, tft.height() - 15, 1);
        } else {
            tft.drawString("L: --- | R: MODE", cx, tft.height() - 15, 1);
        }
    } else {
        tft.drawString("HOLD L: PAIR | R: MODE", cx, tft.height() - 15, 1);
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

    // --- R 버튼 (모드 변경 및 전원 OFF) ---
    bool currR = digitalRead(BTN_R);
    if (currR == LOW) { 
        if (rPressTime == 0) rPressTime = millis(); 
        
        // 💡 1.5초간 눌렀다면 딥슬립 진입
        if (millis() - rPressTime > 1500 && !rLongPressed) {
            rLongPressed = true;
            enterDeepSleep(); // 이 함수를 호출하면 보드 전원이 꺼지며 영영 돌아오지 않습니다.
        }
    } else { 
        if (rPressTime != 0) { 
            // 길게 누르지 않고 짧게 뗐다면 모드 변경
            if (!rLongPressed && millis() - rPressTime > 50) {
                currentMode++;
                if (currentMode > 3) currentMode = 1;
                uiNeedsUpdate = true;
            }
            rPressTime = 0; 
            rLongPressed = false;
        }
    }

    // --- L 버튼 (BLE 켜기 or 녹음 시작) ---
    bool currL = digitalRead(BTN_L);
    if (currL == LOW) {
        if (lPressTime == 0) lPressTime = millis();
        if (currentMode == 3 && !lLongPressed && millis() - lPressTime > 1000) {
            lLongPressed = true;
            if (!isBleOn && !deviceConnected) {
                isBleOn = true;
                BLEDevice::startAdvertising(); 
                uiNeedsUpdate = true;
            }
        }
    } else {
        if (lPressTime != 0) {
            if (!lLongPressed && millis() - lPressTime > 50) {
                if (deviceConnected && (currentMode == 1 || currentMode == 2)) {
                    isRecording5Sec = true;
                    recordStartTime = millis();
                    tft.writecommand(0x10); 
                    delay(5);
                    digitalWrite(TFT_BL, LOW); 
                }
            }
            lPressTime = 0; 
            lLongPressed = false;
        }
    }

    drawUI();

    if ((currentMode == 1 || currentMode == 2) && !isRecording5Sec) {
        drawWaveform(sharedAveragedSample);
        if (currentMode == 1) {
            drawBPM();
            drawBeatIndicator();
        }
    }
}