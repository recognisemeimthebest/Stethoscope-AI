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

const unsigned long RECORD_DURATION_MS = 5000;
const unsigned long TFT_WAKEUP_DELAY_MS = 120;
const unsigned long BPM_UPDATE_INTERVAL_MS = 500;
const unsigned long BEAT_FLASH_DURATION_MS = 100;

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
volatile int currentMode = 3; 
volatile bool uiNeedsUpdate = true; 

volatile bool isRecording5Sec = false; 
unsigned long recordStartTime = 0;

volatile int16_t sharedAveragedSample = 0;

// 💡 [핵심] 표 형식 출력을 위한 환자 구조체
struct PatientInfo {
    String id;
    String name;
    String ward;
};
PatientInfo patientList[10];
int patientCount = 0;
int currentPatIdx = 0;
String selectedPatient = "NONE";

const int SAMPLE_RATE = 16000; 
const int TARGET_RATE = 2000;
const int DECIMATION = SAMPLE_RATE / TARGET_RATE;
const int BLE_BUFFER_SIZE = 50; 
int16_t bleBuffer[BLE_BUFFER_SIZE];
int bufferIndex = 0;

// 💡 파형 구역을 아래로 내려서 글씨와 겹치지 않게 함
#define WAVE_Y_TOP 75
#define WAVE_Y_BOTTOM 150
#define WAVE_HEIGHT (WAVE_Y_BOTTOM - WAVE_Y_TOP)
#define WAVE_ZERO_Y (WAVE_Y_TOP + (WAVE_HEIGHT / 2)) 
#define WAVE_X_START 10
#define WAVE_X_END 125

#define BPM_Y 165
#define BPM_H 40

int graphX = WAVE_X_START; 
int16_t lastY_HEART_WAVE = WAVE_ZERO_Y; 
int16_t lastY_LUNG_WAVE = WAVE_ZERO_Y;  
unsigned long lastDrawTime = 0;
const int drawInterval = 15; 

#define AUDIO_TASK_STACK_SIZE 10000
StackType_t audioTaskStack[AUDIO_TASK_STACK_SIZE];
StaticTask_t audioTaskBuffer;
TaskHandle_t audioTaskHandle;

enum AppState { ACTIVE, SLEEP, POPUP, PAIRING };
volatile AppState currentState = ACTIVE;

void drawUI(); 

// ══════════════════════════════════════════════════════════
//  스마트 버튼 클래스 
// ══════════════════════════════════════════════════════════
class Button {
public:
    int pin;
    bool lastReading;
    unsigned long pressTime;
    bool longPressFired;

    Button(int p) { pin = p; lastReading = HIGH; pressTime = 0; longPressFired = false; }
    void begin() { lastReading = digitalRead(pin); }

    int update() {
        bool reading = digitalRead(pin);
        int event = 0;
        if (reading == LOW && lastReading == HIGH) { pressTime = millis(); longPressFired = false; } 
        else if (reading == LOW && lastReading == LOW) {
            if (pressTime > 0 && !longPressFired && (millis() - pressTime > 1500)) {
                longPressFired = true; event = 2; // 길게 누름
            }
        } 
        else if (reading == HIGH && lastReading == LOW) {
            if (pressTime > 0 && !longPressFired && (millis() - pressTime > 50)) event = 1; // 짧게 누름
            pressTime = 0;
        }
        lastReading = reading;
        return event;
    }
    void reset() { lastReading = digitalRead(pin); pressTime = 0; longPressFired = false; }
};

Button btnL(BTN_L_MODE);
Button btnR(BTN_R_RECORD);

void changeState(AppState newState) {
    currentState = newState;
    uiNeedsUpdate = true;
    drawUI(); 
    btnL.reset(); btnR.reset();
}

// ══════════════════════════════════════════════════════════
//  DSP 필터 생략 (기존 유지)
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
    _env_sum -= _env_buf[_env_idx]; _env_buf[_env_idx] = se; _env_sum += se;
    _env_idx = (_env_idx + 1) % DSP_ENV_WINDOW;
    float envelope = _env_sum / DSP_ENV_WINDOW;
    float old_v = _var_buf[_var_idx];
    _var_sum -= old_v; _var_sq_sum -= old_v * old_v;
    _var_buf[_var_idx] = envelope; _var_sum += envelope; _var_sq_sum += envelope * envelope;
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
    if (!signal_ok) { _running_max *= 0.999f; _samples_since_peak++; return; }
    _running_max *= 0.9998f;
    if (envelope > _running_max) _running_max = envelope;
    float threshold = _running_max * DSP_THRESHOLD_RATIO;
    _samples_since_peak++;
    if (envelope > threshold && envelope > DSP_NOISE_FLOOR && _samples_since_peak > DSP_PEAK_REFRACTORY) {
        dsp_beat_count++; dsp_beat_flag = true;
        if (_last_peak_ms > 0) {
            float rr_ms = (float)(now - _last_peak_ms);
            if (rr_ms > 273.0f && rr_ms < 2000.0f) {
                dsp_instant_bpm = 60000.0f / rr_ms;
                if (dsp_bpm < 1.0f) dsp_bpm = dsp_instant_bpm;
                else dsp_bpm = dsp_bpm * 0.3f + dsp_instant_bpm * 0.7f;
            }
        }
        _last_peak_ms = now; _samples_since_peak = 0;
    }
}

// ══════════════════════════════════════════════════════════
//  BLE Callbacks 
// ══════════════════════════════════════════════════════════
class MyServerCallbacks: public BLEServerCallbacks {
    void onConnect(BLEServer* pServer) { deviceConnected = true; }
    void onDisconnect(BLEServer* pServer) { deviceConnected = false; }
};

class MyCmdCallbacks: public BLECharacteristicCallbacks {
    void onWrite(BLECharacteristic *pCharacteristic) {
        String rxValue = pCharacteristic->getValue().c_str();
        
        // 💡 [핵심] 웹에서 날아온 "PAT:P001|Kim|301,P002|Lee|302" 파싱하기
        if(rxValue.startsWith("PAT:")) {
            String data = rxValue.substring(4);
            patientCount = 0; currentPatIdx = 0;
            
            int start = 0;
            while(start < data.length() && patientCount < 10) {
                int end = data.indexOf(',', start);
                if (end == -1) end = data.length();
                String patStr = data.substring(start, end);

                int p1 = patStr.indexOf('|');
                int p2 = patStr.indexOf('|', p1 + 1);

                if (p1 != -1 && p2 != -1) {
                    patientList[patientCount].id = patStr.substring(0, p1);
                    patientList[patientCount].name = patStr.substring(p1 + 1, p2);
                    patientList[patientCount].ward = patStr.substring(p2 + 1);
                    patientCount++;
                }
                start = end + 1;
            }
            uiNeedsUpdate = true;
        }
    }
};

MyServerCallbacks serverCallbacks;
MyCmdCallbacks cmdCallbacks;
BLE2902 audioDescriptor;
BLE2902 cmdDescriptor;

void enterDeepSleep() {
    tft.writecommand(0x10); 
    delay(5); digitalWrite(TFT_BL, LOW); 
    while(digitalRead(BTN_L_MODE) == LOW || digitalRead(BTN_R_RECORD) == LOW) { delay(10); }
    delay(200); esp_sleep_enable_ext0_wakeup(GPIO_NUM_35, 0); esp_deep_sleep_start(); 
}

// ══════════════════════════════════════════════════════════
//  오디오 처리 태스크 
// ══════════════════════════════════════════════════════════
void audioTask(void *pvParameters) {
    static int prevMode = -1;
    float prev_heart_y = 0; float prev_lung_x = 0; float prev_lung_y = 0;

    for(;;) { 
        int32_t sum = 0;
        for(int i=0; i<DECIMATION; i++) {
            int32_t sample = 0; size_t bytesIn = 0;
            i2s_read(I2S_PORT, &sample, sizeof(sample), &bytesIn, portMAX_DELAY);
            sum += (int16_t)(sample >> 16);
        }
        
        int32_t avg = sum / DECIMATION;
        float current_sample = (float)(avg << 2); 
        int16_t final_audio = 0;

        if (currentMode == 1) { 
            float heart_filtered = 0.24f * current_sample + 0.76f * prev_heart_y;
            prev_heart_y = heart_filtered; final_audio = (int16_t)heart_filtered;
        } 
        else if (currentMode == 2) { 
            float lung_filtered = 0.61f * prev_lung_y + 0.61f * (current_sample - prev_lung_x);
            prev_lung_x = current_sample; prev_lung_y = lung_filtered; final_audio = (int16_t)lung_filtered;
        } 
        else { final_audio = (int16_t)current_sample; }

        if (final_audio > 32767) final_audio = 32767;
        else if (final_audio < -32768) final_audio = -32768;

        sharedAveragedSample = final_audio; 

        if (currentMode != prevMode) { if (currentMode == 1) dsp_reset(); prevMode = currentMode; }
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
    pinMode(BTN_L_MODE, INPUT); pinMode(BTN_R_RECORD, INPUT_PULLUP);
    pinMode(TFT_BL, OUTPUT); digitalWrite(TFT_BL, HIGH); 

    if (esp_sleep_get_wakeup_cause() == ESP_SLEEP_WAKEUP_EXT0) {
        while(digitalRead(BTN_L_MODE) == LOW) { delay(10); } delay(200); 
    }

    tft.init(); tft.writecommand(0x11); delay(TFT_WAKEUP_DELAY_MS);
    tft.setRotation(2); tft.fillScreen(TFT_BLACK); 

    btnL.begin(); btnR.begin();

    const i2s_config_t i2s_config = {
        .mode = i2s_mode_t(I2S_MODE_MASTER | I2S_MODE_RX), .sample_rate = SAMPLE_RATE,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT, .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
        .communication_format = i2s_comm_format_t(I2S_COMM_FORMAT_STAND_I2S),
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1, .dma_buf_count = 8, .dma_buf_len = 64, .use_apll = false
    };
    const i2s_pin_config_t pin_config = {
        .bck_io_num = I2S_SCK, .ws_io_num = I2S_WS, .data_out_num = I2S_PIN_NO_CHANGE, .data_in_num = I2S_SD
    };
    i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL); i2s_set_pin(I2S_PORT, &pin_config);

    BLEDevice::init("AI_Stetho"); // 💡 블루투스 이름 확실하게 고정!
    BLEDevice::setMTU(512);
    pServer = BLEDevice::createServer();
    pServer->setCallbacks(&serverCallbacks); 
    
    BLEService *pService = pServer->createService(SERVICE_UUID);
    pAudioChar = pService->createCharacteristic(AUDIO_CHAR_UUID, BLECharacteristic::PROPERTY_NOTIFY);
    pAudioChar->addDescriptor(&audioDescriptor); 
    
    pCmdChar = pService->createCharacteristic(CMD_CHAR_UUID, BLECharacteristic::PROPERTY_WRITE | BLECharacteristic::PROPERTY_NOTIFY);
    pCmdChar->setCallbacks(&cmdCallbacks); 
    pCmdChar->addDescriptor(&cmdDescriptor);

    pService->start();

    // 💡 웹에서 잘 찾도록 Advertising 옵션 활성화
    BLEAdvertising *pAdvertising = BLEDevice::getAdvertising();
    pAdvertising->addServiceUUID(SERVICE_UUID);
    pAdvertising->setScanResponse(true); 
    pAdvertising->setMinPreferred(0x06); 
    pAdvertising->setMinPreferred(0x12);

    audioTaskHandle = xTaskCreateStaticPinnedToCore(
        audioTask, "AudioTask", AUDIO_TASK_STACK_SIZE, NULL, 2, audioTaskStack, &audioTaskBuffer, 0
    );
}

// ══════════════════════════════════════════════════════════
//  UI Drawing Functions
// ══════════════════════════════════════════════════════════
void drawWaveform(int16_t audioSample) {
    if (currentState != ACTIVE || isRecording5Sec || currentMode == 3) return;
    if (millis() - lastDrawTime < drawInterval) return;
    lastDrawTime = millis();

    static int16_t smoothedSample = 0;
    int displayRange = 8000; uint32_t waveColor; int16_t *lastYPtr;

    if (currentMode == 1) {
        smoothedSample = (smoothedSample * 5 + audioSample) / 6; displayRange = 6000; 
        waveColor = TFT_RED; lastYPtr = &lastY_HEART_WAVE;
    } else {
        smoothedSample = audioSample; displayRange = 10000; 
        waveColor = TFT_CYAN; lastYPtr = &lastY_LUNG_WAVE; 
    }

    long mappedY = map(smoothedSample, -displayRange, displayRange, WAVE_Y_BOTTOM, WAVE_Y_TOP);
    int16_t currY = constrain((int)mappedY, WAVE_Y_TOP, WAVE_Y_BOTTOM);

    tft.drawFastVLine(graphX + 1, WAVE_Y_TOP, WAVE_HEIGHT, TFT_BLACK);
    tft.drawFastVLine(graphX + 2, WAVE_Y_TOP, WAVE_HEIGHT, TFT_BLACK); 
    if (graphX % 5 == 0) tft.drawPixel(graphX + 1, WAVE_ZERO_Y, 0x39E7); 

    tft.drawLine(graphX, *lastYPtr, graphX + 1, currY, waveColor);
    tft.drawLine(graphX, *lastYPtr + 1, graphX + 1, currY + 1, waveColor); 

    *lastYPtr = currY; graphX++;

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
    if (millis() - lastBpmDrawTime < BPM_UPDATE_INTERVAL_MS) return;
    lastBpmDrawTime = millis();

    float bpm = dsp_bpm;
    if (fabsf(bpm - lastDisplayedBPM) < 0.5f && lastDisplayedBPM >= 0) return;
    lastDisplayedBPM = bpm;

    tft.fillRect(0, BPM_Y, tft.width(), BPM_H, TFT_BLACK);
    int cx = tft.width() / 2;

    if (bpm > 20 && bpm < 250) {
        int hx = cx - 40, hy = BPM_Y + 12;
        tft.fillCircle(hx - 4, hy, 4, TFT_RED); tft.fillCircle(hx + 4, hy, 4, TFT_RED); tft.fillTriangle(hx - 8, hy + 1, hx + 8, hy + 1, hx, hy + 10, TFT_RED);
        tft.setTextDatum(ML_DATUM); tft.setTextColor(TFT_WHITE, TFT_BLACK); tft.setTextPadding(60);
        char buf[8]; sprintf(buf, "%d", (int)bpm); tft.drawString(buf, cx - 25, BPM_Y + 15, 4);
        tft.setTextColor(TFT_DARKGREY, TFT_BLACK); tft.setTextPadding(0); tft.drawString("bpm", cx + 20, BPM_Y + 17, 2);
    } else {
        tft.setTextDatum(MC_DATUM); tft.setTextColor(TFT_DARKGREY, TFT_BLACK); tft.setTextPadding(tft.width());
        tft.drawString("measuring...", cx, BPM_Y + 15, 2); tft.setTextPadding(0);
    }
}

void clearBPM() { tft.fillRect(0, BPM_Y, tft.width(), BPM_H, TFT_BLACK); lastDisplayedBPM = -1; lastBpmDrawTime = 0; }

static unsigned long lastBeatFlash = 0;
void drawBeatIndicator() {
    if (currentState != ACTIVE || currentMode != 1) return;
    if (dsp_beat_flag) {
        dsp_beat_flag = false; tft.fillCircle(12, 8, 4, TFT_RED); lastBeatFlash = millis();
    } else if (lastBeatFlash > 0 && millis() - lastBeatFlash > BEAT_FLASH_DURATION_MS) {
        tft.fillCircle(12, 8, 4, TFT_BLACK); lastBeatFlash = 0;
    }
}

void drawStatusBar() {
    tft.fillRect(0, 0, tft.width(), 16, TFT_BLACK);
    uint16_t btColor = deviceConnected ? TFT_CYAN : TFT_DARKGREY;
    if (isBleOn && !deviceConnected) btColor = TFT_ORANGE; 
    int cx = tft.width() - 15; int cy = 8; 
    tft.drawLine(cx, cy-6, cx, cy+6, btColor); tft.drawLine(cx, cy-6, cx+4, cy-3, btColor); tft.drawLine(cx+4, cy-3, cx, cy, btColor);     
    tft.drawLine(cx, cy, cx+4, cy+3, btColor); tft.drawLine(cx+4, cy+3, cx, cy+6, btColor); tft.drawLine(cx-4, cy-2, cx+4, cy+4, btColor); tft.drawLine(cx-4, cy+4, cx+4, cy-2, btColor); 
}

void drawUI() {
    if (!uiNeedsUpdate || isRecording5Sec) return;

    int cx = tft.width() / 2;
    tft.setTextDatum(MC_DATUM); tft.setTextPadding(tft.width()); 

    static int lastDrawnMode = -1; static bool lastDrawnConn = false; static bool lastDrawnBle = false; static AppState lastDrawnState = ACTIVE;
    if (currentMode != lastDrawnMode || deviceConnected != lastDrawnConn || isBleOn != lastDrawnBle || currentState != lastDrawnState) {
        tft.fillRect(0, 16, tft.width(), tft.height() - 52, TFT_BLACK); 
        graphX = WAVE_X_START; lastDrawnMode = currentMode; lastDrawnConn = deviceConnected; lastDrawnBle = isBleOn; lastDrawnState = currentState;
        clearBPM(); drawStatusBar(); 
    }

    if (currentState == POPUP) {
        tft.setTextColor(TFT_ORANGE, TFT_BLACK); tft.drawString("! WARNING !", cx, 40, 2);
        tft.setTextColor(TFT_WHITE, TFT_BLACK); tft.drawString("Bluetooth", cx, 80, 2); tft.drawString("Disconnected", cx, 100, 2);
        tft.setTextColor(TFT_GREEN, TFT_BLACK); tft.drawString("Pair now?", cx, 140, 1);
        tft.drawFastHLine(10, tft.height() - 35, tft.width() - 20, TFT_DARKGREY);
        tft.setTextColor(TFT_LIGHTGREY, TFT_BLACK); tft.drawString("L: NO | R: YES", cx, tft.height() - 15, 1);
    }
    else if (currentState == PAIRING) {
        tft.setTextColor(TFT_YELLOW, TFT_BLACK); tft.drawString("PAIRING", cx, 40, 4);       
        tft.setTextColor(TFT_WHITE, TFT_BLACK); tft.drawString("WAITING...", cx, 100, 2);
        tft.drawFastHLine(10, tft.height() - 35, tft.width() - 20, TFT_DARKGREY);
        tft.setTextColor(TFT_LIGHTGREY, TFT_BLACK); tft.drawString("PRESS R TO CANCEL", cx, tft.height() - 15, 1);
    }
    else if (currentState == ACTIVE) {
        // 💡 측정 모드 렌더링 (파형 위로 글씨 피신)
        if (currentMode == 1) {
            tft.setTextColor(TFT_RED, TFT_BLACK); tft.drawString("HEART", cx, 30, 4); 
            tft.setTextColor(TFT_WHITE, TFT_BLACK); tft.drawString(selectedPatient, cx, 55, 2);
        } 
        else if (currentMode == 2) {
            tft.setTextColor(TFT_CYAN, TFT_BLACK); tft.drawString("LUNG", cx, 30, 4); 
            tft.setTextColor(TFT_WHITE, TFT_BLACK); tft.drawString(selectedPatient, cx, 55, 2);
        } 
        else if (currentMode == 3) {
            // 💡 [핵심] 표 형식의 환자 리스트 UI
            tft.setTextColor(TFT_YELLOW, TFT_BLACK); 
            tft.drawString("PATIENT LIST", cx, 20, 2);       
            tft.drawFastHLine(10, 35, tft.width() - 20, TFT_DARKGREY);

            if (patientCount == 0) {
                tft.setTextColor(TFT_WHITE, TFT_BLACK);
                tft.drawString("No Patients", cx, 80, 2);
            } else {
                int startY = 55;
                int rowHeight = 25;
                int visibleRows = 3; 
                int startIdx = 0;

                if (currentPatIdx >= visibleRows) {
                    startIdx = currentPatIdx - visibleRows + 1;
                }

                tft.setTextDatum(ML_DATUM); 
                tft.setTextColor(TFT_DARKGREY, TFT_BLACK);
                tft.drawString("ID", 20, 42, 1);
                tft.drawString("NAME", 85, 42, 1);
                tft.drawString("WARD", 170, 42, 1);

                for (int i = 0; i < visibleRows && (startIdx + i) < patientCount; i++) {
                    int pIdx = startIdx + i;
                    int y = startY + (i * rowHeight);

                    if (pIdx == currentPatIdx) {
                        tft.fillRect(0, y - 8, tft.width(), rowHeight, 0x03E0); 
                        tft.setTextColor(TFT_WHITE, 0x03E0);
                        tft.drawString(">", 5, y, 2); 
                    } else {
                        tft.fillRect(0, y - 8, tft.width(), rowHeight, TFT_BLACK);
                        tft.setTextColor(TFT_LIGHTGREY, TFT_BLACK);
                    }

                    tft.drawString(patientList[pIdx].id, 20, y, 2);
                    tft.drawString(patientList[pIdx].name, 85, y, 2);
                    tft.drawString(patientList[pIdx].ward, 170, y, 2);
                }
                tft.setTextDatum(MC_DATUM); 
            }
        }

        tft.drawFastHLine(10, tft.height() - 35, tft.width() - 20, TFT_DARKGREY);
        tft.setTextColor(TFT_LIGHTGREY, TFT_BLACK);
        
        if (currentMode == 1 || currentMode == 2) {
            tft.drawString("L: MODE | R: RECORD", cx, tft.height() - 15, 1);
        } else {
            tft.drawString("L: MODE | R(Hold): SEL", cx, tft.height() - 15, 1);
        }
    }
    tft.setTextPadding(0); uiNeedsUpdate = false; 
}

// ══════════════════════════════════════════════════════════
//  Main Loop
// ══════════════════════════════════════════════════════════
void loop() {
    static bool lastConnState = false;
    if (deviceConnected != lastConnState) {
        if (deviceConnected) {
            currentMode = 3; 
            dsp_reset(); changeState(ACTIVE);
        } else {
            if (isRecording5Sec) {
                isRecording5Sec = false; tft.writecommand(0x11); delay(TFT_WAKEUP_DELAY_MS); digitalWrite(TFT_BL, HIGH);
            }
            isBleOn = false; uiNeedsUpdate = true; btnL.reset(); btnR.reset();
        }
        lastConnState = deviceConnected;
    }

    if (isRecording5Sec) {
        if (millis() - recordStartTime >= RECORD_DURATION_MS) {
            isRecording5Sec = false; tft.writecommand(0x11); delay(TFT_WAKEUP_DELAY_MS); digitalWrite(TFT_BL, HIGH); 
            uiNeedsUpdate = true; btnL.reset(); btnR.reset();
        }
        return; 
    }

    int eventL = btnL.update();
    int eventR = btnR.update();

    if (currentState == POPUP) {
        if (eventL == 1) changeState(ACTIVE); 
        else if (eventR == 1) { isBleOn = true; BLEDevice::startAdvertising(); changeState(PAIRING); }
    }
    else if (currentState == PAIRING) {
        if (eventR == 1) { isBleOn = false; BLEDevice::stopAdvertising(); changeState(ACTIVE); }
    }
    else if (currentState == ACTIVE) {
        if (eventL == 2) enterDeepSleep(); 
        else if (eventL == 1) {
            currentMode++; if (currentMode > 3) currentMode = 1;
            uiNeedsUpdate = true; drawUI();
        }

        if (currentMode == 3) {
            if (eventR == 1) {
                if(patientCount > 0) {
                    currentPatIdx = (currentPatIdx + 1) % patientCount;
                    uiNeedsUpdate = true; drawUI();
                }
            } else if (eventR == 2) {
                if(patientCount > 0) {
                    selectedPatient = patientList[currentPatIdx].id; 
                    currentMode = 1; 
                    uiNeedsUpdate = true;
                    
                    tft.fillScreen(TFT_BLACK);
                    tft.setTextColor(TFT_GREEN, TFT_BLACK);
                    tft.drawString("SELECTED!", tft.width()/2, tft.height()/2 - 10, 4);
                    tft.setTextColor(TFT_WHITE, TFT_BLACK);
                    tft.drawString(selectedPatient, tft.width()/2, tft.height()/2 + 20, 2);
                    delay(1000); 
                    
                    btnR.reset(); drawUI();
                }
            }
        } 
        else {
            if (eventR == 1) {
                if (deviceConnected) {
                    String header = "REC:" + selectedPatient + ":" + (currentMode == 1 ? "HEART" : "LUNG");
                    pCmdChar->setValue(header.c_str());
                    pCmdChar->notify();
                    delay(100); 

                    isRecording5Sec = true;
                    recordStartTime = millis();
                    tft.writecommand(0x10); delay(5); digitalWrite(TFT_BL, LOW); 
                } else {
                    changeState(POPUP); 
                }
            }
        }
    }

    drawUI();

    if (currentState == ACTIVE && (currentMode == 1 || currentMode == 2) && !isRecording5Sec) {
        drawWaveform(sharedAveragedSample);
        if (currentMode == 1) { drawBPM(); drawBeatIndicator(); }
    }
}