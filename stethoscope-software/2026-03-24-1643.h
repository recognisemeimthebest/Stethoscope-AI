#include <Arduino.h>
#include <driver/i2s.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>
#include <TFT_eSPI.h> 
#include <SPI.h>

TFT_eSPI tft = TFT_eSPI(); 

#define BTN_R 35 // 오른쪽 (모드/전원)
#define BTN_L 0  // 왼쪽 (AI 모드 페어링 & 5초 녹음)
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

enum AppState { ACTIVE, SLEEP };
volatile AppState currentState = ACTIVE;

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

#define WAVE_Y_TOP 120
#define WAVE_Y_BOTTOM 200
#define WAVE_HEIGHT (WAVE_Y_BOTTOM - WAVE_Y_TOP)
#define WAVE_ZERO_Y (WAVE_Y_TOP + (WAVE_HEIGHT / 2)) 
#define WAVE_X_START 10
#define WAVE_X_END 125

int graphX = WAVE_X_START; 
int16_t lastY_HEART_WAVE = WAVE_ZERO_Y; 
int16_t lastY_LUNG_WAVE = WAVE_ZERO_Y;  

unsigned long lastDrawTime = 0;
const int drawInterval = 15; 

TaskHandle_t audioTaskHandle;

class MyServerCallbacks: public BLEServerCallbacks {
    void onConnect(BLEServer* pServer) { 
        deviceConnected = true; 
        uiNeedsUpdate = true; 
    }
    void onDisconnect(BLEServer* pServer) { 
        deviceConnected = false; 
        isRecording5Sec = false; 
        uiNeedsUpdate = true;
        if (currentState == ACTIVE && isBleOn) BLEDevice::startAdvertising(); 
    }
};

class MyCmdCallbacks: public BLECharacteristicCallbacks {
    void onWrite(BLECharacteristic *pCharacteristic) {}
};

void enterSleepMode() {
    currentState = SLEEP;
    isBleOn = false; 
    isRecording5Sec = false;

    tft.writecommand(0x10); 
    delay(5);
    digitalWrite(TFT_BL, LOW); 
    BLEDevice::stopAdvertising(); 
    
    while(digitalRead(BTN_R) == LOW || digitalRead(BTN_L) == LOW) { delay(10); }
    delay(500); 
}

void wakeUp() {
    while(digitalRead(BTN_R) == LOW || digitalRead(BTN_L) == LOW) { delay(10); }
    delay(100); 

    rPressTime = 0; rLongPressed = false;
    lPressTime = 0; lLongPressed = false;
    
    currentMode = 1; 
    isBleOn = false; 
    isRecording5Sec = false;
    
    tft.writecommand(0x11); 
    delay(120); 
    
    tft.fillScreen(TFT_BLACK);
    uiNeedsUpdate = true; 
    graphX = WAVE_X_START;
    
    digitalWrite(TFT_BL, HIGH); 
    currentState = ACTIVE; 
}

void audioTask(void *pvParameters) {
    for(;;) { 
        if (currentState == ACTIVE) {
            int32_t sum = 0;
            
            for(int i=0; i<DECIMATION; i++) {
                int32_t sample = 0;
                size_t bytesIn = 0;
                i2s_read(I2S_PORT, &sample, sizeof(sample), &bytesIn, portMAX_DELAY);
                int16_t pcm16 = (int16_t)(sample >> 16);
                sum += pcm16;
            }

            int32_t avg = sum / DECIMATION;
            int32_t amplified = avg << 2; 

            if (amplified > 32767) amplified = 32767;
            else if (amplified < -32768) amplified = -32768;

            sharedAveragedSample = (int16_t)amplified; 

            if (deviceConnected && isRecording5Sec) {
                bleBuffer[bufferIndex++] = (int16_t)amplified;

                if (bufferIndex >= BLE_BUFFER_SIZE) {
                    pAudioChar->setValue((uint8_t*)bleBuffer, BLE_BUFFER_SIZE * sizeof(int16_t));
                    pAudioChar->notify();
                    bufferIndex = 0;
                }
            }
        } else {
            vTaskDelay(pdMS_TO_TICKS(50)); 
        }
    }
}

void setup() {
    Serial.begin(115200);

    pinMode(BTN_R, INPUT);        
    pinMode(BTN_L, INPUT_PULLUP);
    pinMode(TFT_BL, OUTPUT);
    digitalWrite(TFT_BL, HIGH); 

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
    if (currentMode == 3 || currentState == SLEEP || isRecording5Sec) return;

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

void drawUI() {
    if (!uiNeedsUpdate || currentState == SLEEP || isRecording5Sec) return;

    int cx = tft.width() / 2;
    tft.setTextDatum(MC_DATUM); 
    tft.setTextPadding(tft.width()); 

    // 💡 연결 상태가 바뀌거나 모드가 바뀔 때만 화면 지우기 (초록 점 갱신용)
    static int lastDrawnMode = -1;
    static bool lastDrawnConn = false;
    if (currentMode != lastDrawnMode || deviceConnected != lastDrawnConn) {
        tft.fillRect(0, 0, tft.width(), tft.height() - 36, TFT_BLACK);
        graphX = WAVE_X_START; 
        lastDrawnMode = currentMode;
        lastDrawnConn = deviceConnected;
    }

    // 🟢 [추가됨] 글로벌 연결 인디케이터 (우측 상단 초록색 점)
    if (deviceConnected) {
        tft.fillCircle(tft.width() - 12, 12, 5, TFT_GREEN);
    }

    // --- 상단 모드 텍스트 ---
    if (currentMode == 1) {
        tft.setTextColor(TFT_RED, TFT_BLACK);
        tft.drawString("HEART", cx, 80, 4); 
    } 
    else if (currentMode == 2) {
        tft.setTextColor(TFT_CYAN, TFT_BLACK);
        tft.drawString("LUNG", cx, 80, 4); 
    } 
    else if (currentMode == 3) {
        tft.setTextColor(TFT_YELLOW, TFT_BLACK);
        tft.drawString("AI", cx, 65, 4);       
        tft.drawString("CONNECT", cx, 95, 4);  
        
        if (deviceConnected) {
            tft.setTextColor(TFT_GREEN, TFT_BLACK);
            tft.drawString("CONNECTED", cx, 150, 4); 
        } else if (isBleOn) {
            tft.setTextColor(TFT_WHITE, TFT_BLACK);
            tft.drawString("WAITING...", cx, 145, 2);
        } else {
            tft.setTextColor(TFT_DARKGREY, TFT_BLACK);
            tft.drawString("HOLD L", cx, 140, 2);
            tft.drawString("TO PAIR", cx, 160, 2);
        }
    }

    // --- 하단 안내 텍스트 동적 변경 ---
    tft.drawFastHLine(10, tft.height() - 35, tft.width() - 20, TFT_DARKGREY);
    tft.setTextColor(TFT_LIGHTGREY, TFT_BLACK);
    
    if (currentMode == 1 || currentMode == 2) {
        if (deviceConnected) {
            tft.drawString("L: RECORD | R: MODE", cx, tft.height() - 15, 1);
        } else {
            // 블루투스 안 켜져 있으면 L버튼 비활성화 표시
            tft.drawString("L: --- | R: MODE", cx, tft.height() - 15, 1);
        }
    } else {
        tft.drawString("HOLD L: PAIR | R: MODE", cx, tft.height() - 15, 1);
    }

    tft.setTextPadding(0); 
    uiNeedsUpdate = false; 
}

void loop() {
    if (currentState == SLEEP) {
        if (digitalRead(BTN_R) == LOW || digitalRead(BTN_L) == LOW) {
            unsigned long pressCheckStart = millis();
            bool isRealPress = true;
            while (millis() - pressCheckStart < 100) {
                if (digitalRead(BTN_R) == HIGH && digitalRead(BTN_L) == HIGH) {
                    isRealPress = false; break; 
                }
                delay(10);
            }
            if (isRealPress) wakeUp();
        }
        return; 
    }

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

    bool currR = digitalRead(BTN_R);
    if (currR == LOW) { 
        if (rPressTime == 0) rPressTime = millis(); 
        if (millis() - rPressTime > 1500 && !rLongPressed) {
            rLongPressed = true;
            enterSleepMode();
        }
    } else { 
        if (rPressTime != 0) { 
            if (!rLongPressed && millis() - rPressTime > 50) {
                currentMode++;
                if (currentMode > 3) currentMode = 1;
                uiNeedsUpdate = true;
            }
            rPressTime = 0; rLongPressed = false;
        }
    }

    // --- L 버튼 조작부 ---
    bool currL = digitalRead(BTN_L);
    if (currL == LOW) {
        if (lPressTime == 0) lPressTime = millis();
        
        // [조건 1] 길게 누르면: AI CONNECT(3) 모드일 때만 페어링 시작
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
            // [조건 2] 짧게 눌렀을 때: 연결됨 & (HEART 또는 LUNG) 일 때만 5초 녹음
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

    if (currentState == ACTIVE && (currentMode == 1 || currentMode == 2) && !isRecording5Sec) {
        drawWaveform(sharedAveragedSample);
    }
}