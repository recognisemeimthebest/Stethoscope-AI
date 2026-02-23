#include <Arduino.h>
#include <Wire.h>
#include <TFT_eSPI.h>
#include <PDM.h>
#include <bluefruit.h>

TFT_eSPI tft = TFT_eSPI();
BLEUart bleuart;

// --- 전역 변수 ---
short sampleBuffer[256];
volatile int samplesRead = 0;
bool pdmRunning = false;
int lastMaxVal = 0;           // 블루투스 전송용 최대값

#define MODE_MAIN 0
#define MODE_HEART 1
#define MODE_LUNG 2

int currentMode = MODE_MAIN;
int waveX = 40;
int lastY = 140;
bool isTouching = false;
int startX = 0;
unsigned long lastTouchCheck = 0;
unsigned long aiPressStartTime = 0;
bool aiLongPressTriggered = false;
bool isAIReady = false;        // ✅ AI 토글 상태 변수

unsigned long lastBLESend = 0; // ✅ 블루투스 전송 타이머

// --- 함수 선언부 ---
void onPDMdata();
void checkTouch();
void drawUI();
void updateStatusText();
void drawMedicalStream();
void drawAIButton();
void handleBLE();
void startPDM();
void stopPDM();

void setup() {
  Serial.begin(115200);
  tft.init();
  tft.setRotation(0);
  tft.fillScreen(TFT_BLACK);

  Wire.begin();
  Wire.setClock(100000);

  // --- 블루투스 초기화 ---
  Bluefruit.begin();
  Bluefruit.setName("AI_STETHOSCOPE");
  bleuart.begin();
  Bluefruit.Advertising.addFlags(BLE_GAP_ADV_FLAGS_LE_ONLY_GENERAL_DISC_MODE);
  Bluefruit.Advertising.addService(bleuart);
  Bluefruit.Advertising.addName();
  Bluefruit.Advertising.restartOnDisconnect(true);
  Bluefruit.Advertising.start(0);

  Serial.println("System Ready: BLE & UI Initialized");
  drawUI();
}

void loop() {
  checkTouch();
  handleBLE(); 

  // 1. 소리 데이터 처리 및 그래프 그리기 (화면 출력은 항상 작동)
  if (pdmRunning && samplesRead >= 128 && !isTouching) {
    drawMedicalStream();
    samplesRead = 0;
  }

  // 2. ✅ 전송 제어: AI READY(초록색)일 때만 서버로 데이터 전송
  if (pdmRunning && (millis() - lastBLESend > 150)) {
    if (isAIReady && Bluefruit.connected() && lastMaxVal > 200) {
      bleuart.println(lastMaxVal); // 서버(파이)로 숫자 전송
      lastMaxVal = 0; 
    } else if (!isAIReady) {
      lastMaxVal = 0; // AI OFF일 때는 데이터를 그냥 버림
    }
    lastBLESend = millis();
  }
}

// 블루투스 수신 처리 (결과 수신용)
void handleBLE() {
  if (bleuart.available()) {
    String input = bleuart.readString();
    input.trim();
    Serial.print("\n[BLE 수신]: "); Serial.println(input);
    
    // 파이에서 보낸 분석 결과 처리 예시 (나중에 사용)
    if (input == "1") {
      // 분석 성공 시 보드에서 피드백 줄 로직
    }
  }
}

void checkTouch() {
  if (millis() - lastTouchCheck < 20) return;
  lastTouchCheck = millis();

  Wire.beginTransmission(0x2E);
  if (Wire.endTransmission() == 0) {
    Wire.requestFrom(0x2E, 5);
    if (Wire.available() >= 5) {
      uint8_t data[5];
      for (int i = 0; i < 5; i++) data[i] = Wire.read();
      
      uint8_t status = data[0];
      int curX = data[2]; 
      int curY = data[4]; 
      bool currentlyPressed = (status == 0x01 || status == 0x02);

      if (currentlyPressed) {
        if (!isTouching) { 
          startX = curX; 
          isTouching = true; 
          aiPressStartTime = millis();
          aiLongPressTriggered = false;
        }

        if (currentMode == MODE_MAIN) {
          // HEART 버튼 영역
          if (curX >= 10 && curX <= 110 && curY >= 80 && curY <= 160) {
            currentMode = MODE_HEART;
            bleuart.println("CMD:START_HEART");
            startPDM();
            drawUI();
            isTouching = false;
            return;
          }
          // LUNG 버튼 영역
          else if (curX >= 130 && curX <= 230 && curY >= 80 && curY <= 160) {
            currentMode = MODE_LUNG;
            bleuart.println("CMD:START_LUNG");
            startPDM();
            drawUI();
            isTouching = false;
            return;
          }
          // AI 토글 버튼 영역 (하단)
          else if (curX >= 70 && curX <= 170 && curY >= 180 && curY <= 235) {
            if (!aiLongPressTriggered && (millis() - aiPressStartTime >= 2000)) {
              isAIReady = !isAIReady; // ✅ 토글
              bleuart.println(isAIReady ? "SET_AI:ON" : "SET_AI:OFF");
              updateStatusText();
              drawAIButton(); 
              aiLongPressTriggered = true;
            }
          }
        }
      } 
      else {
        if (isTouching) {
          // 스캔 모드에서 오른쪽 스와이프 시 메인으로
          if (currentMode != MODE_MAIN && (curX - startX > 40)) { 
            currentMode = MODE_MAIN; 
            bleuart.println("CMD:GOTO_MAIN");
            stopPDM();
            drawUI(); 
          }
          isTouching = false;
          aiLongPressTriggered = false;
        }
      }
    }
  }
}

void startPDM() {
  if (!pdmRunning) {
    PDM.onReceive(onPDMdata);
    if (PDM.begin(1, 16000)) {
      pdmRunning = true;
      Serial.println("[시스템]: 마이크 활성화");
    }
  }
}

void stopPDM() {
  if (pdmRunning) {
    PDM.end();
    pdmRunning = false;
    Serial.println("[시스템]: 마이크 비활성화");
  }
}

// --- UI 드로잉 함수군 ---

void drawAIButton() {
  uint32_t btnColor = isAIReady ? TFT_GREEN : 0x5AEB;
  tft.drawRoundRect(70, 195, 100, 30, 4, btnColor);
  tft.fillRect(71, 196, 98, 28, TFT_BLACK); 
  tft.setTextColor(btnColor);
  tft.drawCentreString(isAIReady ? "AI ON (2s)" : "AI OFF (2s)", 120, 203, 2);
}

void updateStatusText() {
  tft.fillRect(40, 45, 160, 25, TFT_BLACK); 
  tft.setTextColor(isAIReady ? TFT_GREEN : TFT_LIGHTGREY);
  tft.drawCentreString(isAIReady ? "AI READY" : "SYSTEM READY", 120, 45, 2);
}

void drawUI() {
  tft.fillScreen(TFT_BLACK);
  tft.drawRect(5, 5, 230, 230, 0x5AEB);
  waveX = 40; lastY = 140;

  if (currentMode == MODE_MAIN) {
    tft.drawCircle(120, 120, 110, 0x10A2);
    tft.setTextColor(TFT_CYAN);
    tft.drawCentreString("AI STETHOSCOPE", 120, 20, 2);
    updateStatusText();
    
    tft.drawRoundRect(25, 90, 85, 60, 4, TFT_RED);
    tft.setTextColor(TFT_RED);
    tft.drawCentreString("HEART", 67, 105, 2);
    
    tft.drawRoundRect(130, 90, 85, 60, 4, TFT_SKYBLUE);
    tft.setTextColor(TFT_SKYBLUE);
    tft.drawCentreString("LUNG", 172, 105, 2);

    drawAIButton();
    tft.setTextColor(0x5AEB);
    tft.drawCentreString("LONG PRESS TO TOGGLE", 120, 182, 1);
  } else {
    uint32_t color = (currentMode == MODE_HEART) ? TFT_RED : TFT_SKYBLUE;
    tft.drawRoundRect(10, 10, 220, 220, 8, color);
    tft.setTextColor(color);
    String title = (currentMode == MODE_HEART) ? "HEART SCAN" : "LUNG SCAN";
    tft.drawCentreString(isAIReady ? "AI " + title : title, 120, 25, 2);
    for(int i=40; i<=200; i+=20) tft.drawFastVLine(i, 85, 110, 0x0841);
    tft.drawCentreString("SWIPE RIGHT TO EXIT", 120, 210, 1);
  }
}

void drawMedicalStream() {
  int yCenter = 140;
  int soundVal = sampleBuffer[0]; 

  // ✅ 데이터 전송용 피크값 갱신
  if (abs(soundVal) > lastMaxVal) lastMaxVal = abs(soundVal);

  int range = (currentMode == MODE_HEART) ? 1000 : 1800;
  int yVal = map(soundVal, -range, range, yCenter + 50, yCenter - 50);
  yVal = constrain(yVal, yCenter - 55, yCenter + 55);
  uint32_t color = (currentMode == MODE_HEART) ? TFT_RED : TFT_SKYBLUE;
  
  if (isAIReady && abs(soundVal) > range * 0.6) color = TFT_WHITE;
  
  tft.fillRect(waveX + 1, yCenter - 55, 10, 110, TFT_BLACK);
  if ((waveX + 1) % 20 <= 10) tft.drawFastVLine(waveX + (20 - (waveX % 20)), yCenter - 55, 110, 0x0841);
  tft.drawLine(waveX, lastY, waveX + 1, yVal, color);
  lastY = yVal; waveX++;
  if (waveX > 200) waveX = 40;
}

void onPDMdata() {
  int bytesAvailable = PDM.available();
  if (bytesAvailable > 0) {
    PDM.read(sampleBuffer, bytesAvailable);
    samplesRead = bytesAvailable / 2;
  }
}