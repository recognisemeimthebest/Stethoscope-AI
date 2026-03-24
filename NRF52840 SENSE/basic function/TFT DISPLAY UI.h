#include <TFT_eSPI.h>

TFT_eSPI tft = TFT_eSPI();
TFT_eSprite img = TFT_eSprite(&tft);

int x_pos = 0;
int prev_y = 50;

void setup() {
  tft.init();
  tft.setRotation(0);
  tft.fillScreen(TFT_BLACK);

  // 외곽 원
  tft.drawCircle(120, 120, 115, TFT_BLUE);
  tft.drawCircle(120, 120, 113, 0x0044AA);

  // 제목
  tft.setTextColor(TFT_CYAN);
  tft.setTextDatum(MC_DATUM);
  tft.drawString("AI STETHOSCOPE", 120, 35, 2);

  // 구분선
  tft.drawLine(60, 75, 180, 75, 0x003366);

  // 심장 아이콘 (빨간 원)
  tft.fillCircle(120, 108, 18, TFT_RED);
  tft.drawCircle(120, 108, 18, 0xFF6666);

  // READY 텍스트
  tft.setTextColor(TFT_GREEN);
  tft.drawString("READY", 120, 160, 2);

  // BLE 상태
  tft.setTextColor(0x334455);
  tft.drawString("BLE: OFF", 120, 200, 2);

  // 파형 스프라이트 초기화
  img.createSprite(200, 40);
  img.fillSprite(TFT_BLACK);
}

void loop() {
  // 랜덤 파형 (나중에 마이크 데이터로 교체)
  int y_val = 20 + random(-10, 10);
  if (x_pos % 50 == 0) y_val -= 15;

  img.drawLine(x_pos, prev_y, x_pos + 1, y_val, TFT_GREEN);
  img.pushSprite(20, 170);

  prev_y = y_val;
  x_pos++;

  if (x_pos > 200) {
    x_pos = 0;
    img.fillSprite(TFT_BLACK);
  }

  delay(20);
}