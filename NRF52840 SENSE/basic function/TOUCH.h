#include <Wire.h>
#include <TFT_eSPI.h>

TFT_eSPI tft = TFT_eSPI();

void setup() {
  Serial.begin(115200);
  tft.init();
  tft.setRotation(0);
  tft.fillScreen(TFT_BLACK);

  // nRF52840은 기본 Wire.begin()을 사용합니다.
  // 내부적으로 이미 D4(SDA), D5(SCL)에 할당되어 있습니다.
  Wire.begin(); 
  
  tft.setTextColor(TFT_WHITE);
  tft.drawCentreString("Touch Diagnostic", 120, 100, 2);
  Serial.println("I2C Touch Scanner Ready");
}

void loop() {
  // CHSC6X 터치 칩의 I2C 주소는 보통 0x2E입니다.
  Wire.beginTransmission(0x2E); 
  if (Wire.endTransmission() == 0) {
    Wire.requestFrom(0x2E, 5); 
    if (Wire.available() >= 5) {
      uint8_t data[5];
      for (int i = 0; i < 5; i++) data[i] = Wire.read();
      
      // 터치 좌표 해석 (데이터 시트 기반 간략화)
      int x = data[2]; // 칩셋에 따라 인덱스는 다를 수 있음
      int y = data[4];

      if (data[0] == 0x01 || data[0] == 0x02) { // 터치가 감지되었을 때
        Serial.printf("X: %d, Y: %d\n", x, y);
        tft.fillCircle(x, y, 3, TFT_YELLOW);
      }
    }
  }
  delay(20);
}