#include <PDM.h>

short sampleBuffer[256];
volatile int samplesRead = 0;

void onPDMdata();

void setup() {
  Serial.begin(115200);
  while (!Serial);

  PDM.onReceive(onPDMdata);

  if (!PDM.begin(1, 16000)) {
    Serial.println("PDM 초기화 실패!");
    while (1);
  }

  Serial.println("마이크 초기화 성공!");
}

void loop() {
  if (samplesRead > 0) {
    Serial.println(sampleBuffer[0]);
    samplesRead = 0;
  }
  delay(50);
}

void onPDMdata() {
  int bytesAvailable = PDM.available();
  PDM.read(sampleBuffer, bytesAvailable);
  samplesRead = bytesAvailable / 2;
}