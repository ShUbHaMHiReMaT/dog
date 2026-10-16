// Diagnostic: find which GPIO pair the I2C sensor is actually wired to (and whether SDA/SCL are swapped).
// Tries every pair of "safe" ESP32-S3 GPIOs and looks for a device at 0x68/0x69 (BMI270) - prints any hit.
// Excludes strapping (0,3,45,46), USB (19,20), UART0 (43,44), flash/PSRAM (26-37) pins.
#include <Arduino.h>
#include <Wire.h>

const int PINS[] = {1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 21, 38, 39, 40, 41, 42, 47, 48};
const int N = sizeof(PINS) / sizeof(PINS[0]);

uint8_t probe(uint8_t addr) {
  Wire.beginTransmission(addr);
  Wire.write(0x00);
  if (Wire.endTransmission(false) != 0) return 0;
  if (Wire.requestFrom((int)addr, 1) != 1) return 0;
  return Wire.read();
}

void setup() {
  Serial.begin(921600);
  delay(1500);
}

void loop() {
  Serial.println("#I,pin_finder,scanning...");
  int hits = 0;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      if (i == j) continue;
      Wire.end();
      if (!Wire.begin(PINS[i], PINS[j], 100000)) continue;
      Wire.setTimeOut(5);
      for (uint8_t a = 0x68; a <= 0x69; a++) {
        Wire.beginTransmission(a);
        if (Wire.endTransmission() == 0) {
          uint8_t id = probe(a);
          Serial.printf("#FOUND,SDA=GPIO%d,SCL=GPIO%d,addr=0x%02X,chip_id=0x%02X\n", PINS[i], PINS[j], a, id);
          hits++;
        }
      }
    }
  }
  Wire.end();
  Serial.printf("#I,pin_finder,done,hits=%d\n", hits);
  delay(3000);
}
