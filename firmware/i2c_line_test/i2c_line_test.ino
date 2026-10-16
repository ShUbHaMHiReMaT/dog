// Electrical line test for an I2C module on SDA=GPIO8, SCL=GPIO9.
// A powered, connected I2C module pulls both lines HIGH through its pull-up resistors (typ. 2.2-10 kOhm).
//   floating (no pull)   : reads with internal pull-DOWN -> LOW  means nothing is pulling the line up
//   module pull-ups      : reads HIGH even against the internal ~45 kOhm pull-down
// Also checks for a short between SDA and SCL, and runs a full 0x01-0x7F scan at 10 kHz and 100 kHz
// with internal pull-ups on.
#include <Arduino.h>
#include <Wire.h>

const int SDA_PIN = 8, SCL_PIN = 9;

int readWith(int pin, int mode) { pinMode(pin, mode); delay(5); return digitalRead(pin); }

void scan(uint32_t hz) {
  Wire.end();
  Wire.begin(SDA_PIN, SCL_PIN, hz);
  String found;
  for (uint8_t a = 1; a < 127; a++) {
    Wire.beginTransmission(a);
    if (Wire.endTransmission() == 0) { char b[8]; snprintf(b, sizeof(b), " 0x%02X", a); found += b; }
  }
  Serial.printf("#SCAN,%lu Hz:%s\n", (unsigned long)hz, found.length() ? found.c_str() : " none");
  Wire.end();
}

void setup() { Serial.begin(921600); delay(1500); }

void loop() {
  Wire.end();
  int sda_pd = readWith(SDA_PIN, INPUT_PULLDOWN), scl_pd = readWith(SCL_PIN, INPUT_PULLDOWN);
  int sda_pu = readWith(SDA_PIN, INPUT_PULLUP),   scl_pu = readWith(SCL_PIN, INPUT_PULLUP);
  Serial.printf("#LINES,with_pulldown: SDA=%d SCL=%d | with_pullup: SDA=%d SCL=%d\n", sda_pd, scl_pd, sda_pu, scl_pu);
  // short test: drive SDA low, read SCL (with pull-up)
  pinMode(SCL_PIN, INPUT_PULLUP); pinMode(SDA_PIN, OUTPUT); digitalWrite(SDA_PIN, LOW); delay(5);
  int scl_when_sda_low = digitalRead(SCL_PIN);
  pinMode(SDA_PIN, INPUT_PULLUP); pinMode(SCL_PIN, OUTPUT); digitalWrite(SCL_PIN, LOW); delay(5);
  int sda_when_scl_low = digitalRead(SDA_PIN);
  pinMode(SCL_PIN, INPUT_PULLUP);
  Serial.printf("#SHORT,SCL_when_SDA_low=%d SDA_when_SCL_low=%d (0 = lines shorted together)\n",
                scl_when_sda_low, sda_when_scl_low);
  scan(10000);
  scan(100000);
  Serial.println("#DONE");
  delay(3000);
}
