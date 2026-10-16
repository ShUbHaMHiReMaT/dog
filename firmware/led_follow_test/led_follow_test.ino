// Tell apart "VCC on GPIO9" from "GND not connected (ground returns through SCL/GPIO9)".
// GPIO8 held HIGH. GPIO9: HIGH for 3 s, LOW for 0.3 s, repeat.
//   LED mostly ON with short dips  -> module is POWERED BY GPIO9 (VCC wire on GPIO9)
//   LED mostly OFF with short flashes -> module GND is NOT connected; it returns through GPIO9 when low
//   LED steady ON                  -> wiring OK
#include <Arduino.h>

void setup() {
  Serial.begin(921600);
  pinMode(8, OUTPUT); digitalWrite(8, HIGH);
  pinMode(9, OUTPUT);
}

void loop() {
  digitalWrite(9, HIGH); Serial.println("#I,GPIO9=HIGH (3 s)"); delay(3000);
  digitalWrite(9, LOW);  Serial.println("#I,GPIO9=LOW (0.3 s)"); delay(300);
}
