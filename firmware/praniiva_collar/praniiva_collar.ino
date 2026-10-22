// =====================================================================================
// PRANIIVA collar v0 firmware  (ESP32-S3 + Bosch BMI270)
//
// Produces EXACTLY what the PRANIIVA pipeline consumes (praniiva/ingest.py):
//   raw CSV   timestamp_ms,ax,ay,az           epoch ms UTC, acceleration in g, 50 Hz, +-8 g,
//                                             collar frame (x along the neck toward the head)
//   QC CSV    once per minute: vm_std_mg, rest_jitter_mg (loose-collar telemetry), chip temp,
//             samples, dropped samples, battery
//   events    marker presses / serial labels (video sync for behaviour labelling)
//
// Outputs: USB serial stream (tools/collar_logger.py) and/or microSD files.
// Serial commands (newline terminated):
//   T<epoch_ms>   set UTC time (the logger sends this automatically)
//   I<dog_id>     set dog id (stored in flash), e.g. ID0001
//   M<label>      write a marker event, e.g. Mstart_walk
//   S             print status
// =====================================================================================
#include <Arduino.h>
#include <Wire.h>
#include <SPI.h>
#include <SD.h>
#include <Preferences.h>
#include <WiFi.h>
#include <time.h>
#include "esp_timer.h"
#include "src/SparkFun_BMI270/SparkFun_BMI270_Arduino_Library.h"   // bundled copy (SparkFun MIT + Bosch BSD-3)
#include "config.h"

#if USE_MIC
#include <ESP_I2S.h>
#endif

#define FW_VERSION "0.4.0"

BMI270 imu;
Preferences prefs;

struct Sample {
  int64_t t_us;          // esp_timer time at data-ready
  float a[3];            // g, collar frame
  float g[3];            // deg/s (only if RECORD_GYRO)
};

static QueueHandle_t sampleQ;
static TaskHandle_t readerTask = nullptr;
static volatile int64_t isrTimeUs = 0;
static volatile uint32_t droppedSamples = 0, readErrors = 0, queueOverflows = 0, recoveries = 0;
static volatile int64_t lastSampleUs = 0;           // last good sample OR last recovery attempt
static volatile int64_t lastGoodUs = 0;             // last GOOD sample only (restart watchdog)
static volatile bool tempRequest = false;          // loop() asks, reader task reads (single I2C owner)
static volatile float lastTempC = NAN;
static int activeSda = -1, activeScl = -1;

// survive a watchdog restart without losing the clock
RTC_NOINIT_ATTR static int64_t savedEpochMs;
RTC_NOINIT_ATTR static uint32_t savedMagic;
#define TIME_MAGIC 0xC0FFEE42u
#define STALL_RECOVER_MS 500                        // no good sample for this long -> recover the sensor
#define STALL_RESTART_MS 20000                      // still nothing -> restart the ESP32 (clock kept)

static int64_t epochOffsetMs = 0;      // epoch_ms = esp_ms + offset
static bool timeSynced = false;
static String dogId = "ID0000";

// SD state
static bool sdOk = false;
static File rawFile, qcFile, evFile;
static int64_t fileStartMs = 0, lastFlushMs = 0;
static String rawBuf;

// ---------------------------------------------------------------- time helpers
static inline int64_t espMs() { return esp_timer_get_time() / 1000; }
static inline int64_t toEpochMs(int64_t t_us) { return t_us / 1000 + epochOffsetMs; }
static int64_t nowEpochMs() { return toEpochMs(esp_timer_get_time()); }

static String utcStamp(int64_t epochMs) {
  time_t s = (time_t)(epochMs / 1000);
  struct tm tmv;
  gmtime_r(&s, &tmv);
  char b[24];
  strftime(b, sizeof(b), "%Y%m%d-%H%M%SZ", &tmv);
  return String(b);
}

// ---------------------------------------------------------------- BMI270
static void IRAM_ATTR onDataReady() {
  isrTimeUs = esp_timer_get_time();
  BaseType_t woken = pdFALSE;
  if (readerTask) vTaskNotifyGiveFromISR(readerTask, &woken);
  portYIELD_FROM_ISR(woken);
}

static uint8_t imuAddr = 0;
static int accelRangeG = 0;                         // read back from the chip after configuration
static float accelOdrHz = 0;

static uint8_t readChipId(uint8_t addr) {
  Wire.beginTransmission(addr);
  Wire.write(0x00);                                  // CHIP_ID register
  if (Wire.endTransmission(false) != 0) return 0;
  if (Wire.requestFrom((int)addr, 1) != 1) return 0;
  return Wire.read();
}

static void i2cScan() {
  String found;
  for (uint8_t a = 1; a < 127; a++) {
    Wire.beginTransmission(a);
    if (Wire.endTransmission() == 0) { char b[8]; snprintf(b, sizeof(b), " 0x%02X", a); found += b; }
  }
  Serial.printf("#I,i2c_scan:%s\n", found.length() ? found.c_str() : " nothing found (check 3V3/GND/SDA/SCL)");
}

static bool setupImu() {
  const uint8_t candidates[2] = {(uint8_t)(BMI270_I2C_ADDR ? BMI270_I2C_ADDR : 0x68), 0x69};
  const int pinSets[2][2] = {{PIN_I2C_SDA, PIN_I2C_SCL}, {PIN_I2C_SCL, PIN_I2C_SDA}};
  uint8_t id = 0;
  for (int ps = 0; ps < (I2C_AUTO_SWAP ? 2 : 1) && id != 0x24; ps++) {
    Wire.end();
    Wire.begin(pinSets[ps][0], pinSets[ps][1], I2C_FREQ_HZ);
    delay(10);
    Serial.printf("#I,trying SDA=GPIO%d SCL=GPIO%d\n", pinSets[ps][0], pinSets[ps][1]);
    i2cScan();
    for (int i = 0; i < (BMI270_I2C_ADDR ? 1 : 2) && id != 0x24; i++) {
      id = readChipId(candidates[i]);
      if (id) imuAddr = candidates[i];
    }
    if (id == 0x24) { activeSda = pinSets[ps][0]; activeScl = pinSets[ps][1]; }
  }
  Serial.printf("#I,imu_addr=0x%02X chip_id=0x%02X (BMI270 expects 0x24; 0x26 = BMI220; 0x00 = no answer)\n", imuAddr, id);
  if (id != 0x24) return false;

  int8_t err = imu.beginI2C(imuAddr, Wire);          // uploads Bosch config file, enables accel+gyro
  if (err != BMI2_OK) { Serial.printf("#E,bmi270_init_failed,%d\n", err); return false; }

  bmi2_sens_config acc;
  acc.type = BMI2_ACCEL;
  acc.cfg.acc.odr = BMI2_ACC_ODR_50HZ;
  acc.cfg.acc.bwp = BMI2_ACC_NORMAL_AVG4;            // normal filter mode
  acc.cfg.acc.filter_perf = BMI2_PERF_OPT_MODE;      // low-noise performance mode
  acc.cfg.acc.range = BMI2_ACC_RANGE_8G;
  err = imu.setConfig(acc);
  if (err != BMI2_OK) { Serial.printf("#E,accel_config_failed,%d\n", err); return false; }
  // read the configuration BACK from the chip - proves range/rate instead of assuming them
  bmi2_sens_config rb;
  rb.type = BMI2_ACCEL;
  if (imu.getConfig(&rb) == BMI2_OK) {
    static const int rangeG[4] = {2, 4, 8, 16};
    accelRangeG = rangeG[rb.cfg.acc.range & 0x03];
    accelOdrHz = 25.0f * powf(2.0f, (float)rb.cfg.acc.odr - 6.0f);      // ODR code 0x06 = 25 Hz, 0x07 = 50 Hz
    Serial.printf("#I,accel_config_readback,range_g=%d,odr_hz=%.1f\n", accelRangeG, accelOdrHz);
  }

#if RECORD_GYRO
  bmi2_sens_config gyr;
  gyr.type = BMI2_GYRO;
  gyr.cfg.gyr.odr = BMI2_GYR_ODR_50HZ;
  gyr.cfg.gyr.bwp = BMI2_GYR_NORMAL_MODE;
  gyr.cfg.gyr.filter_perf = BMI2_PERF_OPT_MODE;
  gyr.cfg.gyr.noise_perf = BMI2_PERF_OPT_MODE;
  gyr.cfg.gyr.range = BMI2_GYR_RANGE_1000;
  gyr.cfg.gyr.ois_range = BMI2_GYR_OIS_2000;
  err = imu.setConfig(gyr);
  if (err != BMI2_OK) { Serial.printf("#E,gyro_config_failed,%d\n", err); return false; }
#else
  imu.disableFeature(BMI2_GYRO);                     // models do not use gyro; saves power
#endif

#if PIN_BMI_INT1 >= 0
  imu.mapInterruptToPin(BMI2_DRDY_INT, BMI2_INT1);
  bmi2_int_pin_config pc;
  pc.pin_type = BMI2_INT1;
  pc.int_latch = BMI2_INT_NON_LATCH;
  pc.pin_cfg[0].lvl = BMI2_INT_ACTIVE_HIGH;
  pc.pin_cfg[0].od = BMI2_INT_PUSH_PULL;
  pc.pin_cfg[0].output_en = BMI2_INT_OUTPUT_ENABLE;
  pc.pin_cfg[0].input_en = BMI2_INT_INPUT_DISABLE;
  imu.setInterruptPinConfig(pc);
#endif

  // sanity: a still sensor must read |a| ~ 1 g (verifies range/scaling end to end)
  delay(100);
  float sum = 0; int n = 0;
  for (int i = 0; i < 20; i++) {
    if (imu.getSensorData() == BMI2_OK) {
      sum += sqrtf(imu.data.accelX * imu.data.accelX + imu.data.accelY * imu.data.accelY +
                   imu.data.accelZ * imu.data.accelZ);
      n++;
    }
    delay(20);
  }
  float vm = n ? sum / n : 0;
  Serial.printf("#I,startup_|a|=%.3f g (expect 0.95-1.05 if the collar is still)\n", vm);
  if (vm < 0.5f || vm > 1.5f) Serial.println("#W,|a| far from 1 g - check range/scaling or keep the collar still at boot");
  return true;
}

static inline void mapAxes(float sx, float sy, float sz, float out[3]) {
  const float src[3] = {sx, sy, sz};
  out[0] = MAP_X_SGN * src[MAP_X_SRC];
  out[1] = MAP_Y_SGN * src[MAP_Y_SRC];
  out[2] = MAP_Z_SGN * src[MAP_Z_SRC];
}

// Free an I2C bus that a slave is holding low (standard 9-clock recovery + STOP).
static void i2cBusClear(int sda, int scl) {
  if (sda < 0 || scl < 0) return;
  Wire.end();
  pinMode(sda, INPUT_PULLUP);
  pinMode(scl, OUTPUT_OPEN_DRAIN);
  for (int i = 0; i < 9 && digitalRead(sda) == LOW; i++) {
    digitalWrite(scl, LOW); delayMicroseconds(10);
    digitalWrite(scl, HIGH); delayMicroseconds(10);
  }
  pinMode(sda, OUTPUT_OPEN_DRAIN);                    // STOP: SDA low -> high while SCL high
  digitalWrite(sda, LOW); delayMicroseconds(10);
  digitalWrite(scl, HIGH); delayMicroseconds(10);
  digitalWrite(sda, HIGH); delayMicroseconds(10);
  pinMode(sda, INPUT); pinMode(scl, INPUT);
}

// A failed read on a module with a floating SDO pin usually means the address bit flipped (0x68 <-> 0x69).
// The chip and its configuration are fine - just talk to the other address (costs ~1 ms, no re-init).
static volatile uint32_t addrFlips = 0;
static void flipAddress() {
#if BMI270_I2C_ADDR == 0
  imuAddr = (imuAddr == 0x68) ? 0x69 : 0x68;
  imu.setI2CAddress(imuAddr);
  addrFlips++;
#endif
}

// Re-acquire the sensor after a communication loss (loose wire, floating address pin, stuck bus).
static void recoverImu() {
  recoveries++;
  Serial.printf("#W,imu_recovery,%lu - lost contact with BMI270, re-initialising\n", (unsigned long)recoveries);
  i2cBusClear(activeSda, activeScl);
  delay(5);
  if (setupImu()) Serial.println("#I,imu_recovered");
}

// Reader task = the ONLY task that touches I2C. Wakes on data-ready (INT pin) or polls the
// data-ready flag every 1 ms, reads one sample, and recovers the sensor if contact is lost.
static void readerLoop(void*) {
  int64_t lastT = 0;
  const int64_t periodUs = 1000000 / SAMPLE_HZ;
  TickType_t lastWake = xTaskGetTickCount();
  lastSampleUs = esp_timer_get_time();
  for (;;) {
    if (esp_timer_get_time() - lastSampleUs > (int64_t)STALL_RECOVER_MS * 1000) {
      recoverImu();
      lastSampleUs = esp_timer_get_time();         // give the recovered sensor a fresh window
      lastT = 0;                                    // do not count the outage as one giant 'drop'
      continue;
    }
    if (tempRequest) {                              // temperature read here, never from loop()
      float tc = NAN;
      if (imu.getTemperature(&tc) == BMI2_OK && tc > -40.0f && tc < 85.0f) lastTempC = tc;  // -41 = invalid code
      else lastTempC = NAN;
      tempRequest = false;
    }
#if PIN_BMI_INT1 >= 0
    if (ulTaskNotifyTake(pdTRUE, pdMS_TO_TICKS(200)) == 0) { readErrors++; continue; }   // no DRDY in 200 ms
    int64_t t = isrTimeUs;
#else
    // No INT pin: poll the BMI270 status register every 1 ms and read only when a NEW sample is ready
    // (bit 0x80 = drdy_acc). Avoids duplicated/skipped samples from free-running timers; timing error < 1 ms.
    uint8_t st = 0;
    if (imu.getStatus(&st) != BMI2_OK) { readErrors++; flipAddress(); vTaskDelay(1); continue; }
    if (!(st & BMI2_DRDY_ACC)) { vTaskDelay(1); continue; }
    int64_t t = esp_timer_get_time();
    (void)lastWake;
#endif
    if (imu.getSensorData() != BMI2_OK) { readErrors++; flipAddress(); continue; }
    lastSampleUs = t;
    lastGoodUs = t;
    if (lastT) {
      int64_t dt = t - lastT;
      if (dt > periodUs * 3 / 2) droppedSamples += (uint32_t)((dt + periodUs / 2) / periodUs) - 1;
    }
    lastT = t;
    Sample s;
    s.t_us = t;
    mapAxes(imu.data.accelX, imu.data.accelY, imu.data.accelZ, s.a);
#if RECORD_GYRO
    mapAxes(imu.data.gyroX, imu.data.gyroY, imu.data.gyroZ, s.g);
#endif
    if (xQueueSend(sampleQ, &s, 0) != pdTRUE) queueOverflows++;
  }
}

// ---------------------------------------------------------------- SD
// Microphone (sound LEVEL only). Privacy by design: audio samples never leave this task. Every 0.5 s it
// publishes two numbers - RMS and peak level in dBFS - plus whether the mic produced any data at all.
#if USE_MIC
static I2SClass micI2S;
static volatile float micDbRms = NAN, micDbPeak = NAN;
static volatile bool micFresh = false, micDead = false;

static void micLoop(void*) {
  micI2S.setPins(PIN_MIC_SCK, PIN_MIC_WS, -1, PIN_MIC_SD);
  if (!micI2S.begin(I2S_MODE_STD, MIC_RATE, I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_MONO, I2S_STD_SLOT_LEFT)) {
    Serial.println("#E,mic_i2s_begin_failed");
    vTaskDelete(nullptr);
  }
  static int32_t buf[MIC_RATE / 10];                  // 100 ms blocks
  double sumsq = 0;
  float peak = 0;
  uint32_t n = 0, nonzero = 0;
  int blocks = 0;
  for (;;) {
    size_t got = micI2S.readBytes((char*)buf, sizeof(buf)) / sizeof(int32_t);
    if (got == 0) { vTaskDelay(5); continue; }
    double mean = 0;
    for (size_t i = 0; i < got; i++) mean += (double)(buf[i] >> 8);   // INMP441: 24-bit left-justified
    mean /= got;                                      // remove DC offset per block
    for (size_t i = 0; i < got; i++) {
      if (buf[i] != 0) nonzero++;
      float x = (float)(((double)(buf[i] >> 8) - mean) / 8388608.0);
      sumsq += (double)x * x;
      float ax = fabsf(x);
      if (ax > peak) peak = ax;
    }
    n += got;
    if (++blocks >= 5) {                              // publish every 0.5 s
      float rms = sqrtf((float)(sumsq / n));
      micDbRms = 20.0f * log10f(rms + 1e-9f);
      micDbPeak = 20.0f * log10f(peak + 1e-9f);
      micDead = (nonzero == 0);
      micFresh = true;
      sumsq = 0; peak = 0; n = 0; nonzero = 0; blocks = 0;
    }
  }
}
#endif

static void closeFiles() {
  if (rawFile) { rawFile.print(rawBuf); rawBuf = ""; rawFile.close(); }
  if (qcFile) qcFile.close();
  if (evFile) evFile.close();
}

static void openFiles() {
  if (!sdOk) return;
  closeFiles();
  String dir = "/" + dogId;
  if (!SD.exists(dir)) SD.mkdir(dir);
  int64_t now = nowEpochMs();
  String stamp = timeSynced ? utcStamp(now) : ("UNSYNCED-" + String((long)espMs()));
  String base = dir + "/" + dogId + "_" + stamp;
  rawFile = SD.open(base + "_raw.csv", FILE_WRITE);
  qcFile = SD.open(base + "_qc.csv", FILE_WRITE);
  evFile = SD.open(base + "_events.csv", FILE_WRITE);
  if (!rawFile || !qcFile || !evFile) { Serial.println("#E,sd_open_failed"); sdOk = false; return; }
  rawFile.println(RECORD_GYRO ? "timestamp_ms,ax,ay,az,gx,gy,gz" : "timestamp_ms,ax,ay,az");
  qcFile.println("timestamp_ms,vm_std_mg,rest_jitter_mg,temp_c,samples,dropped,read_errors,battery_v,recoveries");
  evFile.println("timestamp_ms,label");
  fileStartMs = espMs();
  Serial.printf("#I,sd_file,%s\n", base.c_str());
}

static void setupSd() {
#if USE_SD
  SPI.begin(PIN_SD_SCK, PIN_SD_MISO, PIN_SD_MOSI, PIN_SD_CS);
  sdOk = SD.begin(PIN_SD_CS, SPI, 20000000);
  Serial.printf("#I,sd_card,%s\n", sdOk ? "ok" : "not_found (serial streaming only)");
  if (sdOk) openFiles();
#endif
}

// ---------------------------------------------------------------- events & time
static void writeEvent(const String& label) {
  int64_t ts = timeSynced ? nowEpochMs() : espMs();
  Serial.printf("#M,%lld,%s\n", (long long)ts, label.c_str());
  if (sdOk && evFile) { evFile.printf("%lld,%s\n", (long long)ts, label.c_str()); evFile.flush(); }
}

static void setTimeFromEpoch(int64_t epochMs) {
  epochOffsetMs = epochMs - espMs();
  bool first = !timeSynced;
  timeSynced = true;
  Serial.printf("#I,time_synced,%lld\n", (long long)epochMs);
  if (first && sdOk) openFiles();                      // replace UNSYNCED files with properly named ones
}

static void tryWifiTime() {
  if (strlen(WIFI_SSID) == 0) return;
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  for (int i = 0; i < 40 && WiFi.status() != WL_CONNECTED; i++) delay(250);
  if (WiFi.status() == WL_CONNECTED) {
    configTime(0, 0, "pool.ntp.org", "time.google.com");
    struct tm tmv;
    if (getLocalTime(&tmv, 10000)) {
      struct timeval tv;
      gettimeofday(&tv, nullptr);
      setTimeFromEpoch((int64_t)tv.tv_sec * 1000 + tv.tv_usec / 1000);
    }
  }
  WiFi.disconnect(true);
  WiFi.mode(WIFI_OFF);                                 // radio off: power
}

// ---------------------------------------------------------------- QC telemetry (per minute)
// vm_std_mg     : median over 2 s windows of SD(|a|)            -> ~0 = frozen sensor
// rest_jitter_mg: median over STILL windows of RMS(diff(|a|))     -> high = loose collar rattling
static float winVm[SAMPLE_HZ * 2];
static int winN = 0;
static float stdList[40], jitList[40];
static int stdN = 0, jitN = 0;
static uint32_t minuteSamples = 0;
static int64_t minuteStartMs = 0;

static float medianOf(float* v, int n) {
  if (n == 0) return NAN;
  for (int i = 1; i < n; i++) { float x = v[i]; int j = i - 1; while (j >= 0 && v[j] > x) { v[j + 1] = v[j]; j--; } v[j + 1] = x; }
  return n % 2 ? v[n / 2] : 0.5f * (v[n / 2 - 1] + v[n / 2]);
}

static void qcAddSample(const float a[3]) {
  winVm[winN++] = sqrtf(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
  minuteSamples++;
  if (winN == SAMPLE_HZ * 2) {
    float m = 0; for (int i = 0; i < winN; i++) m += winVm[i]; m /= winN;
    float v = 0; for (int i = 0; i < winN; i++) v += (winVm[i] - m) * (winVm[i] - m);
    float sd = sqrtf(v / winN);
    if (stdN < 40) stdList[stdN++] = sd * 1000.0f;
    if (sd < 0.030f) {                                  // still window (lying/standing)
      float d = 0; for (int i = 1; i < winN; i++) { float x = winVm[i] - winVm[i - 1]; d += x * x; }
      if (jitN < 40) jitList[jitN++] = sqrtf(d / (winN - 1)) * 1000.0f;
    }
    winN = 0;
  }
}

static float batteryVolts() {
#if PIN_BATTERY_ADC >= 0
  return analogReadMilliVolts(PIN_BATTERY_ADC) * BATTERY_DIVIDER / 1000.0f;
#else
  return NAN;
#endif
}

static void qcEmit() {
  float temp = lastTempC;                              // read by the reader task (single I2C owner)
  tempRequest = true;                                  // ask for a fresh one for the next minute
  int64_t ts = timeSynced ? nowEpochMs() : espMs();
  char line[176];
  snprintf(line, sizeof(line), "%lld,%.2f,%.3f,%.1f,%lu,%lu,%lu,%.2f,%lu", (long long)ts, medianOf(stdList, stdN),
           medianOf(jitList, jitN), temp, (unsigned long)minuteSamples, (unsigned long)droppedSamples,
           (unsigned long)readErrors, batteryVolts(), (unsigned long)recoveries);
  Serial.printf("#Q,%s\n", line);
  if (addrFlips) Serial.printf("#W,i2c_address_flips_last_minute,%lu (floating SDO pin on the module - see HARDWARE_SETUP.md)\n",
                               (unsigned long)addrFlips);
  addrFlips = 0;
  if (sdOk && qcFile) { qcFile.println(line); qcFile.flush(); }
  stdN = jitN = 0;
  minuteSamples = 0;
  droppedSamples = 0;
  readErrors = 0;
  recoveries = 0;
}

// ---------------------------------------------------------------- serial commands
static String cmdBuf;
static void handleCommand(const String& c) {
  if (c.length() == 0) return;
  char k = c[0];
  String v = c.substring(1);
  v.trim();
  if (k == 'T') setTimeFromEpoch(atoll(v.c_str()));
  else if (k == 'I' && v.length()) { dogId = v; prefs.putString("dog", dogId); Serial.printf("#I,dog_id,%s\n", dogId.c_str()); if (sdOk) openFiles(); }
  else if (k == 'M') writeEvent(v.length() ? v : "serial_marker");
  else if (k == 'S') {
    Serial.printf("#I,status,fw=%s,dog=%s,time_synced=%d,sd=%d,queue=%u,overflows=%lu,range_g=%d,odr_hz=%.1f\n",
                  FW_VERSION, dogId.c_str(), timeSynced, sdOk, (unsigned)uxQueueMessagesWaiting(sampleQ),
                  (unsigned long)queueOverflows, accelRangeG, accelOdrHz);
  }
}

// ---------------------------------------------------------------- Arduino
void setup() {
  Serial.begin(SERIAL_BAUD);
  delay(1500);
  Serial.printf("#I,praniiva_collar,fw=%s,reset_reason=%d (1=power-on 3=software 4=panic/crash 5-7=watchdog 9=brownout)\n",
                FW_VERSION, (int)esp_reset_reason());
  prefs.begin("praniiva", false);
  dogId = prefs.getString("dog", "ID0000");
  if (esp_reset_reason() == ESP_RST_SW && savedMagic == TIME_MAGIC) {   // our own stall restart: keep the clock
    epochOffsetMs = savedEpochMs + 300 - espMs();     // ~300 ms between save and this point (estimate)
    timeSynced = true;
    Serial.printf("#W,time_restored_after_restart,%lld (accuracy ~0.5 s)\n", (long long)nowEpochMs());
  }
  savedMagic = 0;
  pinMode(PIN_MARKER_BUTTON, INPUT_PULLUP);

  sampleQ = xQueueCreate(256, sizeof(Sample));
  if (!setupImu()) {
    Serial.println("#E,imu_not_ready - check: 3V3/GND, SDA/SCL wires (see config.h PIN_I2C_SDA/SCL), module powered");
    do {                                               // keep retrying: starts by itself once wiring is fixed
      delay(3000);
      Serial.printf("#E,imu_not_ready,chip_id@0x68=0x%02X,@0x69=0x%02X - retrying\n", readChipId(0x68), readChipId(0x69));
    } while (!setupImu());
    Serial.println("#I,imu_ready_after_retry");
  }
  tryWifiTime();
  setupSd();
  if (!timeSynced) Serial.println("#W,time_not_synced - run tools/collar_logger.py (sends T<epoch_ms>) or set WIFI_SSID");

  xTaskCreatePinnedToCore(readerLoop, "bmi_reader", 4096, nullptr, configMAX_PRIORITIES - 2, &readerTask, 0);
#if USE_MIC
  xTaskCreatePinnedToCore(micLoop, "mic", 4096, nullptr, 2, nullptr, 1);
#endif
#if PIN_BMI_INT1 >= 0
  pinMode(PIN_BMI_INT1, INPUT);
  attachInterrupt(digitalPinToInterrupt(PIN_BMI_INT1), onDataReady, RISING);
#endif
  minuteStartMs = espMs();
  lastFlushMs = espMs();
  Serial.println(RECORD_GYRO ? "#H,timestamp_ms,ax,ay,az,gx,gy,gz" : "#H,timestamp_ms,ax,ay,az");
}

void loop() {
  Sample s;
  char line[112];
  while (xQueueReceive(sampleQ, &s, 0) == pdTRUE) {
    int64_t ts = timeSynced ? toEpochMs(s.t_us) : s.t_us / 1000;
#if RECORD_GYRO
    int n = snprintf(line, sizeof(line), "%lld,%.5f,%.5f,%.5f,%.2f,%.2f,%.2f", (long long)ts, s.a[0], s.a[1], s.a[2],
                     s.g[0], s.g[1], s.g[2]);
#else
    int n = snprintf(line, sizeof(line), "%lld,%.5f,%.5f,%.5f", (long long)ts, s.a[0], s.a[1], s.a[2]);
#endif
#if STREAM_TO_SERIAL
    Serial.println(line);
#endif
    if (sdOk && rawFile) { rawBuf += line; rawBuf += '\n'; }
    (void)n;
    qcAddSample(s.a);
  }

  int64_t now = espMs();
  if (sdOk && rawFile && (now - lastFlushMs >= SD_FLUSH_SECONDS * 1000 || rawBuf.length() > 8192)) {
    rawFile.print(rawBuf);
    rawBuf = "";
    rawFile.flush();
    lastFlushMs = now;
    if (now - fileStartMs >= (int64_t)SD_ROTATE_HOURS * 3600000LL) openFiles();
  }
  if (now - minuteStartMs >= 60000) { qcEmit(); minuteStartMs = now; }

#if USE_MIC
  if (micFresh) {                                      // sound level twice a second: #A,ts,rms_dbfs,peak_dbfs,ok
    micFresh = false;
    int64_t ts = timeSynced ? nowEpochMs() : espMs();
    Serial.printf("#A,%lld,%.1f,%.1f,%d\n", (long long)ts, micDbRms, micDbPeak, micDead ? 0 : 1);
    static int deadCount = 0;
    if (micDead && ++deadCount % 20 == 1)
      Serial.println("#W,mic_no_data - check INMP441 wiring: SCK->GPIO5, WS->GPIO6, SD->GPIO7, L/R->GND, VDD->3V3");
    if (!micDead) deadCount = 0;
  }
#endif

  // last resort: sensor still silent after recovery attempts -> restart the ESP32, keeping the clock
  if (lastGoodUs && (esp_timer_get_time() - lastGoodUs) > (int64_t)STALL_RESTART_MS * 1000) {
    Serial.println("#E,sensor_stalled_20s - restarting");
    writeEvent("watchdog_restart");
    closeFiles();
    savedEpochMs = timeSynced ? nowEpochMs() : 0;
    savedMagic = timeSynced ? TIME_MAGIC : 0;
    Serial.flush();
    esp_restart();
  }

  // marker button (debounced, active low)
  static bool lastBtn = true; static int64_t lastBtnMs = 0;
  bool b = digitalRead(PIN_MARKER_BUTTON);
  if (!b && lastBtn && now - lastBtnMs > 300) { writeEvent("button"); lastBtnMs = now; }
  lastBtn = b;

  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') { handleCommand(cmdBuf); cmdBuf = ""; }
    else if (cmdBuf.length() < 64) cmdBuf += c;
  }
  delay(2);
}
