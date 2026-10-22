// =====================================================================================
// PRANIIVA collar v0 firmware - configuration. Change pins/options here only.
// Board: ESP32-S3 (defaults = ESP32-S3-DevKitC-1).   Sensor: Bosch BMI270 over I2C.
// =====================================================================================
#pragma once

// ---------------- I2C to BMI270 ----------------
#define PIN_I2C_SDA        4        // GPIO8 had a bad contact on this board -> uses 4
#define PIN_I2C_SCL        9
#define I2C_AUTO_SWAP      1        // 1 = also try SDA/SCL the other way round (wires swapped)
#define I2C_FREQ_HZ        100000   // 100 kHz: robust over jumper wires (50 Hz x 6 bytes needs <1% of the bus)
#define BMI270_I2C_ADDR    0        // 0 = auto-detect (tries 0x68 then 0x69). 4-pin modules fix SDO internally.
#define PIN_BMI_INT1       -1       // -1 = no INT pin (4-pin module): poll the data-ready flag every 1 ms.
                                    //  Wire INT1 to a GPIO and set it here for interrupt timing (e.g. 4).

// ---------------- microSD (optional, SPI) ----------------
#define USE_SD             1        // 1 = log to microSD when a card is present
#define PIN_SD_CS          10
#define PIN_SD_MOSI        11
#define PIN_SD_SCK         12
#define PIN_SD_MISO        13
#define SD_ROTATE_HOURS    6        // start a new file every N hours (limits loss if a file corrupts)
#define SD_FLUSH_SECONDS   10       // at most this much data is lost on sudden power loss

// ---------------- INMP441 I2S microphone (research add-on; sound LEVEL only, never raw audio) ----------------
#define USE_MIC            1
#define PIN_MIC_SCK        5        // INMP441 SCK (bit clock)
#define PIN_MIC_WS         6        // INMP441 WS  (word select)
#define PIN_MIC_SD         7        // INMP441 SD  (data out of the mic)
#define MIC_RATE           16000    // Hz; INMP441 L/R pin -> GND = left channel

// ---------------- Marker button (video-sync / labelling) ----------------
#define PIN_MARKER_BUTTON  14       // tactile button: diagonal legs to GPIO14 and GND (internal pull-up)

// ---------------- Optional battery sense (via 1:2 divider) ----------------
#define PIN_BATTERY_ADC    -1       // e.g. 1 ; -1 = not connected
#define BATTERY_DIVIDER    2.0f

// ---------------- Sampling (MUST match praniiva/params.py & DEVICE_SPEC.md) ----------------
#define SAMPLE_HZ          50       // models were built for 50 Hz
#define ACCEL_RANGE_G      8        // +-8 g (head shakes reach 2-3 g)
#define RECORD_GYRO        0        // 0 = gyro off (saves ~0.6 mA; models do not use it); 1 = log gx,gy,gz (deg/s)

// ---------------- Mounting: map sensor axes to the COLLAR frame ----------------
// Collar frame used by every PRANIIVA algorithm:  x = along the neck pointing to the HEAD,
// y = lateral, z = dorso-ventral. If your board is mounted differently, remap here
// (source axis index 0=x,1=y,2=z and sign) so that the device x axis points toward the dog's head.
#define MAP_X_SRC 0
#define MAP_X_SGN +1
#define MAP_Y_SRC 1
#define MAP_Y_SGN +1
#define MAP_Z_SRC 2
#define MAP_Z_SGN +1

// ---------------- Serial ----------------
#define SERIAL_BAUD        921600   // USB-CDC ignores baud; UART bridge uses it
#define STREAM_TO_SERIAL   1        // stream every sample over USB (needed by tools/collar_logger.py)

// ---------------- Optional Wi-Fi time sync (leave SSID empty to disable) ----------------
#define WIFI_SSID          ""
#define WIFI_PASS          ""
#define TZ_INFO            "IST-5:30"   // only used for human-readable logs; data is always UTC epoch ms
