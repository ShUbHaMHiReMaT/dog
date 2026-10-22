# PRANIIVA collar v0 - hardware setup (ESP32-S3 + BMI270)

> **For:** whoever assembles and flashes the Phase 1 research collars. This is a **bench/research build** for
> collecting labelled data - not the final low-power product (see "Limits" at the end).

## 1. Wiring (I2C)

### Bring-up log of the first unit (13 Oct 2026) - read this if a new unit misbehaves
| Symptom | Cause found | Fix |
|---|---|---|
| I2C scan finds nothing; BMI LED blinks in step with GPIO 9 | module GND not making contact (ground returned through SCL) | re-seated / replaced GND jumper |
| SDA line floating (no pull-up seen on GPIO 8) | bad contact at the ESP32's GPIO 8 header | SDA moved to **GPIO 4** (`PIN_I2C_SDA 4`, `PIN_I2C_SCL 9`) |
| sensor answers at 0x68 on one boot and 0x69 on the next; contact lost every 15-30 s | **4-pin module leaves the BMI270 SDO (address) pin floating** - the address flips at runtime | firmware: bundled library retries at the other address on any failed transfer (0 errors in 3 min); hardware fix below |
| library returned "OK" with stale data on short reads | bug in SparkFun `readRegistersI2C` (byte count not checked) | fixed in the bundled copy |
| -41 °C temperature | BMI270's "invalid" code (gyro off) | reported as NaN |

**Recommended hardware fix for the floating address pin:** find the module's `SDO` / `SA0` / `AD0` pad (or the
chip's SDO pin) and solder it to GND (address 0x68) - then set `BMI270_I2C_ADDR 0x68`. The firmware copes without
it, but a defined pin is always better than a floating one, especially on a moving dog.

Current verified wiring of unit #1: **3V3→3V3, GND→GND, SDA→GPIO 4, SCL→GPIO 9**.
Guided bench test (`tools/check_sensor.py --port COM15`, report `data/bench/bench_20261013_212220.json`):
**all checks passed** - 50.0 Hz, 0 % missing, 0 ms jitter, |a| 0.994 g, noise 0.63 mg RMS / 115 µg/√Hz,
LSB 0.24 mg, x-axis toward head OK (ax +0.98 g), range/ODR read back from chip ±8 g / 50 Hz, shake peak 2.57 g
with no clipping, file accepted by the pipeline. 3-minute endurance run: 9,169 samples, 0 gaps, 0 read errors.

**4-pin BMI270 module (3V3, GND, SDA, SCL) - the current build:**

| Module pin | ESP32-S3 |
|---|---|
| 3V3 | 3V3 |
| GND | GND |
| SDA | GPIO 8 |
| SCL | GPIO 9 |

That's all. CS and SDO are tied on the module; the firmware auto-detects address 0x68/0x69 and, with no INT pin,
polls the sensor's data-ready flag every 1 ms (`PIN_BMI_INT1 -1`) - each sample is read exactly once, timing
error < 1 ms. The full table below applies to modules that break out CS/SDO/INT1.

| BMI270 module pin | ESP32-S3 pin (DevKitC-1 default) | Notes |
|---|---|---|
| VCC / VIN / VDD | **3V3** | Use 3.3 V. BMI270 absolute max is 3.6 V - never 5 V unless the module has its own regulator |
| VDDIO (if separate) | 3V3 | |
| GND | GND | |
| SDA | **GPIO 8** | 4.7 kΩ pull-up to 3V3 if the module has none |
| SCL | **GPIO 9** | 4.7 kΩ pull-up to 3V3 if the module has none |
| INT1 | **GPIO 4** | data-ready interrupt → exact 50 Hz timing (set `PIN_BMI_INT1 -1` to poll instead) |
| SDO / SA0 | GND | I2C address **0x68** (to 3V3 → 0x69, then change `BMI270_I2C_ADDR`) |
| CS / CSB | **3V3** | must be HIGH at power-up to select I2C mode |
| INT2 | not connected | |

Optional **microSD** (SPI) for logging while the dog wears it:

| microSD module | ESP32-S3 |
|---|---|
| CS | GPIO 10 |
| MOSI / DI | GPIO 11 |
| SCK / CLK | GPIO 12 |
| MISO / DO | GPIO 13 |
| VCC / GND | 3V3 / GND (use a 3.3 V module or one with a level shifter) |

Optional **battery sense:** LiPo+ → 100 kΩ → GPIO 1 → 100 kΩ → GND, then set `PIN_BATTERY_ADC 1`.

```
            ESP32-S3-DevKitC-1                          BMI270 module
          ┌──────────────────┐                      ┌───────────────┐
          │ 3V3 ─────────────┼──────────┬───────────┤ VCC  (VDDIO)  │
          │                  │          └───────────┤ CS            │
          │ GND ─────────────┼──────────┬───────────┤ GND           │
          │                  │          └───────────┤ SDO  (=0x68)  │
          │ GPIO8  (SDA) ────┼──────────────────────┤ SDA           │
          │ GPIO9  (SCL) ────┼──────────────────────┤ SCL           │
          │ GPIO4  (INT) ────┼──────────────────────┤ INT1          │
          │ GPIO0  BOOT btn  │  = marker button     └───────────────┘
          │ GPIO10-13 ───────┼── microSD (optional)
          └──────────────────┘
```
A different ESP32-S3 board (XIAO, Feather, etc.)? Pick any free GPIOs and change `firmware/praniiva_collar/config.h`.
Avoid GPIO 19/20 (USB) and the PSRAM pins of N16R8 modules (GPIO 35-37).

## 1b. Battery build for demos and dog trials (LiPo + TP4056 USB-C + 3.3 V LDO + microSD)

```
 USB-C (charging)
      │
 ┌────┴─────────────┐        ┌──────────── SLIDE SWITCH (off while the ESP32's own USB is plugged in)
 │ TP4056 (with     │ OUT+ ──┘──────┐
 │ DW01 protection) │ OUT- ─────────┼──────────────────────────────── common GND ───────────────┐
 │  B+  B-          │               │                                                          │
 └──┬───┬───────────┘          ┌────┴─────┐ 3.3 V                                               │
    │   │                      │ 3.3 V LDO├────────┬──────────────┬───────────────┐             │
  LiPo 3.7 V                   │ (HW-764) │        │              │               │             │
  (+)  (-)                     └──────────┘   ESP32-S3 3V3    BMI270 3V3   microSD 3V3*         │
                                                   GND ────────── GND ─────────── GND ───────────┘
 battery sense (optional): TP4056 B+ ── 100 kΩ ──┬── 100 kΩ ── GND
                                                 └── ESP32 GPIO 1   (config.h: PIN_BATTERY_ADC 1)
```

| From | To | Notes |
|---|---|---|
| LiPo + / − | TP4056 **B+ / B−** | single-cell 3.7 V LiPo with protection; check polarity twice - reversed LiPo destroys the TP4056 |
| TP4056 **OUT+** | slide switch → LDO **VIN** | OUT (not B) so the DW01 protection cuts off a flat battery |
| TP4056 **OUT−** | LDO **GND** + common GND | |
| LDO **3.3 V out** | ESP32-S3 **3V3** pin | feeds the 3.3 V rail directly (bypasses the board's own regulator) |
| ESP32 3V3 / GND | BMI270 3V3 / GND | unchanged |
| ESP32 GPIO 4 / 9 | BMI270 SDA / SCL | unchanged |
| ESP32 3V3* / GND | microSD VCC* / GND | *see the SD regulator warning below |
| ESP32 GPIO 10 / 11 / 12 / 13 | microSD **CS / MOSI / SCK / MISO** | `config.h` PIN_SD_* |

**Before connecting the battery - three checks:**
1. **What chip is on the HW-764?** Read the marking on the 3-legged/5-legged chip.
   * `AMS1117-3.3` (or `1117`) → **will not work from a LiPo**: it needs ≥ 4.4 V in, the ESP32 will brown-out and reset.
     Use a low-dropout regulator instead: **ME6211 / RT9013 / XC6220 / MCP1825 (≥ 500 mA)**, or a 3.3 V buck-boost module.
   * `ME6211`, `RT9013`, `XC6220`, `TLV757`, `662K` (XC6206 - only 200 mA, too weak for the ESP32) → check its current ≥ 500 mA.
   * Measure: with the battery connected and no load, LDO output must read **3.25-3.35 V**.
2. **microSD module type.** Most blue 6-pin modules (`Catalex`, with a `74LVC125` chip) have an **AMS1117 for 5 V
   input** - fed from 3.3 V the card gets only ~2.3 V and fails randomly. Either buy a **3.3 V-only microSD breakout
   (no regulator)**, or bridge that module's AMS1117 input to output and feed 3.3 V (never 5 V after bridging).
3. **TP4056 charge current** is 1 A by default (R_PROG 1.2 kΩ). Charge at ≤ 1C: for a battery **smaller than
   1000 mAh** replace R_PROG (e.g. 2.4 kΩ = 500 mA for a 500 mAh cell).

**USB and battery together:** never feed the 3V3 rail from the LDO while the ESP32's own USB is connected (two
regulators fighting). Switch the battery off before plugging USB in - or live-stream over USB without the battery.

**Battery life estimate (to measure, not promise):** ESP32-S3 with radios off + BMI270 + SD writes ≈ 50-90 mA →
1000 mAh ≈ 11-18 h. Fine for day-long trials; the 14-day product needs the low-power redesign (DEVICE_SPEC.md).

**SD card:** FAT32, ≤ 32 GB. Files appear as `/<dog>/<dog>_<UTC time>_{raw,qc,events}.csv`. Sync the clock first
(USB + `tools/collar_logger.py` for a few seconds, then unplug) - or the files are named `UNSYNCED-...` and the
pipeline rejects them on purpose.

## 2. Mounting - this matters for every algorithm
* The board is fixed **rigidly** to the collar housing (no foam, no free play), on the side of the neck.
* The algorithms expect **device x = along the neck, pointing toward the dog's HEAD**. Look at the axis arrows
  printed on the BMI270 module. If x does not point to the head, change `MAP_X_SRC / MAP_X_SGN` (and y, z) in
  `config.h`. `tools/check_sensor.py` step 2 tells you exactly what to set.
* Rotation **around** the neck is fine - the features are built to be invariant to it.
* Fit: two fingers under the collar. A loose collar is detected (rest-jitter telemetry), but its data is discarded.

## 3. Flash the firmware
The BMI270 library is **bundled** inside `firmware/praniiva_collar/src/` (your Arduino library folder path contains
non-ASCII characters, which breaks the ESP32 compiler on Windows).

```powershell
cd d:\dog\firmware\praniiva_collar
arduino-cli compile --fqbn esp32:esp32:esp32s3 .
arduino-cli upload  --fqbn esp32:esp32:esp32s3 -p COM5 .     # your port
```
Verified: compiles with esp32 core 3.3.11 (500 KB flash, 30 KB RAM).
Arduino IDE: board "ESP32S3 Dev Module". If you use the board's native **USB** port (not the UART port), set
*Tools → USB CDC On Boot → Enabled*, otherwise you will see no serial output.

Expected serial output at boot (921600 baud):
```
#I,praniiva_collar,fw=0.3.0
#I,chip_id=0x24 (...)                  <- 0x24 = genuine BMI270
#I,startup_|a|=1.002 g (...)           <- must be ~1.00 with the collar still
#I,sd_card,ok | not_found (serial streaming only)
#W,time_not_synced - run tools/collar_logger.py ...
#H,timestamp_ms,ax,ay,az
```
`chip_id=0x26` means the module carries a BMI220, not a BMI270. `0x00` means no I2C answer: check wiring, CS high,
SDO/address.

## 4. Check the sensor before any dog (≈ 2 minutes)
```powershell
pip install pyserial
python tools/check_sensor.py            # guided: still -> x-axis up -> shake
```
PASS/FAIL against what the pipeline needs:

| Check | Requirement | Why |
|---|---|---|
| sample rate | 50 ± 1 Hz | the behaviour models were built at 50 Hz |
| missing samples | < 0.5 % | gaps become "invalid windows" |
| timestamp jitter p95 | < 5 ms | interrupt timing works |
| \|a\| at rest | 1.00 ± 0.05 g | range/scaling correct |
| noise RMS at rest | < 3 mg | breathing micro-motion is ~1-3 mg |
| noise density | ≤ 400 µg/√Hz | DEVICE_SPEC.md |
| resolution | < 1 mg | 16-bit at ±8 g = 0.244 mg |
| orientation | ax ≈ +1 g with the head-direction up | collar frame |
| shake | peak > 2 g, < 1 % clipped | dynamic range |
| pipeline ingest | file accepted by `praniiva/ingest.py` | end-to-end contract |

## 5. Record data
**USB (bench, labelling sessions with video):**
```powershell
python tools/collar_logger.py --dog ID0001
```
It syncs the collar clock, writes `data/raw/ID0001/ID0001_<date>_{raw,qc,events}.csv`, and lets you type a
label + Enter (or press the BOOT button) to drop a marker. Start every video session with **3 sharp taps on the
collar + a marker** so video and data can be aligned to < 0.1 s.

**microSD (dog wears it at home):** sync the clock first (plug into the PC and run the logger for a few seconds,
or set `WIFI_SSID` for NTP), then unplug. Files rotate every 6 h. Copy the card's `/<dog>/` folder into
`data/raw/<dog>/`. Files from an unsynced boot are named `UNSYNCED-...` and the pipeline rejects them - by design.

## 6. Process
```powershell
python tools/process_collar_data.py --dog ID0001   # first run creates dog.json - fill it in
```
Day 1-35 report `insufficient_data`: that is the personal-baseline learning period, not a bug.

## 7. Data contract (what the system needs - checked automatically by `ingest.py`)

| Item | Required | Enforced by |
|---|---|---|
| columns | `timestamp_ms,ax,ay,az` (extra columns ignored) | `load_raw` |
| time | epoch **milliseconds UTC**; days are split on India local date | unit sniffing + tz conversion |
| units | g (m/s² auto-converted, anything else rejected) | median \|a\| check |
| rate | 50 Hz nominal (20-200 Hz accepted, resampled, warned) | sample-rate estimate |
| range / resolution | ±8 g, 16-bit | bench test |
| axes | x along the neck toward the head | bench test step 2 |
| gaps | > 1 s gaps never interpolated into motion | whole-window validity mask |
| coverage | ≥ 75 % of the day for daily totals; ≥ 50 % of 00-06 h for night metrics / breathing rate | `daily_from_raw` |
| telemetry | per-minute `rest_jitter_mg`, `vm_std_mg` (loose collar / frozen sensor detection) | `*_qc.csv` → `load_qc` |
| labels (Phase 1) | ≥ 20 min per behaviour per dog, video + markers, two annotators | protocol, not code |

## 8. Limits of this build (be honest about them)
* **Power:** the ESP32-S3 runs continuously at roughly 40-80 mA (radios off), so a 1000 mAh LiPo lasts well under a
  day (estimate - measure it). That is fine for Phase 1 labelling sessions, **not** for the 14-day product target, which needs FIFO
  batching + deep sleep or a BLE-SoC class MCU (DEVICE_SPEC.md).
* **Breathing rate from the collar is experimental** until validated against reference counts.
* **Loose-collar thresholds** (`fit_jitter_mg`) were set in simulation; calibrate them on the bench with the
  collar deliberately loose vs. snug.
* No BLE app yet - USB and microSD only.
