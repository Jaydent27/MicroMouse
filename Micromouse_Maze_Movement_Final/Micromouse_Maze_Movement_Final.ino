/*
  MTRN3100 Micromouse
  LIDAR-ASSISTED ROUTE EXECUTOR
  -----------------------------

  The Python planner still uploads the complete route in the same format:

      ROUTE:turn,mode,distance_mm;turn,mode,distance_mm;...

  Execution remains deliberately simple:
    - Turns use the onboard MPU6050 Z-angle instead of wheel-step geometry.
    - Straight motion uses wheel steps for the exact uploaded nominal distance in mm.
    - Front + dual-left VL6180X sensors provide feedback while driving straight.
    - When the left wall disappears, MPU6050 heading hold keeps the robot aligned.
    - The Arduino motion loop/steppers stay on one ESP32 core.
    - A dedicated FreeRTOS task owns ALL moving-time I2C reads on the opposite core.
    - The MPU6050 is calibrated after the website START command and before movement.
    - NO Wi-Fi processing while the robot is moving.

  LEFT-WALL FEEDBACK
  ------------------
  Left 1 is the FRONT-left sensor.
  Left 2 is the REAR-left sensor.

  Left 2 reads about 35 mm short, so:
      corrected Left2 = raw Left2 + 35 mm

  When BOTH corrected left readings are <= 90 mm, the robot uses their
  difference to stay parallel to the wall.

  If the robot gets closer than 35 mm to the left wall, normal two-sensor
  parallel correction is temporarily abandoned. The robot uses Left 1 only
  and steers AWAY from the wall until Left 2 is back in a usable range.

  If either left sensor sees no usable wall (> 90 mm), left-wall correction
  is ignored and the robot drives by equal wheel speed.

  FRONT-WALL FEEDBACK
  -------------------
  Front-wall readings are guarded by route geometry, IMU heading, and a
  stationary re-check routine:
    - A new <=50 mm front reading immediately pauses forward motion.
    - The mouse rotates IN PLACE back to the planned straight heading and must
      settle within +/-1 deg before the reading can be classified.
    - Any front sample captured during that correction is discarded.
    - The mouse then requires 5 consecutive NEW <=50 mm front samples while
      stationary before accepting the obstacle as real.
    - If that stationary test does not confirm the obstacle, the in-place
      correction steps are removed from straight-distance odometry and the
      mouse resumes the same straight segment.
    - A confirmed wall can end the straight early only within 80 mm of the
      planned endpoint and only when the NEXT route segment contains a turn.
    - A confirmed obstacle somewhere the route does not permit one causes a
      safety stop rather than an incorrect/unplanned turn.
    - A 60..180 mm approach still requires heading alignment + multiple fresh
      samples and is only used near an expected planned turn.

  A hard extension limit is retained only as a fail-safe so a failed front
  sensor cannot make the robot drive forever.

  Dual-core feedback revision generated with assistance from ChatGPT (OpenAI), 13 Aug 2026.
*/

#include <WiFi.h>
#include <WebServer.h>
#include <Wire.h>
#include <VL6180X.h>
#include <MPU6050_light.h>
#include <AccelStepper.h>
#include <esp_system.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <math.h>

// ============================================================
// WIFI
// ============================================================

const char *WIFI_AP_NAME = "MicroMouse";
const char *WIFI_AP_PASSWORD = "micromouse3100";

WebServer webServer(80);

bool webRoutesConfigured = false;
bool wifiActive = false;

bool startPending = false;
uint32_t startAcceptedAtMs = 0;

// Give the HTTP response time to reach the browser before Wi-Fi is disabled.
constexpr uint32_t START_HANDOFF_DELAY_MS = 700;

// ============================================================
// VL6180X FEEDBACK SENSORS
// ============================================================

// Same I2C wiring used by MicromouseMK2.
constexpr uint8_t SDA_PIN = 21;
constexpr uint8_t SCL_PIN = 22;

// Shutdown / enable pins.
constexpr uint8_t FRONT_CE_PIN = 25;
constexpr uint8_t LEFT1_CE_PIN = 26;
constexpr uint8_t LEFT2_SHUT_PIN = 4;

// The right sensor is not used by this executor, but it MUST be held in
// shutdown while the other VL6180X devices are assigned new I2C addresses;
// otherwise it would also respond at the default 0x29 address.
constexpr uint8_t RIGHT_CE_PIN = 27;

// Unique runtime I2C addresses.
constexpr uint8_t FRONT_ADDRESS = 0x30;
constexpr uint8_t LEFT1_ADDRESS = 0x31;
constexpr uint8_t LEFT2_ADDRESS = 0x32;

VL6180X frontSensor;
VL6180X left1Sensor;
VL6180X left2Sensor;

bool frontWorking = false;
bool left1Working = false;
bool left2Working = false;

// ============================================================
// MPU6050 HEADING SENSOR
// ============================================================

// Same MPU6050 + MPU6050_light setup used by MicromouseMK2.
constexpr uint8_t MPU_ADDRESS = 0x68;
MPU6050 mpu(Wire);
bool imuWorking = false;

// Convert raw MPU6050 Z-angle direction into the planner convention:
//   positive = LEFT / counter-clockwise
//   negative = RIGHT / clockwise
//
// With the normal upright MPU6050 mounting this should be +1. If a physical
// LEFT turn makes the reported planner heading decrease, change this to -1.
constexpr float IMU_YAW_SIGN = 1.0f;

// MPU data is generated by the opposite-core sensor task.
constexpr uint32_t IMU_READING_MAX_AGE_MS = 150;

// Straight heading hold is used ONLY when normal left-wall control is not
// usable, so the already-working wall-following behaviour is left untouched.
constexpr float IMU_STRAIGHT_KP_SPEED_PER_DEG = 12.0f;
constexpr float IMU_STRAIGHT_MAX_SPEED_CORRECTION = 180.0f;

// Safety guard while left-wall following is active.  The LiDAR controller may
// trim the heading, but it is never allowed to drag the mouse far away from
// the route heading.  Above ENTER the IMU temporarily takes priority; normal
// wall following resumes after the heading recovers below EXIT.
constexpr float IMU_WALL_GUARD_ENTER_DEG = 12.0f;
constexpr float IMU_WALL_GUARD_EXIT_DEG = 6.0f;

// If the robot arrives at a planned turn already badly off the previous route
// heading, do NOT make the next absolute turn absorb that whole error.  Rebase
// the route heading to the real IMU heading, then execute only the requested
// relative turn.  This prevents a requested ~90 deg turn becoming ~180 deg.
constexpr float IMU_PRETURN_REBASE_DEG = 15.0f;

// Hard turn fail-safe: a physical in-place turn may exceed the requested angle
// only by this small allowance before the motors are stopped and the route is
// aborted.
constexpr float IMU_TURN_EXTRA_TRAVEL_GUARD_DEG = 25.0f;

// Allow the full signed turn range generated by the path planner, including
// a legitimate 180 degree initial reorientation when the robot starts facing
// directly away from the first route segment.
constexpr float MAX_ROUTE_TURN_DEGREES = 180.0f;

// IMU turn controller settings.
constexpr float IMU_TURN_TOLERANCE_DEG = 1.5f;
constexpr float IMU_TURN_MEDIUM_ERROR_DEG = 20.0f;
constexpr float IMU_TURN_SLOW_ERROR_DEG = 6.0f;
constexpr uint32_t IMU_TURN_SETTLE_MS = 100;
constexpr uint32_t IMU_TURN_TIMEOUT_MS = 15000;

// The raw Z angle at the end of the START-time calibration. All route
// headings are measured relative to this value.
float imuRunZeroRawDeg = 0.0f;
float plannedHeadingDeg = 0.0f;

// Calibration temporarily asks the sensor task to stop at a safe point so
// mpu.calcOffsets() has exclusive access to the shared Wire bus.
volatile bool sensorTaskPauseRequested = false;
volatile bool sensorTaskPaused = false;

// Left 2 is the rear-left sensor and historically reads ~35 mm SHORT.
constexpr float LEFT2_OFFSET_MM = 35.0f;

// ============================================================
// LEFT-WALL TRACKING
// ============================================================
//
// The left wall is used as a feedback reference while executing the planner's
// straight segments. The robot still completes the route's commanded forward
// distance; this controller only adds a differential steering correction.
//
// Desired side-wall distance:
constexpr float LEFT_TARGET_DISTANCE_MM = 30.0f;
constexpr float LEFT_DISTANCE_TOLERANCE_MM = 5.0f;

// More than one 180 mm maze cell away is treated as open space.
// If a trustworthy common left wall is not within this range, wall following
// is disabled and the robot drives the planned straight normally.
constexpr float LEFT_WALL_IGNORE_MM = 90.0f;

// Left 2 is the offset sensor. Below about 35 mm it is not trusted for
// parallel-angle measurement, so Left 1 alone controls distance in that case.
constexpr float LEFT2_MIN_USABLE_MM = 35.0f;

// Parallel controller tuning.
// parallel error = front-left - corrected rear-left
//
// error > 0 : nose is farther from wall -> steer LEFT (toward wall)
// error < 0 : nose is closer to wall -> steer RIGHT (away from wall)
constexpr float PARALLEL_DEADBAND_MM = 2.0f;
constexpr float PARALLEL_KP_SPEED_PER_MM = 9.0f;

// Distance controller tuning.
// distance error = measured left distance - 30 mm
//
// error > 0 : robot is too far from left wall -> steer LEFT
// error < 0 : robot is too close to left wall -> steer RIGHT
//
// The +/-5 mm tolerance is removed before applying proportional correction,
// making the correction enter smoothly at 25 mm and 35 mm.
constexpr float DISTANCE_KP_SPEED_PER_MM = 8.0f;

// Limit the TOTAL correction from distance + parallel control.
constexpr float WALL_MAX_SPEED_CORRECTION = 220.0f;

// Front-wall turn timing.
constexpr float FRONT_TURN_TRIGGER_MM = 50.0f;
constexpr float FRONT_APPROACH_START_MM = 60.0f;
constexpr float FRONT_APPROACH_MAX_MM = 180.0f;

// FRONT-WALL DISCRIMINATION
// -------------------------
// A short front range is only allowed to influence turn timing when the robot
// is very close to the heading that the planner says this straight should have.
// This prevents the front VL6180X from clipping a side wall while the mouse is
// yawed and treating that side wall as the wall at the end of the corridor.
constexpr float FRONT_WALL_HEADING_TOLERANCE_DEG = 5.0f;

// When a possible front feature is seen while the heading is outside the
// acceptance band, temporarily give the IMU heading controller priority until
// the mouse is nearly square again, then re-test the range.
constexpr float FRONT_WALL_HEADING_RECOVER_DEG = 2.5f;

// A 60..180 mm approach warning still needs more than one distinct LiDAR
// update before it can extend a straight at an expected end wall.
constexpr uint8_t FRONT_APPROACH_CONFIRM_SAMPLES = 2;

// STATIONARY FRONT-WALL VERIFICATION
// ----------------------------------
// A <=50 mm sample is treated only as a reason to STOP AND CHECK. It is never
// allowed to trigger the next route turn directly.
constexpr float FRONT_VERIFY_ALIGN_TOLERANCE_DEG = 1.0f;
constexpr float FRONT_VERIFY_SAMPLE_HEADING_TOLERANCE_DEG = 1.5f;

// Recovery should only ever correct a modest straight-line heading error. If
// the robot is this far from the planned heading, something more serious than
// a side-wall clip has occurred and the route is stopped.
constexpr float FRONT_VERIFY_MAX_INITIAL_HEADING_ERROR_DEG = 20.0f;
constexpr float FRONT_VERIFY_MAX_ROTATION_TRAVEL_DEG = 30.0f;

constexpr uint32_t FRONT_VERIFY_ALIGN_SETTLE_MS = 180;
constexpr uint32_t FRONT_VERIFY_ALIGN_TIMEOUT_MS = 6000;
constexpr uint32_t FRONT_VERIFY_POST_ALIGN_SETTLE_MS = 150;

// After realignment, every accepted sample must have a NEW frontUpdatedMs.
// Five consecutive <=50 mm samples are required. Up to eight fresh samples are
// observed, so one or two noisy/non-close readings can clear a false detection.
constexpr uint8_t FRONT_VERIFY_REQUIRED_CLOSE_SAMPLES = 5;
constexpr uint8_t FRONT_VERIFY_MAX_SAMPLES = 8;
constexpr uint32_t FRONT_VERIFY_SAMPLE_TIMEOUT_MS = 400;

constexpr uint8_t FRONT_VERIFY_RESULT_CLEARED = 0;
constexpr uint8_t FRONT_VERIFY_RESULT_CONFIRMED = 1;
constexpr uint8_t FRONT_VERIFY_RESULT_FAILED = 2;

// A planned end wall is only allowed to terminate the straight slightly before
// the wheel-distance endpoint. A confirmed close obstacle earlier than this is
// treated as a route/safety mismatch instead of causing an unplanned early turn.
constexpr float FRONT_EARLY_ACCEPT_WINDOW_MM = 80.0f;

// Continuous ranging keeps sensor reads much shorter than repeatedly starting
// a new single-shot measurement while the steppers are running.
constexpr uint16_t LIDAR_CONTINUOUS_PERIOD_MS = 50;

// The dedicated LiDAR task reads one sensor per slot. Each of the three
// sensors is therefore refreshed roughly every 60 ms.
constexpr uint32_t LIDAR_SENSOR_SLOT_MS = 20;

// The motor core copies the shared range cache at this interval. This is only
// a memory copy; NO I2C operation occurs on the motor core.
constexpr uint32_t RANGE_CACHE_COPY_PERIOD_MS = 2;

// If a measurement is older than this, do not use it for control.
constexpr uint32_t LIDAR_READING_MAX_AGE_MS = 180;

// FreeRTOS task used only for LiDAR/I2C. The task is pinned at runtime to the
// core opposite the Arduino setup()/loop() task, guaranteeing that range reads
// and step generation do not share a CPU core on a dual-core ESP32.
constexpr uint32_t LIDAR_TASK_STACK_SIZE = 4096;
constexpr UBaseType_t LIDAR_TASK_PRIORITY = 2;
TaskHandle_t lidarTaskHandle = nullptr;
BaseType_t lidarTaskCore = 0;

// Fail-safe only: maximum extra forward travel once the robot has deliberately
// postponed a turn because the front wall is between 60 mm and 180 mm.
constexpr float FRONT_APPROACH_MAX_EXTRA_MM = 180.0f;

struct RangeSnapshot {
  bool frontValid = false;
  uint16_t frontMm = 0;
  uint32_t frontUpdatedMs = 0;

  bool left1Valid = false;
  uint16_t left1Mm = 0;
  uint32_t left1UpdatedMs = 0;

  bool left2Valid = false;
  uint16_t left2RawMm = 0;
  float left2CorrectedMm = 0.0f;
  uint32_t left2UpdatedMs = 0;

  bool imuValid = false;
  float imuRawAngleZDeg = 0.0f;
  uint32_t imuUpdatedMs = 0;
};

// Shared only between the LiDAR task and the motor/route core. Protect the
// complete structure so the motor core never sees half-updated readings.
RangeSnapshot ranges;
portMUX_TYPE rangeMux = portMUX_INITIALIZER_UNLOCKED;

// ============================================================
// MOTOR PINS
// ============================================================

// 28BYJ-48 + ULN2003
//
// AccelStepper HALF4WIRE constructor ordering:
// IN1, IN3, IN2, IN4

constexpr int LEFT_IN1 = 16;
constexpr int LEFT_IN2 = 17;
constexpr int LEFT_IN3 = 18;
constexpr int LEFT_IN4 = 19;

constexpr int RIGHT_IN1 = 32;
constexpr int RIGHT_IN2 = 33;
constexpr int RIGHT_IN3 = 13;
constexpr int RIGHT_IN4 = 14;

AccelStepper leftMotor(
  AccelStepper::HALF4WIRE,
  LEFT_IN1,
  LEFT_IN3,
  LEFT_IN2,
  LEFT_IN4
);

AccelStepper rightMotor(
  AccelStepper::HALF4WIRE,
  RIGHT_IN1,
  RIGHT_IN3,
  RIGHT_IN2,
  RIGHT_IN4
);

// ============================================================
// PHYSICAL ROBOT CONSTANTS
// ============================================================

constexpr float PI_F = 3.14159265358979323846f;

// The Python planner converts image geometry to millimetres before upload,
// so the ESP32 no longer assumes movement occurs in 180 mm cell increments.

// User supplied geometry.
constexpr float WHEEL_DIAMETER_MM = 70.0f;
constexpr float WHEEL_TRACK_MM = 100.0f;

// 28BYJ-48 in half-step mode.
// This is the theoretical value. Real gearboxes can differ slightly.
constexpr float HALF_STEPS_PER_WHEEL_REV = 4096.0f;

// Straight wheel travel conversion.
constexpr float STEPS_PER_MM =
  HALF_STEPS_PER_WHEEL_REV /
  (PI_F * WHEEL_DIAMETER_MM);

// In-place turn:
//
// each wheel travels:
//   PI * wheelTrack * angle / 360
//
// divide by wheel circumference to obtain wheel revolutions.
constexpr float TURN_STEPS_PER_DEGREE =
  HALF_STEPS_PER_WHEEL_REV *
  WHEEL_TRACK_MM /
  (360.0f * WHEEL_DIAMETER_MM);

// ============================================================
// SPEEDS
// ============================================================

// Straight driving speed in half-steps/second.
constexpr float DRIVE_SPEED = 900.0f;

// Turning speed.
constexpr float TURN_FAST_SPEED = 300.0f;
constexpr float TURN_MEDIUM_SPEED = 180.0f;
constexpr float TURN_SLOW_SPEED = 100.0f;

constexpr long TURN_MEDIUM_REMAINING_STEPS = 300;
constexpr long TURN_SLOW_REMAINING_STEPS = 100;

// Ignore tiny planner angle noise.
constexpr float MIN_TURN_DEGREES = 2.0f;

// Planner convention: positive angle = LEFT, negative angle = RIGHT.
// This robot requires the opposite raw motor sign from the previous build,
// so -1 maps planner LEFT to physical LEFT and planner RIGHT to physical RIGHT.
constexpr float TURN_DIRECTION_SIGN = -1.0f;

// Very occasional scheduler break while a blocking motor loop runs. The old
// code yielded for 1 ms every 8 ms, which noticeably reduced runSpeed()'s
// achievable average step rate. LiDAR now lives on the other core, so the
// motor core only gives FreeRTOS this small watchdog/idle opportunity.
constexpr uint32_t SCHEDULER_BREAK_PERIOD_MS = 250;

// ============================================================
// ROUTE STORAGE
// ============================================================

// Increased route storage for full-maze paths.
constexpr int MAX_ROUTE_SEGMENTS = 200;

struct RouteSegment {
  float turnDegrees;

  // Accepted from Python only for compatibility.
  // Execution ignores this value.
  char uploadedMode;

  float distanceMm;
};

RouteSegment route[MAX_ROUTE_SEGMENTS];
int routeLength = 0;

enum RobotState {
  WAITING_FOR_ROUTE,
  WAITING_FOR_START,
  RUNNING_ROUTE
};

RobotState robotState = WAITING_FOR_ROUTE;

// ============================================================
// SMALL HELPERS
// ============================================================

String resetReasonText()
{
  switch (esp_reset_reason()) {
    case ESP_RST_POWERON:
      return "POWER_ON";

    case ESP_RST_EXT:
      return "EXTERNAL_RESET";

    case ESP_RST_SW:
      return "SOFTWARE_RESET";

    case ESP_RST_PANIC:
      return "PANIC_EXCEPTION";

    case ESP_RST_INT_WDT:
      return "INTERRUPT_WATCHDOG";

    case ESP_RST_TASK_WDT:
      return "TASK_WATCHDOG";

    case ESP_RST_WDT:
      return "WATCHDOG";

    case ESP_RST_BROWNOUT:
      return "BROWNOUT";

    default:
      return "OTHER";
  }
}

const char *stateText()
{
  switch (robotState) {
    case WAITING_FOR_ROUTE:
      return "WAITING - upload complete route";

    case WAITING_FOR_START:
      return "READY - complete route loaded";

    case RUNNING_ROUTE:
      return "RUNNING";

    default:
      return "UNKNOWN";
  }
}

void stopMotorsHold()
{
  leftMotor.setSpeed(0.0f);
  rightMotor.setSpeed(0.0f);

  // Do not disable outputs.
  // Keeping outputs enabled maintains holding torque.
}

void schedulerBreakIfNeeded(uint32_t &lastBreakMs)
{
  uint32_t now = millis();

  if (now - lastBreakMs >= SCHEDULER_BREAK_PERIOD_MS) {
    /*
      Keep the motor core responsive to the RTOS watchdog/idle task without
      repeatedly stealing time from AccelStepper. All LiDAR/I2C work is on the
      opposite core, so this yield can be very infrequent.
    */
    delay(1);
    lastBreakMs = millis();
  }
}

// ============================================================
// MPU6050 INITIALISATION + START-TIME CALIBRATION
// ============================================================

bool initialiseMPU6050Hardware()
{
  Serial.println();
  Serial.println("Initialising MPU-6050...");

  byte status = mpu.begin();

  Serial.print("MPU-6050 status: ");
  Serial.println(status);

  if (status != 0) {
    Serial.println("ERROR: MPU-6050 failed.");
    return false;
  }

  Serial.println(
    "MPU-6050 connected. Calibration is deferred until website START."
  );

  return true;
}

bool calibrateMPU6050ForRun()
{
  if (!imuWorking) {
    Serial.println(
      "ERROR: cannot calibrate MPU-6050 because it did not initialise."
    );
    return false;
  }

  Serial.println();
  Serial.println("===========================================");
  Serial.println("MPU-6050 START-TIME CALIBRATION");
  Serial.println("KEEP THE MICROMOUSE COMPLETELY STILL");
  Serial.println("===========================================");

  // Ask the opposite-core task to finish its current I2C transaction and
  // pause before touching Wire from this core.
  sensorTaskPauseRequested = true;

  uint32_t pauseWaitStart = millis();

  while (!sensorTaskPaused) {
    if (millis() - pauseWaitStart > 1000) {
      sensorTaskPauseRequested = false;
      Serial.println(
        "ERROR: sensor task did not pause for IMU calibration."
      );
      return false;
    }

    delay(1);
  }

  // Same calibration sequence used by the supplied MicromouseMK2 example.
  delay(1000);
  mpu.calcOffsets();

  // Refresh the Z angle once with the new gyro offsets. Any accumulated
  // pre-calibration angle is harmless because this value becomes our zero.
  mpu.update();

  imuRunZeroRawDeg = mpu.getAngleZ();
  plannedHeadingDeg = 0.0f;

  uint32_t now = millis();

  portENTER_CRITICAL(&rangeMux);
  ranges.imuRawAngleZDeg = imuRunZeroRawDeg;
  ranges.imuUpdatedMs = now;
  ranges.imuValid = true;
  portEXIT_CRITICAL(&rangeMux);

  sensorTaskPauseRequested = false;

  // Wait for the sensor task to acknowledge that it has resumed.
  uint32_t resumeWaitStart = millis();

  while (sensorTaskPaused) {
    if (millis() - resumeWaitStart > 1000) {
      Serial.println(
        "WARNING: sensor task resume acknowledgement timed out."
      );
      break;
    }

    delay(1);
  }

  Serial.print("MPU calibration complete. Raw Z zero = ");
  Serial.print(imuRunZeroRawDeg, 2);
  Serial.println(" deg");
  Serial.println("Planned route heading = 0.0 deg");

  return true;
}

bool getPlannerHeadingDeg(float &headingDeg)
{
  bool valid;
  float rawAngleZDeg;
  uint32_t updatedMs;

  portENTER_CRITICAL(&rangeMux);
  valid = ranges.imuValid;
  rawAngleZDeg = ranges.imuRawAngleZDeg;
  updatedMs = ranges.imuUpdatedMs;
  portEXIT_CRITICAL(&rangeMux);

  if (
    !valid ||
    millis() - updatedMs > IMU_READING_MAX_AGE_MS
  ) {
    return false;
  }

  headingDeg =
    IMU_YAW_SIGN *
    (rawAngleZDeg - imuRunZeroRawDeg);

  return true;
}

// ============================================================
// VL6180X INITIALISATION + READING
// ============================================================

bool i2cDevicePresent(uint8_t address)
{
  Wire.beginTransmission(address);
  return Wire.endTransmission() == 0;
}

void disableAllRangeSensors()
{
  pinMode(FRONT_CE_PIN, OUTPUT);
  pinMode(LEFT1_CE_PIN, OUTPUT);
  pinMode(LEFT2_SHUT_PIN, OUTPUT);
  pinMode(RIGHT_CE_PIN, OUTPUT);

  digitalWrite(FRONT_CE_PIN, LOW);
  digitalWrite(LEFT1_CE_PIN, LOW);
  digitalWrite(LEFT2_SHUT_PIN, LOW);
  digitalWrite(RIGHT_CE_PIN, LOW);

  delay(150);
}

bool initialiseOneRangeSensor(
  VL6180X &sensor,
  uint8_t shutdownPin,
  uint8_t newAddress,
  const char *sensorName
)
{
  Serial.print("Starting ");
  Serial.print(sensorName);
  Serial.println("...");

  digitalWrite(shutdownPin, HIGH);
  delay(150);

  if (!i2cDevicePresent(0x29)) {
    Serial.print("ERROR: ");
    Serial.print(sensorName);
    Serial.println(" did not respond at default address 0x29.");
    return false;
  }

  sensor.init();
  sensor.configureDefault();

  // Short timeout: a failed range reading must not stall the LiDAR task for long.
  sensor.setTimeout(20);

  sensor.setAddress(newAddress);
  delay(50);

  if (!i2cDevicePresent(newAddress)) {
    Serial.print("ERROR: ");
    Serial.print(sensorName);
    Serial.println(" did not respond at its assigned address.");
    return false;
  }

  sensor.startRangeContinuous(LIDAR_CONTINUOUS_PERIOD_MS);

  Serial.print(sensorName);
  Serial.print(" ready at 0x");
  Serial.println(newAddress, HEX);

  return true;
}

void initialiseFeedbackRangeSensors()
{
  Serial.println();
  Serial.println("Initialising route-feedback VL6180X sensors...");

  disableAllRangeSensors();

  frontWorking = initialiseOneRangeSensor(
    frontSensor,
    FRONT_CE_PIN,
    FRONT_ADDRESS,
    "Front"
  );

  left1Working = initialiseOneRangeSensor(
    left1Sensor,
    LEFT1_CE_PIN,
    LEFT1_ADDRESS,
    "Left 1 (front-left)"
  );

  left2Working = initialiseOneRangeSensor(
    left2Sensor,
    LEFT2_SHUT_PIN,
    LEFT2_ADDRESS,
    "Left 2 (rear-left)"
  );

  // RIGHT_CE_PIN deliberately remains LOW.

  Serial.print("Front LiDAR: ");
  Serial.println(frontWorking ? "READY" : "FAILED");

  Serial.print("Left 1 LiDAR: ");
  Serial.println(left1Working ? "READY" : "FAILED");

  Serial.print("Left 2 LiDAR: ");
  Serial.println(left2Working ? "READY" : "FAILED");
}

bool readContinuousRange(
  VL6180X &sensor,
  bool sensorWorking,
  uint16_t &distanceMm
)
{
  if (!sensorWorking) {
    return false;
  }

  distanceMm = sensor.readRangeContinuousMillimeters();

  if (sensor.timeoutOccurred()) {
    return false;
  }

  return true;
}

bool readingFresh(
  bool valid,
  uint32_t updatedMs
)
{
  return (
    valid &&
    (millis() - updatedMs <= LIDAR_READING_MAX_AGE_MS)
  );
}

void updateFrontRange()
{
  uint16_t value = 0;
  bool valid =
    readContinuousRange(
      frontSensor,
      frontWorking,
      value
    );

  uint32_t now = millis();

  portENTER_CRITICAL(&rangeMux);

  if (valid) {
    ranges.frontMm = value;
    ranges.frontUpdatedMs = now;
    ranges.frontValid = true;
  }
  else {
    ranges.frontValid = false;
  }

  portEXIT_CRITICAL(&rangeMux);
}

void updateLeft1Range()
{
  uint16_t value = 0;
  bool valid =
    readContinuousRange(
      left1Sensor,
      left1Working,
      value
    );

  uint32_t now = millis();

  portENTER_CRITICAL(&rangeMux);

  if (valid) {
    ranges.left1Mm = value;
    ranges.left1UpdatedMs = now;
    ranges.left1Valid = true;
  }
  else {
    ranges.left1Valid = false;
  }

  portEXIT_CRITICAL(&rangeMux);
}

void updateLeft2Range()
{
  uint16_t value = 0;
  bool valid =
    readContinuousRange(
      left2Sensor,
      left2Working,
      value
    );

  uint32_t now = millis();

  portENTER_CRITICAL(&rangeMux);

  if (valid) {
    ranges.left2RawMm = value;
    ranges.left2CorrectedMm =
      (float)value + LEFT2_OFFSET_MM;

    ranges.left2UpdatedMs = now;
    ranges.left2Valid = true;
  }
  else {
    ranges.left2Valid = false;
  }

  portEXIT_CRITICAL(&rangeMux);
}

void lidarTask(void *parameter)
{
  (void)parameter;

  uint8_t sensorSlot = 0;

  for (;;) {
    // Calibration is performed on the motion core only after this task has
    // reached this safe point between I2C transactions.
    if (sensorTaskPauseRequested) {
      sensorTaskPaused = true;

      while (sensorTaskPauseRequested) {
        vTaskDelay(pdMS_TO_TICKS(1));
      }

      sensorTaskPaused = false;
    }

    // Keep the MPU6050 angle integration updated on the SAME core/task that
    // owns the VL6180X I2C reads. This prevents concurrent Wire access.
    if (imuWorking) {
      mpu.update();

      uint32_t now = millis();
      float rawAngleZDeg = mpu.getAngleZ();

      portENTER_CRITICAL(&rangeMux);
      ranges.imuRawAngleZDeg = rawAngleZDeg;
      ranges.imuUpdatedMs = now;
      ranges.imuValid = true;
      portEXIT_CRITICAL(&rangeMux);
    }

    switch (sensorSlot) {
      case 0:
        updateFrontRange();
        break;

      case 1:
        updateLeft1Range();
        break;

      default:
        updateLeft2Range();
        break;
    }

    sensorSlot =
      (sensorSlot + 1) % 3;

    vTaskDelay(
      pdMS_TO_TICKS(LIDAR_SENSOR_SLOT_MS)
    );
  }
}

bool startLidarTaskOnOppositeCore()
{
  if (lidarTaskHandle != nullptr) {
    return true;
  }

  BaseType_t motionCore =
    xPortGetCoreID();

  // setup() and loop() run on the same Arduino task/core. Pin the LiDAR task
  // to the other CPU so I2C cannot hold up AccelStepper::runSpeed().
  lidarTaskCore =
    motionCore == 0
      ? 1
      : 0;

  BaseType_t result =
    xTaskCreatePinnedToCore(
      lidarTask,
      "LidarTask",
      LIDAR_TASK_STACK_SIZE,
      nullptr,
      LIDAR_TASK_PRIORITY,
      &lidarTaskHandle,
      lidarTaskCore
    );

  if (result != pdPASS) {
    lidarTaskHandle = nullptr;
    return false;
  }

  Serial.print("Motion/stepper core: ");
  Serial.println(motionCore);

  Serial.print("LiDAR/I2C core: ");
  Serial.println(lidarTaskCore);

  return true;
}

float clampFloat(
  float value,
  float low,
  float high
)
{
  if (value < low) {
    return low;
  }

  if (value > high) {
    return high;
  }

  return value;
}

// Return target-current as the equivalent shortest physical heading error.
// This also protects against cumulative headings that differ by whole 360 deg
// revolutions: +360 deg and 0 deg are the same physical orientation.
float shortestHeadingErrorDeg(
  float targetHeadingDeg,
  float currentHeadingDeg
)
{
  float errorDeg =
    targetHeadingDeg - currentHeadingDeg;

  while (errorDeg > 180.0f) {
    errorDeg -= 360.0f;
  }

  while (errorDeg < -180.0f) {
    errorDeg += 360.0f;
  }

  return errorDeg;
}

// ============================================================
// STRICT ROUTE PARSING
// ============================================================

bool parseFloatStrict(
  const String &input,
  float &value
)
{
  String s = input;
  s.trim();

  if (s.length() == 0) {
    return false;
  }

  char buffer[32];

  if (s.length() >= sizeof(buffer)) {
    return false;
  }

  s.toCharArray(
    buffer,
    sizeof(buffer)
  );

  char *endPtr = nullptr;

  value = strtof(
    buffer,
    &endPtr
  );

  if (endPtr == buffer) {
    return false;
  }

  while (*endPtr == ' ' || *endPtr == '\t') {
    endPtr++;
  }

  return *endPtr == '\0' && isfinite(value);
}

bool parsePositiveUIntStrict(
  const String &input,
  uint16_t &value
)
{
  String s = input;
  s.trim();

  if (s.length() == 0) {
    return false;
  }

  char buffer[24];

  if (s.length() >= sizeof(buffer)) {
    return false;
  }

  s.toCharArray(
    buffer,
    sizeof(buffer)
  );

  char *endPtr = nullptr;

  long parsed = strtol(
    buffer,
    &endPtr,
    10
  );

  if (endPtr == buffer) {
    return false;
  }

  while (*endPtr == ' ' || *endPtr == '\t') {
    endPtr++;
  }

  if (*endPtr != '\0') {
    return false;
  }

  if (parsed <= 0 || parsed > 100) {
    return false;
  }

  value = (uint16_t)parsed;
  return true;
}

/*
  Transactional route parser.

  It NEVER modifies the active route until the entire uploaded payload
  has been successfully validated.

  Expected format:

    ROUTE:0.0,D,243.0;81.2,D,392.4;-42.7,W,176.4

  W and D are both accepted for compatibility with the Python planner.
  LiDAR safety/feedback rules are applied to every straight segment.
*/
bool parseCompleteRouteTransactionally(
  const String &payload,
  String &errorMessage
)
{
  String input = payload;
  input.trim();

  if (!input.startsWith("ROUTE:")) {
    errorMessage = "Payload must begin with ROUTE:";
    return false;
  }

  String body = input.substring(6);
  body.trim();

  if (body.length() == 0) {
    errorMessage = "Route contains no segments.";
    return false;
  }

  // Temporary route. Active route remains untouched unless all segments pass.
  // Static storage keeps the larger 200-segment validation buffer off the
  // function stack while preserving the existing transactional validation.
  static RouteSegment candidate[MAX_ROUTE_SEGMENTS];
  int candidateCount = 0;

  int startIndex = 0;

  while (startIndex < body.length()) {
    if (candidateCount >= MAX_ROUTE_SEGMENTS) {
      errorMessage =
        "Too many route segments. Maximum is " +
        String(MAX_ROUTE_SEGMENTS) +
        ".";
      return false;
    }

    int semicolon =
      body.indexOf(
        ';',
        startIndex
      );

    String segmentText;

    if (semicolon < 0) {
      segmentText =
        body.substring(startIndex);

      startIndex =
        body.length();
    }
    else {
      segmentText =
        body.substring(
          startIndex,
          semicolon
        );

      startIndex =
        semicolon + 1;
    }

    segmentText.trim();

    if (segmentText.length() == 0) {
      errorMessage = "Empty route segment detected.";
      return false;
    }

    int comma1 =
      segmentText.indexOf(',');

    int comma2 =
      comma1 >= 0
        ? segmentText.indexOf(',', comma1 + 1)
        : -1;

    if (comma1 < 0 || comma2 < 0) {
      errorMessage =
        "Segment missing fields: " +
        segmentText;

      return false;
    }

    // Do not permit an unexpected fourth field.
    if (segmentText.indexOf(',', comma2 + 1) >= 0) {
      errorMessage =
        "Segment contains too many fields: " +
        segmentText;

      return false;
    }

    String turnText =
      segmentText.substring(
        0,
        comma1
      );

    String modeText =
      segmentText.substring(
        comma1 + 1,
        comma2
      );

    String distanceText =
      segmentText.substring(
        comma2 + 1
      );

    turnText.trim();
    modeText.trim();
    distanceText.trim();

    float turnDegrees = 0.0f;

    if (!parseFloatStrict(turnText, turnDegrees)) {
      errorMessage =
        "Invalid turn angle in segment: " +
        segmentText;

      return false;
    }

    if (fabsf(turnDegrees) > MAX_ROUTE_TURN_DEGREES) {
      errorMessage =
        "Turn angle exceeds safe +/-" +
        String(MAX_ROUTE_TURN_DEGREES, 0) +
        " degree route limit. Check START facing/path; U-turn rejected.";

      return false;
    }

    if (modeText.length() != 1) {
      errorMessage =
        "Mode must be D or W.";

      return false;
    }

    char mode =
      toupper(modeText.charAt(0));

    if (mode != 'D' && mode != 'W') {
      errorMessage =
        "Mode must be D or W.";

      return false;
    }

    float distanceMm = 0.0f;

    if (!parseFloatStrict(distanceText, distanceMm)) {
      errorMessage =
        "Invalid distance in segment: " +
        segmentText;

      return false;
    }

    if (distanceMm <= 0.0f || distanceMm > 10000.0f) {
      errorMessage =
        "Distance must be > 0 and <= 10000 mm.";

      return false;
    }

    candidate[candidateCount].turnDegrees =
      turnDegrees;

    candidate[candidateCount].uploadedMode =
      mode;

    candidate[candidateCount].distanceMm =
      distanceMm;

    candidateCount++;
  }

  if (candidateCount <= 0) {
    errorMessage = "No valid route segments found.";
    return false;
  }

  // ----------------------------------------------------------
  // COMMIT POINT
  // ----------------------------------------------------------
  // Only now, after complete validation, replace the active route.
  for (int i = 0; i < candidateCount; i++) {
    route[i] = candidate[i];
  }

  routeLength = candidateCount;
  robotState = WAITING_FOR_START;

  return true;
}

// ============================================================
// BLOCKING ROUTE MOTION WITH LIDAR STRAIGHT-LINE FEEDBACK
// ============================================================

long averageMotorTravel(
  long startLeft,
  long startRight
)
{
  long leftTravel =
    labs(
      leftMotor.currentPosition() -
      startLeft
    );

  long rightTravel =
    labs(
      rightMotor.currentPosition() -
      startRight
    );

  return (leftTravel + rightTravel) / 2;
}

void executeTurnByStepsFallback(float turnDegrees)
{
  if (fabsf(turnDegrees) < MIN_TURN_DEGREES) {
    return;
  }

  long targetSteps =
    max(
      1L,
      (long)(
        TURN_STEPS_PER_DEGREE *
        fabsf(turnDegrees) +
        0.5f
      )
    );

  float direction =
    turnDegrees >= 0.0f
      ? 1.0f
      : -1.0f;

  direction *= TURN_DIRECTION_SIGN;

  long startLeft =
    leftMotor.currentPosition();

  long startRight =
    rightMotor.currentPosition();

  uint32_t lastBreakMs =
    millis();

  while (true) {
    long travelled =
      averageMotorTravel(
        startLeft,
        startRight
      );

    long remaining =
      targetSteps -
      travelled;

    if (remaining <= 0) {
      break;
    }

    float speed =
      TURN_FAST_SPEED;

    if (remaining <= TURN_SLOW_REMAINING_STEPS) {
      speed =
        TURN_SLOW_SPEED;
    }
    else if (remaining <= TURN_MEDIUM_REMAINING_STEPS) {
      speed =
        TURN_MEDIUM_SPEED;
    }

    /*
      Equal motor command signs were established on this robot as
      in-place rotation.
    */
    leftMotor.setSpeed(
      direction * speed
    );

    rightMotor.setSpeed(
      direction * speed
    );

    leftMotor.runSpeed();
    rightMotor.runSpeed();

    schedulerBreakIfNeeded(lastBreakMs);
  }

  stopMotorsHold();

  delay(80);
}

bool executeTurnToPlannedHeading(
  float targetHeadingDeg,
  float expectedTurnDeg
)
{
  if (!imuWorking) {
    return false;
  }

  uint32_t turnStartMs = millis();
  uint32_t settleStartMs = 0;
  uint32_t lastBreakMs = millis();

  bool initialErrorCaptured = false;
  float initialAbsErrorDeg = 0.0f;

  bool startHeadingCaptured = false;
  float turnStartHeadingDeg = 0.0f;

  // A 90 degree instruction, for example, can physically travel at most
  // 115 degrees before this fail-safe stops it.  Therefore an unexpected
  // 180 degree in-place rotation cannot complete silently.
  float maximumPhysicalTurnDeg =
    fabsf(expectedTurnDeg) +
    IMU_TURN_EXTRA_TRAVEL_GUARD_DEG;

  while (true) {
    float currentHeadingDeg = 0.0f;

    if (!getPlannerHeadingDeg(currentHeadingDeg)) {
      stopMotorsHold();

      if (millis() - turnStartMs > IMU_TURN_TIMEOUT_MS) {
        Serial.println(
          "ERROR: IMU heading became unavailable during turn."
        );
        return false;
      }

      schedulerBreakIfNeeded(lastBreakMs);
      continue;
    }

    if (!startHeadingCaptured) {
      turnStartHeadingDeg = currentHeadingDeg;
      startHeadingCaptured = true;

      Serial.print("Turn start IMU heading: ");
      Serial.print(turnStartHeadingDeg, 2);
      Serial.println(" deg");
    }

    // MPU6050_light supplies a continuous Z angle, so the difference from the
    // turn-start reading is a reliable physical rotation travelled during this
    // one in-place turn.
    float physicalTurnTravelDeg =
      fabsf(currentHeadingDeg - turnStartHeadingDeg);

    if (
      physicalTurnTravelDeg >
      maximumPhysicalTurnDeg
    ) {
      stopMotorsHold();

      Serial.print(
        "ERROR: in-place turn exceeded safe travel. Requested="
      );
      Serial.print(expectedTurnDeg, 1);
      Serial.print(" deg, travelled=");
      Serial.print(physicalTurnTravelDeg, 1);
      Serial.println(" deg");
      Serial.println(
        "Route aborted before an unintended large/180-degree turn could continue."
      );

      return false;
    }

    // Always use the shortest physically equivalent heading error.  This keeps
    // cumulative absolute headings from ever requesting an unnecessary whole
    // revolution merely because they differ by +/-360 degrees.
    float errorDeg =
      shortestHeadingErrorDeg(
        targetHeadingDeg,
        currentHeadingDeg
      );

    float absErrorDeg = fabsf(errorDeg);

    if (!initialErrorCaptured) {
      initialAbsErrorDeg = absErrorDeg;
      initialErrorCaptured = true;

      Serial.print("Initial physical turn error: ");
      Serial.print(errorDeg, 2);
      Serial.println(" deg");
    }

    // If the configured MPU Z sign is backwards, a commanded turn makes the
    // measured heading error GROW instead of shrink. Stop early rather than
    // allowing the mouse to keep rotating until the general timeout.
    if (
      millis() - turnStartMs > 500 &&
      absErrorDeg > initialAbsErrorDeg + 5.0f
    ) {
      stopMotorsHold();
      Serial.println(
        "ERROR: IMU turn is moving away from its target."
      );
      Serial.println(
        "Check IMU_YAW_SIGN near the top of the sketch; change +1 to -1 if needed."
      );
      return false;
    }

    if (absErrorDeg <= IMU_TURN_TOLERANCE_DEG) {
      stopMotorsHold();

      if (settleStartMs == 0) {
        settleStartMs = millis();
      }

      if (
        millis() - settleStartMs >=
        IMU_TURN_SETTLE_MS
      ) {
        delay(80);
        return true;
      }

      schedulerBreakIfNeeded(lastBreakMs);
      continue;
    }

    settleStartMs = 0;

    if (millis() - turnStartMs > IMU_TURN_TIMEOUT_MS) {
      stopMotorsHold();
      Serial.print("ERROR: IMU turn timed out. Remaining error: ");
      Serial.print(errorDeg, 2);
      Serial.println(" deg");
      return false;
    }

    float speed = TURN_FAST_SPEED;

    if (absErrorDeg <= IMU_TURN_SLOW_ERROR_DEG) {
      speed = TURN_SLOW_SPEED;
    }
    else if (absErrorDeg <= IMU_TURN_MEDIUM_ERROR_DEG) {
      speed = TURN_MEDIUM_SPEED;
    }

    // error > 0 means the robot must rotate LEFT in planner coordinates.
    // TURN_DIRECTION_SIGN converts that planner direction to this robot's
    // established raw motor sign.
    float direction =
      errorDeg >= 0.0f
        ? 1.0f
        : -1.0f;

    direction *= TURN_DIRECTION_SIGN;

    // Same raw sign on both motor objects is an in-place rotation on this
    // chassis because straight-forward is left positive / right negative.
    leftMotor.setSpeed(direction * speed);
    rightMotor.setSpeed(direction * speed);

    leftMotor.runSpeed();
    rightMotor.runSpeed();

    schedulerBreakIfNeeded(lastBreakMs);
  }
}

// ============================================================
// STATIONARY FRONT-WALL VERIFICATION
// ============================================================

bool realignToPlannedHeadingForFrontCheck(
  float targetHeadingDeg
)
{
  stopMotorsHold();

  float startHeadingDeg = 0.0f;

  if (!getPlannerHeadingDeg(startHeadingDeg)) {
    Serial.println(
      "FRONT VERIFY FAILED: no fresh IMU heading before realignment."
    );
    return false;
  }

  float initialErrorDeg =
    shortestHeadingErrorDeg(
      targetHeadingDeg,
      startHeadingDeg
    );

  float initialAbsErrorDeg =
    fabsf(initialErrorDeg);

  Serial.print("Front verify: planned heading = ");
  Serial.print(targetHeadingDeg, 2);
  Serial.print(" deg, current = ");
  Serial.print(startHeadingDeg, 2);
  Serial.print(" deg, error = ");
  Serial.print(initialErrorDeg, 2);
  Serial.println(" deg");

  if (
    initialAbsErrorDeg >
      FRONT_VERIFY_MAX_INITIAL_HEADING_ERROR_DEG
  ) {
    Serial.println(
      "FRONT VERIFY FAILED: heading error is too large for a safe local realignment."
    );
    return false;
  }

  uint32_t realignStartMs = millis();
  uint32_t settleStartMs = 0;
  uint32_t lastBreakMs = millis();

  while (true) {
    float currentHeadingDeg = 0.0f;

    if (!getPlannerHeadingDeg(currentHeadingDeg)) {
      stopMotorsHold();

      if (
        millis() - realignStartMs >
          FRONT_VERIFY_ALIGN_TIMEOUT_MS
      ) {
        Serial.println(
          "FRONT VERIFY FAILED: IMU became unavailable during realignment."
        );
        return false;
      }

      schedulerBreakIfNeeded(lastBreakMs);
      continue;
    }

    float physicalRotationDeg =
      fabsf(currentHeadingDeg - startHeadingDeg);

    if (
      physicalRotationDeg >
        FRONT_VERIFY_MAX_ROTATION_TRAVEL_DEG
    ) {
      stopMotorsHold();
      Serial.println(
        "FRONT VERIFY FAILED: realignment exceeded the allowed physical rotation."
      );
      return false;
    }

    float errorDeg =
      shortestHeadingErrorDeg(
        targetHeadingDeg,
        currentHeadingDeg
      );

    float absErrorDeg =
      fabsf(errorDeg);

    // Catch a reversed IMU sign / motor direction before a small correction
    // can grow into a large turn.
    if (
      initialAbsErrorDeg > FRONT_VERIFY_ALIGN_TOLERANCE_DEG &&
      millis() - realignStartMs > 500 &&
      absErrorDeg > initialAbsErrorDeg + 3.0f
    ) {
      stopMotorsHold();
      Serial.println(
        "FRONT VERIFY FAILED: realignment is moving away from the planned heading."
      );
      return false;
    }

    if (
      absErrorDeg <=
        FRONT_VERIFY_ALIGN_TOLERANCE_DEG
    ) {
      stopMotorsHold();

      if (settleStartMs == 0) {
        settleStartMs = millis();
      }

      if (
        millis() - settleStartMs >=
          FRONT_VERIFY_ALIGN_SETTLE_MS
      ) {
        Serial.print(
          "Front verify: heading aligned within +/-"
        );
        Serial.print(
          FRONT_VERIFY_ALIGN_TOLERANCE_DEG,
          1
        );
        Serial.println(" deg.");
        return true;
      }

      schedulerBreakIfNeeded(lastBreakMs);
      continue;
    }

    settleStartMs = 0;

    if (
      millis() - realignStartMs >
        FRONT_VERIFY_ALIGN_TIMEOUT_MS
    ) {
      stopMotorsHold();
      Serial.print(
        "FRONT VERIFY FAILED: realignment timed out; remaining error = "
      );
      Serial.print(errorDeg, 2);
      Serial.println(" deg");
      return false;
    }

    float speed =
      absErrorDeg <= IMU_TURN_SLOW_ERROR_DEG
        ? TURN_SLOW_SPEED
        : TURN_MEDIUM_SPEED;

    float direction =
      errorDeg >= 0.0f
        ? 1.0f
        : -1.0f;

    direction *= TURN_DIRECTION_SIGN;

    // On this chassis equal raw signs produce an in-place rotation.
    leftMotor.setSpeed(direction * speed);
    rightMotor.setSpeed(direction * speed);

    leftMotor.runSpeed();
    rightMotor.runSpeed();

    schedulerBreakIfNeeded(lastBreakMs);
  }
}

bool waitForNewFrontVerificationSample(
  uint32_t &lastSeenUpdatedMs,
  uint16_t &distanceMm
)
{
  uint32_t waitStartMs = millis();

  while (
    millis() - waitStartMs <
      FRONT_VERIFY_SAMPLE_TIMEOUT_MS
  ) {
    bool valid = false;
    uint16_t value = 0;
    uint32_t updatedMs = 0;

    portENTER_CRITICAL(&rangeMux);
    valid = ranges.frontValid;
    value = ranges.frontMm;
    updatedMs = ranges.frontUpdatedMs;
    portEXIT_CRITICAL(&rangeMux);

    if (
      valid &&
      updatedMs != 0 &&
      updatedMs != lastSeenUpdatedMs &&
      millis() - updatedMs <=
        LIDAR_READING_MAX_AGE_MS
    ) {
      lastSeenUpdatedMs = updatedMs;
      distanceMm = value;
      return true;
    }

    delay(1);
  }

  return false;
}

uint8_t verifyFrontWallStationary(
  float targetHeadingDeg,
  uint32_t &lastConsumedFrontUpdatedMs
)
{
  stopMotorsHold();

  Serial.println();
  Serial.println(
    "FRONT CHECK: suspicious <=50 mm reading; forward motion paused."
  );

  if (
    !realignToPlannedHeadingForFrontCheck(
      targetHeadingDeg
    )
  ) {
    return FRONT_VERIFY_RESULT_FAILED;
  }

  // Let mechanical vibration die away before establishing the timestamp from
  // which NEW front samples are allowed to count.
  stopMotorsHold();
  delay(FRONT_VERIFY_POST_ALIGN_SETTLE_MS);

  uint32_t baselineUpdatedMs = 0;

  portENTER_CRITICAL(&rangeMux);
  baselineUpdatedMs = ranges.frontUpdatedMs;
  portEXIT_CRITICAL(&rangeMux);

  lastConsumedFrontUpdatedMs =
    baselineUpdatedMs;

  uint8_t consecutiveCloseSamples = 0;

  for (
    uint8_t sampleNumber = 1;
    sampleNumber <= FRONT_VERIFY_MAX_SAMPLES;
    sampleNumber++
  ) {
    uint16_t frontMm = 0;

    if (
      !waitForNewFrontVerificationSample(
        lastConsumedFrontUpdatedMs,
        frontMm
      )
    ) {
      Serial.println(
        "FRONT VERIFY FAILED: timed out waiting for a new front LiDAR sample."
      );
      return FRONT_VERIFY_RESULT_FAILED;
    }

    // The robot should remain stationary and aligned during sampling. If the
    // heading is no longer tight enough, do not trust the classification.
    float currentHeadingDeg = 0.0f;

    if (!getPlannerHeadingDeg(currentHeadingDeg)) {
      Serial.println(
        "FRONT VERIFY FAILED: IMU heading unavailable during stationary sampling."
      );
      return FRONT_VERIFY_RESULT_FAILED;
    }

    float sampleHeadingErrorDeg =
      fabsf(
        shortestHeadingErrorDeg(
          targetHeadingDeg,
          currentHeadingDeg
        )
      );

    if (
      sampleHeadingErrorDeg >
        FRONT_VERIFY_SAMPLE_HEADING_TOLERANCE_DEG
    ) {
      Serial.print(
        "FRONT VERIFY FAILED: heading drifted to "
      );
      Serial.print(sampleHeadingErrorDeg, 2);
      Serial.println(
        " deg during stationary sampling."
      );
      return FRONT_VERIFY_RESULT_FAILED;
    }

    Serial.print("Front verify sample ");
    Serial.print(sampleNumber);
    Serial.print("/");
    Serial.print(FRONT_VERIFY_MAX_SAMPLES);
    Serial.print(": ");
    Serial.print(frontMm);
    Serial.println(" mm");

    if (
      (float)frontMm <=
        FRONT_TURN_TRIGGER_MM
    ) {
      consecutiveCloseSamples++;
    }
    else {
      consecutiveCloseSamples = 0;
    }

    if (
      consecutiveCloseSamples >=
        FRONT_VERIFY_REQUIRED_CLOSE_SAMPLES
    ) {
      Serial.println(
        "FRONT CHECK CONFIRMED: 5 consecutive fresh <=50 mm samples while aligned."
      );
      return FRONT_VERIFY_RESULT_CONFIRMED;
    }
  }

  Serial.println(
    "FRONT CHECK CLEARED: obstacle was not confirmed after exact realignment; resuming straight."
  );

  return FRONT_VERIFY_RESULT_CLEARED;
}

enum StraightEndReason {
  STRAIGHT_END_ODOMETRY,
  STRAIGHT_END_FRONT_WALL_EARLY,
  STRAIGHT_END_FRONT_APPROACH,
  STRAIGHT_END_APPROACH_FAILSAFE,
  STRAIGHT_END_UNEXPECTED_FRONT_OBSTACLE,
  STRAIGHT_END_FRONT_VERIFICATION_FAILED
};

const char *straightEndReasonText(
  uint8_t reason
)
{
  switch (reason) {
    case STRAIGHT_END_ODOMETRY:
      return "nominal wheel distance reached";

    case STRAIGHT_END_FRONT_WALL_EARLY:
      return "confirmed heading-aligned front wall <= 50 mm near planned turn";

    case STRAIGHT_END_FRONT_APPROACH:
      return "confirmed front approach continued until confirmed <= 50 mm";

    case STRAIGHT_END_APPROACH_FAILSAFE:
      return "front approach fail-safe distance reached";

    case STRAIGHT_END_UNEXPECTED_FRONT_OBSTACLE:
      return "stationary-verified front obstacle where route does not permit an early turn";

    case STRAIGHT_END_FRONT_VERIFICATION_FAILED:
      return "front-wall stationary verification failed safely";

    default:
      return "unknown";
  }
}

bool calculateStraightWheelSpeeds(
  bool left1Valid,
  uint16_t left1Mm,
  uint32_t left1UpdatedMs,
  bool left2Valid,
  float left2CorrectedMm,
  uint32_t left2UpdatedMs,
  float &leftSpeed,
  float &rightMagnitude
)
{
  // Default: execute the planner's straight segment with no wall correction.
  leftSpeed = DRIVE_SPEED;
  rightMagnitude = DRIVE_SPEED;

  bool left1Fresh =
    readingFresh(
      left1Valid,
      left1UpdatedMs
    );

  bool left2Fresh =
    readingFresh(
      left2Valid,
      left2UpdatedMs
    );

  // Left 1 is the non-offset sensor and is the minimum requirement for
  // side-distance control.
  if (!left1Fresh) {
    return false;
  }

  float frontLeftMm =
    (float)left1Mm;

  // If Left 1 sees no wall within one maze cell, ignore wall following.
  if (frontLeftMm > LEFT_WALL_IGNORE_MM) {
    return false;
  }

  bool left2WithinWallRange =
    left2Fresh &&
    left2CorrectedMm <= LEFT_WALL_IGNORE_MM;

  /*
    If Left 2 is fresh and says the wall is more than 180 mm away while
    Left 1 sees something closer, the two sensors are probably not looking
    at one continuous wall (for example the end of a wall). Do not steer
    toward that ambiguous feature.
  */
  if (
    left2Fresh &&
    left2CorrectedMm > LEFT_WALL_IGNORE_MM
  ) {
    return false;
  }

  bool dualLidarUsable =
    left2WithinWallRange &&
    left2CorrectedMm >= LEFT2_MIN_USABLE_MM;

  // ----------------------------------------------------------
  // DISTANCE CONTROL: target approximately 30 mm from left wall
  // ----------------------------------------------------------
  float measuredDistanceMm;

  if (dualLidarUsable) {
    // Average front and rear readings so the distance controller is not
    // confused merely because the robot is slightly angled.
    measuredDistanceMm =
      0.5f *
      (
        frontLeftMm +
        left2CorrectedMm
      );
  }
  else {
    // When the offset sensor is inside its unreliable <35 mm region (or is
    // temporarily unavailable), use the trustworthy Left 1 sensor alone.
    measuredDistanceMm =
      frontLeftMm;
  }

  float distanceErrorMm =
    measuredDistanceMm -
    LEFT_TARGET_DISTANCE_MM;

  float effectiveDistanceErrorMm = 0.0f;

  if (
    distanceErrorMm >
    LEFT_DISTANCE_TOLERANCE_MM
  ) {
    effectiveDistanceErrorMm =
      distanceErrorMm -
      LEFT_DISTANCE_TOLERANCE_MM;
  }
  else if (
    distanceErrorMm <
    -LEFT_DISTANCE_TOLERANCE_MM
  ) {
    effectiveDistanceErrorMm =
      distanceErrorMm +
      LEFT_DISTANCE_TOLERANCE_MM;
  }

  float distanceCorrection =
    effectiveDistanceErrorMm *
    DISTANCE_KP_SPEED_PER_MM;

  // ----------------------------------------------------------
  // PARALLEL CONTROL: only when both sensors are trustworthy
  // ----------------------------------------------------------
  float parallelCorrection = 0.0f;

  if (dualLidarUsable) {
    float parallelErrorMm =
      frontLeftMm -
      left2CorrectedMm;

    if (
      fabsf(parallelErrorMm) >
      PARALLEL_DEADBAND_MM
    ) {
      parallelCorrection =
        parallelErrorMm *
        PARALLEL_KP_SPEED_PER_MM;
    }
  }

  /*
    Both correction terms use the same physical sign:

      positive correction:
        - too far from left wall, and/or
        - nose points away from left wall
        -> steer LEFT

      negative correction:
        - too close to left wall, and/or
        - nose points toward left wall
        -> steer RIGHT

    Forward motor convention:
      left  = positive
      right = negative

    Therefore:
      steer LEFT  -> slow left, speed right
      steer RIGHT -> speed left, slow right

    Because the speed adjustment is symmetric, the average commanded forward
    wheel speed remains DRIVE_SPEED. The robot therefore continues completing
    the planner's straight-distance instruction while wall feedback trims its
    heading and lateral spacing.
  */
  float totalCorrection =
    clampFloat(
      distanceCorrection +
      parallelCorrection,
      -WALL_MAX_SPEED_CORRECTION,
      WALL_MAX_SPEED_CORRECTION
    );

  leftSpeed =
    DRIVE_SPEED -
    totalCorrection;

  rightMagnitude =
    DRIVE_SPEED +
    totalCorrection;

  return true;
}

uint8_t executeStraightByDistanceMm(
  float distanceMm,
  bool plannedTurnAfterStraight
)
{
  long targetSteps =
    (long)(
      distanceMm *
      STEPS_PER_MM +
      0.5f
    );

  targetSteps = max(1L, targetSteps);

  long startLeft =
    leftMotor.currentPosition();

  long startRight =
    rightMotor.currentPosition();

  long maxExtraApproachSteps =
    (long)(
      FRONT_APPROACH_MAX_EXTRA_MM *
      STEPS_PER_MM +
      0.5f
    );

  long frontEarlyAcceptWindowSteps =
    (long)(
      FRONT_EARLY_ACCEPT_WINDOW_MM *
      STEPS_PER_MM +
      0.5f
    );

  uint32_t lastBreakMs =
    millis();

  bool frontApproachActive = false;

  long frontApproachStartTravel = 0;

  // When LiDAR steering has pulled the mouse too far away from the route
  // heading, temporarily let the IMU take control until the heading is safe.
  bool imuWallHeadingRecoveryActive = false;

  // For 60..180 mm approach readings, nearby geometry seen while yawed still
  // gives the IMU heading controller priority. A <=50 mm sample is handled by
  // the stronger stationary verification routine below instead.
  bool frontHeadingRecoveryActive = false;

  // Approach confirmation uses distinct front LiDAR updates. A <=50 mm update
  // does not count here; it immediately enters stationary verification.
  uint8_t frontApproachConfirmCount = 0;
  uint32_t lastProcessedFrontUpdatedMs = 0;

  // The LiDAR task is already sampling continuously on the opposite core.
  // The motor core keeps only primitive cached values so Arduino's automatic
  // .ino prototype generation cannot trip over a custom struct type.
  bool cachedFrontValid = false;
  uint16_t cachedFrontMm = 0;
  uint32_t cachedFrontUpdatedMs = 0;

  bool cachedLeft1Valid = false;
  uint16_t cachedLeft1Mm = 0;
  uint32_t cachedLeft1UpdatedMs = 0;

  bool cachedLeft2Valid = false;
  float cachedLeft2CorrectedMm = 0.0f;
  uint32_t cachedLeft2UpdatedMs = 0;

  // Force an immediate first cache copy.
  uint32_t lastRangeCacheCopyMs =
    millis() - RANGE_CACHE_COPY_PERIOD_MS;

  while (true) {
    long travelled =
      averageMotorTravel(
        startLeft,
        startRight
      );

    uint32_t now = millis();

    if (
      now - lastRangeCacheCopyMs >=
        RANGE_CACHE_COPY_PERIOD_MS
    ) {
      // Very short cross-core critical section: copy RAM only.
      // No Wire/VL6180X calls occur on this motor core.
      portENTER_CRITICAL(&rangeMux);

      cachedFrontValid = ranges.frontValid;
      cachedFrontMm = ranges.frontMm;
      cachedFrontUpdatedMs = ranges.frontUpdatedMs;

      cachedLeft1Valid = ranges.left1Valid;
      cachedLeft1Mm = ranges.left1Mm;
      cachedLeft1UpdatedMs = ranges.left1UpdatedMs;

      cachedLeft2Valid = ranges.left2Valid;
      cachedLeft2CorrectedMm = ranges.left2CorrectedMm;
      cachedLeft2UpdatedMs = ranges.left2UpdatedMs;

      portEXIT_CRITICAL(&rangeMux);

      lastRangeCacheCopyMs = now;
    }

    bool frontFresh =
      readingFresh(
        cachedFrontValid,
        cachedFrontUpdatedMs
      );

    // Read the IMU BEFORE making any front-wall decision. In the old code the
    // <=50 mm test happened before heading was checked, so a side wall clipped
    // by the front sensor could immediately end the straight.
    float currentHeadingDeg = 0.0f;
    bool headingAvailable =
      getPlannerHeadingDeg(currentHeadingDeg);

    float headingErrorDeg = 0.0f;
    float absHeadingErrorDeg = 999.0f;

    if (headingAvailable) {
      headingErrorDeg =
        shortestHeadingErrorDeg(
          plannedHeadingDeg,
          currentHeadingDeg
        );

      absHeadingErrorDeg =
        fabsf(headingErrorDeg);
    }

    bool frontHeadingAligned =
      headingAvailable &&
      absHeadingErrorDeg <=
        FRONT_WALL_HEADING_TOLERANCE_DEG;

    bool nearPlannedStraightEnd =
      travelled + frontEarlyAcceptWindowSteps >=
        targetSteps;

    // If a nearby feature appears while the mouse is yawed, do not classify
    // it yet. Straighten toward the planned corridor heading and re-measure.
    if (
      frontFresh &&
      (float)cachedFrontMm < FRONT_APPROACH_MAX_MM &&
      headingAvailable &&
      !frontHeadingAligned
    ) {
      frontHeadingRecoveryActive = true;
    }
    else if (
      frontHeadingRecoveryActive &&
      (
        !frontFresh ||
        (float)cachedFrontMm >= FRONT_APPROACH_MAX_MM ||
        (
          headingAvailable &&
          absHeadingErrorDeg <=
            FRONT_WALL_HEADING_RECOVER_DEG
        )
      )
    ) {
      frontHeadingRecoveryActive = false;
    }

    // Process each DISTINCT front-sensor measurement only once. The motor loop
    // can execute many times before the LiDAR task publishes another sample.
    if (
      frontFresh &&
      cachedFrontUpdatedMs != lastProcessedFrontUpdatedMs
    ) {
      lastProcessedFrontUpdatedMs =
        cachedFrontUpdatedMs;

      // ------------------------------------------------------
      // <=50 mm NEVER TURNS DIRECTLY: STOP, SQUARE UP, RE-CHECK
      // ------------------------------------------------------
      if (
        (float)cachedFrontMm <=
          FRONT_TURN_TRIGGER_MM
      ) {
        stopMotorsHold();

        // In-place realignment moves both stepper position counters. Capture
        // those correction steps and shift the straight's odometry baselines by
        // the same amounts so rotation is NEVER mistaken for forward travel.
        long correctionStartLeft =
          leftMotor.currentPosition();

        long correctionStartRight =
          rightMotor.currentPosition();

        uint32_t lastVerificationFrontUpdatedMs =
          cachedFrontUpdatedMs;

        uint8_t verificationResult =
          verifyFrontWallStationary(
            plannedHeadingDeg,
            lastVerificationFrontUpdatedMs
          );

        long correctionDeltaLeft =
          leftMotor.currentPosition() -
          correctionStartLeft;

        long correctionDeltaRight =
          rightMotor.currentPosition() -
          correctionStartRight;

        // Remove the in-place correction from the straight-distance estimate.
        startLeft += correctionDeltaLeft;
        startRight += correctionDeltaRight;

        // The verification routine consumed newer samples than the local cache.
        // Mark them processed so the same stationary samples cannot immediately
        // retrigger another verification when this loop resumes.
        lastProcessedFrontUpdatedMs =
          lastVerificationFrontUpdatedMs;

        frontApproachConfirmCount = 0;
        frontHeadingRecoveryActive = false;
        imuWallHeadingRecoveryActive = false;

        if (
          verificationResult ==
            FRONT_VERIFY_RESULT_FAILED
        ) {
          stopMotorsHold();
          Serial.println(
            "SAFETY STOP: front-wall verification could not complete reliably."
          );
          return STRAIGHT_END_FRONT_VERIFICATION_FAILED;
        }

        if (
          verificationResult ==
            FRONT_VERIFY_RESULT_CLEARED
        ) {
          // Recompute all cached ranges, heading, wall control, and odometry
          // from scratch before applying forward motor power again.
          continue;
        }

        // It survived strict realignment + fresh-sample verification. Only NOW
        // may the route decide whether this real wall is supposed to end the
        // current straight.
        travelled =
          averageMotorTravel(
            startLeft,
            startRight
          );

        nearPlannedStraightEnd =
          travelled + frontEarlyAcceptWindowSteps >=
            targetSteps;

        if (!plannedTurnAfterStraight) {
          Serial.println(
            "SAFETY STOP: stationary-verified front obstacle, but the uploaded "
            "route has no turn after this straight."
          );
          return STRAIGHT_END_UNEXPECTED_FRONT_OBSTACLE;
        }

        if (!nearPlannedStraightEnd) {
          Serial.println(
            "SAFETY STOP: stationary-verified front obstacle is too early for "
            "the planned straight endpoint."
          );
          return STRAIGHT_END_UNEXPECTED_FRONT_OBSTACLE;
        }

        if (frontApproachActive) {
          return STRAIGHT_END_FRONT_APPROACH;
        }

        return STRAIGHT_END_FRONT_WALL_EARLY;
      }

      // 60..180 mm approach extension remains conservative: it only counts
      // when aligned and already near an expected straight endpoint.
      if (
        frontHeadingAligned &&
        nearPlannedStraightEnd &&
        (float)cachedFrontMm > FRONT_APPROACH_START_MM &&
        (float)cachedFrontMm < FRONT_APPROACH_MAX_MM
      ) {
        if (
          frontApproachConfirmCount <
            FRONT_APPROACH_CONFIRM_SAMPLES
        ) {
          frontApproachConfirmCount++;
        }
      }
      else {
        frontApproachConfirmCount = 0;
      }
    }

    bool nominalDistanceReached =
      travelled >= targetSteps;

    // --------------------------------------------------------
    // NOMINAL DISTANCE REACHED: SHOULD THE NEXT TURN WAIT?
    // --------------------------------------------------------
    if (
      nominalDistanceReached &&
      !frontApproachActive
    ) {
      // Only wait for a wall if the NEXT uploaded segment actually contains a
      // turn. Otherwise a side feature can never make this straight overrun.
      if (
        plannedTurnAfterStraight &&
        frontApproachConfirmCount >=
          FRONT_APPROACH_CONFIRM_SAMPLES
      ) {
        frontApproachActive = true;
        frontApproachStartTravel = travelled;

        Serial.print(
          "Nominal distance reached with confirmed, heading-aligned front wall "
        );
        Serial.print(cachedFrontMm);
        Serial.println(
          " mm away. Delaying planned turn until confirmed <= 50 mm."
        );
      }
      else {
        stopMotorsHold();
        delay(80);

        return STRAIGHT_END_ODOMETRY;
      }
    }

    // Fail-safe only. Normal behaviour is to remain in front-approach mode
    // until the sensor reaches the <= 50 mm trigger.
    if (
      frontApproachActive &&
      travelled - frontApproachStartTravel >=
        maxExtraApproachSteps
    ) {
      stopMotorsHold();
      delay(80);

      Serial.println(
        "WARNING: front-approach fail-safe reached before <= 50 mm."
      );

      return STRAIGHT_END_APPROACH_FAILSAFE;
    }

    float leftSpeed = DRIVE_SPEED;
    float rightMagnitude = DRIVE_SPEED;

    bool wallControlActive =
      calculateStraightWheelSpeeds(
        cachedLeft1Valid,
        cachedLeft1Mm,
        cachedLeft1UpdatedMs,
        cachedLeft2Valid,
        cachedLeft2CorrectedMm,
        cachedLeft2UpdatedMs,
        leftSpeed,
        rightMagnitude
      );

    // The IMU is now also a SAFETY GUARD while wall following is active.
    // Normal LiDAR steering remains untouched while heading error is small.
    // If LiDAR ever drags the mouse >12 deg from the route heading, IMU control
    // temporarily takes priority until the error is back below 6 deg.
    if (headingAvailable) {
      if (wallControlActive) {
        if (
          !imuWallHeadingRecoveryActive &&
          absHeadingErrorDeg >= IMU_WALL_GUARD_ENTER_DEG
        ) {
          imuWallHeadingRecoveryActive = true;
          Serial.print(
            "IMU wall-follow guard ACTIVE; heading error="
          );
          Serial.print(headingErrorDeg, 1);
          Serial.println(" deg");
        }
        else if (
          imuWallHeadingRecoveryActive &&
          absHeadingErrorDeg <= IMU_WALL_GUARD_EXIT_DEG
        ) {
          imuWallHeadingRecoveryActive = false;
          Serial.println(
            "IMU wall-follow guard released; LiDAR steering resumed."
          );
        }
      }
      else {
        // No trustworthy wall: heading hold is the normal controller.
        imuWallHeadingRecoveryActive = false;
      }
    }

    bool useImuHeadingControl =
      !wallControlActive ||
      imuWallHeadingRecoveryActive ||
      frontHeadingRecoveryActive;

    if (
      useImuHeadingControl &&
      headingAvailable
    ) {
      float imuCorrection =
        clampFloat(
          headingErrorDeg *
            IMU_STRAIGHT_KP_SPEED_PER_DEG,
          -IMU_STRAIGHT_MAX_SPEED_CORRECTION,
          IMU_STRAIGHT_MAX_SPEED_CORRECTION
        );

      // Positive correction steers LEFT; negative correction steers RIGHT.
      // Both wheel magnitudes remain forward, so this can never become an
      // in-place turn.
      leftSpeed = DRIVE_SPEED - imuCorrection;
      rightMagnitude = DRIVE_SPEED + imuCorrection;
    }

    /*
      Forward direction established from the earlier motor tests:

        left  = positive
        right = negative
    */
    leftMotor.setSpeed(
      leftSpeed
    );

    rightMotor.setSpeed(
      -rightMagnitude
    );

    leftMotor.runSpeed();
    rightMotor.runSpeed();

    schedulerBreakIfNeeded(lastBreakMs);
  }
}

void executeEntireRouteBlocking()
{
  robotState = RUNNING_ROUTE;

  Serial.println();
  Serial.println("===========================================");
  Serial.println("AUTONOMOUS ROUTE START");
  Serial.println("Wi-Fi: OFF");
  Serial.println("IMU: MPU6050 ACTIVE");
  Serial.println("Turns: MPU6050 Z-ANGLE");
  Serial.println("Straights: WHEEL STEPS + LIDAR + IMU + STATIONARY FRONT-WALL VERIFICATION");
  Serial.println("===========================================");

  bool routeAborted = false;

  for (int i = 0; i < routeLength; i++) {
    RouteSegment &segment =
      route[i];

    Serial.println();
    Serial.print("SEGMENT ");
    Serial.print(i + 1);
    Serial.print("/");
    Serial.println(routeLength);

    Serial.print("Turn: ");
    Serial.print(
      segment.turnDegrees,
      1
    );
    Serial.println(" deg");

    Serial.print("Planner mode: ");
    Serial.print(
      segment.uploadedMode
    );
    Serial.println(
      " (retained; feedback rules apply to all straights)"
    );

    Serial.print("Nominal straight distance: ");
    Serial.print(segment.distanceMm, 1);
    Serial.println(" mm");

    if (
      fabsf(segment.turnDegrees) >=
      MIN_TURN_DEGREES
    ) {
      float currentHeadingBeforeTurnDeg = 0.0f;

      if (!getPlannerHeadingDeg(currentHeadingBeforeTurnDeg)) {
        routeAborted = true;
        Serial.println(
          "ROUTE ABORTED: no fresh MPU6050 heading before turn."
        );
        break;
      }

      // How far are we from the heading the previous straight was supposed to
      // maintain?  Previously this error was silently added to the next turn.
      // Example: already 90 deg off + requested 90 deg => 180 deg in place.
      float preTurnHeadingErrorDeg =
        shortestHeadingErrorDeg(
          plannedHeadingDeg,
          currentHeadingBeforeTurnDeg
        );

      Serial.print("IMU heading before turn: ");
      Serial.print(currentHeadingBeforeTurnDeg, 1);
      Serial.println(" deg");

      Serial.print("Pre-turn route heading error: ");
      Serial.print(preTurnHeadingErrorDeg, 1);
      Serial.println(" deg");

      if (
        fabsf(preTurnHeadingErrorDeg) >
        IMU_PRETURN_REBASE_DEG
      ) {
        Serial.println(
          "WARNING: large pre-turn heading error; rebasing to real IMU heading."
        );
        Serial.println(
          "Only the planner-requested relative turn will be executed."
        );

        plannedHeadingDeg =
          currentHeadingBeforeTurnDeg;
      }

      float targetHeadingDeg =
        plannedHeadingDeg +
        segment.turnDegrees;

      float requiredPhysicalTurnDeg =
        shortestHeadingErrorDeg(
          targetHeadingDeg,
          currentHeadingBeforeTurnDeg
        );

      Serial.print("Planner requested relative turn: ");
      Serial.print(segment.turnDegrees, 1);
      Serial.println(" deg");

      Serial.print("Actual physical turn required now: ");
      Serial.print(requiredPhysicalTurnDeg, 1);
      Serial.println(" deg");

      Serial.print("Executing IMU turn to heading: ");
      Serial.print(targetHeadingDeg, 1);
      Serial.println(" deg");

      bool turnOk =
        executeTurnToPlannedHeading(
          targetHeadingDeg,
          segment.turnDegrees
        );

      if (!turnOk) {
        routeAborted = true;
        Serial.println(
          "ROUTE ABORTED: MPU6050 turn control failed/safety guard triggered."
        );
        break;
      }

      // The successfully reached target becomes the new straight-line heading.
      plannedHeadingDeg =
        targetHeadingDeg;

      Serial.println("IMU turn complete.");
    }
    else {
      Serial.println(
        "No turn required."
      );
    }

    Serial.print(
      "Executing nominal straight target: "
    );

    Serial.print(
      (long)(
        segment.distanceMm *
        STEPS_PER_MM +
        0.5f
      )
    );

    Serial.println(" steps");

    // A segment stores its turn BEFORE its straight. Therefore the turn that
    // follows this straight is route[i + 1].turnDegrees.
    bool plannedTurnAfterStraight =
      (i + 1 < routeLength) &&
      fabsf(route[i + 1].turnDegrees) >=
        MIN_TURN_DEGREES;

    uint8_t endReason =
      executeStraightByDistanceMm(
        segment.distanceMm,
        plannedTurnAfterStraight
      );

    Serial.print("Straight complete: ");
    Serial.println(
      straightEndReasonText(endReason)
    );

    if (endReason == STRAIGHT_END_APPROACH_FAILSAFE) {
      routeAborted = true;

      Serial.println(
        "ROUTE ABORTED: front wall never reached the <= 50 mm trigger "
        "within the allowed extra approach distance."
      );

      break;
    }

    if (endReason == STRAIGHT_END_UNEXPECTED_FRONT_OBSTACLE) {
      routeAborted = true;

      Serial.println(
        "ROUTE ABORTED: stationary-verified front obstacle did not match "
        "the location/turn expected by the uploaded path."
      );

      break;
    }

    if (endReason == STRAIGHT_END_FRONT_VERIFICATION_FAILED) {
      routeAborted = true;

      Serial.println(
        "ROUTE ABORTED: front-wall verification lost reliable IMU/LiDAR "
        "confirmation or could not realign safely."
      );

      break;
    }

    /*
      If the front wall ended this straight early, the for-loop now advances
      to the next segment. Its turnDegrees value is therefore the "next
      rotation instruction" requested by the planner.
    */
  }

  stopMotorsHold();

  Serial.println();
  Serial.println("===========================================");

  if (routeAborted) {
    Serial.println("ROUTE ABORTED / SAFETY STOP");
  }
  else {
    Serial.println("ENTIRE ROUTE COMPLETE");
  }

  Serial.println("===========================================");

  routeLength = 0;
  robotState = WAITING_FOR_ROUTE;
}

// ============================================================
// WEB PAGE
// ============================================================


String buildControlPage()
{
  String page;

  // A 200-segment route can produce a substantially larger confirmation page.
  // Reserve enough capacity up front to reduce repeated String reallocations.
  page.reserve(7000 + routeLength * 100);

  page +=
    "<!DOCTYPE html>"
    "<html>"
    "<head>"
    "<meta name='viewport' "
    "content='width=device-width,initial-scale=1'>"
    "<style>"
    "body{font-family:Arial;background:#f1f1f1;margin:0;padding:18px;}"
    ".card{max-width:760px;margin:auto;background:white;"
    "padding:28px;border-radius:20px;"
    "box-shadow:0 3px 18px rgba(0,0,0,.15);}"
    "h1{text-align:center;font-size:42px;margin-top:5px;}"
    ".status{padding:18px;border-radius:14px;background:#eee;"
    "text-align:center;font-size:23px;margin-bottom:20px;}"
    ".route{font-size:17px;margin:8px 0;padding:10px 0;"
    "border-bottom:1px solid #ddd;}"
    ".note{background:#eef4ff;padding:14px;border-radius:12px;"
    "margin:16px 0;line-height:1.5;}"
    "button{width:100%;padding:22px;border:0;border-radius:14px;"
    "font-size:30px;font-weight:bold;color:white;}"
    ".start{background:#19a34a;}"
    ".disabled{background:#999;}"
    "</style>"
    "</head>"
    "<body>"
    "<div class='card'>"
    "<h1>MicroMouse</h1>";

  page +=
    "<div class='status'><b>Status:</b><br>";

  page +=
    stateText();

  page +=
    "</div>";

  page +=
    "<div class='note'>"
    "<b>LiDAR + MPU6050 route executor</b><br>"
    "IMU: calibrated after START, before any movement<br>"
    "Turns: MPU6050 Z-angle control<br>"
    "Straights: wheel steps + front/left LiDAR; IMU heading hold if left wall is lost<br>"
    "Left2 correction: +35 mm<br>"
    "Left wall control active only when both left readings are <=90 mm<br>"
    "Front <=50 mm: requires IMU alignment + 3 new samples + planned-turn match<br>"
    "Front 60-180 mm: approach extension requires alignment + 2 new samples<br>"
    "Wi-Fi: switches OFF while route runs"
    "</div>";

  page +=
    "<div class='note'>"
    "<b>Last ESP32 reset:</b> ";

  page +=
    resetReasonText();

  page +=
    "</div>";

  if (
    robotState == WAITING_FOR_START &&
    routeLength > 0
  ) {
    page +=
      "<p><b>Complete route confirmed:</b> ";

    page +=
      String(routeLength);

    page +=
      " segments</p>";

    for (int i = 0; i < routeLength; i++) {
      page +=
        "<div class='route'>";

      page +=
        String(i + 1);

      page +=
        ": turn ";

      page +=
        String(
          route[i].turnDegrees,
          1
        );

      page +=
        "&deg; | NOMINAL DRIVE | ";

      page +=
        String(route[i].distanceMm, 1);

      page +=
        " mm";

      page +=
        "</div>";
    }

    page +=
      "<div class='note'>"
      "The START button is enabled because the ESP32 has already "
      "received, parsed, validated, and committed the ENTIRE route."
      "</div>";

    page +=
      "<form method='POST' action='/start'>"
      "<button class='start' type='submit'>START</button>"
      "</form>";
  }
  else {
    page +=
      "<div class='note'>"
      "START remains disabled until a complete valid route is uploaded."
      "</div>"
      "<button class='disabled' disabled>START</button>";
  }

  page +=
    "</div>"
    "</body>"
    "</html>";

  return page;
}

// ============================================================
// WEB HANDLERS
// ============================================================

void handleRoot()
{
  webServer.send(
    200,
    "text/html",
    buildControlPage()
  );
}

void handleRouteUpload()
{
  // Running routes cannot receive uploads because Wi-Fi is off,
  // but keep this check for completeness.
  if (robotState == RUNNING_ROUTE) {
    webServer.send(
      409,
      "text/plain",
      "Robot is running."
    );
    return;
  }

  String payload =
    webServer.arg("plain");

  String errorMessage;

  bool valid =
    parseCompleteRouteTransactionally(
      payload,
      errorMessage
    );

  if (!valid) {
    webServer.send(
      400,
      "text/plain",
      "ROUTE REJECTED: " +
      errorMessage
    );

    return;
  }

  Serial.println();
  Serial.println("===========================================");
  Serial.println("FULL ROUTE RECEIVED AND COMMITTED");
  Serial.print("Segments: ");
  Serial.println(routeLength);
  Serial.println("START is now available.");
  Serial.println("===========================================");

  webServer.send(
    200,
    "text/plain",
    "ROUTE ACCEPTED: complete route loaded; " +
    String(routeLength) +
    " segments; START enabled."
  );
}

void handleStart()
{
  if (!imuWorking) {
    webServer.send(
      503,
      "text/html",
      "<html><body>"
      "<h1>START refused</h1>"
      "<p>The MPU6050 did not initialise, so IMU-guided navigation cannot run.</p>"
      "<p><a href='/'>Back</a></p>"
      "</body></html>"
    );

    return;
  }

  if (
    robotState != WAITING_FOR_START ||
    routeLength <= 0
  ) {
    webServer.send(
      409,
      "text/html",
      "<html><body>"
      "<h1>START refused</h1>"
      "<p>A complete validated route has not been loaded.</p>"
      "<p><a href='/'>Back</a></p>"
      "</body></html>"
    );

    return;
  }

  if (startPending) {
    webServer.send(
      200,
      "text/plain",
      "START already accepted."
    );

    return;
  }

  /*
    Send the browser response FIRST.

    The route itself is already entirely in ESP32 RAM at this point.
    No network data is required after this response.
  */
  webServer.send(
    200,
    "text/html",
    "<!DOCTYPE html>"
    "<html><head>"
    "<meta name='viewport' "
    "content='width=device-width,initial-scale=1'>"
    "</head>"
    "<body style='font-family:Arial;text-align:center;padding:40px'>"
    "<h1>START accepted</h1>"
    "<p>The complete route is already stored in the ESP32.</p>"
    "<p><b>Keep the MicroMouse completely still.</b></p>"
    "<p>The MPU6050 will calibrate before any movement begins.</p>"
    "<p>Wi-Fi will turn off for calibration and the autonomous route.</p>"
    "<p>The MicroMouse Wi-Fi network will return after completion.</p>"
    "</body></html>"
  );

  startPending = true;
  startAcceptedAtMs = millis();

  Serial.println();
  Serial.println("START accepted.");
  Serial.println(
    "Complete route already stored in ESP32 RAM."
  );
}

// ============================================================
// WIFI LIFECYCLE
// ============================================================

void configureWebRoutesOnce()
{
  if (webRoutesConfigured) {
    return;
  }

  webServer.on(
    "/",
    HTTP_GET,
    handleRoot
  );

  webServer.on(
    "/route",
    HTTP_POST,
    handleRouteUpload
  );

  webServer.on(
    "/start",
    HTTP_POST,
    handleStart
  );

  webRoutesConfigured = true;
}

void startWiFiServer()
{
  if (wifiActive) {
    return;
  }

  WiFi.mode(WIFI_AP);

  bool ok =
    WiFi.softAP(
      WIFI_AP_NAME,
      WIFI_AP_PASSWORD
    );

  if (!ok) {
    Serial.println(
      "ERROR: could not start MicroMouse Wi-Fi."
    );

    wifiActive = false;
    return;
  }

  configureWebRoutesOnce();

  webServer.begin();

  wifiActive = true;

  Serial.println();
  Serial.println("===========================================");
  Serial.println("MicroMouse Wi-Fi READY");
  Serial.print("Network: ");
  Serial.println(WIFI_AP_NAME);
  Serial.print("Control page: http://");
  Serial.println(WiFi.softAPIP());
  Serial.println("===========================================");
}

void stopWiFiForAutonomousRun()
{
  if (!wifiActive) {
    return;
  }

  Serial.println(
    "Stopping web server + Wi-Fi before autonomous movement..."
  );

  webServer.stop();

  WiFi.softAPdisconnect(true);

  WiFi.mode(WIFI_OFF);

  wifiActive = false;

  /*
    Give ESP32 Wi-Fi background tasks a short moment to settle completely
    before motor movement begins.
  */
  delay(100);

  Serial.println("Wi-Fi OFF.");
}

// ============================================================
// SETUP / LOOP
// ============================================================

void setup()
{
  Serial.begin(115200);
  delay(500);

  Serial.println();
  Serial.println("===========================================");
  Serial.println("DUAL-CORE LIDAR + IMU MICROMOUSE ROUTE EXECUTOR");
  Serial.println("===========================================");

  Serial.print("Last reset reason: ");
  Serial.println(resetReasonText());

  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(100000);

  Serial.println("I2C: SDA=GPIO21, SCL=GPIO22");

  Serial.print("Wheel diameter: ");
  Serial.print(WHEEL_DIAMETER_MM, 1);
  Serial.println(" mm");

  Serial.print("Wheel track: ");
  Serial.print(WHEEL_TRACK_MM, 1);
  Serial.println(" mm");

  Serial.print("Steps/mm: ");
  Serial.println(STEPS_PER_MM, 4);

  Serial.print("Turn steps/degree: ");
  Serial.println(
    TURN_STEPS_PER_DEGREE,
    4
  );

  Serial.print("90 degree turn target: ");
  Serial.println(
    (long)(
      TURN_STEPS_PER_DEGREE *
      90.0f +
      0.5f
    )
  );

  // Allow enough headroom for the combined distance + parallel correction.
  float maximumStraightWheelSpeed =
    DRIVE_SPEED +
    WALL_MAX_SPEED_CORRECTION;

  float configuredMotorMaxSpeed =
    max(
      maximumStraightWheelSpeed,
      TURN_FAST_SPEED
    );

  leftMotor.setMaxSpeed(
    configuredMotorMaxSpeed
  );

  rightMotor.setMaxSpeed(
    configuredMotorMaxSpeed
  );

  leftMotor.setSpeed(0.0f);
  rightMotor.setSpeed(0.0f);

  leftMotor.enableOutputs();
  rightMotor.enableOutputs();

  // Initialise the MPU6050 using the same hardware/library setup as the
  // supplied MicromouseMK2 example. Offset calibration happens only after
  // the website START command, while the robot is stationary.
  imuWorking = initialiseMPU6050Hardware();

  initialiseFeedbackRangeSensors();

  if (!startLidarTaskOnOppositeCore()) {
    Serial.println(
      "ERROR: could not start dedicated LiDAR task. Feedback will remain unavailable."
    );
  }
  else {
    // Give the first round-robin sensor samples time to populate the cache.
    delay(100);
  }

  robotState = WAITING_FOR_ROUTE;
  routeLength = 0;

  startWiFiServer();
}

void loop()
{
  // While idle, only service Wi-Fi.
  if (wifiActive) {
    webServer.handleClient();
  }

  /*
    The START HTTP response has already been sent.

    Wait long enough for it to leave the network stack, then shut down
    the entire wireless stack BEFORE running any movement code.
  */
  if (
    startPending &&
    millis() - startAcceptedAtMs >=
      START_HANDOFF_DELAY_MS
  ) {
    startPending = false;

    stopWiFiForAutonomousRun();

    // START-time calibration happens after the HTTP response has been sent
    // and before ANY movement. The robot must remain completely still here.
    bool calibrationOk =
      calibrateMPU6050ForRun();

    if (calibrationOk) {
      /*
        This call intentionally BLOCKS until the whole stored route finishes.

        During this function there are:
          - no web requests
          - no Wi-Fi
          - steppers/route execution on the Arduino motion core
          - all moving-time I2C reads on the opposite-core sensor task
          - LiDAR and MPU6050 values copied from the shared cache
          - no path parsing
      */
      executeEntireRouteBlocking();
    }
    else {
      stopMotorsHold();
      Serial.println();
      Serial.println(
        "AUTONOMOUS START CANCELLED: MPU6050 calibration failed."
      );
      Serial.println(
        "The uploaded route has been retained; fix the IMU issue and press START again."
      );
    }

    // After a completed route the old route is cleared. If calibration failed
    // the route is intentionally retained and START remains available.
    startWiFiServer();
  }

  delay(1);
}
