#include <Adafruit_Sensor.h>
#include <DPEng_BMX160.h>
#include <Mahony_BMX160.h>
#include <Wire.h>

// DPEng_BMX160 talks to the BMI160 accel/gyro registers, which are
// identical between BMI160 and BMX160 (same die). There is no
// magnetometer on this board, so mag data is never read here and the
// fusion filter runs in 6-DOF (gyro+accel) mode: roll/pitch are accurate,
// yaw has no absolute heading reference and will drift.
DPEng_BMX160 dpEng = DPEng_BMX160(0x160A, 0x160B, 0x160C);

float gyro_zero_offsets[3] = {0.0F, 0.0F, 0.0F};
Mahony_BMX160 filter;

unsigned long previousMillis = 0;
const int interval = 10;  // 100 Hz -> 10 ms per loop iteration

void setup()
{
  Serial.begin(115200);
  Serial.println(F("BMI160 AHRS Fusion (no magnetometer)"));
  if (!dpEng.begin(BMX160_ACCELRANGE_4G, GYRO_RANGE_250DPS)) {
    Serial.println("Ooops, no BMI160 detected ... Check your wiring!");
    while (1);
  }
  filter.begin();
}

void loop()
{
  unsigned long currentMillis = millis();
  if (currentMillis - previousMillis >= interval) {
    previousMillis = currentMillis;

    sensors_event_t accel_event, gyro_event, mag_event;
    dpEng.getEvent(&accel_event, &gyro_event, &mag_event);

    float gx = gyro_event.gyro.x + gyro_zero_offsets[0];
    float gy = gyro_event.gyro.y + gyro_zero_offsets[1];
    float gz = gyro_event.gyro.z + gyro_zero_offsets[2];

    // mag_event.timestamp is the BMX160/BMI160 chip's own sensortime decoded
    // to microseconds (see DPEng_BMX160::getEvent) -- finer-grained than the
    // millis()-based accel_event/gyro_event timestamps, kept here purely as
    // the fusion filter's clock even though mag_event's magnetic.* fields
    // are unused (no magnetometer on this chip).
    filter.updateIMU(
      gx, gy, gz, accel_event.acceleration.x, accel_event.acceleration.y,
      accel_event.acceleration.z, mag_event.timestamp);

    gx *= 0.0174533f;
    gy *= 0.0174533f;
    gz *= 0.0174533f;
    Serial.print("G X: ");
    Serial.print(gx);
    Serial.print(" Y: ");
    Serial.print(gy);
    Serial.print(" Z: ");
    Serial.print(gz);
    Serial.println(" rad/s");

    Serial.print("A X: ");
    Serial.print(accel_event.acceleration.x);
    Serial.print(" Y: ");
    Serial.print(accel_event.acceleration.y);
    Serial.print(" Z: ");
    Serial.print(accel_event.acceleration.z);
    Serial.println(" m/s^2");

    float qx, qy, qz, qw;
    filter.getQuaternion(&qw, &qx, &qy, &qz);
    Serial.print("R X: ");
    Serial.print(qx);
    Serial.print(" Y: ");
    Serial.print(qy);
    Serial.print(" Z: ");
    Serial.print(qz);
    Serial.print(" W: ");
    Serial.print(qw);
    Serial.println("");
  }
}
