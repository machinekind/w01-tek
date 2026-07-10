#include <Adafruit_Sensor.h>
#include <DPEng_BMX160.h>
#include <Madgwick_BMX160.h>
#include <Mahony_BMX160.h>
#include <Wire.h>

// Create sensor instance.
DPEng_BMX160 dpEng = DPEng_BMX160(0x160A, 0x160B, 0x160C);

float mag_offsets[3] = {9.83F, 4.42F, -6.97F};
float mag_softiron_matrix[3][3] = {
  {0.586, 0.006, 0.001}, {0.006, 0.601, -0.002}, {0.001, -0.002, 2.835}};
float mag_field_strength = 56.33F;
float gyro_zero_offsets[3] = {0.0F, 0.0F, 0.0F};
Mahony_BMX160 filter;

unsigned long previousMillis = 0;
const int interval = 10;  // 100 Hz -> 10 ms per loop iteration

void setup()
{
  Serial.begin(115200);
  Serial.println(F("DPEng AHRS Fusion Example"));
  if (!dpEng.begin(BMX160_ACCELRANGE_4G, GYRO_RANGE_250DPS)) {
    Serial.println("Ooops, no BMX160 detected ... Check your wiring!");
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

    float x = mag_event.magnetic.x - mag_offsets[0];
    float y = mag_event.magnetic.y - mag_offsets[1];
    float z = mag_event.magnetic.z - mag_offsets[2];

    float mx = x * mag_softiron_matrix[0][0] + y * mag_softiron_matrix[0][1] +
               z * mag_softiron_matrix[0][2];
    float my = x * mag_softiron_matrix[1][0] + y * mag_softiron_matrix[1][1] +
               z * mag_softiron_matrix[1][2];
    float mz = x * mag_softiron_matrix[2][0] + y * mag_softiron_matrix[2][1] +
               z * mag_softiron_matrix[2][2];

    float gx = gyro_event.gyro.x + gyro_zero_offsets[0];
    float gy = gyro_event.gyro.y + gyro_zero_offsets[1];
    float gz = gyro_event.gyro.z + gyro_zero_offsets[2];

    filter.updateIMU(
      gx, gy, gz, accel_event.acceleration.x, accel_event.acceleration.y,
      accel_event.acceleration.z, mag_event.timestamp);

    float roll = filter.getRollRadians();
    float pitch = filter.getPitchRadians();
    float heading = filter.getYawRadians();

    Serial.print("M X: ");
    Serial.print(mx);
    Serial.print(" Y: ");
    Serial.print(my);
    Serial.print(" Z: ");
    Serial.print(mz);
    Serial.println(" uT");

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
