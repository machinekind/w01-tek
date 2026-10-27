// The robot's roll, pitch and heading from a raw IMU reading.
//
// The IMU chip is bolted on upright but turned a quarter turn, so its +y
// axis points forward and its +x axis points to the right. The driver
// publishes what the chip measures in the chip's own axes, unchanged, and
// the page reads that topic straight off the bridge. So the raw numbers are
// the chip's attitude and the chip's turn rates, not the robot's: tilt the
// robot nose down and they call it a roll. This takes the mount back out.
// The gateway's hello frame says how the chip is mounted (imu_mount_rpy).
//
// The same sums as wojtek_policy/imu_mount.py, one function for one
// function; the two must stay the same maths. Quaternions are {x, y, z, w}
// like the bridge decodes them, vectors are [x, y, z].

// The quaternion for a URDF roll, pitch and yaw. URDF turns about fixed
// axes: roll about x first, then pitch about y, then yaw about z.
export function quatFromRpy(r, p, y) {
  const cr = Math.cos(r / 2), sr = Math.sin(r / 2);
  const cp = Math.cos(p / 2), sp = Math.sin(p / 2);
  const cy = Math.cos(y / 2), sy = Math.sin(y / 2);
  return {
    x: sr * cp * cy - cr * sp * sy,
    y: cr * sp * cy + sr * cp * sy,
    z: cr * cp * sy - sr * sp * cy,
    w: cr * cp * cy + sr * sp * sy,
  };
}

// a times b: turn by b first, then by a (the usual Hamilton product).
export function quatMul(a, b) {
  return {
    x: a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
    y: a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
    z: a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
    w: a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
  };
}

// The opposite turn of a unit quaternion.
export function quatConj(q) {
  return { x: -q.x, y: -q.y, z: -q.z, w: q.w };
}

// The vector v turned by the unit quaternion q: q * (v, 0) * conj(q).
export function rotate(q, v) {
  const r = quatMul(quatMul(q, { x: v[0], y: v[1], z: v[2], w: 0 }), quatConj(q));
  return [r.x, r.y, r.z];
}

// [roll, pitch, yaw] in radians. The pitch is clamped before the arcsine so
// rounding cannot push it out of range, and the all-zero quaternion (the
// driver's "no fusion yet") reads as level.
export function rpyFromQuat(q) {
  const roll = Math.atan2(2 * (q.w * q.x + q.y * q.z), 1 - 2 * (q.x * q.x + q.y * q.y));
  const pitch = Math.asin(Math.max(-1, Math.min(1, 2 * (q.w * q.y - q.z * q.x))));
  const yaw = Math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z));
  return [roll, pitch, yaw];
}

// The robot's attitude and turn rates from the IMU's: q is the topic's
// orientation {x, y, z, w}, w its angular_velocity {x, y, z} (or [x, y, z]),
// mountRpy how the sensor sits in base_link. Returns
// { roll, pitch, yaw, gyro: [x, y, z] }, all about the robot's own axes.
export function attitudeInBase(q, w, mountRpy) {
  const mount = quatFromRpy(mountRpy[0], mountRpy[1], mountRpy[2]);
  // The sensor's attitude is the robot's with the mount added on top, so
  // the robot's is the sensor's with the mount taken off again.
  const body = quatMul(q, quatConj(mount));
  const [roll, pitch, yaw] = rpyFromQuat(body);
  // A turn rate about the sensor's axes, said in the robot's axes.
  const gyro = rotate(mount, Array.isArray(w) ? w : [w.x, w.y, w.z]);
  return { roll, pitch, yaw, gyro };
}
