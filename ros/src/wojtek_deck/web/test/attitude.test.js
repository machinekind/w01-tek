// node --test ros/src/wojtek_deck/web/test/*.test.js   (plain JS, no browser)
//
// The page's half of the IMU mount: the same cases as wojtek_policy's
// test_imu_mount.py, because attitude.js and imu_mount.py must stay the
// same maths. A reading is built the way the simulator builds it (the
// body's attitude with the mount added on top, rates turned into the
// chip's axes) and the robot's own numbers must come back out.
import { test } from "node:test";
import assert from "node:assert/strict";
import { attitudeInBase, quatConj, quatFromRpy, quatMul, rotate, rpyFromQuat } from "../attitude.js";

const DEG = Math.PI / 180;
const MOUNT = [0, 0, -1.5707963];   // body.urdf.xacro's imu_joint

function near(actual, expected, tol, what) {
  assert.equal(actual.length, expected.length, what);
  actual.forEach((v, i) => assert.ok(Math.abs(v - expected[i]) <= tol,
    `${what}[${i}]: ${v} is not within ${tol} of ${expected[i]}`));
}

test("with no mount the reading passes through", () => {
  const q = quatFromRpy(0.2, -0.3, 1.1);
  const a = attitudeInBase(q, { x: 0.4, y: -0.5, z: 0.6 }, [0, 0, 0]);
  near([a.roll, a.pitch, a.yaw], rpyFromQuat(q), 1e-12, "rpy");
  near(a.gyro, [0.4, -0.5, 0.6], 1e-12, "gyro");
});

test("the yawed mount comes out of the attitude", () => {
  const robot = [3 * DEG, 10 * DEG, 30 * DEG];
  const qWs = quatMul(quatFromRpy(...robot), quatFromRpy(...MOUNT));   // site * mount, like the sim
  const a = attitudeInBase(qWs, { x: 0, y: 0, z: 0 }, MOUNT);
  near([a.roll, a.pitch, a.yaw], robot, 1e-6, "rpy");
  // Read raw, the robot's pitch shows up as a roll: what the page used to draw.
  const raw = rpyFromQuat(qWs);
  assert.ok(Math.abs(raw[0] - -robot[1]) < 0.5 * DEG, `raw roll ${raw[0]} should be minus the pitch`);
});

test("the yawed mount comes out of the turn rates", () => {
  const robot = [0.1, 0.2, 0.3];
  const wS = rotate(quatConj(quatFromRpy(...MOUNT)), robot);   // into the chip's axes, like the sim
  near(wS, [-0.2, 0.1, 0.3], 1e-6, "chip rates");
  const a = attitudeInBase({ x: 0, y: 0, z: 0, w: 1 }, { x: wS[0], y: wS[1], z: wS[2] }, MOUNT);
  near(a.gyro, robot, 1e-9, "gyro");
});

test("the driver's all-zero quaternion still reads level", () => {
  const a = attitudeInBase({ x: 0, y: 0, z: 0, w: 0 }, { x: 0, y: 0, z: 0 }, MOUNT);
  near([a.roll, a.pitch, a.yaw], [0, 0, 0], 0, "rpy");
  assert.ok(a.gyro.every(Number.isFinite), "gyro is finite");
});

test("takes the bridge's decoded message shape", () => {
  // The bridge hands over plain objects with more fields than the maths
  // needs; an [x, y, z] array for the rates works as well.
  const orientation = { x: 0, y: 0, z: 0, w: 1, extra: "ignored" };
  const fromObject = attitudeInBase(orientation, { x: 0.1, y: 0.2, z: 0.3 }, MOUNT);
  const fromArray = attitudeInBase(orientation, [0.1, 0.2, 0.3], MOUNT);
  near(fromObject.gyro, fromArray.gyro, 0, "gyro");
  near(fromObject.gyro, [0.2, -0.1, 0.3], 1e-6, "gyro");
  for (const k of ["roll", "pitch", "yaw"]) assert.equal(typeof fromObject[k], "number", k);
});
