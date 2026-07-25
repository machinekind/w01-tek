# Architektura ROS stacka (`ros/src`)

Stan na 2026-07-18 (branch `jakuc/CANdle_hat+BMI_9DOF`).

Stack dzieli się na trzy warstwy:

* **warstwa sprzętowa** — pluginy ros2_control (C++), działają na RPi wewnątrz `ros2_control_node`,
* **warstwa sterowania** — Python: polityka RL + adapter I/O (RPi),
* **warstwa PC** — symulacja MuJoCo, RViz/PlotJuggler, konsola operatora.

Wspólnym językiem między nodami są **absolutne kąty URDF** na topicach `wojtek/*`.

## Przepływ danych — realny robot

```mermaid
flowchart LR
    subgraph RPI["Raspberry Pi — stack RT (taskset, rdzenie izolowane)"]
        subgraph R2C["ros2_control_node @ 400 Hz"]
            MD80["md80_hardware_interface\n12× MD80, CAN 8M przez CANdle HAT (SPI)"]
            JSB["joint_state_broadcaster\n200 Hz"]
            FPC["forward_position_controller\nJointGroupPositionController"]
            IMUB["imu_sensor_broadcaster\n(use_imu:=false — wyłączony)"]
        end
        RIO["real_io_node\n(offset boot↔abs, arm/zero/rampy)"]
        POL["policy_node\n50 Hz, MLP numpy"]
        RSP["robot_state_publisher"]
        BAG["rosbag (serwis: cały run)"]
    end
    subgraph PC["PC — kontener dev (viz.launch.py)"]
        RVIZ["RViz"]
        CON["operator console (PyQt5)"]
    end

    MD80 --> JSB
    JSB -- "/joint_states (boot-relative)" --> RIO
    RIO -- "/wojtek/joint_states_abs (abs URDF)" --> POL
    RIO -- "/wojtek/joint_states_abs" --> RSP
    IMUB -. "/imu_sensor_broadcaster/imu" .-> POL
    POL -- "/wojtek/joint_targets (abs URDF)" --> RIO
    RIO -- "/forward_position_controller/commands\n(Float64MultiArray, boot-relative)" --> FPC
    FPC --> MD80
    CON -- "/cmd_vel (Twist)" --> POL
    CON -- "usługi /wojtek/*" --> RIO
    RSP -- "/tf, /robot_description" --> RVIZ
    RIO -- "/wojtek/joint_states_abs" --> CON
```

## Przepływ danych — symulacja (`sim.launch.py`)

`mujoco_sim_node` zastępuje cały lewy słupek (hardware + broadcastery + real_io_node).
`policy_node` jest **identyczny** — różnią się tylko remapy i parametry.

```mermaid
flowchart LR
    SIM["mujoco_sim_node\nscene_mjx.xml (fizyka treningowa)\nkp=20 kd=1, ±6 Nm, dt=0.004"]
    POL["policy_node @ 50 Hz"]
    RSP["robot_state_publisher"]
    RVIZ["RViz"]
    TEL["teleop (Twist)"]

    SIM -- "/joint_states (aktuowane + pasywne)" --> POL
    SIM -- "/imu/data (ground truth)" --> POL
    POL -- "/wojtek/joint_targets" --> SIM
    TEL -- "/cmd_vel" --> POL
    SIM -- "TF odom→base_link" --> RVIZ
    SIM -- "/joint_states" --> RSP
    RSP --> RVIZ
```

---

## 1. `wojtek_policy` — mózg (Python, ament_python)

Runtime wytrenowanej polityki RL + node ROS. Działa na RPi i w symulacji.

**Pliki:**

* `wojtek_policy/policy.py` — `WojtekPolicy`: czysty numpy runtime (bez importów ROS, testowalny unit-testami). Ładuje `policy.npz` (wagi MLP z SiLU + tanh) i `policy_meta.json`; składa wektor obserwacji wg `obs_layout` z meta (dzięki temu jeden runtime obsługuje rodziny polityk z IMU i bez — "springy" jest proprioceptywna), liczy `targets = clip(anchor + tanh_mlp(obs)·action_scale, target_low, target_high)`. Ma clampy bezpieczeństwa na kolano (singularność czworoboku — snap-through może złamać mechanizm) i abdukcję.
* `wojtek_policy/policy_node.py` — node ROS (opis niżej).
* `wojtek_policy/joint_map.py` — `JointMap`: afiniczne mapowanie konwencji MuJoCo↔URDF (`q_urdf = sign·q_mjc + offset`), wczytywane z `config/joint_map.yaml`.
* `wojtek_policy/poses.py` — nazwane pozy współdzielone przez real i sim: `FOLDED_KNEE_RAD = 0.425` (poza boot/zerowania), `folded_ctrl()`, oraz `PASSIVE_FROM_KNEE` — wielomiany 8. stopnia dające kąty pasywnych przegubów czworoboku (fourth/fifth) z kąta kolana (dopasowane do domknięcia MJCF, błąd ≤ 7e-5 rad).
* `config/` — `policy.npz`, `policy_meta.json`, `joint_map.yaml`; `test/test_policy.py`.

**Node: `policy_node`** (nazwa: `wojtek_policy`) — pętla 50 Hz (z `ctrl_dt` w meta): sensory → cele pozycyjne stawów.

| Kierunek | Interfejs | Typ |
|---|---|---|
| sub | `/joint_states` (absolutne URDF) | `sensor_msgs/JointState` |
| sub | `/imu/data` | `sensor_msgs/Imu` |
| sub | `/cmd_vel` (przycinany do boxa komend z treningu; `linear.z > 0` = komenda wysokości stania dla polityk z komendą 4-D) | `geometry_msgs/Twist` |
| pub | `/wojtek/joint_targets` (absolutne URDF, 12 stawów) | `sensor_msgs/JointState` |
| srv | `/wojtek/enable` | `std_srvs/SetBool` |
| srv | `/wojtek/reset` | `std_srvs/Trigger` |

Parametry: `policy_dir`, `joint_map_yaml`, `imu_mount_rpy` (rotacja IMU→base_link; real: `[0, π, 0]`, sim: zera), `clamp_knee`, `auto_enable`, `soft_start_s` (blend od pozy mierzonej do wyjścia polityki), `watchdog_timeout_s` (0.2 s — przy nieświeżych danych **wstrzymuje publikację**; MD80 trzyma ostatni cel), `gravity_from_accel` (filtr komplementarny zamiast kwaternionu IMU, dla IMU bez fuzji).

Zabezpieczenia: watchdog świeżości per-sensor (polityka bez IMU przeżywa dropout IMU), odrzucanie NaN w wejściach (NaN zatruwałby stan przez pętlę `last_action`), reset stanu polityki na krawędzi hold→run.

## 2. `wojtek_bringup` — strona robota (Python, ament_python)

Bring-up realnego robota: ros2_control + adapter + procedura operacyjna.

**Pliki:**

* `launch/robot.launch.py` — **kanoniczny launch na RPi** (bez GUI). Startuje: `ros2_control_node` (URDF z `wojtek_real.urdf.xacro` + `config/real_controllers.yaml`), spawnery kontrolerów, `robot_state_publisher` (remap na `/wojtek/joint_states_abs` — RViz dostaje kąty absolutne), statyczny TF `odom→base_link` (brak odometrii na realu), `real_io_node`, `policy_node` (z remapami `joint_states→wojtek/joint_states_abs`, `imu/data→imu_sensor_broadcaster/imu`), opcjonalnie rosbag.
  Argumenty: `max_torque` (dom. 6.0 — tyle co w treningu), `bus` (dom. `spi` = CANdle HAT), `can_baud` (dom. `8` — napędy przeflashowane na 8M od 2026-07-17), `use_imu` (dom. `false` — szeregowy IMU wycofany na rzecz I2C Adafruit 5543, jeszcze nie zintegrowanego), `imu_port`, `dry_run`, `boot_pose` (`home`/`folded`), `bag`, `bag_dir`, `bag_cpus` (affinity nagrywarki poza rdzeniami RT).
* `launch/real.launch.py` — starsza wersja tego samego (z ery IMU BMI160 po USB); `robot.launch.py` go zastąpił.
* `config/real_controllers.yaml` — controller_manager @ **400 Hz**; `joint_state_broadcaster` (200 Hz), `imu_sensor_broadcaster` (100 Hz, sensor `imu`, frame `imu_link`), `forward_position_controller` (`JointGroupPositionController`) z listą 12 stawów **w kolejności aktuatorów polityki** (rear_left, rear_right, front_right, front_left × first/second/third).
* `urdf/wojtek_real.urdf.xacro` — składa opis robota z `wojtek_description` + ros2_control; `urdf/wojtek_ros2_control.urdf.xacro` — makra: `wojtek_ros2_control` (12 stawów MD80, per staw: `can_id` (10-12/20-22/30-32/40-42), `max_torque`, gainy impedancji `ddq_kp=20, ddq_kd=1` — dokładnie model aktuatora z treningu) oraz makra IMU (BMX160/BMI160 — obecnie nieużywane).
* `wojtek_bringup/robot.py` — **CLI `ros2 run wojtek_bringup robot`**: jednokomendowy bring-up z kontenera na PC. Odpala stack na RPi przez SSH (domyślnie: systemd `wojtek-robot.service` z RT-pinningiem na izolowane rdzenie; `--dry-run` = launch bez RT; `--sim` = MuJoCo lokalnie), plus lokalnie viz i konsolę operatora; Ctrl-C składa wszystko. Wstrzykuje `ROS_DOMAIN_ID=42` + CycloneDDS do zdalnej sesji (bez tego RPi ląduje na domenie 0 i PC nie widzi topiców).

**Node: `real_io_node`** (nazwa: `wojtek_real_io`) — adapter między polityką a ros2_control. Powód istnienia: MD80 raportuje pozycje **względem pozy aktywacji**, a polityka pracuje w **absolutnych kątach URDF** — node przesuwa o offset w obie strony i jest bramką bezpieczeństwa.

| Kierunek | Interfejs | Typ |
|---|---|---|
| sub | `/joint_states` (boot-relative) | `sensor_msgs/JointState` |
| sub | `/wojtek/joint_targets` (absolutne) | `sensor_msgs/JointState` |
| pub | `/wojtek/joint_states_abs` (12 stawów + osobna wiadomość z pasywnymi przegubami czworoboku z `PASSIVE_FROM_KNEE` — bez tego RViz rysuje "połamane" nogi) | `sensor_msgs/JointState` |
| pub | `/forward_position_controller/commands` (boot-relative) | `std_msgs/Float64MultiArray` |
| srv | `/wojtek/arm` — dopiero to przepuszcza komendy do silników | `std_srvs/SetBool` |
| srv | `/wojtek/stand_up`, `/wojtek/lie_down` — wolne rampy cosinusowe do pozy home/folded | `std_srvs/Trigger` |
| srv | `/wojtek/zero` — "robot jest TERAZ fizycznie w pozie boot", przelicza offset | `std_srvs/Trigger` |

Parametry: `policy_dir`, `joint_map_yaml`, `max_arm_jump_rad` (0.15 — odmowa ARM, gdy robot daleko od home), `dry_run` (loguje komendy zamiast publikować), `boot_pose`, `folded_knee_rad`, `ramp_duration_s` (4.0), `ramp_rate_hz` (50).

Zabezpieczenia: startuje DISARMED; arm/rampa/zero wzajemnie się wykluczają; filtr non-finite tuż przed napędami (ostatnia linia obrony — topic komend jest otwarty dla każdego).

## 3. `wojtek_viz` — strona PC (Python, ament_python)

Symulacja, wizualizacja i ręczne sterowanie. Nie buduje się na RPi.

**Pliki:** `launch/sim.launch.py`, `launch/viz.launch.py`, `config/{scene_mjx,wojtek_mjx}.xml` (kopie modeli MJX), `urdf/wojtek_sim.urdf.xacro`.

**Node: `mujoco_sim_node`** (nazwa: `wojtek_mujoco_sim`) — symulator real-time zamykający pętlę z policy_node na dokładnie tej fizyce, na której trenowano (`scene_mjx.xml`: serwa kp=20/kd=1, forcerange ±6, dt=0.004), krokowany do zegara ściennego (z limitem kroków — brak spirali śmierci przy lagu).

| Kierunek | Interfejs | Typ |
|---|---|---|
| sub | `/wojtek/joint_targets` | `sensor_msgs/JointState` |
| pub | `/joint_states` (aktuowane + pasywne, z efforts) | `sensor_msgs/JointState` |
| pub | `/imu/data` (ground truth z sensorów MuJoCo, frame base_link) | `sensor_msgs/Imu` |
| pub | TF `odom→base_link` (ground-truth poza bazy) | tf2 |
| pub | `/odom_vel` (debug) | `geometry_msgs/Twist` |
| srv | `/sim/reset` | `std_srvs/Trigger` |

Parametry: `model_xml` (puste = przygotuj MJX z share z przepisaniem meshdir), `joint_map_yaml`, `publish_rate_hz` (100), `realtime_factor`, `initial_pose` (`home`/`folded` — folded uzyskiwane przez fizyczne "osiadanie" z home, żeby domknięcie czworoboku było spójne), `folded_knee_rad`.

**Node: `console`** (`operator_console.py`, nazwa: `operator_console`) — GUI PyQt5, jedno okno na wszystkie ręczne operacje (zamiast `ros2 service call`). Cztery moduły:

* **M1** — przyciski serwisów `/wojtek/{arm,zero,stand_up,lie_down,enable,reset}`,
* **M2** — telemetria na żywo z `/wojtek/joint_states_abs` i `/imu/data`,
* **M3** — jog per-staw: slidery publikują `/wojtek/joint_targets` z własną rampą 0.8 rad/s (wymaga: armed + policy DISABLED),
* **M4** — pad XY (vx, yaw) + slider strafe + slider wysokości → `/cmd_vel` @ 20 Hz (wysokość na `linear.z`), puszczenie pada/strafe = stop, wysokość trzyma nastawę (wymaga: armed + policy ENABLED).

Nie ma topicu statusu arm/enable — konsola śledzi stan lokalnie z odpowiedzi serwisów. rclpy spinuje w wątku tła, do GUI przez sygnały Qt.

**Launche:**

* `sim.launch.py` — robot_state_publisher + mujoco_sim_node + policy_node (imu_mount_rpy=0, soft_start 0.5 s, bez clamp_knee) + RViz. Argumenty: `rviz`, `initial_pose`.
* `viz.launch.py` — czyste PC-side dla żywego robota: RViz (czyta `/robot_description` i `/tf` z RPi po DDS), opcjonalnie PlotJuggler, oraz rosbag całego runu **na żądanie** (`bag:=true`, do `~/wojtek_bags/run_<timestamp>`; domyślnie wyłączony). Zero hardware'u, zero RSP.

## 4. `md80_hardware_interface` — napędy (C++, plugin ros2_control)

**Nie jest nodem** — to plugin `SystemInterface` ładowany przez `ros2_control_node` (`md80_hardware_interface/MD80HardwareInterface`). Gada z 12 napędami MD80 (AK80-9) przez CANdle.

* Parametry hardware (z URDF): `bus` (`spi`|`usb`; SPI = CANdle HAT, domyślne urządzenie `/dev/spidev0.0`), `can_baud` (1|2|5|8 Mbps — **musi** zgadzać się z flashem napędów; teraz 8M), `usb_port`, `spi_device`. Per staw: `can_id`, `max_torque`, gainy impedancji `ddq_kp`/`ddq_kd` (parametry `q_*`/`dq_*` wymagane przez parser, nieużywane w IMPEDANCE).
* Eksportuje: stany position/velocity/effort, komendę position (→ tryb IMPEDANCE: `tau = kp·(q_t − q) − kd·dq`, clamp do `max_torque`; velocity→VELOCITY_PID, effort→RAW_TORQUE też wspierane).
* Ważne poprawki forka: retry `addMd80()` ×3 (na USB ~5% transakcji flakowało i co drugi launch padał) oraz przesunięcie komend w `write()` o offset aktywacji, żeby read i write były w jednej ramce (bez tego `/wojtek/zero` nie byłby dokładny).
* `scripts/flash_motors.py` — jednorazowe `mdtool setup` konfiguracji AK80-9 na wszystkie 12 ID.

## 5. `bmi160_serial_hardware_interface` / `bmx160_serial_hardware_interface` — IMU (C++, pluginy ros2_control)

Dwa bliźniacze pluginy `SensorInterface` czytające IMU przez mostek serial (MCU z firmware Mahony w `firmware/`): BMX160 = pełny 9-DoF z magnetometrem, BMI160 = "loaner" bez magnetometru. Parametry: `port`, `baud_rate` (115200; naprawiony bug `cfsetospeed`). Eksportują `orientation.*`, `angular_velocity.*`, `linear_acceleration.*` (BMX też `magnetometer.*`); odczyt z deadline'em — brak danych degraduje do stale zamiast NaN.

**Status: de facto martwe** — sprzęt odłączony (`use_imu:=false` domyślnie), zastępuje je IMU I2C (Adafruit 5543: LSM6DS3TR-C + LIS3MDL), którego bench-testy w `ros/hw_tests/imu_i2c` przeszły, ale paczka ROS jeszcze nie istnieje (w backlogu: nowa paczka + skasowanie obu serialowych).

## 6. `wojtek_description` — model robota (CMake, tylko dane)

Bez nodów. Źródło prawdy o geometrii:

* `urdf/` — modularne xacro: `body`, `leg`, `inertia`, `wojtek.urdf.xacro` — z pasywnymi przegubami fourth/fifth **bez** domknięcia czworoboku (URDF go nie umie),
* `mujoco/` — `wojtek.xml`, `wojtek_mjx.xml` + sceny; tu domknięcie istnieje jako equality connect; wersje MJX to te treningowe,
* meshe oraz historyczne launche (`bringup`, `simulation` — Gazebo, nieużywane w obecnym flow).

---

## Rzeczy nieoczywiste, które warto wiedzieć

1. **Trzy układy współrzędnych stawów**: MuJoCo/policy (kolejność aktuatorów, konwencja treningu) ↔ URDF absolutne (wspólny język topiców `wojtek/*`) ↔ boot-relative (to, co widzi/przyjmuje MD80). Konwersje: `JointMap` (policy_node, sim) i `_offset_urdf` (real_io_node).
2. **Podwójna bramka bezpieczeństwa**: `/wojtek/enable` (czy polityka liczy) jest niezależne od `/wojtek/arm` (czy komendy idą do silników). Na realu `auto_enable:=true`, bo bramką jest arm.
3. **Kolejność stawów w `forward_position_controller` musi się zgadzać** z `policy_meta.json` — real_io_node publikuje goły `Float64MultiArray` bez nazw.
4. **RViz na realu** karmiony jest z `/wojtek/joint_states_abs` (remap RSP), nie z surowego `/joint_states` — i to real_io_node dolicza pasywne przeguby czworoboku.
5. **Bagi**: realne jazdy nagrywa robot — serwis przekazuje `bag:=true bag_cpus:=0,1` (bezstratnie, localhost); PC nagrywa na żądanie (`viz.launch.py bag:=true` po DDS, `real.launch.py` domyślnie). Rotacji na razie brak — przed długą jazdą sprawdź wolne miejsce na karcie RPi.
