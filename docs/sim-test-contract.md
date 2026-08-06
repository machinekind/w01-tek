# Kontrakt testowy symulacji

Co znaczy „sprawdzone w symulacji" i czego symulacja **nie** sprawdza. Cel:
run na robocie ma być formalnością, a nie pierwszym uruchomieniem kodu. Bez
tej listy nie ma kryterium „wierne dość" i praca nad symulacją nigdy się nie
kończy.

Uruchomienie (w kontenerze — wszystko ROS-owe na PC działa w kontenerze):

```bash
./ros/dev.sh
ros2 launch wojtek_pc sim.launch.py hw:=mock rviz:=false     # logika stacku
ros2 launch wojtek_pc sim.launch.py hw:=mujoco               # + fizyka
```

## A. Co musi przejść przed każdym runem na robocie

Kolumna „plant" mówi, czy wystarczy `hw:=mock` (szybko, bez fizyki), czy
potrzebny jest `hw:=mujoco`.

| # | Scenariusz | Kryterium | Plant |
|---|---|---|---|
| 1 | Bring-up | 4 kontrolery `active` (`joint_state_broadcaster`, `imu_sensor_broadcaster`, `magnetometer_broadcaster`, `forward_position_controller`), komponent `WojtekSim` aktywny, zero ERROR w logu | mock |
| 2 | Kolejność spawnów | spawner nie wywala się na braku interfejsu; broadcastery łapią sensor `imu` | mock |
| 3 | Stan po starcie | `real_io_node` zgłasza DISARMED, `/forward_position_controller/commands` **milczy** | mock |
| 4 | `zero` | serwis zwraca `success`, korekta ≈ 0 gdy robot jest w `boot_pose` | mock |
| 5 | `stand_up` | rampa startuje, po niej pozycja = home | mock |
| 6 | `arm` | po `stand_up` przechodzi; komendy zaczynają płynąć | mock |
| 7 | Odmowa: `arm` w trakcie rampy | `success=False`, „ramp running" | mock |
| 8 | Odmowa: `lie_down` gdy ARMED | `success=False`, „armed" | mock |
| 9 | Odmowa: `arm` przy skoku > `max_arm_jump_rad` | `success=False` (ustaw pozę daleko od home i spróbuj) | mujoco |
| 10 | `disarm` → `lie_down` | rampa do folded, komendy przestają się odświeżać | mock |
| 11 | Watchdog `policy_node` | po ucięciu `/cmd_vel` polityka wchodzi w stan bezpieczny w `watchdog_timeout_s` | mock |
| 12 | `dry_run:=true` | stany lecą, moment nie jest aplikowany | mujoco |
| 13 | Rate'y | `/joint_states` i `/imu_sensor_broadcaster/imu` na oczekiwanej częstotliwości (patrz uwaga niżej) | mock |
| 14 | Bag | `bag:=true` tworzy `bag_dir/run_<stamp>` i nagrywa | mock |
| 15 | Pad / teleop | `/cmd_vel` z pada i z `teleop_twist_keyboard` rusza polityką | mock |
| 16 | `text_commander` | `/wojtek/nav_command` → `/cmd_vel`, dead-man po 2 s | mock |
| 17 | Chód | robot stoi po `stand_up` i idzie na komendę bez upadku | mujoco |
| 18 | `boot_pose:=folded` | pełna sekwencja folded → zero → stand_up → arm | mujoco |
| 19 | Kamera | `/camera/camera/depth/*` i `/color/*` publikują, `cloud_reduce` zwraca siatkę 8x8 | mujoco |

## B. Czego symulacja NIE pokryje

„Przeszło w sim" **nie** znaczy „przejdzie na robocie". Te rzeczy weryfikuje
się wyłącznie na sprzęcie (`robot.launch.py dry_run:=true`, potem jazda):

- **candle/CAN**: losowe timeouty MD80, flakujący zapis do flasha, terminacja
  magistrali, baudrate napędów.
- **Real-time na RPi**: jitter pętli 400 Hz przy kernelu RT, affinity/izolacja
  rdzeni, throttling termiczny SoC.
- **Sieć**: DDS po wifi, zerwania linku, discovery między maszynami.
- **Fizyka, której model nie ma**: tarcie i luz w przekładniach, ripple
  momentu, podatność konstrukcji, poślizg stopy na realnej podłodze.
- **Czujniki**: hard-iron magnetometru od magnesów w napędach, bias żyra,
  kwantyzacja enkoderów, opóźnienia I2C/CAN. (Wchodzą do symulacji jako
  parametry w kolejnym etapie — do tego czasu są poza kontraktem.)

## C. Uwagi z pierwszego przejścia (2026-08-06)

- Punkty 1-8, 10 i 13 przeszły na `hw:=mock` przy pierwszym uruchomieniu.
- **Rate'y (13) wymagają decyzji, nie tylko odczytu.** `real_controllers.yaml`
  ustawia `publish_rate: 200.0` dla `joint_state_broadcaster` i `100.0` dla
  `imu_sensor_broadcaster`, ale w Jazzy tempo publikacji bierze się z
  `update_rate` kontrolera — `publish_rate` jest ignorowany. Zmierzone: oba
  publikują **400 Hz** (`magnetometer_broadcaster`, który ma `update_rate:
  100`, faktycznie chodzi 100 Hz). Czyli robot też publikuje 2× i 4× częściej
  niż zakładano, co obciąża DDS na RPi. Poprawka to zmiana klucza w yamlu,
  ale zmienia zachowanie robota — do świadomej decyzji.
- Kontener nie ma `huggingface_hub`, więc bring-up pada na starcie
  (`No module named 'huggingface_hub'`), a domyślna referencja polityki w
  `launch_common.py` wskazuje nieistniejącą rewizję. Do symulacji trzeba dziś
  podać `policy:=<HF_ORGANIZATION>/wojtek-stiff-height-locomotion@2f385eb...`.
  To ten sam otwarty problem, który blokuje autostart na robocie.
