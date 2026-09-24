"""Pomocniki kursu `wojtek_rl_guide.ipynb`.

Wszystko, co w notebooku powtarzałoby się między krokami: przebieg polityki
w czystym MuJoCo (ta sama pętla, którą robot wykonuje na sprzęcie), widok z
uproszczonymi siatkami, podsumowanie przebiegu, uruchomienie treningu i
eksportu jako podprocesów. Bez JAX-a poza treningiem i eksportem.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np

from lowpoly import render_model
from wojtek_rl import paths
from wojtek_rl.np_policy import actuator_addresses, gravity_from_quat

TRAINING = paths.PROJECT_DIR
UPADEK_WYSOKOSC, UPADEK_GZ = 0.06, -0.4   # test upadku jak w środowisku (task.env.fall)
PRESET = "course_locomotion"


def konfiguracja(preset: str = PRESET):
    """Konfiguracja Hydry przepisu, złożona jak w `wojtek_rl.train` (bez treningu)."""
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=str(TRAINING / "wojtek_rl" / "conf"), version_base=None):
        return compose(config_name="config", overrides=[f"+experiment={preset}"])


def srodowisko(preset: str = PRESET):
    """Pełna konfiguracja środowiska przepisu: domyślne wartości z env.py plus nakładka z YAML.

    Tak samo składa ją `wojtek_rl.train`. Import środowiska ciągnie JAX; kernel
    notebooka trzyma JAX na CPU (JAX_PLATFORMS), żeby nie zajmować pamięci karty
    treningowi w podprocesie (`trenuj` zdejmuje tę zmienną).
    """
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    if "jax" in sys.modules:                      # JAX czyta JAX_PLATFORMS tylko przy imporcie
        import jax
        from jax._src import xla_bridge

        if xla_bridge._backends:
            raise RuntimeError("JAX w kernelu już zajął kartę (operacja JAX przed importem kursu): "
                               "Runtime → Restart session i uruchom komórki od początku")
        jax.config.update("jax_platforms", "cpu")
    from omegaconf import OmegaConf

    from wojtek_rl.registry import TASKS, _apply_overrides

    h = konfiguracja(preset)
    cfg = TASKS[h.task.name][1]()
    _apply_overrides(cfg, OmegaConf.to_container(h.task.env, resolve=True) or {})
    return cfg


def ustawienia(preset: str = PRESET) -> dict:
    """Serwo i mapowanie akcji przepisu, tak jak środowisko nakłada je na model.

    `serwo`: kp/kd/max_torque (0 w przepisie = wartości z XML); `skala`: 12 skal
    akcji [rad]; `dol`/`gora`: limity celów po ograniczeniu odwodzenia i kolana.
    """
    env = srodowisko(preset)
    m = mujoco.MjModel.from_xml_path(str(paths.SCENE_XML))
    serwo = {"kp": float(env.pd_kp or m.actuator_gainprm[0, 0]),
             "kd": float(env.pd_kd or -m.actuator_biasprm[0, 2]),
             "max_torque": float(env.max_torque or m.actuator_forcerange[0, 1])}
    skala = np.asarray(env.action_scale, np.float32)
    skala = np.tile(skala, 4) if skala.ndim else np.full(12, float(skala), np.float32)
    dol, gora = m.actuator_ctrlrange.T.copy()
    if env.abduction_ctrl_limit:
        lim = float(env.abduction_ctrl_limit)
        dol[0::3], gora[0::3] = np.maximum(dol[0::3], -lim), np.minimum(gora[0::3], lim)
    if env.knee_target_max:
        gora[2::3] = np.minimum(gora[2::3], float(env.knee_target_max))
    return {"serwo": serwo, "skala": skala, "dol": dol, "gora": gora}


try:
    SERWO = ustawienia()["serwo"]           # serwo PD przepisu PRESET
except Exception:                           # noqa: BLE001  (np. brak Hydry): wartości z XML
    _m = mujoco.MjModel.from_xml_path(str(paths.SCENE_XML))
    SERWO = {"kp": float(_m.actuator_gainprm[0, 0]), "kd": float(-_m.actuator_biasprm[0, 2]),
             "max_torque": float(_m.actuator_forcerange[0, 1])}


# -- widok ---------------------------------------------------------------------

class Widok:
    """Renderer na kopii sceny z uproszczonymi siatkami (Colab rysuje programowo)."""

    def __init__(self, cache: Path, height: int = 360, width: int = 480):
        self.rmodel = render_model(paths.SCENE_XML, cache)
        self.rdata = mujoco.MjData(self.rmodel)
        self.renderer = mujoco.Renderer(self.rmodel, height=height, width=width)

    def klatka(self, data: mujoco.MjData, camera: str = "track") -> np.ndarray:
        self.rdata.qpos[:] = data.qpos
        mujoco.mj_forward(self.rmodel, self.rdata)
        self.renderer.update_scene(self.rdata, camera=camera)
        return self.renderer.render().copy()


# -- symulacja -----------------------------------------------------------------

def model_z_serwem(pd: dict | None = None) -> mujoco.MjModel:
    """Scena treningowa z serwem PD z kontraktu polityki (kp, kd, max_torque)."""
    m = mujoco.MjModel.from_xml_path(str(paths.SCENE_XML))
    if pd:
        m.actuator_gainprm[:, 0] = pd["kp"]
        m.actuator_biasprm[:, 1] = -pd["kp"]
        m.actuator_biasprm[:, 2] = -pd["kd"]
        m.actuator_forcerange[:, 0] = -pd["max_torque"]
        m.actuator_forcerange[:, 1] = pd["max_torque"]
    return m


def przebieg(polityka, komenda, sekundy: float = 4.0, widok: Widok | None = None,
             pchniecie_s: float | None = None, pchniecie_ms: float = 0.6, seed: int = 0,
             opoznienie: int = 1) -> dict:
    """Polityka steruje robotem w czystym MuJoCo, jak na sprzęcie.

    `polityka`: obiekt z `step(gyro, grawitacja, q, dq, komenda) -> 12 celów`,
    `reset()`, `ctrl_dt`, `meta` (WojtekPolicy z eksportu albo własna klasa).
    `komenda`: (vx, vy, wz) lub funkcja krok -> (vx, vy, wz). Czwarty element,
    wysokość stania, dopełnia kontrakt polityki. `opoznienie`: o tyle kroków
    sterowania później cele trafiają do serw (1 = jak w treningu, task.env.action_delay).
    Zwraca słownik z przebiegiem (czas, komenda, prędkość w układzie robota,
    obrót, wysokość, momenty, prędkości przegubów), klatkami i krokiem upadku.
    """
    m = model_z_serwem(polityka.meta.get("pd"))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
    mujoco.mj_forward(m, d)
    qadr, vadr = actuator_addresses(m)
    substeps = round(polityka.ctrl_dt / m.opt.timestep)
    gyro_adr = m.sensor("angular-velocity").adr[0]
    tau_ff = bool(getattr(polityka, "tau_ff_enabled", False))
    rng = np.random.default_rng(seed)
    polityka.reset()

    n = int(round(sekundy / polityka.ctrl_dt))
    log = {k: [] for k in ("t", "komenda", "v", "wz", "h", "moment", "dq", "xy")}
    klatki, upadek = [], None
    qinv, v_body = np.zeros(4), np.zeros(3)
    kolejka = [(d.ctrl.copy(), np.zeros(12, np.float32))] * opoznienie   # cele z poprzednich kroków (home na start)
    for i in range(n):
        cmd = np.asarray(komenda(i) if callable(komenda) else komenda, np.float32)
        gyro = d.sensordata[gyro_adr:gyro_adr + 3].copy()
        graw = gravity_from_quat(*d.qpos[3:7])
        cele = np.asarray(polityka.step(gyro, graw, d.qpos[qadr].copy(), d.qvel[vadr].copy(), cmd), np.float32)
        tau = polityka.last_tau_ff.copy() if tau_ff else np.zeros(12, np.float32)
        kolejka.append((cele, tau))
        cele, tau = kolejka.pop(0)
        d.ctrl[:] = cele
        if tau_ff:
            d.qfrc_applied[:] = 0.0
            d.qfrc_applied[vadr] = tau
        if pchniecie_s is not None and i == int(pchniecie_s / polityka.ctrl_dt):
            kierunek = rng.uniform(-1, 1, 2)
            d.qvel[:2] += kierunek / (np.linalg.norm(kierunek) + 1e-6) * pchniecie_ms
        for _ in range(substeps):
            mujoco.mj_step(m, d)
        mujoco.mju_negQuat(qinv, d.qpos[3:7])
        mujoco.mju_rotVecQuat(v_body, d.qvel[:3], qinv)
        log["t"].append(i * polityka.ctrl_dt)
        log["komenda"].append(cmd[:3])
        log["v"].append(v_body.copy())
        log["wz"].append(d.sensordata[gyro_adr + 2])
        log["h"].append(d.qpos[2])
        log["moment"].append(d.actuator_force.copy())
        log["dq"].append(d.qvel[vadr].copy())
        log["xy"].append(d.qpos[:2].copy())
        if widok is not None and i % 2 == 0:
            klatki.append(widok.klatka(d))
        if d.qpos[2] < UPADEK_WYSOKOSC or gravity_from_quat(*d.qpos[3:7])[2] > UPADEK_GZ:
            upadek = i
            break
    out = {k: np.asarray(v) for k, v in log.items()}
    out["klatki"], out["upadek"], out["dt"] = klatki, upadek, polityka.ctrl_dt
    return out


def podsumuj(p: dict) -> dict:
    """Liczby, którymi porównuje się przebiegi (jak w baterii testów treningu)."""
    cmd, v, wz, dt = p["komenda"], p["v"], p["wz"], p["dt"]
    ruch = np.abs(cmd).max(axis=1) > 0.05
    blad_v = np.hypot(v[:, 0] - cmd[:, 0], v[:, 1] - cmd[:, 1])
    dq = p["dq"] - p["dq"].mean(axis=0, keepdims=True)
    moc = np.abs(np.fft.rfft(dq, axis=0)) ** 2
    f = np.fft.rfftfreq(dq.shape[0], d=dt)
    xy = p["xy"]
    return {
        "przebyte [m]": float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1))),   # długość drogi, nie przesunięcie
        "obrót łącznie [rad]": float(np.sum(wz) * dt),
        "błąd prędkości RMS [m/s]": float(np.sqrt(np.mean(blad_v[ruch] ** 2))) if ruch.any() else float("nan"),
        "błąd obrotu RMS [rad/s]": float(np.sqrt(np.mean((wz[ruch] - cmd[ruch, 2]) ** 2))) if ruch.any() else float("nan"),
        "wysokość [m]": float(p["h"].mean()),
        "moment p90 [N·m]": float(np.percentile(np.abs(p["moment"]), 90)),
        "drgania >5 Hz": float(moc[f > 5.0].sum() / max(moc[f > 0].sum(), 1e-12)),
        "upadek": "nie" if p["upadek"] is None else f"{p['upadek'] * dt:.1f} s",
    }


def tabela(przebiegi: dict) -> str:
    """Jedna tabela tekstowa: kolumna na przebieg."""
    if not przebiegi:
        raise ValueError("brak przebiegów")
    nazwy = list(przebiegi)
    pods = {n: podsumuj(p) for n, p in przebiegi.items()}
    szer = max(len(k) for k in next(iter(pods.values())))
    kol = {n: max(14, len(n) + 2) for n in nazwy}          # kolumna mieści nazwę przebiegu
    wiersze = [" " * szer + "".join(f"{n:>{kol[n]}s}" for n in nazwy)]
    for k in next(iter(pods.values())):
        kom = "".join(f"{pods[n][k]:>{kol[n]}.3f}" if isinstance(pods[n][k], float) else f"{pods[n][k]:>{kol[n]}s}"
                      for n in nazwy)
        wiersze.append(f"{k:{szer}s}{kom}")
    return "\n".join(wiersze)


def obok_siebie(przebiegi: dict) -> list:
    """Klatki kilku przebiegów zszyte w jeden kadr; upadły zastyga na ostatniej."""
    bez = [n for n, p in przebiegi.items() if not p["klatki"]]
    if bez:
        raise ValueError(f"przebieg bez klatek (wywołaj przebieg(..., widok=widok)): {', '.join(bez)}")
    n = max(len(p["klatki"]) for p in przebiegi.values())
    return [np.hstack([p["klatki"][min(k, len(p["klatki"]) - 1)] for p in przebiegi.values()]) for k in range(n)]


# -- trening i eksport ---------------------------------------------------------

def status(run_name: str):
    """'complete', 'running' (trening trwa lub został przerwany) albo None, gdy runu nie ma."""
    f = TRAINING / "runs" / run_name / "run.json"
    return json.loads(f.read_text()).get("status") if f.exists() else None


def jest_gpu() -> bool:
    """Sprawdzenie bez JAX-a: import JAX-a w kernelu zająłby pamięć karty, którą trening potrzebuje w podprocesie."""
    if shutil.which("nvidia-smi") is None:
        return False
    return subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).returncode == 0


def trenuj(run_name: str, kroki: int, envs: int = 2048, preset: str = PRESET, seed: int = 0,
           dodatkowe: tuple = (), start: str | Path | None = None) -> Path:
    """Jeden trening PPO jako podproces; drukuje linie ewaluacji. Zwraca katalog runu.

    Dokończony trening o tej nazwie nie startuje ponownie; przerwany (zerwana sesja,
    Stop, błąd) jest kasowany i liczony od nowa. `dodatkowe` to nadpisania Hydry,
    np. ("++task.env.reward.scales.action_rate=-1.0",). `start` to katalog
    checkpointu, od którego trening rusza zamiast od losowych wag (dostrajanie).
    """
    run_dir = TRAINING / "runs" / run_name
    overrides = [f"+experiment={preset}", f"run_name={run_name}", f"seed={seed}",
                 f"++ppo.num_timesteps={kroki}", f"++ppo.num_envs={envs}", "wandb.enable=false", *dodatkowe]
    if start is not None:
        overrides.append(f"restore={Path(start).resolve()}")
    st = status(run_name)
    if st == "complete":
        zapis = run_dir / "trenuj.json"
        if zapis.exists() and json.loads(zapis.read_text()).get("overrides") != overrides:
            raise ValueError(f"trening {run_name} istnieje z inną konfiguracją; podaj nową nazwę "
                             f"albo skasuj {run_dir}")
        print("trening już jest:", run_dir)
        pokaz_krzywa(run_name)
        return run_dir
    if st is not None:
        print(f"poprzedni trening {run_name} nie dokończył się; liczę od nowa")
        shutil.rmtree(run_dir)
    if not jest_gpu():
        raise RuntimeError("brak GPU: Runtime → Change runtime type → T4 GPU, potem od kroku 0 "
                           "(trening na CPU nie ma sensu; bez GPU zostają kroki 0-3)")
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    env.pop("JAX_PLATFORMS", None)              # kernel trzyma JAX na CPU; trening ma dostać GPU
    proba = subprocess.run([sys.executable, "-c", "import jax; print(jax.default_backend())"],
                           env=env, capture_output=True, text=True)
    if proba.stdout.strip().splitlines()[-1:] != ["gpu"]:
        print(proba.stderr[-1500:])
        raise RuntimeError("JAX w podprocesie nie widzi GPU: Runtime → Restart session i uruchom od kroku 0")
    print("polecenie: python -m wojtek_rl.train " + " ".join(overrides))
    print("kompilacja trwa 1-3 min, pierwsza linia pojawi się po niej; potem jedna linia na ewaluację")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "trenuj.json").write_text(json.dumps({"overrides": overrides}))
    with open(run_dir / "train.log", "w") as log:
        proc = subprocess.Popen([sys.executable, "-m", "wojtek_rl.train", *overrides], cwd=TRAINING, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in proc.stdout:
                log.write(line)
                if line.startswith(("steps", "done")) or "Error" in line:
                    print(line, end="")
            proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            proc.wait()
            print("trening przerwany; kolejne trenuj() z tą nazwą zacznie od nowa")
            raise
    if proc.returncode:
        tail = (run_dir / "train.log").read_text().splitlines()[-12:]
        print("\n".join(tail))
        raise RuntimeError(f"trening {run_name} zakończył się błędem (kod {proc.returncode}); koniec train.log wyżej")
    return run_dir


def pokaz_krzywa(run_name: str) -> None:
    """Ewaluacje z train.log w jednej tabeli: kroki, nagroda z epizodu, długość epizodu."""
    rows = krzywa(run_name)
    if not rows:
        print("brak train.log dla", run_name)
        return
    print(f"{'kroki':>13s} {'nagroda':>9s} {'ep_len':>7s}")
    for k, r, l in rows:
        print(f"{k:>13,d} {r:>9.1f} {l:>7.0f}")


def eksportuj(run_name: str) -> Path:
    """policy.npz + policy_meta.json z ostatniego checkpointu (na CPU, 1-2 min). Zwraca katalog."""
    run_dir = TRAINING / "runs" / run_name
    out = run_dir / "deploy"
    st = status(run_name)
    if st is None:
        raise FileNotFoundError(f"brak treningu {run_name}: najpierw trenuj() (wymaga GPU)")
    if st != "complete":
        raise RuntimeError(f"trening {run_name} nie dokończył się: uruchom trenuj() ponownie")
    if not (out / "policy_meta.json").exists():
        print("eksport (budowa środowiska na CPU i dwie walidacje, 1-2 min)...")
        tmp = run_dir / "deploy.tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        res = subprocess.run([sys.executable, "-m", "wojtek_rl.export_policy", "--run", f"runs/{run_name}",
                              "--out", str(tmp)],
                             cwd=TRAINING, env=dict(os.environ, JAX_PLATFORMS="cpu"), capture_output=True, text=True)
        if res.returncode:
            shutil.rmtree(tmp, ignore_errors=True)
            raise RuntimeError("eksport nie powiódł się:\n" + (res.stderr or res.stdout)[-2000:])
        shutil.rmtree(out, ignore_errors=True)
        tmp.rename(out)                          # gotowe pliki pojawiają się dopiero po udanej walidacji
        print("\n".join(l for l in res.stdout.splitlines() if l.startswith(("validated", "wrote"))))
    return out


def krzywa(run_name: str) -> list:
    """(kroki, nagroda, długość epizodu) z train.log."""
    import re

    log = TRAINING / "runs" / run_name / "train.log"
    if not log.exists():
        return []
    return [(int(s.replace(",", "")), float(r), float(l)) for s, r, l in
            re.findall(r"steps\s+([\d,]+)\s+reward\s+([-\d.]+)\s+ep_len\s+([\d.]+)", log.read_text())]


def checkpoint_z_hf(repo_id: str, revision: str | None = None) -> Path:
    """Katalog checkpointu Brax opublikowanej polityki z Hugging Face (do `trenuj(start=...)`).

    Repozytoria są prywatne: potrzebny HF_TOKEN w środowisku. Pobiera tylko pliki
    checkpointu i wskazuje katalog z `ppo_network_config.json`.
    """
    from huggingface_hub import snapshot_download

    root = Path(snapshot_download(repo_id, revision=revision, allow_patterns=["*checkpoint*/**", "run.json"]))
    cfgs = sorted(root.rglob("ppo_network_config.json"))
    if not cfgs:
        raise FileNotFoundError(f"w {repo_id} nie ma katalogu checkpointu (ppo_network_config.json)")
    return cfgs[-1].parent
