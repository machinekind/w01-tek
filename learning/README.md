# Wojtek: uczenie chodu krok po kroku

`wojtek_rl_guide.ipynb`, notebook na Google Colab. Krok po kroku: robot w MuJoCo → środowisko → trening → eksport → Twoja polityka kontra ta na robocie. Każdy krok kończy się widokiem z MuJoCo. Kroki: 1 Wojtek stoi w MuJoCo, 2 polityka na początku treningu, 3 nagroda, 4 trening na GPU, 5 wytrenowana polityka w MuJoCo, 6 zadania, 7 zadanie główne (panel sterowania kierunkiem), 8 finał: porównanie z polityką z robota (Hugging Face, sekrety `HF_ORGANIZATION` i `HF_TOKEN`). Pomocniki: `wojtek_kurs.py` (przebieg w MuJoCo, tabela, trening, eksport) i `lowpoly.py` (uproszczone siatki do renderowania, bo Colab nie ma OpenGL NVIDII). Przepis treningu: `training/wojtek_rl/conf/experiment/course_locomotion.yaml` (zmierzone na T4: 16 mln kroków idzie do przodu, 52 mln wykonuje pełny zakres komend). Cały notebook przechodzi na Colab T4 w ok. 45 min; przetestowany 2026-09-24.

[![Otwórz w Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machinekind/w01-tek/blob/main/learning/wojtek_rl_guide.ipynb)

1. Otwórz link, wybierz środowisko **GPU**.
2. Uruchamiaj komórki po kolei. Krok 0 klonuje repozytorium i instaluje `training/`; jeśli import nie zadziała, zrestartuj sesję raz.

Notebook jest w repozytorium bez wyników komórek. Nic w nim nie wgrywa polityki na robota ani go nie uzbraja.
