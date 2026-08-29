"""Navigation evaluation for Wojtek: kinematic fast loop, occupancy mapping,
episode generation, metrics, and the spoken-instruction chain.

Tier A of the two-tier eval: iterate the VLM/nav stack here (no legged
physics), gate on the wojtek_rl.room_app physics loop before hardware.
"""
