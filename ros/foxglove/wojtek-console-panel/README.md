# Wojtek console — Foxglove panel

Embeds the Wojtek web operator console (drive pad, arm/pose buttons, jog,
telemetry, VLM text box) as a Foxglove panel, so a macOS/Foxglove session is
one window instead of two.

The panel is nothing but an iframe onto the page `web_console` already serves
on <http://localhost:8080>. It keeps its own websocket to that port and never
touches the Foxglove bridge — the console works identically outside Foxglove
(a plain browser tab, RViz sessions, a phone on the robot's AP), and losing
the panel loses nothing.

## Build and install

```bash
npm install
npm run local-install   # builds + installs into ~/.foxglove-studio/extensions
```

Restart the Foxglove desktop app, then add the **Wojtek console** panel to
your layout. `npm run package` produces a shareable
`machinekind.wojtek-console-panel-<version>.foxe` instead (install via
Extensions → drag the file in).

The console URL is a panel setting (gear icon) for the day the console is not
on `localhost:8080`.
