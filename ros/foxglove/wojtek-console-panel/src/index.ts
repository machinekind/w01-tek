// The Wojtek web operator console, inside Foxglove. The panel is nothing but
// an iframe onto the console the stack already serves on :8080 -- the page
// keeps its own websocket to that port, so this never touches the Foxglove
// bridge and works identically outside Foxglove (a plain browser tab, RViz
// sessions, a phone on the robot's AP).

import { ExtensionContext, PanelExtensionContext, SettingsTreeAction } from "@foxglove/extension";

const DEFAULT_URL = "http://localhost:8080";

type State = { url: string };

export function activate(extensionContext: ExtensionContext): void {
  extensionContext.registerPanel({ name: "Wojtek console", initPanel });
}

function initPanel(context: PanelExtensionContext): () => void {
  const state: State = { url: DEFAULT_URL, ...(context.initialState as Partial<State>) };

  const iframe = document.createElement("iframe");
  iframe.src = state.url;
  iframe.style.width = "100%";
  iframe.style.height = "100%";
  iframe.style.border = "none";
  iframe.style.display = "block";
  // The console needs pointer input for the drive pad and reads a gamepad in
  // the browser; nothing else.
  iframe.allow = "gamepad";
  context.panelElement.style.overflow = "hidden";
  context.panelElement.appendChild(iframe);

  const updateSettings = () => {
    context.updatePanelSettingsEditor({
      actionHandler: (action: SettingsTreeAction) => {
        if (
          action.action === "update" &&
          action.payload.path.join(".") === "general.url" &&
          typeof action.payload.value === "string"
        ) {
          state.url = action.payload.value.length > 0 ? action.payload.value : DEFAULT_URL;
          context.saveState(state);
          iframe.src = state.url;
          updateSettings();
        }
      },
      nodes: {
        general: {
          label: "Console",
          fields: {
            url: {
              label: "URL",
              input: "string",
              value: state.url,
              help: "Where the wojtek web console is served (web_console, port 8080).",
            },
          },
        },
      },
    });
  };
  updateSettings();

  return () => {
    iframe.remove();
  };
}
