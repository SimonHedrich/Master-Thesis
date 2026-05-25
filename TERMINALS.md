# Terminal Setup (tmux + VS Code Remote SSH)

Terminals persist across SSH disconnects via tmux. When the project folder is opened in VS Code, five terminal tabs open automatically, each attached to a named tmux session.

## Sessions

| Name    | tmux session |
|---------|-------------|
| alpha   | `alpha`     |
| bravo   | `bravo`     |
| charlie | `charlie`   |
| delta   | `delta`     |
| echo    | `echo`      |

Configured in `.vscode/tasks.json` with `runOn: folderOpen`. VS Code will prompt once to allow automatic tasks — click **Allow** to enable.

## Manual attach (plain SSH)

```bash
tmux attach -t alpha   # or bravo, charlie, delta, echo
tmux ls                # list all running sessions
```

## Useful tmux shortcuts

| Keys | Action |
|------|--------|
| `Ctrl+b, d` | Detach (leave session running) |
| `Ctrl+b, c` | New window inside session |
| `Ctrl+b, n/p` | Next / previous window |
| `Ctrl+b, [` | Scroll mode (exit with `q`) |

## Re-enabling automatic tasks

If you accidentally dismissed the prompt: `Ctrl+Shift+P` → **"Tasks: Manage Automatic Tasks"** → Allow.

To open a session manually: `Ctrl+Shift+P` → **"Tasks: Run Task"** → select `tmux: <name>`.
