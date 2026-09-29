#!/usr/bin/env bash
# ============================================================================
# tmux-status.sh — which of my tmux sessions are still working?
#
# `tmux ls` only tells you a session EXISTS, not whether anything is running
# in it. The trick is `#{pane_current_command}`: it reports the program
# currently in the foreground of the pane.
#
#     bash / zsh / sh   -> nothing running, the job finished, safe to kill
#     python / python3  -> still working
#
# USAGE:   ./tmux-status.sh
# ============================================================================
set -uo pipefail

if ! tmux has-session 2>/dev/null; then
  echo "No tmux sessions at all."
  exit 0
fi

printf '%-14s %-10s %-12s %s\n' SESSION STATUS RUNNING "LAST LINE OF OUTPUT"
printf '%-14s %-10s %-12s %s\n' "-------------" "---------" "-----------" "-------------------"

idle_list=""

while IFS='|' read -r name cmd; do
  case "${cmd}" in
    bash|zsh|sh|fish|-bash|-zsh)
      status="idle"
      idle_list="${idle_list} ${name}"
      ;;
    *)
      status="BUSY"
      ;;
  esac

  # Last non-empty line the pane printed, trimmed so the table stays readable.
  last=$(tmux capture-pane -p -t "${name}" 2>/dev/null \
         | grep -v '^[[:space:]]*$' | tail -1 | cut -c1-58)

  printf '%-14s %-10s %-12s %s\n' "${name}" "${status}" "${cmd}" "${last}"
done < <(tmux list-panes -a -F '#{session_name}|#{pane_current_command}')

echo
if [ -n "${idle_list}" ]; then
  echo "Finished (safe to close):${idle_list}"
  echo
  for s in ${idle_list}; do
    echo "    tmux kill-session -t ${s}"
  done
else
  echo "Every session still has something running - do not kill any."
fi

echo
echo "Reminders:"
echo "    tmux attach -t NAME     step into a session   (Ctrl-b then d to leave)"
echo "    tmux capture-pane -p -t NAME | tail -20       peek without attaching"
