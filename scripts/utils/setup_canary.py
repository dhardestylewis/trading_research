"""Set up Windows Task Scheduler for the canary tick.

Creates a scheduled task that runs canary_tick.py every hour.
Run this script once (admin recommended for StartWhenAvailable):

    python setup_canary.py

Key behaviors:
  - Runs every hour
  - Catches up after sleep/shutdown (StartWhenAvailable)
  - Runs on battery power
  - Survives reboots (persistent scheduled task)

To remove the scheduled task later:
    schtasks /delete /tn "TradingCanaryTick" /f
"""
from __future__ import annotations
import subprocess
import sys
from pathlib import Path


def main():
    project_dir = Path(__file__).resolve().parent.parent.parent
    python_exe = sys.executable
    script = project_dir / "scripts" / "canaries" / "canary_tick.py"
    log_file = project_dir / "logs" / "canary_tick.log"

    task_name = "TradingCanaryTick"

    print(f"Project directory: {project_dir}")
    print(f"Python: {python_exe}")
    print(f"Script: {script}")
    print(f"Log file: {log_file}")
    print(f"Task name: {task_name}")
    print()

    # Use PowerShell to create the task — gives us access to:
    #   StartWhenAvailable = catch up missed ticks after sleep/shutdown
    #   AllowStartIfOnBatteries = run even on battery
    #   DontStopIfGoingOnBatteries = don't kill mid-tick
    #   WorkingDirectory = so imports resolve correctly
    ps_script = f'''
$taskName = "{task_name}"
$action = New-ScheduledTaskAction `
    -Execute '"{python_exe}"' `
    -Argument '"{script}" --dry-run >> "{log_file}" 2>&1' `
    -WorkingDirectory "{project_dir}"

$trigger = New-ScheduledTaskTrigger `
    -Once `
    -At (Get-Date).Date.AddHours((Get-Date).Hour + 1) `
    -RepetitionInterval (New-TimeSpan -Hours 1) `
    -RepetitionDuration ([TimeSpan]::MaxValue)

$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 5) `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 10)

$principal = New-ScheduledTaskPrincipal -UserId "$env:USERNAME" -LogonType S4U

Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue
Register-ScheduledTask `
    -TaskName $taskName `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Principal $principal `
    -Description "Canary tick: hourly signal check for SOL-USD trading strategy"

Write-Host "TASK_CREATED_OK"
'''

    print("Creating scheduled task via PowerShell...")
    print()

    try:
        result = subprocess.run(
            ["powershell", "-ExecutionPolicy", "Bypass", "-Command", ps_script],
            capture_output=True,
            text=True,
        )

        if "TASK_CREATED_OK" in result.stdout:
            print("=" * 60)
            print("  CANARY SCHEDULED TASK CREATED SUCCESSFULLY")
            print("=" * 60)
            print()
            print("Behaviors:")
            print("  - Runs every hour")
            print("  - Catches up after sleep/shutdown (StartWhenAvailable)")
            print("  - Runs on battery power")
            print("  - Auto-retries 3x on failure (5 min apart)")
            print("  - Survives reboots")
            print()
            print("What it does each tick:")
            print("  1. Fetch latest SOL-USD bars from Binance")
            print("  2. Run LightGBM model (auto-retrains weekly)")
            print("  3. If signal fires -> append to canary log")
            print()
            print("Commands:")
            print(f"  Check results:   python run_exp007.py --mode live")
            print(f"  Manual tick:     python canary_tick.py --dry-run")
            print(f"  View log:        type {log_file}")
            print(f"  Stop canary:     schtasks /delete /tn \"{task_name}\" /f")
        else:
            print(f"Output: {result.stdout}")
            if result.stderr:
                print(f"Errors: {result.stderr}")
            print()
            print("If you got access errors, try running as Administrator.")
            print("Or create the task manually in Task Scheduler:")
            print(f'  Action: "{python_exe}" "{script}" --dry-run')
            print(f"  Working dir: {project_dir}")
            print(f"  Trigger: Every 1 hour, repeat indefinitely")
            print(f"  Settings: Start when available = ON")
    except FileNotFoundError:
        print("PowerShell not found. For Linux/Mac, add to crontab (crontab -e):")
        print(f"  0 * * * * cd {project_dir} && {python_exe} {script} --dry-run >> /tmp/canary.log 2>&1")


if __name__ == "__main__":
    main()
