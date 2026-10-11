#!/usr/bin/env bash
# Runs unattended: waits for the bake-off, then builds dashboard data, add-on test, final report.
cd /d/dog
while powershell -NoProfile -Command "if (Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*run_bakeoff*' }) { exit 0 } else { exit 1 }"; do sleep 30; done
echo "bakeoff finished $(date)"
python -u scripts/run_simulation.py > reports_simulation.log 2>&1 && echo "simulation ok $(date)"
python -u scripts/test_feature_addons.py > reports_addons.log 2>&1 && echo "addons ok $(date)"
python -u scripts/scripts_report.py > reports_final.log 2>&1 && echo "report ok $(date)"
echo ALL DONE
