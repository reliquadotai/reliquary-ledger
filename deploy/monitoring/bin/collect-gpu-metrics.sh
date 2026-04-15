#!/usr/bin/env bash
set -euo pipefail

TEXTFILE_DIR="${TEXTFILE_DIR:-/var/lib/node_exporter/textfile_collector}"
OUTPUT_FILE="${OUTPUT_FILE:-${TEXTFILE_DIR}/reliquary_gpu.prom}"
TMP_FILE="${OUTPUT_FILE}.tmp"

mkdir -p "${TEXTFILE_DIR}"

{
  echo "# HELP reliquary_gpu_available Whether nvidia-smi is available on the host."
  echo "# TYPE reliquary_gpu_available gauge"
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "reliquary_gpu_available 1"
    echo "# HELP reliquary_gpu_utilization_percent GPU utilization percent."
    echo "# TYPE reliquary_gpu_utilization_percent gauge"
    echo "# HELP reliquary_gpu_memory_used_bytes GPU memory currently used in bytes."
    echo "# TYPE reliquary_gpu_memory_used_bytes gauge"
    echo "# HELP reliquary_gpu_memory_total_bytes GPU memory total in bytes."
    echo "# TYPE reliquary_gpu_memory_total_bytes gauge"
    echo "# HELP reliquary_gpu_temperature_celsius GPU temperature in Celsius."
    echo "# TYPE reliquary_gpu_temperature_celsius gauge"
    echo "# HELP reliquary_gpu_power_draw_watts GPU power draw in watts."
    echo "# TYPE reliquary_gpu_power_draw_watts gauge"
    echo "# HELP reliquary_gpu_process_count Active GPU process count."
    echo "# TYPE reliquary_gpu_process_count gauge"
    gpu_process_map="$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader,nounits 2>/dev/null | sort | uniq -c || true)"
    while IFS=, read -r index uuid name util mem_used mem_total temp power; do
      clean_name="$(printf '%s' "${name}" | sed 's/\\/\\\\/g; s/"/\\"/g')"
      uuid="$(printf '%s' "${uuid}" | xargs)"
      util="$(printf '%s' "${util}" | xargs)"
      mem_used="$(printf '%s' "${mem_used}" | xargs)"
      mem_total="$(printf '%s' "${mem_total}" | xargs)"
      temp="$(printf '%s' "${temp}" | xargs)"
      power="$(printf '%s' "${power}" | xargs)"
      process_count="$(printf '%s\n' "${gpu_process_map}" | awk -v needle="${uuid}" '$2 == needle {print $1}' | head -n1)"
      process_count="${process_count:-0}"
      echo "reliquary_gpu_utilization_percent{gpu=\"${index}\",name=\"${clean_name}\"} ${util}"
      echo "reliquary_gpu_memory_used_bytes{gpu=\"${index}\",name=\"${clean_name}\"} $((mem_used * 1024 * 1024))"
      echo "reliquary_gpu_memory_total_bytes{gpu=\"${index}\",name=\"${clean_name}\"} $((mem_total * 1024 * 1024))"
      echo "reliquary_gpu_temperature_celsius{gpu=\"${index}\",name=\"${clean_name}\"} ${temp}"
      echo "reliquary_gpu_power_draw_watts{gpu=\"${index}\",name=\"${clean_name}\"} ${power}"
      echo "reliquary_gpu_process_count{gpu=\"${index}\",name=\"${clean_name}\"} ${process_count}"
    done < <(nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits)
  else
    echo "reliquary_gpu_available 0"
  fi

  echo "# HELP reliquary_systemd_unit_active Whether the named systemd unit is active."
  echo "# TYPE reliquary_systemd_unit_active gauge"
  echo "# HELP reliquary_systemd_unit_nrestarts Number of times the named systemd unit has restarted."
  echo "# TYPE reliquary_systemd_unit_nrestarts gauge"
  for unit in inference-miner.service inference-validator.service reliquary-metrics-exporter.service; do
    if systemctl list-unit-files "${unit}" >/dev/null 2>&1; then
      active="$(systemctl is-active "${unit}" >/dev/null 2>&1 && echo 1 || echo 0)"
      restarts="$(systemctl show "${unit}" --property=NRestarts --value 2>/dev/null || echo 0)"
      echo "reliquary_systemd_unit_active{unit=\"${unit}\"} ${active}"
      echo "reliquary_systemd_unit_nrestarts{unit=\"${unit}\"} ${restarts:-0}"
    fi
  done
} > "${TMP_FILE}"

mv "${TMP_FILE}" "${OUTPUT_FILE}"
