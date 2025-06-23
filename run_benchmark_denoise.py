import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

# Define parameter sets
noisecone_params = ["-1", "-2", "-3", "-4"]
percentile_params = ["1", "5", "10", "15", "20"]

# Common command parts
base_cmd = [
    "unhuddle-denoise",
    "--base_path", "demodata",
    "--nuclear_markers", "DNA1", "DNA2", "HistoneH3",
    "--max_workers", "11",
    "--create_adata",
    "--use_denoise",
    "--coord_cols", "optsne_1", "optsne_2",
    "--add_dimensionreduction_coords", "demodata-tsne",
    "--denoise_regress", "perimeter",
    "--normalization", "sensormarker",
    "--save_reallocation_debug",
    "--add_original_compiled_sum",
]

# Build all jobs
jobs = []
for param in noisecone_params:
    outdir = f"benchmark1/noisecone_{param}"
    cmd = base_cmd + [
        "--output_base_path", outdir,
        "--denoise_method", "noisecone",
        "--denoise_x_anchor_multiplier", param,
    ]
    jobs.append((f"noisecone_{param}", cmd))

for param in percentile_params:
    outdir = f"benchmark1/percentile_{param}"
    cmd = base_cmd + [
        "--output_base_path", outdir,
        "--denoise_method", "percentile",
        "--percentile", param,
    ]
    jobs.append((f"percentile_{param}", cmd))


def run_job(job):
    name, cmd = job
    print(f"[START] {name}: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"[DONE]  {name}")
        else:
            print(f"[FAIL]  {name} (exit {result.returncode})\nSTDERR:\n{result.stderr}")
    except Exception as e:
        print(f"[ERROR] {name}: {e}")

if __name__ == "__main__":
    max_parallel = 4
    with ProcessPoolExecutor(max_workers=max_parallel) as executor:
        futures = {executor.submit(run_job, job): job[0] for job in jobs}
        for future in as_completed(futures):
            name = futures[future]
            try:
                future.result()
            except Exception as exc:
                print(f"[EXCEPTION] {name}: {exc}") 