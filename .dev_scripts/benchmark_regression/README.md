# Scripts for development

```
# benchmark based on same image size
docker compose exec clshub python3 .dev_scripts/benchmark_regression/1-benchmark_speed_same_image_size.py --dump benchmarks/benchmark_256_bs1.yml --batch-size 1
# benchmark based on config image size
docker compose exec clshub python3 .dev_scripts/benchmark_regression/2-benchmark_speed.py --dump benchmarks/benchmark_bs1.yml --batch-size 1
```
