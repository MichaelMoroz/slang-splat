from __future__ import annotations

import argparse
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import slangpy as spy

ROOT = Path(__file__).resolve().parent.parent
SHADER_PATH = ROOT / "shaders" / "utility" / "blur" / "separable_gaussian_blur.slang"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import create_default_device
from src.filter import SeparableGaussianBlur


@dataclass(frozen=True, slots=True)
class BenchmarkStats:
    group_size: int
    channel_pack: int
    forward_avg_ms: float
    forward_min_ms: float
    adjoint_avg_ms: float
    adjoint_min_ms: float
    forward_bandwidth_gbps: float
    adjoint_bandwidth_gbps: float

    @property
    def total_avg_ms(self) -> float:
        return self.forward_avg_ms + self.adjoint_avg_ms

    @property
    def total_bandwidth_gbps(self) -> float:
        return min(self.forward_bandwidth_gbps, self.adjoint_bandwidth_gbps)


class BlurBenchmark:
    def __init__(self, width: int, height: int, channel_count: int, warmup: int, iterations: int) -> None:
        self.width = int(width)
        self.height = int(height)
        self.channel_count = int(channel_count)
        self.warmup = int(warmup)
        self.iterations = int(iterations)
        self.device = create_default_device(device_type=spy.DeviceType.vulkan, enable_debug_layers=False)

    def _create_shader_variant(self, group_size: int, channel_pack: int) -> Path:
        temp_dir = Path(tempfile.gettempdir())
        path = temp_dir / f"benchmark_gaussian_blur_{int(group_size)}_{int(channel_pack)}.slang"
        path.write_text(
            (
                f"#define BLUR_GROUP_SIZE_OVERRIDE {int(group_size)}u\n"
                f"#define BLUR_CHANNEL_PACK_OVERRIDE {int(channel_pack)}u\n"
                "#include \"utility/blur/separable_gaussian_blur.slang\"\n"
            ),
            encoding="ascii",
        )
        return path

    def _create_blur(self, group_size: int, channel_pack: int) -> SeparableGaussianBlur:
        return SeparableGaussianBlur(
            self.device,
            width=self.width,
            height=self.height,
            shader_path=self._create_shader_variant(group_size, channel_pack),
            channel_pack=channel_pack,
        )

    def _create_buffers(self, blur: SeparableGaussianBlur) -> tuple[spy.Buffer, spy.Buffer]:
        return blur.make_buffer(self.channel_count), blur.make_buffer(self.channel_count)

    def _upload_inputs(self, input_buffer: spy.Buffer, output_buffer: spy.Buffer) -> None:
        rng = np.random.default_rng(0)
        values = rng.random(self.width * self.height * self.channel_count, dtype=np.float32)
        input_buffer.copy_from_numpy(values)
        output_buffer.copy_from_numpy(values[::-1].copy())

    def _execute(self, blur: SeparableGaussianBlur, input_buffer: spy.Buffer, output_buffer: spy.Buffer) -> tuple[float, float]:
        query_pool = self.device.create_query_pool(spy.QueryType.timestamp, 3)
        encoder = self.device.create_command_encoder()
        encoder.write_timestamp(query_pool, 0)
        blur.blur(encoder, input_buffer, output_buffer, self.channel_count)
        encoder.write_timestamp(query_pool, 1)
        blur.blur_adjoint(encoder, output_buffer, input_buffer, self.channel_count)
        encoder.write_timestamp(query_pool, 2)
        self.device.submit_command_buffer(encoder.finish())
        self.device.wait()
        start, mid, end = query_pool.get_timestamp_results(0, 3)
        return (mid - start) * 1000.0, (end - mid) * 1000.0

    def _warm_up(self, blur: SeparableGaussianBlur, input_buffer: spy.Buffer, output_buffer: spy.Buffer) -> None:
        for _ in range(self.warmup):
            self._execute(blur, input_buffer, output_buffer)

    def _pass_bytes(self) -> int:
        return self.width * self.height * self.channel_count * 4 * 4

    def _bandwidth_gbps(self, elapsed_ms: float) -> float:
        return self._pass_bytes() / (float(elapsed_ms) * 1.0e6)

    def run_case(self, group_size: int, channel_pack: int) -> BenchmarkStats:
        blur = self._create_blur(group_size, channel_pack)
        input_buffer, output_buffer = self._create_buffers(blur)
        self._upload_inputs(input_buffer, output_buffer)
        self._warm_up(blur, input_buffer, output_buffer)
        forward_ms: list[float] = []
        adjoint_ms: list[float] = []
        for _ in range(self.iterations):
            forward, adjoint = self._execute(blur, input_buffer, output_buffer)
            forward_ms.append(forward)
            adjoint_ms.append(adjoint)
        return BenchmarkStats(
            group_size=int(group_size),
            channel_pack=int(channel_pack),
            forward_avg_ms=float(np.mean(forward_ms)),
            forward_min_ms=float(np.min(forward_ms)),
            adjoint_avg_ms=float(np.mean(adjoint_ms)),
            adjoint_min_ms=float(np.min(adjoint_ms)),
            forward_bandwidth_gbps=self._bandwidth_gbps(float(np.mean(forward_ms))),
            adjoint_bandwidth_gbps=self._bandwidth_gbps(float(np.mean(adjoint_ms))),
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark the separable Gaussian blur forward and adjoint passes with Slangpy GPU timestamps.")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--channels", type=int, default=15)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--group-sizes", type=int, nargs="+", default=[8, 12, 16, 24, 32, 64, 128])
    parser.add_argument("--channel-packs", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    return parser


def _print_results(stats: list[BenchmarkStats]) -> None:
    print(f"shader={SHADER_PATH}")
    print("group_size channel_pack forward_avg_ms forward_bw_gbps adjoint_avg_ms adjoint_bw_gbps total_avg_ms")
    for row in stats:
        print(
            f"{row.group_size:10d} {row.channel_pack:12d} {row.forward_avg_ms:14.6f} {row.forward_bandwidth_gbps:15.3f} "
            f"{row.adjoint_avg_ms:15.6f} {row.adjoint_bandwidth_gbps:15.3f} {row.total_avg_ms:12.6f}"
        )
    best = min(stats, key=lambda item: item.total_avg_ms)
    print(
        f"best_group_size={best.group_size} best_channel_pack={best.channel_pack} "
        f"best_total_avg_ms={best.total_avg_ms:.6f} best_min_bw_gbps={best.total_bandwidth_gbps:.3f}"
    )


def main() -> int:
    args = build_parser().parse_args()
    benchmark = BlurBenchmark(
        width=args.width,
        height=args.height,
        channel_count=args.channels,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    results = [benchmark.run_case(group_size, channel_pack) for group_size in args.group_sizes for channel_pack in args.channel_packs]
    _print_results(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
