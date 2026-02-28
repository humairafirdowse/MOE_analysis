import { GpuType } from "../state/useConfigStore";

export interface GpuSpec {
  id: GpuType;
  name: string;
  fp16Tflops: number;
  fp32Tflops: number;
  /** FP8 peak TFLOPS (Hopper: ~2x FP16). Used for inference / theoretical peak. */
  fp8Tflops?: number;
  /** FP8 effective for training (memory-bandwidth limited). If set, used for GPU-hours when precision=fp8. */
  fp8TrainingTflops?: number;
  hbmGB: number;
  memBandwidthGBs: number;
  /** Per-GPU unidirectional interconnect bandwidth in GB/s (NVLink or PCIe). */
  interconnectBWGBs: number;
  costPerHourUSD: number;
}

export const GPU_SPECS: Record<GpuType, GpuSpec> = {
  "A100-80G": {
    id: "A100-80G",
    name: "NVIDIA A100 80GB",
    fp16Tflops: 312,
    fp32Tflops: 19.5,
    hbmGB: 80,
    memBandwidthGBs: 2039,
    interconnectBWGBs: 300,
    costPerHourUSD: 5.0
  },
  "H100-80G": {
    id: "H100-80G",
    name: "NVIDIA H100 80GB",
    fp16Tflops: 989,
    fp32Tflops: 67,
    fp8Tflops: 1978,
    hbmGB: 80,
    memBandwidthGBs: 3350,
    interconnectBWGBs: 450,
    costPerHourUSD: 8.5
  },
  "H800-80G": {
    id: "H800-80G",
    name: "NVIDIA H800 80GB (China export variant)",
    fp16Tflops: 756,
    fp32Tflops: 51,
    fp8Tflops: 1512,
    fp8TrainingTflops: 738,
    hbmGB: 80,
    memBandwidthGBs: 2000,
    interconnectBWGBs: 200,
    costPerHourUSD: 2.0
  },
  "H200-141G": {
    id: "H200-141G",
    name: "NVIDIA H200 141GB",
    fp16Tflops: 989,
    fp32Tflops: 67,
    fp8Tflops: 1978,
    hbmGB: 141,
    memBandwidthGBs: 4800,
    interconnectBWGBs: 450,
    costPerHourUSD: 12.0
  },
  "B200-192G": {
    id: "B200-192G",
    name: "NVIDIA B200 192GB",
    fp16Tflops: 2250,
    fp32Tflops: 180,
    fp8Tflops: 4500,
    hbmGB: 192,
    memBandwidthGBs: 8000,
    interconnectBWGBs: 900,
    costPerHourUSD: 18.0
  },
  "RTX-4090": {
    id: "RTX-4090",
    name: "NVIDIA RTX 4090",
    fp16Tflops: 165,
    fp32Tflops: 82.6,
    hbmGB: 24,
    memBandwidthGBs: 1008,
    interconnectBWGBs: 32,
    costPerHourUSD: 0.7
  },
  "M4-Max": {
    id: "M4-Max",
    name: "Apple M4 Max (unified memory)",
    fp16Tflops: 19.4,
    fp32Tflops: 9.7,
    hbmGB: 128,
    memBandwidthGBs: 546,
    interconnectBWGBs: 100,
    costPerHourUSD: 0
  }
};

export function getGpuSpec(type: GpuType): GpuSpec {
  return GPU_SPECS[type];
}
