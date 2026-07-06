# Official rocprofv2 Kernel Summary

- output_dir: `/public/home/xdzs2026_c166/codex_logs/profile_runs/rocprofv2_8_16K_20260622_161546`
- rows: 99839
- trace_span_s: 646.496
- note: benchmark main run failed after EngineCore died; this summary is from official rocprofv2 trace CSV only.

## full

- rows: 99839
- summed_kernel_ms: 19828.493

| rank | total_ms | pct | count | avg_us | max_ms | kernel |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 5580.268 | 28.14 | 23717 | 235.286 | 0.253 | `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array<char*, 1ul>>(int, at::native::FillFunctor<int>...` |
| 2 | 3864.034 | 19.49 | 27 | 143112.361 | 143.132 | `void flash_fwd_kernel_16x64_prefetch<Flash_fwd_kernel_16x64_prefetch_traits_dim96<96, 128, 64, 4, cutlass::bfloat16_t, 3, 96, Flash_kerne...` |
| 3 | 2305.005 | 11.62 | 218 | 10573.419 | 152.672 | `kernel_unified_attention_2d.kd` |
| 4 | 1537.535 | 7.75 | 14633 | 105.073 | 1.356 | `chunk_fwd_kernel_o.kd` |
| 5 | 847.952 | 4.28 | 736 | 1152.109 | 3.158 | `Cijk_Alik_Bljk_BBH_MT256x256x16_SE_AMAS3_BW_DTL0_BL1_DRB16_ALT_KME0_ETSP_EPS1_FL0_GRVW8_GSU1_ISA936_IU1_KME_GRCMn1_K1_KLA_LDL1_LRVW8_MAC_...` |
| 6 | 780.670 | 3.94 | 456 | 1711.996 | 4.514 | `Cijk_Alik_Bljk_BBH_MT256x256x16_SE_AMAS3_BW_DTL0_BL1_DRB16_ALT_KME0_ETSP_EPS1_FL0_GRVW8_GSU1_ISA936_IU1_KME_GRCMn1_K1_KLA_LDL1_LRVW8_MAC_...` |
| 7 | 475.429 | 2.40 | 1249 | 380.648 | 2.488 | `chunk_gated_delta_rule_fwd_kernel_h_blockdim64.kd` |
| 8 | 437.663 | 2.21 | 1472 | 297.325 | 0.755 | `Cijk_Alik_Bljk_BBH_MT32x64x128_SE_AMAS3_BW_DTL0_BL1_DRB16_ALT_KME0_ETSP_EPS1_FL0_GRVW4_GSU1_ISA936_IU1_KME_GRCMn1_K1_KLA_LDL1_LRVW4_MAC_M...` |
| 9 | 415.670 | 2.10 | 1120 | 371.134 | 0.658 | `Cijk_Alik_Bljk_BBH_MT128x64x32_SE_AMAS3_BW_DTL0_BL1_DRB16_ALT_KME0_ETSP_EPS1_FL0_GRVW4_GSU1_ISA936_IU1_KME_GRCMn1_K1_KLA_LDL1_LRVW4_MAC_M...` |
| 10 | 356.742 | 1.80 | 265 | 1346.195 | 2.151 | `Cijk_Alik_Bljk_BBH_MT256x256x16_SE_AMAS3_BW_DTL0_BL1_DRB16_ALT_KME0_ETSP_EPS1_FL0_GRVW8_GSU1_ISA936_IU1_KME_GRCMn1_K1_KLA_LDL1_LRVW8_MAC_...` |
| 11 | 292.403 | 1.47 | 768 | 380.733 | 0.880 | `Cijk_Alik_Bljk_BBH_MT32x80x128_SE_AMAS3_BW_DTL0_BL1_DRB16_ALT_KME0_ETSP_EPS1_FL0_GRVW2_GSU1_ISA936_IU1_KME_GRCMn1_K1_KLA_LDL1_LRVW2_MAC_M...` |
| 12 | 272.020 | 1.37 | 2612 | 104.143 | 0.593 | `merge_16x16_to_64x64_inverse_kernel.kd` |
| 13 | 255.873 | 1.29 | 8306 | 30.806 | 0.195 | `chunk_scaled_dot_kkt_fwd_kernel.kd` |
| 14 | 193.405 | 0.98 | 1957 | 98.827 | 0.430 | `recompute_w_u_fwd_kernel.kd` |
| 15 | 154.482 | 0.78 | 384 | 402.298 | 0.615 | `Cijk_Alik_Bljk_BBH_MT32x16x256_SE_AMAS3_BW_DTL0_BL1_DRB16_ALT_KME0_ETSP_EPS1_FL0_GRVW2_GSU1_ISA936_IU1_KME_GRCMn1_K1_KLA_LDL1_LRVW2_MAC_M...` |
| 16 | 137.680 | 0.69 | 384 | 358.543 | 0.688 | `Cijk_Alik_Bljk_BBH_MT64x128x32_SE_AMAS3_BW_DTL0_BL1_DRB16_ALT_KME0_ETSP_EPS0_FL0_GRVW4_GSU8_ISA936_IU1_KME_GRCMn1_K1_KLA_LDL1_LRVW4_MAC_M...` |
| 17 | 132.099 | 0.67 | 912 | 144.845 | 0.343 | `fused_recurrent_gated_delta_rule_packed_decode_kernel.kd` |
| 18 | 123.914 | 0.62 | 256 | 484.038 | 0.681 | `Cijk_Alik_Bljk_BBH_MT32x32x128_SE_AMAS3_BW_DTL0_BL1_DRB16_ALT_KME0_ETSP_EPS1_FL0_GRVW4_GSU1_ISA936_IU1_KME_GRCMn1_K1_KLA_LDL1_LRVW4_MAC_M...` |
| 19 | 123.609 | 0.62 | 704 | 175.581 | 0.312 | `Cijk_Alik_Bljk_BBH_MT32x32x128_SE_AMAS3_BW_DTL0_BL1_DRB16_ALT_KME0_ETSP_EPS1_FL0_GRVW4_GSU1_ISA936_IU1_KME_GRCMn1_K1_KLA_LDL1_LRVW4_MAC_M...` |
| 20 | 112.781 | 0.57 | 384 | 293.701 | 0.407 | `Cijk_Alik_Bljk_BBH_MT64x64x64_SE_AMAS3_BW_DTL0_BL1_DRB16_ALT_KME0_ETSP_EPS1_FL0_GRVW4_GSU1_ISA936_IU1_KME_GRCMn1_K1_KLA_LDL1_LRVW4_MAC_MM...` |
| 21 | 100.348 | 0.51 | 832 | 120.611 | 0.300 | `Cijk_Alik_Bljk_BBH_MT32x32x4_SE_AMAS3_BW_DTL0_BL1_DRB16_ALT_KME0_ETSP_EPS1_FL0_GRVW4_GSU1_ISA936_IU1_KME_GRCMn1_K1_KLA_LDL1_LRVW4_MAC_MMA...` |
| 22 | 99.490 | 0.50 | 400 | 248.725 | 0.928 | `void at::native::elementwise_kernel_manual_unroll<128, 8, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at:...` |
| 23 | 87.644 | 0.44 | 224 | 391.269 | 0.528 | `Cijk_Alik_Bljk_BBH_MT64x32x32_SE_AMAS3_BW_DTL0_BL1_DRB16_ALT_KME0_ETSP_EPS1_FL0_GRVW4_GSU1_ISA936_IU1_KME_GRCMn2_K1_KLA_LDL1_LRVW4_MAC_MM...` |
| 24 | 87.014 | 0.44 | 1152 | 75.533 | 0.309 | `Cijk_Alik_Bljk_BBH_MT16x16x256_SE_AMAS3_BW_DTL0_BL1_DRB16_ALT_KME0_ETSP_EPS1_FL0_GRVW2_GSU1_ISA936_IU1_KME_GRCMn1_K1_KLA_LDL1_LRVW2_MAC_M...` |
| 25 | 86.918 | 0.44 | 929 | 93.561 | 1.899 | `Cijk_Alik_Bljk_BBH_MT32x16x4_SE_AMAS3_BW_DTL0_BL1_DRB16_ALT_KME0_ETSP_EPS1_FL0_GRVW2_GSU1_ISA936_IU1_KME_GRCMn1_K1_KLA_LDL1_LRVW2_MAC_MMA...` |
| 26 | 77.122 | 0.39 | 544 | 141.768 | 0.293 | `Cijk_Alik_Bljk_BBH_MT32x16x128_SE_AMAS3_BW_DTL0_BL1_DRB16_ALT_KME0_ETSP_EPS1_FL0_GRVW2_GSU1_ISA936_IU1_KME_GRCMn1_K1_KLA_LDL1_LRVW2_MAC_M...` |
| 27 | 70.439 | 0.36 | 768 | 91.718 | 0.215 | `Cijk_Alik_Bljk_BBH_MT128x128x32_SE_AMAS3_BW_DTL0_BL1_DRB16_ALT_KME0_ETSP_EPS1_FL0_GRVW8_GSU1_ISA936_IU1_KME_GRCMn1_K1_KLA_LDL1_LRVW8_MAC_...` |
| 28 | 65.287 | 0.33 | 2 | 32643.262 | 33.111 | `_topk_topp_kernel.kd` |
| 29 | 63.097 | 0.32 | 28 | 2253.472 | 2.391 | `void at::native::vectorized_elementwise_kernel<8, at::native::GeluCUDAKernelImpl(at::TensorIteratorBase&, at::native::GeluType)::'lambda0...` |
| 30 | 47.583 | 0.24 | 27 | 1762.328 | 1.925 | `rotary_kernel.kd` |
| 31 | 43.654 | 0.22 | 1364 | 32.005 | 0.394 | `triton_poi_fused_mul_rocm_unquantized_gemm_silu_slice_3.kd` |
| 32 | 41.865 | 0.21 | 55 | 761.184 | 0.797 | `void at::native::(anonymous namespace)::vectorized_layer_norm_kernel<c10::BFloat16, float, false>(int, float, c10::BFloat16 const*, c10::...` |
| 33 | 30.333 | 0.15 | 78 | 388.880 | 0.412 | `_causal_conv1d_fwd_kernel.kd` |
| 34 | 30.103 | 0.15 | 1364 | 22.069 | 0.257 | `triton_red_fused__to_copy_add_copy__mean_mul_pow_rsqrt_4.kd` |
| 35 | 29.142 | 0.15 | 1365 | 21.349 | 0.256 | `triton_poi_fused__to_copy__unsafe_view_add_clone_mean_mul_pow_reshape_rocm_unquantized_gemm_rsqrt_silu_view_1.kd` |
| 36 | 28.253 | 0.14 | 383 | 73.767 | 0.083 | `Cijk_Alik_Bljk_BBH_MT128x64x32_SE_AMAS3_BW_DTL0_BL1_DRB16_ALT_KME0_ETSP_EPS1_FL0_GRVW4_GSU1_ISA936_IU1_KME_GRCMn1_K1_KLA_LDL1_LRVW4_MAC_M...` |
| 37 | 21.885 | 0.11 | 682 | 32.090 | 0.394 | `triton_poi_fused_mul_rocm_unquantized_gemm_silu_slice_4.kd` |
| 38 | 21.838 | 0.11 | 682 | 32.021 | 0.393 | `triton_poi_fused_mul_rocm_unquantized_gemm_silu_slice_2.kd` |
| 39 | 21.636 | 0.11 | 256 | 84.515 | 0.092 | `Cijk_Alik_Bljk_BBH_MT128x128x32_SE_AMAS3_BW_DTL0_BL1_DRB16_ALT_KME0_ETSP_EPS1_FL0_GRVW8_GSU1_ISA936_IU1_KME_GRCMn1_K1_KLA_LDL1_LRVW8_MAC_...` |
| 40 | 19.620 | 0.10 | 912 | 21.514 | 0.087 | `_causal_conv1d_update_kernel.kd` |

## last_60s

- rows: 7297
- summed_kernel_ms: 5891.755

| rank | total_ms | pct | count | avg_us | max_ms | kernel |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 2287.135 | 38.82 | 26 | 87966.715 | 152.672 | `kernel_unified_attention_2d.kd` |
| 2 | 1334.274 | 22.65 | 1915 | 696.749 | 1.356 | `chunk_fwd_kernel_o.kd` |
| 3 | 451.036 | 7.66 | 331 | 1362.647 | 2.488 | `chunk_gated_delta_rule_fwd_kernel_h_blockdim64.kd` |
| 4 | 447.016 | 7.59 | 104 | 4298.233 | 4.514 | `Cijk_Alik_Bljk_BBH_MT256x256x16_SE_AMAS3_BW_DTL0_BL1_DRB16_ALT_KME0_ETSP_EPS1_FL0_GRVW8_GSU1_ISA936_IU1_KME_GRCMn1_K1_KLA_LDL1_LRVW8_MAC_...` |
| 5 | 443.522 | 7.53 | 2002 | 221.539 | 0.253 | `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array<char*, 1ul>>(int, at::native::FillFunctor<int>...` |
| 6 | 335.188 | 5.69 | 209 | 1603.771 | 2.875 | `Cijk_Alik_Bljk_BBH_MT256x256x16_SE_AMAS3_BW_DTL0_BL1_DRB16_ALT_KME0_ETSP_EPS1_FL0_GRVW8_GSU1_ISA936_IU1_KME_GRCMn1_K1_KLA_LDL1_LRVW8_MAC_...` |
| 7 | 208.338 | 3.54 | 104 | 2003.253 | 2.094 | `Cijk_Alik_Bljk_BBH_MT256x256x16_SE_AMAS3_BW_DTL0_BL1_DRB16_ALT_KME0_ETSP_EPS1_FL0_GRVW8_GSU1_ISA936_IU1_KME_GRCMn1_K1_KLA_LDL1_LRVW8_MAC_...` |
| 8 | 153.195 | 2.60 | 459 | 333.759 | 0.430 | `recompute_w_u_fwd_kernel.kd` |
| 9 | 41.982 | 0.71 | 79 | 531.419 | 0.533 | `merge_16x16_to_64x64_inverse_kernel.kd` |
| 10 | 29.925 | 0.51 | 77 | 388.633 | 0.412 | `_causal_conv1d_fwd_kernel.kd` |
| 11 | 19.462 | 0.33 | 52 | 374.261 | 0.393 | `triton_poi_fused_mul_rocm_unquantized_gemm_silu_slice_3.kd` |
| 12 | 12.188 | 0.21 | 52 | 234.378 | 0.257 | `triton_red_fused__to_copy_add_copy__mean_mul_pow_rsqrt_4.kd` |
| 13 | 12.185 | 0.21 | 231 | 52.748 | 0.090 | `void at::native::elementwise_kernel_manual_unroll<128, 8, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at:...` |
| 14 | 12.139 | 0.21 | 53 | 229.038 | 0.254 | `triton_poi_fused__to_copy__unsafe_view_add_clone_mean_mul_pow_reshape_rocm_unquantized_gemm_rsqrt_silu_view_1.kd` |
| 15 | 9.737 | 0.17 | 26 | 374.485 | 0.393 | `triton_poi_fused_mul_rocm_unquantized_gemm_silu_slice_2.kd` |
| 16 | 9.605 | 0.16 | 26 | 369.434 | 0.393 | `triton_poi_fused_mul_rocm_unquantized_gemm_silu_slice_4.kd` |
| 17 | 7.116 | 0.12 | 79 | 90.070 | 0.092 | `chunk_scaled_dot_kkt_fwd_kernel.kd` |
| 18 | 6.881 | 0.12 | 156 | 44.108 | 0.074 | `l2norm_fwd_kernel2.kd` |
| 19 | 6.385 | 0.11 | 77 | 82.922 | 0.083 | `fused_gdn_gating_kernel.kd` |
| 20 | 6.151 | 0.10 | 26 | 236.591 | 0.258 | `triton_red_fused__to_copy_add_copy__mean_mul_pow_rsqrt_3.kd` |
| 21 | 6.113 | 0.10 | 26 | 235.126 | 0.258 | `triton_red_fused__to_copy_add_copy__mean_mul_pow_rocm_unquantized_gemm_rsqrt_5.kd` |
| 22 | 6.113 | 0.10 | 53 | 115.336 | 0.158 | `triton_red_fused__to_copy_add_mean_mul_pow_rocm_unquantized_gemm_rsqrt_2.kd` |
| 23 | 5.993 | 0.10 | 26 | 230.498 | 0.274 | `triton_poi_fused__to_copy__unsafe_view_add_clone_mean_mul_pow_reshape_rocm_unquantized_gemm_rsqrt_silu_view_2.kd` |
| 24 | 5.018 | 0.09 | 78 | 64.330 | 0.067 | `Cijk_Alik_Bljk_BBH_MT128x64x32_SE_AMAS3_BW_DTL0_BL1_DRB16_ALT_KME0_ETSP_EPS1_FL0_GRVW4_GSU1_ISA936_IU1_KME_GRCMn1_K1_KLA_LDL1_LRVW4_MAC_M...` |
| 25 | 3.894 | 0.07 | 26 | 149.766 | 0.151 | `triton_poi_fused_8.kd` |
| 26 | 3.749 | 0.06 | 26 | 144.190 | 0.168 | `triton_poi_fused_mul_rocm_unquantized_gemm_sigmoid_view_0.kd` |
| 27 | 3.308 | 0.06 | 53 | 62.418 | 0.080 | `triton_per_fused__to_copy_mean_pow_view_0.kd` |
| 28 | 3.075 | 0.05 | 26 | 118.258 | 0.152 | `triton_red_fused__to_copy_add_mean_mul_pow_rocm_unquantized_gemm_rsqrt_1.kd` |
| 29 | 2.984 | 0.05 | 26 | 114.751 | 0.152 | `triton_red_fused__to_copy_add_mean_mul_pow_rocm_unquantized_gemm_rsqrt_3.kd` |
| 30 | 2.142 | 0.04 | 26 | 82.400 | 0.099 | `triton_red_fused_7.kd` |
| 31 | 1.979 | 0.03 | 26 | 76.129 | 0.077 | `triton_poi_fused_6.kd` |
| 32 | 1.899 | 0.03 | 1 | 1899.359 | 1.899 | `Cijk_Alik_Bljk_BBH_MT32x16x4_SE_AMAS3_BW_DTL0_BL1_DRB16_ALT_KME0_ETSP_EPS1_FL0_GRVW2_GSU1_ISA936_IU1_KME_GRCMn1_K1_KLA_LDL1_LRVW2_MAC_MMA...` |
| 33 | 1.677 | 0.03 | 79 | 21.225 | 0.023 | `void at::native::vectorized_elementwise_kernel<8, at::native::FillFunctor<c10::BFloat16>, std::array<char*, 1ul>>(int, at::native::FillFu...` |
| 34 | 1.646 | 0.03 | 26 | 63.311 | 0.080 | `triton_per_fused_1.kd` |
| 35 | 1.512 | 0.03 | 77 | 19.641 | 0.023 | `void at::native::index_elementwise_kernel<128, 4, void at::native::gpu_index_kernel<void at::native::index_kernel_impl<at::native::Opaque...` |
| 36 | 1.490 | 0.03 | 80 | 18.620 | 0.022 | `void at::native::index_elementwise_kernel<128, 4, void at::native::gpu_index_kernel<void at::native::index_put_kernel_impl<at::native::Op...` |
| 37 | 1.031 | 0.02 | 26 | 39.643 | 0.084 | `reshape_and_cache_kernel_flash.kd` |
| 38 | 0.936 | 0.02 | 161 | 5.812 | 0.012 | `void at::native::elementwise_kernel_manual_unroll<128, 4, void at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::Tensor...` |
| 39 | 0.911 | 0.02 | 78 | 11.674 | 0.012 | `chunk_local_cumsum_scalar_kernel.kd` |
| 40 | 0.903 | 0.02 | 77 | 11.726 | 0.014 | `void at::native::elementwise_kernel_manual_unroll<128, 4, void at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::mask...` |

## last_180s

- rows: 11195
- summed_kernel_ms: 6703.937

| rank | total_ms | pct | count | avg_us | max_ms | kernel |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 2287.135 | 34.12 | 26 | 87966.715 | 152.672 | `kernel_unified_attention_2d.kd` |
| 2 | 1334.274 | 19.90 | 1915 | 696.749 | 1.356 | `chunk_fwd_kernel_o.kd` |
| 3 | 856.633 | 12.78 | 3733 | 229.476 | 0.253 | `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array<char*, 1ul>>(int, at::native::FillFunctor<int>...` |
| 4 | 451.036 | 6.73 | 331 | 1362.647 | 2.488 | `chunk_gated_delta_rule_fwd_kernel_h_blockdim64.kd` |
| 5 | 447.016 | 6.67 | 104 | 4298.233 | 4.514 | `Cijk_Alik_Bljk_BBH_MT256x256x16_SE_AMAS3_BW_DTL0_BL1_DRB16_ALT_KME0_ETSP_EPS1_FL0_GRVW8_GSU1_ISA936_IU1_KME_GRCMn1_K1_KLA_LDL1_LRVW8_MAC_...` |
| 6 | 335.188 | 5.00 | 209 | 1603.771 | 2.875 | `Cijk_Alik_Bljk_BBH_MT256x256x16_SE_AMAS3_BW_DTL0_BL1_DRB16_ALT_KME0_ETSP_EPS1_FL0_GRVW8_GSU1_ISA936_IU1_KME_GRCMn1_K1_KLA_LDL1_LRVW8_MAC_...` |
| 7 | 236.058 | 3.52 | 638 | 369.997 | 0.593 | `merge_16x16_to_64x64_inverse_kernel.kd` |
| 8 | 210.489 | 3.14 | 105 | 2004.657 | 2.151 | `Cijk_Alik_Bljk_BBH_MT256x256x16_SE_AMAS3_BW_DTL0_BL1_DRB16_ALT_KME0_ETSP_EPS1_FL0_GRVW8_GSU1_ISA936_IU1_KME_GRCMn1_K1_KLA_LDL1_LRVW8_MAC_...` |
| 9 | 190.582 | 2.84 | 1380 | 138.103 | 0.195 | `chunk_scaled_dot_kkt_fwd_kernel.kd` |
| 10 | 167.671 | 2.50 | 512 | 327.482 | 0.430 | `recompute_w_u_fwd_kernel.kd` |
| 11 | 30.333 | 0.45 | 78 | 388.880 | 0.412 | `_causal_conv1d_fwd_kernel.kd` |
| 12 | 19.462 | 0.29 | 52 | 374.261 | 0.393 | `triton_poi_fused_mul_rocm_unquantized_gemm_silu_slice_3.kd` |
| 13 | 12.356 | 0.18 | 234 | 52.805 | 0.096 | `void at::native::elementwise_kernel_manual_unroll<128, 8, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at:...` |
| 14 | 12.188 | 0.18 | 52 | 234.378 | 0.257 | `triton_red_fused__to_copy_add_copy__mean_mul_pow_rsqrt_4.kd` |
| 15 | 12.139 | 0.18 | 53 | 229.038 | 0.254 | `triton_poi_fused__to_copy__unsafe_view_add_clone_mean_mul_pow_reshape_rocm_unquantized_gemm_rsqrt_silu_view_1.kd` |
| 16 | 9.737 | 0.15 | 26 | 374.485 | 0.393 | `triton_poi_fused_mul_rocm_unquantized_gemm_silu_slice_2.kd` |
| 17 | 9.605 | 0.14 | 26 | 369.434 | 0.393 | `triton_poi_fused_mul_rocm_unquantized_gemm_silu_slice_4.kd` |
| 18 | 6.984 | 0.10 | 158 | 44.199 | 0.076 | `l2norm_fwd_kernel2.kd` |
| 19 | 6.477 | 0.10 | 78 | 83.042 | 0.092 | `fused_gdn_gating_kernel.kd` |
| 20 | 6.151 | 0.09 | 26 | 236.591 | 0.258 | `triton_red_fused__to_copy_add_copy__mean_mul_pow_rsqrt_3.kd` |
| 21 | 6.113 | 0.09 | 26 | 235.126 | 0.258 | `triton_red_fused__to_copy_add_copy__mean_mul_pow_rocm_unquantized_gemm_rsqrt_5.kd` |
| 22 | 6.113 | 0.09 | 53 | 115.336 | 0.158 | `triton_red_fused__to_copy_add_mean_mul_pow_rocm_unquantized_gemm_rsqrt_2.kd` |
| 23 | 5.993 | 0.09 | 26 | 230.498 | 0.274 | `triton_poi_fused__to_copy__unsafe_view_add_clone_mean_mul_pow_reshape_rocm_unquantized_gemm_rsqrt_silu_view_2.kd` |
| 24 | 5.084 | 0.08 | 79 | 64.356 | 0.067 | `Cijk_Alik_Bljk_BBH_MT128x64x32_SE_AMAS3_BW_DTL0_BL1_DRB16_ALT_KME0_ETSP_EPS1_FL0_GRVW4_GSU1_ISA936_IU1_KME_GRCMn1_K1_KLA_LDL1_LRVW4_MAC_M...` |
| 25 | 4.345 | 0.06 | 295 | 14.728 | 0.025 | `chunk_local_cumsum_scalar_kernel.kd` |
| 26 | 3.894 | 0.06 | 26 | 149.766 | 0.151 | `triton_poi_fused_8.kd` |
| 27 | 3.749 | 0.06 | 26 | 144.190 | 0.168 | `triton_poi_fused_mul_rocm_unquantized_gemm_sigmoid_view_0.kd` |
| 28 | 3.308 | 0.05 | 53 | 62.418 | 0.080 | `triton_per_fused__to_copy_mean_pow_view_0.kd` |
| 29 | 3.075 | 0.05 | 26 | 118.258 | 0.152 | `triton_red_fused__to_copy_add_mean_mul_pow_rocm_unquantized_gemm_rsqrt_1.kd` |
| 30 | 2.984 | 0.04 | 26 | 114.751 | 0.152 | `triton_red_fused__to_copy_add_mean_mul_pow_rocm_unquantized_gemm_rsqrt_3.kd` |
| 31 | 2.142 | 0.03 | 26 | 82.400 | 0.099 | `triton_red_fused_7.kd` |
| 32 | 1.979 | 0.03 | 26 | 76.129 | 0.077 | `triton_poi_fused_6.kd` |
| 33 | 1.899 | 0.03 | 1 | 1899.359 | 1.899 | `Cijk_Alik_Bljk_BBH_MT32x16x4_SE_AMAS3_BW_DTL0_BL1_DRB16_ALT_KME0_ETSP_EPS1_FL0_GRVW2_GSU1_ISA936_IU1_KME_GRCMn1_K1_KLA_LDL1_LRVW2_MAC_MMA...` |
| 34 | 1.699 | 0.03 | 80 | 21.238 | 0.023 | `void at::native::vectorized_elementwise_kernel<8, at::native::FillFunctor<c10::BFloat16>, std::array<char*, 1ul>>(int, at::native::FillFu...` |
| 35 | 1.646 | 0.02 | 26 | 63.311 | 0.080 | `triton_per_fused_1.kd` |
| 36 | 1.534 | 0.02 | 78 | 19.662 | 0.023 | `void at::native::index_elementwise_kernel<128, 4, void at::native::gpu_index_kernel<void at::native::index_kernel_impl<at::native::Opaque...` |
| 37 | 1.490 | 0.02 | 80 | 18.620 | 0.022 | `void at::native::index_elementwise_kernel<128, 4, void at::native::gpu_index_kernel<void at::native::index_put_kernel_impl<at::native::Op...` |
| 38 | 1.031 | 0.02 | 26 | 39.643 | 0.084 | `reshape_and_cache_kernel_flash.kd` |
| 39 | 0.956 | 0.01 | 163 | 5.863 | 0.014 | `void at::native::elementwise_kernel_manual_unroll<128, 4, void at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::Tensor...` |
| 40 | 0.916 | 0.01 | 78 | 11.737 | 0.014 | `void at::native::elementwise_kernel_manual_unroll<128, 4, void at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::mask...` |

## last_360s

- rows: 28074
- summed_kernel_ms: 8159.132

| rank | total_ms | pct | count | avg_us | max_ms | kernel |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 2302.939 | 28.23 | 202 | 11400.687 | 152.672 | `kernel_unified_attention_2d.kd` |
| 2 | 1334.274 | 16.35 | 1915 | 696.749 | 1.356 | `chunk_fwd_kernel_o.kd` |
| 3 | 856.689 | 10.50 | 3751 | 228.389 | 0.253 | `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array<char*, 1ul>>(int, at::native::FillFunctor<int>...` |
| 4 | 465.748 | 5.71 | 199 | 2340.441 | 4.514 | `Cijk_Alik_Bljk_BBH_MT256x256x16_SE_AMAS3_BW_DTL0_BL1_DRB16_ALT_KME0_ETSP_EPS1_FL0_GRVW8_GSU1_ISA936_IU1_KME_GRCMn1_K1_KLA_LDL1_LRVW8_MAC_...` |
| 5 | 451.036 | 5.53 | 331 | 1362.647 | 2.488 | `chunk_gated_delta_rule_fwd_kernel_h_blockdim64.kd` |
| 6 | 363.629 | 4.46 | 353 | 1030.110 | 2.875 | `Cijk_Alik_Bljk_BBH_MT256x256x16_SE_AMAS3_BW_DTL0_BL1_DRB16_ALT_KME0_ETSP_EPS1_FL0_GRVW8_GSU1_ISA936_IU1_KME_GRCMn1_K1_KLA_LDL1_LRVW8_MAC_...` |
| 7 | 236.058 | 2.89 | 638 | 369.997 | 0.593 | `merge_16x16_to_64x64_inverse_kernel.kd` |
| 8 | 218.817 | 2.68 | 736 | 297.306 | 0.755 | `Cijk_Alik_Bljk_BBH_MT32x64x128_SE_AMAS3_BW_DTL0_BL1_DRB16_ALT_KME0_ETSP_EPS1_FL0_GRVW4_GSU1_ISA936_IU1_KME_GRCMn1_K1_KLA_LDL1_LRVW4_MAC_M...` |
| 9 | 216.212 | 2.65 | 137 | 1578.192 | 2.151 | `Cijk_Alik_Bljk_BBH_MT256x256x16_SE_AMAS3_BW_DTL0_BL1_DRB16_ALT_KME0_ETSP_EPS1_FL0_GRVW8_GSU1_ISA936_IU1_KME_GRCMn1_K1_KLA_LDL1_LRVW8_MAC_...` |
| 10 | 190.582 | 2.34 | 1380 | 138.103 | 0.195 | `chunk_scaled_dot_kkt_fwd_kernel.kd` |
| 11 | 173.246 | 2.12 | 496 | 349.287 | 0.524 | `Cijk_Alik_Bljk_BBH_MT128x64x32_SE_AMAS3_BW_DTL0_BL1_DRB16_ALT_KME0_ETSP_EPS1_FL0_GRVW4_GSU1_ISA936_IU1_KME_GRCMn1_K1_KLA_LDL1_LRVW4_MAC_M...` |
| 12 | 167.671 | 2.06 | 512 | 327.482 | 0.430 | `recompute_w_u_fwd_kernel.kd` |
| 13 | 146.175 | 1.79 | 384 | 380.665 | 0.879 | `Cijk_Alik_Bljk_BBH_MT32x80x128_SE_AMAS3_BW_DTL0_BL1_DRB16_ALT_KME0_ETSP_EPS1_FL0_GRVW2_GSU1_ISA936_IU1_KME_GRCMn1_K1_KLA_LDL1_LRVW2_MAC_M...` |
| 14 | 117.342 | 1.44 | 864 | 135.812 | 0.320 | `fused_recurrent_gated_delta_rule_packed_decode_kernel.kd` |
| 15 | 77.254 | 0.95 | 192 | 402.365 | 0.612 | `Cijk_Alik_Bljk_BBH_MT32x16x256_SE_AMAS3_BW_DTL0_BL1_DRB16_ALT_KME0_ETSP_EPS1_FL0_GRVW2_GSU1_ISA936_IU1_KME_GRCMn1_K1_KLA_LDL1_LRVW2_MAC_M...` |
| 16 | 68.850 | 0.84 | 192 | 358.596 | 0.645 | `Cijk_Alik_Bljk_BBH_MT64x128x32_SE_AMAS3_BW_DTL0_BL1_DRB16_ALT_KME0_ETSP_EPS0_FL0_GRVW4_GSU8_ISA936_IU1_KME_GRCMn1_K1_KLA_LDL1_LRVW4_MAC_M...` |
| 17 | 62.006 | 0.76 | 128 | 484.420 | 0.681 | `Cijk_Alik_Bljk_BBH_MT32x32x128_SE_AMAS3_BW_DTL0_BL1_DRB16_ALT_KME0_ETSP_EPS1_FL0_GRVW4_GSU1_ISA936_IU1_KME_GRCMn1_K1_KLA_LDL1_LRVW4_MAC_M...` |
| 18 | 61.756 | 0.76 | 352 | 175.444 | 0.309 | `Cijk_Alik_Bljk_BBH_MT32x32x128_SE_AMAS3_BW_DTL0_BL1_DRB16_ALT_KME0_ETSP_EPS1_FL0_GRVW4_GSU1_ISA936_IU1_KME_GRCMn1_K1_KLA_LDL1_LRVW4_MAC_M...` |
| 19 | 50.220 | 0.62 | 416 | 120.721 | 0.300 | `Cijk_Alik_Bljk_BBH_MT32x32x4_SE_AMAS3_BW_DTL0_BL1_DRB16_ALT_KME0_ETSP_EPS1_FL0_GRVW4_GSU1_ISA936_IU1_KME_GRCMn1_K1_KLA_LDL1_LRVW4_MAC_MMA...` |
| 20 | 44.435 | 0.54 | 465 | 95.558 | 1.899 | `Cijk_Alik_Bljk_BBH_MT32x16x4_SE_AMAS3_BW_DTL0_BL1_DRB16_ALT_KME0_ETSP_EPS1_FL0_GRVW2_GSU1_ISA936_IU1_KME_GRCMn1_K1_KLA_LDL1_LRVW2_MAC_MMA...` |
| 21 | 43.864 | 0.54 | 112 | 391.643 | 0.528 | `Cijk_Alik_Bljk_BBH_MT64x32x32_SE_AMAS3_BW_DTL0_BL1_DRB16_ALT_KME0_ETSP_EPS1_FL0_GRVW4_GSU1_ISA936_IU1_KME_GRCMn2_K1_KLA_LDL1_LRVW4_MAC_MM...` |
| 22 | 43.014 | 0.53 | 527 | 81.621 | 0.309 | `Cijk_Alik_Bljk_BBH_MT16x16x256_SE_AMAS3_BW_DTL0_BL1_DRB16_ALT_KME0_ETSP_EPS1_FL0_GRVW2_GSU1_ISA936_IU1_KME_GRCMn1_K1_KLA_LDL1_LRVW2_MAC_M...` |
| 23 | 38.584 | 0.47 | 272 | 141.852 | 0.291 | `Cijk_Alik_Bljk_BBH_MT32x16x128_SE_AMAS3_BW_DTL0_BL1_DRB16_ALT_KME0_ETSP_EPS1_FL0_GRVW2_GSU1_ISA936_IU1_KME_GRCMn1_K1_KLA_LDL1_LRVW2_MAC_M...` |
| 24 | 35.619 | 0.44 | 128 | 278.274 | 0.307 | `Cijk_Alik_Bljk_BBH_MT64x64x64_SE_AMAS3_BW_DTL0_BL1_DRB16_ALT_KME0_ETSP_EPS1_FL0_GRVW4_GSU1_ISA936_IU1_KME_GRCMn1_K1_KLA_LDL1_LRVW4_MAC_MM...` |
| 25 | 32.176 | 0.39 | 1 | 32175.662 | 32.176 | `_topk_topp_kernel.kd` |
| 26 | 30.333 | 0.37 | 78 | 388.880 | 0.412 | `_causal_conv1d_fwd_kernel.kd` |
| 27 | 28.748 | 0.35 | 320 | 89.839 | 0.175 | `Cijk_Alik_Bljk_BBH_MT128x128x32_SE_AMAS3_BW_DTL0_BL1_DRB16_ALT_KME0_ETSP_EPS1_FL0_GRVW8_GSU1_ISA936_IU1_KME_GRCMn1_K1_KLA_LDL1_LRVW8_MAC_...` |
| 28 | 25.191 | 0.31 | 660 | 38.168 | 0.393 | `triton_poi_fused_mul_rocm_unquantized_gemm_silu_slice_3.kd` |
| 29 | 17.620 | 0.22 | 864 | 20.393 | 0.086 | `_causal_conv1d_update_kernel.kd` |
| 30 | 16.970 | 0.21 | 660 | 25.712 | 0.257 | `triton_red_fused__to_copy_add_copy__mean_mul_pow_rsqrt_4.kd` |
| 31 | 16.504 | 0.20 | 661 | 24.968 | 0.254 | `triton_poi_fused__to_copy__unsafe_view_add_clone_mean_mul_pow_reshape_rocm_unquantized_gemm_rsqrt_silu_view_1.kd` |
| 32 | 15.089 | 0.18 | 207 | 72.892 | 0.082 | `Cijk_Alik_Bljk_BBH_MT128x64x32_SE_AMAS3_BW_DTL0_BL1_DRB16_ALT_KME0_ETSP_EPS1_FL0_GRVW4_GSU1_ISA936_IU1_KME_GRCMn1_K1_KLA_LDL1_LRVW4_MAC_M...` |
| 33 | 12.608 | 0.15 | 330 | 38.205 | 0.393 | `triton_poi_fused_mul_rocm_unquantized_gemm_silu_slice_2.kd` |
| 34 | 12.478 | 0.15 | 330 | 37.811 | 0.393 | `triton_poi_fused_mul_rocm_unquantized_gemm_silu_slice_4.kd` |
| 35 | 12.356 | 0.15 | 234 | 52.805 | 0.096 | `void at::native::elementwise_kernel_manual_unroll<128, 8, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at:...` |
| 36 | 10.849 | 0.13 | 128 | 84.756 | 0.091 | `Cijk_Alik_Bljk_BBH_MT128x128x32_SE_AMAS3_BW_DTL0_BL1_DRB16_ALT_KME0_ETSP_EPS1_FL0_GRVW8_GSU1_ISA936_IU1_KME_GRCMn1_K1_KLA_LDL1_LRVW8_MAC_...` |
| 37 | 10.065 | 0.12 | 661 | 15.227 | 0.158 | `triton_red_fused__to_copy_add_mean_mul_pow_rocm_unquantized_gemm_rsqrt_2.kd` |
| 38 | 8.655 | 0.11 | 330 | 26.227 | 0.258 | `triton_red_fused__to_copy_add_copy__mean_mul_pow_rsqrt_3.kd` |
| 39 | 8.632 | 0.11 | 330 | 26.158 | 0.258 | `triton_red_fused__to_copy_add_copy__mean_mul_pow_rocm_unquantized_gemm_rsqrt_5.kd` |
| 40 | 8.309 | 0.10 | 48 | 173.103 | 0.260 | `Cijk_Alik_Bljk_BBH_MT64x16x128_SE_AMAS3_BW_DTL0_BL1_DRB16_ALT_KME0_ETSP_EPS1_FL0_GRVW2_GSU1_ISA936_IU1_KME_GRCMn1_K1_KLA_LDL1_LRVW2_MAC_M...` |

## Focused Kernel Names

| rank | total_ms | count | avg_us | max_ms | kernel |
|---:|---:|---:|---:|---:|---|
| 1 | 5580.268 | 23717 | 235.286 | 0.253 | `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array<char*, 1ul>>(int, at::native::FillFunctor<int>, std::array<char*, 1ul>) (.kd)` |
| 2 | 1537.535 | 14633 | 105.073 | 1.356 | `chunk_fwd_kernel_o.kd` |
| 3 | 475.429 | 1249 | 380.648 | 2.488 | `chunk_gated_delta_rule_fwd_kernel_h_blockdim64.kd` |
| 4 | 255.873 | 8306 | 30.806 | 0.195 | `chunk_scaled_dot_kkt_fwd_kernel.kd` |
| 5 | 132.099 | 912 | 144.845 | 0.343 | `fused_recurrent_gated_delta_rule_packed_decode_kernel.kd` |
| 6 | 99.490 | 400 | 248.725 | 0.928 | `void at::native::elementwise_kernel_manual_unroll<128, 8, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::'lambda1'()::oper...` |
| 7 | 63.097 | 28 | 2253.472 | 2.391 | `void at::native::vectorized_elementwise_kernel<8, at::native::GeluCUDAKernelImpl(at::TensorIteratorBase&, at::native::GeluType)::'lambda0'()::operator()() const::'lambda2'()::op...` |
| 8 | 43.654 | 1364 | 32.005 | 0.394 | `triton_poi_fused_mul_rocm_unquantized_gemm_silu_slice_3.kd` |
| 9 | 41.865 | 55 | 761.184 | 0.797 | `void at::native::(anonymous namespace)::vectorized_layer_norm_kernel<c10::BFloat16, float, false>(int, float, c10::BFloat16 const*, c10::BFloat16 const*, c10::BFloat16 const*, f...` |
| 10 | 30.333 | 78 | 388.880 | 0.412 | `_causal_conv1d_fwd_kernel.kd` |
| 11 | 30.103 | 1364 | 22.069 | 0.257 | `triton_red_fused__to_copy_add_copy__mean_mul_pow_rsqrt_4.kd` |
| 12 | 29.142 | 1365 | 21.349 | 0.256 | `triton_poi_fused__to_copy__unsafe_view_add_clone_mean_mul_pow_reshape_rocm_unquantized_gemm_rsqrt_silu_view_1.kd` |
| 13 | 21.885 | 682 | 32.090 | 0.394 | `triton_poi_fused_mul_rocm_unquantized_gemm_silu_slice_4.kd` |
| 14 | 21.838 | 682 | 32.021 | 0.393 | `triton_poi_fused_mul_rocm_unquantized_gemm_silu_slice_2.kd` |
| 15 | 19.620 | 912 | 21.514 | 0.087 | `_causal_conv1d_update_kernel.kd` |
| 16 | 19.511 | 55 | 354.746 | 0.393 | `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat16>, std::array<char*, 3ul>>(int, at::native::CUDAFunctor_add<c10::BFloat16>, std::array...` |
| 17 | 18.286 | 1365 | 13.397 | 0.158 | `triton_red_fused__to_copy_add_mean_mul_pow_rocm_unquantized_gemm_rsqrt_2.kd` |
| 18 | 15.384 | 682 | 22.558 | 0.151 | `triton_poi_fused_8.kd` |
| 19 | 15.280 | 682 | 22.405 | 0.258 | `triton_red_fused__to_copy_add_copy__mean_mul_pow_rocm_unquantized_gemm_rsqrt_5.kd` |
| 20 | 15.120 | 682 | 22.170 | 0.258 | `triton_red_fused__to_copy_add_copy__mean_mul_pow_rsqrt_3.kd` |
| 21 | 14.553 | 682 | 21.338 | 0.274 | `triton_poi_fused__to_copy__unsafe_view_add_clone_mean_mul_pow_reshape_rocm_unquantized_gemm_rsqrt_silu_view_2.kd` |
| 22 | 14.336 | 682 | 21.020 | 0.024 | `triton_poi_fused_0.kd` |
| 23 | 13.879 | 3280 | 4.232 | 0.047 | `void at::native::vectorized_elementwise_kernel<8, at::native::FillFunctor<c10::BFloat16>, std::array<char*, 1ul>>(int, at::native::FillFunctor<c10::BFloat16>, std::array<char*, ...` |
| 24 | 13.028 | 1365 | 9.544 | 0.080 | `triton_per_fused__to_copy_mean_pow_view_0.kd` |
| 25 | 10.761 | 682 | 15.779 | 0.168 | `triton_poi_fused_mul_rocm_unquantized_gemm_sigmoid_view_0.kd` |
| 26 | 10.664 | 2019 | 5.282 | 0.025 | `chunk_local_cumsum_scalar_kernel.kd` |
| 27 | 9.216 | 682 | 13.514 | 0.156 | `triton_red_fused__to_copy_add_mean_mul_pow_rocm_unquantized_gemm_rsqrt_3.kd` |
| 28 | 9.162 | 682 | 13.435 | 0.157 | `triton_red_fused__to_copy_add_mean_mul_pow_rocm_unquantized_gemm_rsqrt_1.kd` |
| 29 | 7.303 | 2650 | 2.756 | 0.004 | `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<long>, std::array<char*, 1ul>>(int, at::native::FillFunctor<long>, std::array<char*, 1ul>) (.kd)` |
| 30 | 6.889 | 682 | 10.101 | 0.080 | `triton_per_fused_1.kd` |
| 31 | 6.519 | 682 | 9.559 | 0.099 | `triton_red_fused_7.kd` |
| 32 | 6.477 | 78 | 83.042 | 0.092 | `fused_gdn_gating_kernel.kd` |
| 33 | 6.118 | 16 | 382.400 | 0.420 | `void at::native::vectorized_elementwise_kernel<16, at::native::FillFunctor<signed char>, std::array<char*, 1ul>>(int, at::native::FillFunctor<signed char>, std::array<char*, 1ul...` |
| 34 | 5.999 | 682 | 8.796 | 0.127 | `triton_poi_fused_6.kd` |
| 35 | 5.716 | 721 | 7.927 | 0.044 | `void at::native::(anonymous namespace)::distribution_elementwise_grid_stride_kernel<float, 4, void at::native::templates::cuda::normal_and_transform<c10::BFloat16, float, at::CU...` |
| 36 | 4.912 | 1364 | 3.601 | 0.005 | `triton_poi_fused_5.kd` |
| 37 | 3.094 | 1 | 3093.758 | 3.094 | `void at::native::reduce_kernel<128, 4, at::native::ReduceOp<c10::BFloat16, at::native::func_wrapper_t<c10::BFloat16, at::native::sum_functor<c10::BFloat16, float, c10::BFloat16>...` |
| 38 | 2.754 | 770 | 3.577 | 0.005 | `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctorOnSelf_add<long>, std::array<char*, 2ul>>(int, at::native::CUDAFunctorOnSelf_add<long>, std::array<char*...` |
| 39 | 2.312 | 640 | 3.612 | 0.005 | `triton_poi_fused_4.kd` |
| 40 | 1.718 | 384 | 4.474 | 0.005 | `void at::native::vectorized_elementwise_kernel<4, at::native::BUnaryFunctor<long, long, long, at::native::binary_internal::div_floor_kernel_cuda(at::TensorIteratorBase&)::'lambd...` |
| 41 | 1.534 | 78 | 19.662 | 0.023 | `void at::native::index_elementwise_kernel<128, 4, void at::native::gpu_index_kernel<void at::native::index_kernel_impl<at::native::OpaqueType<4>>(at::TensorIteratorBase&, c10::A...` |
| 42 | 1.490 | 80 | 18.620 | 0.022 | `void at::native::index_elementwise_kernel<128, 4, void at::native::gpu_index_kernel<void at::native::index_put_kernel_impl<at::native::OpaqueType<4>>(at::TensorIterator&, c10::A...` |
| 43 | 1.431 | 79 | 18.118 | 0.982 | `void at::native::vectorized_gather_kernel<16, long>(char*, char*, long*, int, long, long, long, long, bool) (.kd)` |
| 44 | 1.217 | 1 | 1217.120 | 1.217 | `void at::native::elementwise_kernel_manual_unroll<128, 8, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, at::nati...` |
| 45 | 0.979 | 165 | 5.932 | 0.014 | `void at::native::elementwise_kernel_manual_unroll<128, 4, void at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::'lambda1'()::operator()(...` |
| 46 | 0.916 | 78 | 11.737 | 0.014 | `void at::native::elementwise_kernel_manual_unroll<128, 4, void at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10:...` |
| 47 | 0.858 | 144 | 5.960 | 0.008 | `void at::native::(anonymous namespace)::CatArrayBatchedCopy_contig<at::native::(anonymous namespace)::OpaqueType<8u>, unsigned int, 1, 128, 1>(at::native::(anonymous namespace):...` |
| 48 | 0.851 | 2 | 425.359 | 0.426 | `void at::native::(anonymous namespace)::cunn_SoftMaxForwardGmem<4, float, float, float, at::native::(anonymous namespace)::SoftMaxForwardWithMulEpilogue, long>(float*, float con...` |
| 49 | 0.836 | 221 | 3.783 | 0.006 | `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, std::array<char*, 1ul>>(int, at::native::FillFunctor<float>, std::array<char*, 1ul>) (.kd)` |
| 50 | 0.751 | 206 | 3.646 | 0.005 | `void at::native::unrolled_elementwise_kernel<at::native::CUDAFunctor_add<int>, std::array<char*, 3ul>, 4, TrivialOffsetCalculator<2, unsigned int>, TrivialOffsetCalculator<1, un...` |
| 51 | 0.594 | 43 | 13.812 | 0.102 | `triton_red_fused__to_copy_add_mean_mul_pow_rocm_unquantized_gemm_rsqrt_0.kd` |
| 52 | 0.589 | 5 | 117.792 | 0.143 | `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::ArgMaxOps<float>, unsigned int, long, 4, 4>>(at::native::ReduceOp<float, at::native::ArgMaxOps<flo...` |
| 53 | 0.584 | 2 | 292.080 | 0.294 | `void at::native::vectorized_elementwise_kernel<4, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::DivFunctor<float>>, std::array<char*, 3ul>>(int, at...` |
| 54 | 0.561 | 3 | 186.987 | 0.540 | `void at::native::(anonymous namespace)::CatArrayBatchedCopy_contig<at::native::(anonymous namespace)::OpaqueType<4u>, unsigned int, 2, 128, 1>(at::native::(anonymous namespace):...` |
| 55 | 0.545 | 144 | 3.783 | 0.005 | `void at::native::unrolled_elementwise_kernel<at::native::CUDAFunctor_add<long>, std::array<char*, 3ul>, 4, TrivialOffsetCalculator<2, unsigned int>, TrivialOffsetCalculator<1, u...` |
| 56 | 0.531 | 2 | 265.520 | 0.266 | `void at::native::(anonymous namespace)::distribution_elementwise_grid_stride_kernel<float, 4, void at::native::templates::cuda::uniform_and_transform<float, float, at::CUDAGener...` |
| 57 | 0.458 | 78 | 5.871 | 0.007 | `void at::native::vectorized_elementwise_kernel<16, at::native::bitwise_not_kernel_cuda(at::TensorIteratorBase&)::'lambda'(bool), std::array<char*, 2ul>>(int, at::native::bitwise...` |
| 58 | 0.428 | 2 | 214.160 | 0.215 | `void at::native::elementwise_kernel_manual_unroll<128, 4, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::Div...` |
| 59 | 0.409 | 120 | 3.409 | 0.004 | `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<int>, std::array<char*, 3ul>>(int, at::native::CUDAFunctor_add<int>, std::array<char*, 3ul>) (.kd)` |
| 60 | 0.405 | 2 | 202.320 | 0.399 | `void at::native::vectorized_elementwise_kernel<4, at::native::sin_kernel_cuda(at::TensorIteratorBase&)::'lambda0'()::operator()() const::'lambda0'()::operator()() const::'lambda...` |
| 61 | 0.396 | 2 | 197.760 | 0.389 | `void at::native::vectorized_elementwise_kernel<4, at::native::cos_kernel_cuda(at::TensorIteratorBase&)::'lambda0'()::operator()() const::'lambda0'()::operator()() const::'lambda...` |
| 62 | 0.312 | 3 | 103.893 | 0.154 | `void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16tofloat32_copy_kernel_cuda(at::TensorIteratorBase&)::'lambda'(c10::BFloat16), std::array<char*, 2ul>>(int, ...` |
| 63 | 0.309 | 3 | 102.933 | 0.300 | `void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::'lambda'(float), std::array<char*, 2ul>>(int, at::native::bfloa...` |
| 64 | 0.247 | 3 | 82.346 | 0.229 | `void at::native::elementwise_kernel_manual_unroll<128, 4, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::Mul...` |
| 65 | 0.220 | 65 | 3.382 | 0.004 | `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctorOnSelf_add<int>, std::array<char*, 2ul>>(int, at::native::CUDAFunctorOnSelf_add<int>, std::array<char*, ...` |
| 66 | 0.155 | 43 | 3.602 | 0.004 | `triton_poi_fused_1.kd` |
| 67 | 0.137 | 7 | 19.543 | 0.038 | `void at::native::index_elementwise_kernel<128, 4, void at::native::gpu_index_kernel<void at::native::index_kernel_impl<at::native::OpaqueType<2>>(at::TensorIteratorBase&, c10::A...` |
| 68 | 0.092 | 11 | 8.407 | 0.011 | `void at::native::elementwise_kernel_manual_unroll<128, 4, void at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::'lambda1'()::operator()(...` |
| 69 | 0.049 | 6 | 8.240 | 0.009 | `void at::native::vectorized_elementwise_kernel<4, void at::native::compare_scalar_kernel<int>(at::TensorIteratorBase&, at::native::(anonymous namespace)::OpType, int)::'lambda'(...` |
| 70 | 0.037 | 4 | 9.280 | 0.028 | `void (anonymous namespace)::elementwise_kernel_with_index<int, at::native::arange_cuda_out(c10::Scalar const&, c10::Scalar const&, c10::Scalar const&, at::Tensor&)::'lambda'()::...` |
| 71 | 0.037 | 4 | 9.240 | 0.010 | `void at::native::vectorized_elementwise_kernel<4, void at::native::compare_scalar_kernel<float>(at::TensorIteratorBase&, at::native::(anonymous namespace)::OpType, float)::'lamb...` |
| 72 | 0.029 | 2 | 14.480 | 0.015 | `void at::native::elementwise_kernel_manual_unroll<128, 4, void at::native::gpu_kernel_impl<at::native::CUDAFunctor_add<float>>(at::TensorIteratorBase&, at::native::CUDAFunctor_a...` |
| 73 | 0.024 | 2 | 12.160 | 0.012 | `void at::native::(anonymous namespace)::distribution_elementwise_grid_stride_kernel<float, 4, void at::native::templates::cuda::uniform_and_transform<c10::BFloat16, float, at::C...` |
| 74 | 0.023 | 2 | 11.280 | 0.011 | `void at::native::elementwise_kernel_manual_unroll<128, 4, void at::native::gpu_kernel_impl<at::native::(anonymous namespace)::pow_tensor_tensor_kernel(at::TensorIteratorBase&)::...` |
| 75 | 0.020 | 2 | 10.080 | 0.011 | `void at::native::(anonymous namespace)::CatArrayBatchedCopy<at::native::(anonymous namespace)::OpaqueType<8u>, unsigned int, 2, 64, 64>(at::native::(anonymous namespace)::Opaque...` |
| 76 | 0.019 | 2 | 9.360 | 0.010 | `void at::native::elementwise_kernel_manual_unroll<128, 4, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::'lambda1'()::oper...` |
| 77 | 0.018 | 4 | 4.560 | 0.005 | `void at::native::vectorized_elementwise_kernel<4, at::native::BUnaryFunctor<int, int, int, at::native::binary_internal::div_floor_kernel_cuda(at::TensorIteratorBase&)::'lambda'(...` |
| 78 | 0.015 | 4 | 3.680 | 0.005 | `void at::native::vectorized_elementwise_kernel<16, at::native::FillFunctor<bool>, std::array<char*, 1ul>>(int, at::native::FillFunctor<bool>, std::array<char*, 1ul>) (.kd)` |
| 79 | 0.014 | 2 | 7.120 | 0.008 | `void at::native::elementwise_kernel_manual_unroll<128, 4, void at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<float>>(at::TensorIteratorBase&, at::native::CUDAFu...` |
| 80 | 0.013 | 2 | 6.640 | 0.008 | `void at::native::(anonymous namespace)::CatArrayBatchedCopy_contig<at::native::(anonymous namespace)::OpaqueType<4u>, unsigned int, 1, 128, 1>(at::native::(anonymous namespace):...` |
