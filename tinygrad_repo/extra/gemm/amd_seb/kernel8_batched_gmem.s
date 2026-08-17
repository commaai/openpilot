	.text
	.amdgcn_target "amdgcn-amd-amdhsa--gfx1100"
	.protected	kernel                  ; -- Begin function kernel
	.globl	kernel
	.p2align	8
	.type	kernel,@function
kernel:                                 ; @kernel
; %bb.0:                                ; %.preheader193

	;; Init code for matrix A and B buffer Loads  - START
	s_load_b128 s[20:23], s[0:1], 0x0 ; Matrix A and B
	s_waitcnt lgkmcnt(0)

	; Matrix B offsets:
	; input is s[22:23]
	; offset base addresses s[24:39]

	s_add_u32 s24, s22, 0x0000
	s_addc_u32 s25, s23, 0
	s_add_u32 s26, s22, 0x4000
	s_addc_u32 s27, s23, 0
	s_add_u32 s28, s22, 0x8000
	s_addc_u32 s29, s23, 0
	s_add_u32 s30, s22, 0xc000
	s_addc_u32 s31, s23, 0
	s_add_u32 s32, s22, 0x10000
	s_addc_u32 s33, s23, 0
	s_add_u32 s34, s22, 0x14000
	s_addc_u32 s35, s23, 0
	s_add_u32 s36, s22, 0x18000
	s_addc_u32 s37, s23, 0
	s_add_u32 s38, s22, 0x1c000
	s_addc_u32 s39, s23, 0

	; compute Matrix B offset
	s_lshl_b32 s19, s14, 7         		; BN * blockIdx.x


	v_add_nc_u32_e32 v203, s19, v0 		; index = BN * blockIdx.x + threadIdx.x
	v_lshlrev_b32_e32  v203,2, v203     ; offset = 4*index (VPGR offset in global_load are in bytes when using SPGR addressing)

	; Matrix A offsets:
	; input is s[20:21]
	; offset base addresses s[40:55]
	s_add_u32 s40, s20, 0x0000
	s_addc_u32 s41, s21, 0
	s_add_u32 s42, s20, 0x40000
	s_addc_u32 s43, s21, 0
	s_add_u32 s44, s20, 0x80000
	s_addc_u32 s45, s21, 0
	s_add_u32 s46, s20, 0xc0000
	s_addc_u32 s47, s21, 0
	s_add_u32 s48, s20, 0x100000
	s_addc_u32 s49, s21, 0
	s_add_u32 s50, s20, 0x140000
	s_addc_u32 s51, s21, 0
	s_add_u32 s52, s20, 0x180000
	s_addc_u32 s53, s21, 0
	s_add_u32 s54, s20, 0x1c0000
	s_addc_u32 s55, s21, 0

	; compute Matrix A offset
	s_lshl_b32 s19, s15, 19          ; 4096 * 128 * blockIdx.y
	v_lshrrev_b32_e32 v1, 3, v0		 ; threadIdx.x / 8
	v_lshlrev_b32_e32 v1, 12, v1     ; 4096 * (threadIdx.x/8)
	v_and_b32_e32 v215, 7, v0 		 ; threadIdx.x % 8
	v_add_u32_e32 v215, v1, v215     ; index = 4096*(threadIdx.x/8) + threadIdx.x % 8
	v_add_nc_u32_e32 v215, s19, v215 ; index += 4096*128*blockIdx.y
	v_lshlrev_b32_e32  v215,2, v215  ; offset = 4*index




	;; Init code for matrix A and B buffer Loads  - END


	s_clause 0x1
	; s_load_b128 s[4:7], s[0:1], 0x18
	; N=4096, alpha=1.0, beta=0.0
	s_mov_b32 s4, 4096
	s_mov_b32 s5, 0x3F800000
	s_mov_b32 s6, 0
	s_load_b128 s[8:11], s[0:1], 0x0
	s_lshl_b32 s2, s14, 7
	v_lshrrev_b32_e32 v4, 3, v0
	v_or_b32_e32 v1, s2, v0
	s_lshl_b32 s3, s15, 7
	v_and_b32_e32 v118, 7, v0
	s_bfe_i32 s12, s15, 0x10018
	v_or_b32_e32 v22, s3, v4
	v_ashrrev_i32_e32 v2, 31, v1
	s_lshr_b32 s12, s12, 25
	s_load_b64 s[0:1], s[0:1], 0x10
	v_lshlrev_b32_e32 v135, 2, v118
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_lshlrev_b64 v[5:6], 2, v[1:2]
	s_waitcnt lgkmcnt(0)
	v_add_nc_u32_e32 v3, s4, v1
	v_mul_lo_u32 v119, v22, s4
	v_add_co_u32 v5, vcc_lo, s10, v5
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_add_co_ci_u32_e32 v6, vcc_lo, s11, v6, vcc_lo
	v_add_nc_u32_e32 v7, s4, v3
	v_ashrrev_i32_e32 v4, 31, v3
	s_lshl_b32 s7, s4, 4
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_nc_u32_e32 v125, s7, v119
	v_add_nc_u32_e32 v2, s4, v7
	v_ashrrev_i32_e32 v8, 31, v7
	v_lshlrev_b64 v[9:10], 2, v[3:4]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_add_nc_u32_e32 v130, s7, v125
	v_add_nc_u32_e32 v11, s4, v2
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshlrev_b64 v[7:8], 2, v[7:8]
	v_add_co_u32 v9, vcc_lo, s10, v9
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_add_nc_u32_e32 v13, s4, v11
	v_ashrrev_i32_e32 v12, 31, v11
	v_lshlrev_b64 v[2:3], 2, v[2:3]
	v_add_co_ci_u32_e32 v10, vcc_lo, s11, v10, vcc_lo
	v_ashrrev_i32_e32 v14, 31, v13
	v_add_co_u32 v7, vcc_lo, s10, v7
	v_lshlrev_b64 v[11:12], 2, v[11:12]
	v_add_co_ci_u32_e32 v8, vcc_lo, s11, v8, vcc_lo
	v_add_nc_u32_e32 v15, s4, v13
	v_add_co_u32 v2, vcc_lo, s10, v2
	v_lshlrev_b64 v[13:14], 2, v[13:14]
	v_add_co_ci_u32_e32 v3, vcc_lo, s11, v3, vcc_lo
	v_add_co_u32 v11, vcc_lo, s10, v11
	v_add_nc_u32_e32 v4, s4, v15
	v_add_co_ci_u32_e32 v12, vcc_lo, s11, v12, vcc_lo
	v_add_co_u32 v13, vcc_lo, s10, v13
	v_ashrrev_i32_e32 v16, 31, v15
	v_add_nc_u32_e32 v134, s7, v130
	v_add_co_ci_u32_e32 v14, vcc_lo, s11, v14, vcc_lo
	s_clause 0x5
	global_load_b32 v23, v[5:6], off
	global_load_b32 v24, v[9:10], off
	global_load_b32 v25, v[7:8], off
	global_load_b32 v26, v[2:3], off
	global_load_b32 v27, v[11:12], off
	global_load_b32 v28, v[13:14], off
	v_add_nc_u32_e32 v6, v119, v118
	v_ashrrev_i32_e32 v5, 31, v4
	v_add_nc_u32_e32 v8, v125, v118
	v_lshlrev_b64 v[2:3], 2, v[15:16]
	v_add_nc_u32_e32 v137, s7, v134
	v_ashrrev_i32_e32 v7, 31, v6
	v_add_nc_u32_e32 v10, v130, v118
	v_lshlrev_b64 v[4:5], 2, v[4:5]
	v_ashrrev_i32_e32 v9, 31, v8
	v_add_nc_u32_e32 v12, v134, v118
	v_add_nc_u32_e32 v138, s7, v137
	v_add_co_u32 v2, vcc_lo, s10, v2
	v_lshlrev_b64 v[6:7], 2, v[6:7]
	v_ashrrev_i32_e32 v11, 31, v10
	v_add_co_ci_u32_e32 v3, vcc_lo, s11, v3, vcc_lo
	v_add_nc_u32_e32 v14, v137, v118
	v_add_co_u32 v4, vcc_lo, s10, v4
	v_lshlrev_b64 v[8:9], 2, v[8:9]
	v_ashrrev_i32_e32 v13, 31, v12
	v_add_nc_u32_e32 v139, s7, v138
	v_add_co_ci_u32_e32 v5, vcc_lo, s11, v5, vcc_lo
	v_add_co_u32 v6, vcc_lo, s8, v6
	v_lshlrev_b64 v[10:11], 2, v[10:11]
	v_ashrrev_i32_e32 v15, 31, v14
	v_add_co_ci_u32_e32 v7, vcc_lo, s9, v7, vcc_lo
	v_add_nc_u32_e32 v16, v138, v118
	v_add_co_u32 v8, vcc_lo, s8, v8
	v_lshlrev_b64 v[12:13], 2, v[12:13]
	v_add_nc_u32_e32 v140, s7, v139
	v_add_co_ci_u32_e32 v9, vcc_lo, s9, v9, vcc_lo
	v_add_nc_u32_e32 v18, v139, v118
	v_add_co_u32 v10, vcc_lo, s8, v10
	v_lshlrev_b64 v[14:15], 2, v[14:15]
	v_ashrrev_i32_e32 v17, 31, v16
	v_add_co_ci_u32_e32 v11, vcc_lo, s9, v11, vcc_lo
	v_add_nc_u32_e32 v20, v140, v118
	v_add_co_u32 v12, vcc_lo, s8, v12
	v_ashrrev_i32_e32 v19, 31, v18
	v_add_co_ci_u32_e32 v13, vcc_lo, s9, v13, vcc_lo
	v_add_co_u32 v14, vcc_lo, s8, v14
	v_lshlrev_b64 v[16:17], 2, v[16:17]
	v_ashrrev_i32_e32 v21, 31, v20
	v_add_co_ci_u32_e32 v15, vcc_lo, s9, v15, vcc_lo
	s_clause 0x4
	global_load_b32 v29, v[6:7], off
	global_load_b32 v30, v[8:9], off
	global_load_b32 v31, v[10:11], off
	global_load_b32 v12, v[12:13], off
	global_load_b32 v13, v[14:15], off
	v_lshlrev_b64 v[6:7], 2, v[18:19]
	v_add_co_u32 v8, vcc_lo, s8, v16
	v_lshlrev_b64 v[10:11], 2, v[20:21]
	v_add_co_ci_u32_e32 v9, vcc_lo, s9, v17, vcc_lo
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_co_u32 v6, vcc_lo, s8, v6
	v_add_co_ci_u32_e32 v7, vcc_lo, s9, v7, vcc_lo
	v_add_co_u32 v10, vcc_lo, s8, v10
	v_add_co_ci_u32_e32 v11, vcc_lo, s9, v11, vcc_lo
	s_clause 0x1
	global_load_b32 v3, v[2:3], off
	global_load_b32 v4, v[4:5], off
	s_clause 0x2
	global_load_b32 v5, v[8:9], off
	global_load_b32 v6, v[6:7], off
	global_load_b32 v7, v[10:11], off
	v_add_nc_u32_e32 v9, s12, v22
	v_or_b32_e32 v10, 16, v22
	v_or_b32_e32 v11, 32, v22
	v_or_b32_e32 v14, 48, v22
	v_or_b32_e32 v15, 64, v22
	v_or_b32_e32 v16, 0x50, v22
	v_or_b32_e32 v17, 0x60, v22
	v_or_b32_e32 v18, 0x70, v22
	s_bfe_i32 s7, s14, 0x10018
	v_and_b32_e32 v9, 0x3fffff80, v9
	s_lshr_b32 s7, s7, 25
	v_add_nc_u32_e32 v19, s12, v10
	v_add_nc_u32_e32 v8, s7, v1
	v_add_nc_u32_e32 v20, s12, v11
	v_add_nc_u32_e32 v21, s12, v14
	v_add_nc_u32_e32 v32, s12, v15
	v_add_nc_u32_e32 v33, s12, v16
	v_add_nc_u32_e32 v34, s12, v17
	v_add_nc_u32_e32 v35, s12, v18
	v_and_b32_e32 v8, 0x3fffff80, v8
	v_sub_nc_u32_e32 v9, v22, v9
	v_and_b32_e32 v19, 0x3fffff80, v19
	v_and_b32_e32 v20, 0x3fffff80, v20
	v_and_b32_e32 v21, 0x3fffff80, v21
	v_and_b32_e32 v22, 0x3fffff80, v32
	v_and_b32_e32 v32, 0x3fffff80, v33
	v_and_b32_e32 v33, 0x3fffff80, v34
	v_and_b32_e32 v34, 0x3fffff80, v35
	v_sub_nc_u32_e32 v8, v1, v8
	v_lshlrev_b32_e32 v9, 2, v9
	v_sub_nc_u32_e32 v10, v10, v19
	v_sub_nc_u32_e32 v11, v11, v20
	v_sub_nc_u32_e32 v14, v14, v21
	v_sub_nc_u32_e32 v15, v15, v22
	v_sub_nc_u32_e32 v16, v16, v32
	v_sub_nc_u32_e32 v17, v17, v33
	v_sub_nc_u32_e32 v18, v18, v34
	v_bfe_u32 v2, v0, 3, 2
	v_lshlrev_b32_e32 v8, 2, v8
	v_mad_u32_u24 v141, 0x210, v118, v9
	v_lshlrev_b32_e32 v9, 2, v10
	v_lshlrev_b32_e32 v10, 2, v11
	v_lshlrev_b32_e32 v11, 2, v14
	v_lshlrev_b32_e32 v14, 2, v15
	v_lshlrev_b32_e32 v15, 2, v16
	v_lshlrev_b32_e32 v16, 2, v17
	v_lshlrev_b32_e32 v17, 2, v18
	v_lshlrev_b32_e32 v136, 2, v2
	v_add_nc_u32_e32 v8, 0x80, v8
	v_mad_u32_u24 v142, 0x210, v118, v9
	v_mad_u32_u24 v143, 0x210, v118, v10
	v_mad_u32_u24 v144, 0x210, v118, v11
	v_mad_u32_u24 v145, 0x210, v118, v14
	v_mad_u32_u24 v146, 0x210, v118, v15
	v_mad_u32_u24 v147, 0x210, v118, v16
	v_mad_u32_u24 v148, 0x210, v118, v17
	s_mov_b32 s7, 0
	s_cmp_gt_i32 s4, 0
	s_waitcnt vmcnt(14)
	ds_store_2addr_stride64_b32 v8, v23, v24 offset0:16 offset1:18
	s_waitcnt vmcnt(9)
	ds_store_b32 v141, v29
	ds_store_2addr_stride64_b32 v8, v25, v26 offset0:20 offset1:22
	s_waitcnt vmcnt(8)
	ds_store_b32 v142, v30
	s_waitcnt vmcnt(7)
	ds_store_b32 v143, v31
	ds_store_2addr_stride64_b32 v8, v27, v28 offset0:24 offset1:26
	s_waitcnt vmcnt(6)
	ds_store_b32 v144, v12
	s_waitcnt vmcnt(5)
	ds_store_b32 v145, v13
	s_waitcnt vmcnt(3)
	ds_store_2addr_stride64_b32 v8, v3, v4 offset0:28 offset1:30
	s_waitcnt vmcnt(2)
	ds_store_b32 v146, v5
	s_waitcnt vmcnt(1)
	ds_store_b32 v147, v6
	s_waitcnt vmcnt(0)
	ds_store_b32 v148, v7
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cbranch_scc1 .LBB0_3
; %bb.1:                                ; %.preheader193..preheader180_crit_edge
	v_lshlrev_b32_e32 v149, 2, v118
	v_lshlrev_b32_e32 v150, 2, v2
	s_mov_b32 s12, 0
	s_and_not1_b32 vcc_lo, exec_lo, s7
	s_cbranch_vccz .LBB0_4
; %bb.2:
	v_dual_mov_b32 v2, s12 :: v_dual_mov_b32 v3, s12
	v_dual_mov_b32 v4, s12 :: v_dual_mov_b32 v5, s12
	v_dual_mov_b32 v18, s12 :: v_dual_mov_b32 v19, s12
	v_dual_mov_b32 v20, s12 :: v_dual_mov_b32 v21, s12
	v_dual_mov_b32 v34, s12 :: v_dual_mov_b32 v35, s12
	v_dual_mov_b32 v36, s12 :: v_dual_mov_b32 v37, s12
	v_dual_mov_b32 v50, s12 :: v_dual_mov_b32 v51, s12
	v_dual_mov_b32 v52, s12 :: v_dual_mov_b32 v53, s12
	v_dual_mov_b32 v6, s12 :: v_dual_mov_b32 v7, s12
	v_dual_mov_b32 v8, s12 :: v_dual_mov_b32 v9, s12
	v_dual_mov_b32 v22, s12 :: v_dual_mov_b32 v23, s12
	v_dual_mov_b32 v24, s12 :: v_dual_mov_b32 v25, s12
	v_dual_mov_b32 v38, s12 :: v_dual_mov_b32 v39, s12
	v_dual_mov_b32 v40, s12 :: v_dual_mov_b32 v41, s12
	v_dual_mov_b32 v54, s12 :: v_dual_mov_b32 v55, s12
	v_dual_mov_b32 v56, s12 :: v_dual_mov_b32 v57, s12
	v_dual_mov_b32 v10, s12 :: v_dual_mov_b32 v11, s12
	v_dual_mov_b32 v12, s12 :: v_dual_mov_b32 v13, s12
	v_dual_mov_b32 v26, s12 :: v_dual_mov_b32 v27, s12
	v_dual_mov_b32 v28, s12 :: v_dual_mov_b32 v29, s12
	v_dual_mov_b32 v42, s12 :: v_dual_mov_b32 v43, s12
	v_dual_mov_b32 v44, s12 :: v_dual_mov_b32 v45, s12
	v_dual_mov_b32 v58, s12 :: v_dual_mov_b32 v59, s12
	v_dual_mov_b32 v60, s12 :: v_dual_mov_b32 v61, s12
	v_dual_mov_b32 v14, s12 :: v_dual_mov_b32 v15, s12
	v_dual_mov_b32 v16, s12 :: v_dual_mov_b32 v17, s12
	v_dual_mov_b32 v30, s12 :: v_dual_mov_b32 v31, s12
	v_dual_mov_b32 v32, s12 :: v_dual_mov_b32 v33, s12
	v_dual_mov_b32 v46, s12 :: v_dual_mov_b32 v47, s12
	v_dual_mov_b32 v48, s12 :: v_dual_mov_b32 v49, s12
	v_dual_mov_b32 v62, s12 :: v_dual_mov_b32 v63, s12
	v_dual_mov_b32 v64, s12 :: v_dual_mov_b32 v65, s12
	v_dual_mov_b32 v66, s12 :: v_dual_mov_b32 v67, s12
	v_dual_mov_b32 v68, s12 :: v_dual_mov_b32 v69, s12
	v_dual_mov_b32 v82, s12 :: v_dual_mov_b32 v83, s12
	v_dual_mov_b32 v84, s12 :: v_dual_mov_b32 v85, s12
	v_dual_mov_b32 v98, s12 :: v_dual_mov_b32 v99, s12
	v_dual_mov_b32 v100, s12 :: v_dual_mov_b32 v101, s12
	v_dual_mov_b32 v114, s12 :: v_dual_mov_b32 v115, s12
	v_dual_mov_b32 v116, s12 :: v_dual_mov_b32 v117, s12
	v_dual_mov_b32 v70, s12 :: v_dual_mov_b32 v71, s12
	v_dual_mov_b32 v72, s12 :: v_dual_mov_b32 v73, s12
	v_dual_mov_b32 v86, s12 :: v_dual_mov_b32 v87, s12
	v_dual_mov_b32 v88, s12 :: v_dual_mov_b32 v89, s12
	v_dual_mov_b32 v102, s12 :: v_dual_mov_b32 v103, s12
	v_dual_mov_b32 v104, s12 :: v_dual_mov_b32 v105, s12
	v_dual_mov_b32 v120, s12 :: v_dual_mov_b32 v121, s12
	v_dual_mov_b32 v122, s12 :: v_dual_mov_b32 v123, s12
	v_dual_mov_b32 v74, s12 :: v_dual_mov_b32 v75, s12
	v_dual_mov_b32 v76, s12 :: v_dual_mov_b32 v77, s12
	v_dual_mov_b32 v90, s12 :: v_dual_mov_b32 v91, s12
	v_dual_mov_b32 v92, s12 :: v_dual_mov_b32 v93, s12
	v_dual_mov_b32 v106, s12 :: v_dual_mov_b32 v107, s12
	v_dual_mov_b32 v108, s12 :: v_dual_mov_b32 v109, s12
	v_dual_mov_b32 v126, s12 :: v_dual_mov_b32 v127, s12
	v_dual_mov_b32 v128, s12 :: v_dual_mov_b32 v129, s12
	v_dual_mov_b32 v78, s12 :: v_dual_mov_b32 v79, s12
	v_dual_mov_b32 v80, s12 :: v_dual_mov_b32 v81, s12
	v_dual_mov_b32 v94, s12 :: v_dual_mov_b32 v95, s12
	v_dual_mov_b32 v96, s12 :: v_dual_mov_b32 v97, s12
	v_dual_mov_b32 v110, s12 :: v_dual_mov_b32 v111, s12
	v_dual_mov_b32 v112, s12 :: v_dual_mov_b32 v113, s12
	v_dual_mov_b32 v131, s12 :: v_dual_mov_b32 v132, s12
	v_dual_mov_b32 v133, s12 :: v_dual_mov_b32 v124, s12
	s_branch .LBB0_13
.LBB0_3:
	s_mov_b32 s7, -1
                                        ; implicit-def: $sgpr12
                                        ; implicit-def: $vgpr149
                                        ; implicit-def: $vgpr150
.LBB0_4:                                ; %.lr.ph
	s_ashr_i32 s7, s2, 31
	v_dual_mov_b32 v133, 0 :: v_dual_lshlrev_b32 v2, 4, v2
	s_lshr_b32 s7, s7, 25
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_dual_mov_b32 v124, 0 :: v_dual_add_nc_u32 v3, s7, v1
	v_ashrrev_i32_e32 v149, 31, v119
	v_ashrrev_i32_e32 v150, 31, v125
	v_ashrrev_i32_e32 v151, 31, v130
	v_dual_mov_b32 v132, 0 :: v_dual_and_b32 v3, 0x3fffff80, v3
	v_ashrrev_i32_e32 v152, 31, v134
	v_ashrrev_i32_e32 v153, 31, v137
	v_ashrrev_i32_e32 v154, 31, v138
	v_ashrrev_i32_e32 v156, 31, v139
	v_sub_nc_u32_e32 v3, v1, v3
	v_ashrrev_i32_e32 v157, 31, v140
	v_lshl_or_b32 v166, v118, 4, 0x1080
	v_dual_mov_b32 v131, 0 :: v_dual_mov_b32 v110, 0
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v155, v3, 2, 0x1080
	v_dual_mov_b32 v112, 0 :: v_dual_lshlrev_b32 v3, 2, v0
	v_dual_mov_b32 v113, 0 :: v_dual_mov_b32 v96, 0
	v_lshl_add_u32 v158, 1, 9, v155
	v_lshl_add_u32 v159, 2, 9, v155
	v_lshl_add_u32 v160, 3, 9, v155
	v_lshl_add_u32 v161, 4, 9, v155
	v_lshl_add_u32 v162, 5, 9, v155
	v_lshl_add_u32 v163, 6, 9, v155
	v_lshl_add_u32 v164, 7, 9, v155
	v_and_or_b32 v165, 0x180, v3, v2
	v_dual_mov_b32 v111, 0 :: v_dual_mov_b32 v94, 0
	v_dual_mov_b32 v97, 0 :: v_dual_mov_b32 v80, 0
	v_dual_mov_b32 v95, 0 :: v_dual_mov_b32 v78, 0
	v_dual_mov_b32 v81, 0 :: v_dual_mov_b32 v128, 0
	v_dual_mov_b32 v79, 0 :: v_dual_mov_b32 v126, 0
	v_dual_mov_b32 v129, 0 :: v_dual_mov_b32 v108, 0
	v_dual_mov_b32 v127, 0 :: v_dual_mov_b32 v106, 0
	v_dual_mov_b32 v109, 0 :: v_dual_mov_b32 v92, 0
	v_dual_mov_b32 v107, 0 :: v_dual_mov_b32 v90, 0
	v_dual_mov_b32 v93, 0 :: v_dual_mov_b32 v76, 0
	v_dual_mov_b32 v91, 0 :: v_dual_mov_b32 v74, 0
	v_dual_mov_b32 v77, 0 :: v_dual_mov_b32 v122, 0
	v_dual_mov_b32 v75, 0 :: v_dual_mov_b32 v120, 0
	v_dual_mov_b32 v123, 0 :: v_dual_mov_b32 v104, 0
	v_dual_mov_b32 v121, 0 :: v_dual_mov_b32 v102, 0
	v_dual_mov_b32 v105, 0 :: v_dual_mov_b32 v88, 0
	v_dual_mov_b32 v103, 0 :: v_dual_mov_b32 v86, 0
	v_dual_mov_b32 v89, 0 :: v_dual_mov_b32 v72, 0
	v_dual_mov_b32 v87, 0 :: v_dual_mov_b32 v70, 0
	v_dual_mov_b32 v73, 0 :: v_dual_mov_b32 v116, 0
	v_dual_mov_b32 v71, 0 :: v_dual_mov_b32 v114, 0
	v_dual_mov_b32 v117, 0 :: v_dual_mov_b32 v100, 0
	v_dual_mov_b32 v115, 0 :: v_dual_mov_b32 v98, 0
	v_dual_mov_b32 v101, 0 :: v_dual_mov_b32 v84, 0
	v_dual_mov_b32 v99, 0 :: v_dual_mov_b32 v82, 0
	v_dual_mov_b32 v85, 0 :: v_dual_mov_b32 v68, 0
	v_dual_mov_b32 v83, 0 :: v_dual_mov_b32 v66, 0
	v_dual_mov_b32 v69, 0 :: v_dual_mov_b32 v64, 0
	v_dual_mov_b32 v67, 0 :: v_dual_mov_b32 v62, 0
	v_dual_mov_b32 v65, 0 :: v_dual_mov_b32 v48, 0
	v_dual_mov_b32 v63, 0 :: v_dual_mov_b32 v46, 0
	v_dual_mov_b32 v49, 0 :: v_dual_mov_b32 v32, 0
	v_dual_mov_b32 v47, 0 :: v_dual_mov_b32 v30, 0
	v_dual_mov_b32 v33, 0 :: v_dual_mov_b32 v16, 0
	v_dual_mov_b32 v31, 0 :: v_dual_mov_b32 v14, 0
	v_dual_mov_b32 v17, 0 :: v_dual_mov_b32 v60, 0
	v_dual_mov_b32 v15, 0 :: v_dual_mov_b32 v58, 0
	v_dual_mov_b32 v61, 0 :: v_dual_mov_b32 v44, 0
	v_dual_mov_b32 v59, 0 :: v_dual_mov_b32 v42, 0
	v_dual_mov_b32 v45, 0 :: v_dual_mov_b32 v28, 0
	v_dual_mov_b32 v43, 0 :: v_dual_mov_b32 v26, 0
	v_dual_mov_b32 v29, 0 :: v_dual_mov_b32 v12, 0
	v_dual_mov_b32 v27, 0 :: v_dual_mov_b32 v10, 0
	v_dual_mov_b32 v13, 0 :: v_dual_mov_b32 v56, 0
	v_dual_mov_b32 v11, 0 :: v_dual_mov_b32 v54, 0
	v_dual_mov_b32 v57, 0 :: v_dual_mov_b32 v40, 0
	v_dual_mov_b32 v55, 0 :: v_dual_mov_b32 v38, 0
	v_dual_mov_b32 v41, 0 :: v_dual_mov_b32 v24, 0
	v_dual_mov_b32 v39, 0 :: v_dual_mov_b32 v22, 0
	v_dual_mov_b32 v25, 0 :: v_dual_mov_b32 v8, 0
	v_dual_mov_b32 v23, 0 :: v_dual_mov_b32 v6, 0
	v_dual_mov_b32 v9, 0 :: v_dual_mov_b32 v52, 0
	v_dual_mov_b32 v7, 0 :: v_dual_mov_b32 v50, 0
	v_dual_mov_b32 v53, 0 :: v_dual_mov_b32 v36, 0
	v_dual_mov_b32 v51, 0 :: v_dual_mov_b32 v34, 0
	v_dual_mov_b32 v37, 0 :: v_dual_mov_b32 v20, 0
	v_dual_mov_b32 v35, 0 :: v_dual_mov_b32 v18, 0
	v_dual_mov_b32 v21, 0 :: v_dual_mov_b32 v4, 0
	v_dual_mov_b32 v19, 0 :: v_dual_mov_b32 v2, 0
	;setting v214 as well (extra bank2 vpgr used by output matrix C)
	v_mov_b32 v214,0

	v_mov_b32_e32 v5, 0
	v_mov_b32_e32 v3, 0
	s_add_i32 s7, s4, -8
	s_add_u32 s8, s8, 32
	s_addc_u32 s9, s9, 0
	s_mov_b32 s12, 0
                                        ; implicit-def: $vgpr175
                                        ; implicit-def: $vgpr176
                                        ; implicit-def: $vgpr177
                                        ; implicit-def: $vgpr178
                                        ; implicit-def: $vgpr179
                                        ; implicit-def: $vgpr180
                                        ; implicit-def: $vgpr181
                                        ; implicit-def: $vgpr182
                                        ; implicit-def: $vgpr167
                                        ; implicit-def: $vgpr168
                                        ; implicit-def: $vgpr169
                                        ; implicit-def: $vgpr170
                                        ; implicit-def: $vgpr171
                                        ; implicit-def: $vgpr172
                                        ; implicit-def: $vgpr173
                                        ; implicit-def: $vgpr174
	s_branch .LBB0_6
.LBB0_5:                                ;   in Loop: Header=BB0_6 Depth=1
	s_add_i32 s12, s12, 8
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_ge_i32 s12, s4
	s_cbranch_scc1 .LBB0_12
.LBB0_6:                                ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_9 Depth 2
	s_cmp_lt_i32 s12, s7
	s_cselect_b32 s13, -1, 0
	s_cmp_ge_i32 s12, s7
	s_cbranch_scc1 .LBB0_8
; %bb.7:                                ; %.preheader192
                                        ;   in Loop: Header=BB0_6 Depth=1

	; Global memory read for matrix B
	v_add_nc_u32_e32 v203, 0x20000, v203
	v_add_nc_u32_e32 v215, 0x20, v215
	s_setprio 0
	global_load_b32	 v167, v203, s[24:25]
	global_load_b32	 v168, v203, s[26:27]
	global_load_b32	 v169, v203, s[28:29]
	global_load_b32	 v170, v203, s[30:31]




.LBB0_8:                                ; %.loopexit
                                        ;   in Loop: Header=BB0_6 Depth=1
	v_mov_b32_e32 v183, v165
	s_mov_b32 s14, 0

	v_mov_b32 v202,v166
.LBB0_9:                                ; %.preheader188
                                        ;   Parent Loop BB0_6 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2


	s_clause 0xB;
 ;A on bank 2-3
	ds_load_b64 v[186:187], v183
	ds_load_b64 v[190:191], v183 offset: 8
	ds_load_b64 v[194:195], v183 offset: 64
	ds_load_b64 v[198:199], v183 offset: 72

 ;B on bank 0-1
	ds_load_b64 v[184:185], v202
	ds_load_b64 v[188:189], v202 offset: 8
	ds_load_b64 v[192:193], v202 offset: 128
	ds_load_b64 v[196:197], v202 offset: 136
	ds_load_b64 v[200:201], v202 offset: 256
	ds_load_b64 v[204:205], v202 offset: 264
	ds_load_b64 v[208:209], v202 offset: 384
	ds_load_b64 v[212:213], v202 offset: 392

	v_add_nc_u32_e32 v183, 0x210, v183
	v_add_nc_u32_e32 v202, 0x200, v202
	s_waitcnt lgkmcnt(0)

;	; new vpgrs allocation

v_dual_fmac_f32 v5, v186, v184 :: v_dual_fmac_f32 v2, v187, v185
s_setprio 1
v_dual_fmac_f32 v3, v186, v185 :: v_dual_fmac_f32 v4, v187, v184
v_dual_fmac_f32 v9, v186, v188 :: v_dual_fmac_f32 v6, v187, v189
v_dual_fmac_f32 v7, v187, v188 :: v_dual_fmac_f32 v8, v186, v189
v_dual_fmac_f32 v13, v190, v188 :: v_dual_fmac_f32 v10, v191, v189
v_dual_fmac_f32 v11, v190, v189 :: v_dual_fmac_f32 v12, v191, v188
v_dual_fmac_f32 v17, v190, v184 :: v_dual_fmac_f32 v14, v191, v185
v_dual_fmac_f32 v15, v191, v184 :: v_dual_fmac_f32 v16, v190, v185
v_dual_fmac_f32 v21, v194, v184 :: v_dual_fmac_f32 v18, v195, v185
v_dual_fmac_f32 v19, v194, v185 :: v_dual_fmac_f32 v20, v195, v184
v_dual_fmac_f32 v25, v194, v188 :: v_dual_fmac_f32 v22, v195, v189
v_dual_fmac_f32 v23, v195, v188 :: v_dual_fmac_f32 v24, v194, v189
v_dual_fmac_f32 v29, v198, v188 :: v_dual_fmac_f32 v26, v199, v189
v_dual_fmac_f32 v27, v198, v189 :: v_dual_fmac_f32 v28, v199, v188
v_dual_fmac_f32 v33, v198, v192 :: v_dual_fmac_f32 v30, v199, v193
v_dual_fmac_f32 v31, v199, v192 :: v_dual_fmac_f32 v32, v198, v193
v_dual_fmac_f32 v37, v186, v192 :: v_dual_fmac_f32 v34, v187, v193
v_dual_fmac_f32 v35, v186, v193 :: v_dual_fmac_f32 v36, v187, v192
v_dual_fmac_f32 v41, v186, v196 :: v_dual_fmac_f32 v38, v187, v197
v_dual_fmac_f32 v39, v187, v196 :: v_dual_fmac_f32 v40, v186, v197
v_dual_fmac_f32 v45, v190, v196 :: v_dual_fmac_f32 v42, v191, v197
v_dual_fmac_f32 v43, v190, v197 :: v_dual_fmac_f32 v44, v191, v196
v_dual_fmac_f32 v49, v190, v192 :: v_dual_fmac_f32 v46, v191, v193
v_dual_fmac_f32 v47, v191, v192 :: v_dual_fmac_f32 v48, v190, v193
v_dual_fmac_f32 v53, v194, v192 :: v_dual_fmac_f32 v50, v195, v193
v_dual_fmac_f32 v51, v194, v193 :: v_dual_fmac_f32 v52, v195, v192
v_dual_fmac_f32 v57, v194, v196 :: v_dual_fmac_f32 v54, v195, v197
v_dual_fmac_f32 v55, v195, v196 :: v_dual_fmac_f32 v56, v194, v197
v_dual_fmac_f32 v61, v198, v196 :: v_dual_fmac_f32 v58, v199, v197
v_dual_fmac_f32 v59, v198, v197 :: v_dual_fmac_f32 v60, v199, v196
v_dual_fmac_f32 v65, v198, v200 :: v_dual_fmac_f32 v62, v199, v201
v_dual_fmac_f32 v63, v199, v200 :: v_dual_fmac_f32 v64, v198, v201
v_dual_fmac_f32 v69, v186, v200 :: v_dual_fmac_f32 v66, v187, v201
v_dual_fmac_f32 v67, v186, v201 :: v_dual_fmac_f32 v68, v187, v200
v_dual_fmac_f32 v73, v186, v204 :: v_dual_fmac_f32 v70, v187, v205
v_dual_fmac_f32 v71, v187, v204 :: v_dual_fmac_f32 v72, v186, v205
v_dual_fmac_f32 v77, v190, v204 :: v_dual_fmac_f32 v74, v191, v205
v_dual_fmac_f32 v75, v190, v205 :: v_dual_fmac_f32 v76, v191, v204
v_dual_fmac_f32 v81, v190, v200 :: v_dual_fmac_f32 v78, v191, v201
v_dual_fmac_f32 v79, v191, v200 :: v_dual_fmac_f32 v80, v190, v201
v_dual_fmac_f32 v85, v194, v200 :: v_dual_fmac_f32 v82, v195, v201
v_dual_fmac_f32 v83, v194, v201 :: v_dual_fmac_f32 v84, v195, v200
v_dual_fmac_f32 v89, v194, v204 :: v_dual_fmac_f32 v86, v195, v205
v_dual_fmac_f32 v87, v195, v204 :: v_dual_fmac_f32 v88, v194, v205
v_dual_fmac_f32 v93, v198, v204 :: v_dual_fmac_f32 v90, v199, v205
v_dual_fmac_f32 v91, v198, v205 :: v_dual_fmac_f32 v92, v199, v204
v_dual_fmac_f32 v97, v198, v208 :: v_dual_fmac_f32 v94, v199, v209
v_dual_fmac_f32 v95, v199, v208 :: v_dual_fmac_f32 v96, v198, v209
v_dual_fmac_f32 v101, v186, v208 :: v_dual_fmac_f32 v98, v187, v209
v_dual_fmac_f32 v99, v186, v209 :: v_dual_fmac_f32 v100, v187, v208
v_dual_fmac_f32 v105, v186, v212 :: v_dual_fmac_f32 v102, v187, v213
v_dual_fmac_f32 v103, v187, v212 :: v_dual_fmac_f32 v104, v186, v213
v_dual_fmac_f32 v109, v190, v212 :: v_dual_fmac_f32 v106, v191, v213
v_dual_fmac_f32 v107, v190, v213 :: v_dual_fmac_f32 v108, v191, v212
v_dual_fmac_f32 v113, v190, v208 :: v_dual_fmac_f32 v110, v191, v209
v_dual_fmac_f32 v111, v191, v208 :: v_dual_fmac_f32 v112, v190, v209
v_dual_fmac_f32 v117, v194, v208 :: v_dual_fmac_f32 v114, v195, v209
v_dual_fmac_f32 v115, v194, v209 :: v_dual_fmac_f32 v116, v195, v208
v_dual_fmac_f32 v121, v194, v212 :: v_dual_fmac_f32 v122, v195, v213
v_dual_fmac_f32 v123, v195, v212 :: v_dual_fmac_f32 v120, v194, v213
v_dual_fmac_f32 v129, v198, v212 :: v_dual_fmac_f32 v126, v199, v213
v_dual_fmac_f32 v127, v198, v213 :: v_dual_fmac_f32 v124, v199, v212
v_dual_fmac_f32 v133, v198, v184 :: v_dual_fmac_f32 v214, v199, v185
v_dual_fmac_f32 v131, v199, v184 :: v_dual_fmac_f32 v128, v198, v185

s_setprio 0
	s_clause 0x1
	global_load_b32	 v171, v203, s[32:33]
	global_load_b32	 v172, v203, s[34:35]

	s_clause 0xB;
 ;A on bank 2-3
	ds_load_b64 v[186:187], v183
	ds_load_b64 v[190:191], v183 offset: 8
	ds_load_b64 v[194:195], v183 offset: 64
	ds_load_b64 v[198:199], v183 offset: 72


 ;B on bank 0-1
	ds_load_b64 v[184:185], v202
	ds_load_b64 v[188:189], v202 offset: 8
	ds_load_b64 v[192:193], v202 offset: 128
	ds_load_b64 v[196:197], v202 offset: 136
	ds_load_b64 v[200:201], v202 offset: 256
	ds_load_b64 v[204:205], v202 offset: 264
	ds_load_b64 v[208:209], v202 offset: 384
	ds_load_b64 v[212:213], v202 offset: 392

	v_add_nc_u32_e32 v183, 0x210, v183
	v_add_nc_u32_e32 v202, 0x200, v202
	;s_cmpk_lg_i32 s14, 0x1000
	s_waitcnt lgkmcnt(0)

;	; new vpgrs allocation

v_dual_fmac_f32 v5, v186, v184 :: v_dual_fmac_f32 v2, v187, v185
s_setprio 1
v_dual_fmac_f32 v3, v186, v185 :: v_dual_fmac_f32 v4, v187, v184
v_dual_fmac_f32 v9, v186, v188 :: v_dual_fmac_f32 v6, v187, v189
v_dual_fmac_f32 v7, v187, v188 :: v_dual_fmac_f32 v8, v186, v189
v_dual_fmac_f32 v13, v190, v188 :: v_dual_fmac_f32 v10, v191, v189
v_dual_fmac_f32 v11, v190, v189 :: v_dual_fmac_f32 v12, v191, v188
v_dual_fmac_f32 v17, v190, v184 :: v_dual_fmac_f32 v14, v191, v185
v_dual_fmac_f32 v15, v191, v184 :: v_dual_fmac_f32 v16, v190, v185
v_dual_fmac_f32 v21, v194, v184 :: v_dual_fmac_f32 v18, v195, v185
v_dual_fmac_f32 v19, v194, v185 :: v_dual_fmac_f32 v20, v195, v184
v_dual_fmac_f32 v25, v194, v188 :: v_dual_fmac_f32 v22, v195, v189
v_dual_fmac_f32 v23, v195, v188 :: v_dual_fmac_f32 v24, v194, v189
v_dual_fmac_f32 v29, v198, v188 :: v_dual_fmac_f32 v26, v199, v189
v_dual_fmac_f32 v27, v198, v189 :: v_dual_fmac_f32 v28, v199, v188
v_dual_fmac_f32 v33, v198, v192 :: v_dual_fmac_f32 v30, v199, v193
v_dual_fmac_f32 v31, v199, v192 :: v_dual_fmac_f32 v32, v198, v193
v_dual_fmac_f32 v37, v186, v192 :: v_dual_fmac_f32 v34, v187, v193
v_dual_fmac_f32 v35, v186, v193 :: v_dual_fmac_f32 v36, v187, v192
v_dual_fmac_f32 v41, v186, v196 :: v_dual_fmac_f32 v38, v187, v197
v_dual_fmac_f32 v39, v187, v196 :: v_dual_fmac_f32 v40, v186, v197
v_dual_fmac_f32 v45, v190, v196 :: v_dual_fmac_f32 v42, v191, v197
v_dual_fmac_f32 v43, v190, v197 :: v_dual_fmac_f32 v44, v191, v196
v_dual_fmac_f32 v49, v190, v192 :: v_dual_fmac_f32 v46, v191, v193
v_dual_fmac_f32 v47, v191, v192 :: v_dual_fmac_f32 v48, v190, v193
v_dual_fmac_f32 v53, v194, v192 :: v_dual_fmac_f32 v50, v195, v193
v_dual_fmac_f32 v51, v194, v193 :: v_dual_fmac_f32 v52, v195, v192
v_dual_fmac_f32 v57, v194, v196 :: v_dual_fmac_f32 v54, v195, v197
v_dual_fmac_f32 v55, v195, v196 :: v_dual_fmac_f32 v56, v194, v197
v_dual_fmac_f32 v61, v198, v196 :: v_dual_fmac_f32 v58, v199, v197
v_dual_fmac_f32 v59, v198, v197 :: v_dual_fmac_f32 v60, v199, v196
v_dual_fmac_f32 v65, v198, v200 :: v_dual_fmac_f32 v62, v199, v201
v_dual_fmac_f32 v63, v199, v200 :: v_dual_fmac_f32 v64, v198, v201
v_dual_fmac_f32 v69, v186, v200 :: v_dual_fmac_f32 v66, v187, v201
v_dual_fmac_f32 v67, v186, v201 :: v_dual_fmac_f32 v68, v187, v200
v_dual_fmac_f32 v73, v186, v204 :: v_dual_fmac_f32 v70, v187, v205
v_dual_fmac_f32 v71, v187, v204 :: v_dual_fmac_f32 v72, v186, v205
v_dual_fmac_f32 v77, v190, v204 :: v_dual_fmac_f32 v74, v191, v205
v_dual_fmac_f32 v75, v190, v205 :: v_dual_fmac_f32 v76, v191, v204
v_dual_fmac_f32 v81, v190, v200 :: v_dual_fmac_f32 v78, v191, v201
v_dual_fmac_f32 v79, v191, v200 :: v_dual_fmac_f32 v80, v190, v201
v_dual_fmac_f32 v85, v194, v200 :: v_dual_fmac_f32 v82, v195, v201
v_dual_fmac_f32 v83, v194, v201 :: v_dual_fmac_f32 v84, v195, v200
v_dual_fmac_f32 v89, v194, v204 :: v_dual_fmac_f32 v86, v195, v205
v_dual_fmac_f32 v87, v195, v204 :: v_dual_fmac_f32 v88, v194, v205
v_dual_fmac_f32 v93, v198, v204 :: v_dual_fmac_f32 v90, v199, v205
v_dual_fmac_f32 v91, v198, v205 :: v_dual_fmac_f32 v92, v199, v204
v_dual_fmac_f32 v97, v198, v208 :: v_dual_fmac_f32 v94, v199, v209
v_dual_fmac_f32 v95, v199, v208 :: v_dual_fmac_f32 v96, v198, v209
v_dual_fmac_f32 v101, v186, v208 :: v_dual_fmac_f32 v98, v187, v209
v_dual_fmac_f32 v99, v186, v209 :: v_dual_fmac_f32 v100, v187, v208
v_dual_fmac_f32 v105, v186, v212 :: v_dual_fmac_f32 v102, v187, v213
v_dual_fmac_f32 v103, v187, v212 :: v_dual_fmac_f32 v104, v186, v213
v_dual_fmac_f32 v109, v190, v212 :: v_dual_fmac_f32 v106, v191, v213
v_dual_fmac_f32 v107, v190, v213 :: v_dual_fmac_f32 v108, v191, v212
v_dual_fmac_f32 v113, v190, v208 :: v_dual_fmac_f32 v110, v191, v209
v_dual_fmac_f32 v111, v191, v208 :: v_dual_fmac_f32 v112, v190, v209
v_dual_fmac_f32 v117, v194, v208 :: v_dual_fmac_f32 v114, v195, v209
v_dual_fmac_f32 v115, v194, v209 :: v_dual_fmac_f32 v116, v195, v208
v_dual_fmac_f32 v121, v194, v212 :: v_dual_fmac_f32 v122, v195, v213
v_dual_fmac_f32 v123, v195, v212 :: v_dual_fmac_f32 v120, v194, v213
v_dual_fmac_f32 v129, v198, v212 :: v_dual_fmac_f32 v126, v199, v213
v_dual_fmac_f32 v127, v198, v213 :: v_dual_fmac_f32 v124, v199, v212
v_dual_fmac_f32 v133, v198, v184 :: v_dual_fmac_f32 v214, v199, v185
v_dual_fmac_f32 v131, v199, v184 :: v_dual_fmac_f32 v128, v198, v185

s_setprio 0


	s_clause 0x1
	global_load_b32	 v173, v203, s[36:37]
	global_load_b32	 v174, v203, s[38:39]

	s_clause 0xB;
 ;A on bank 2-3
	ds_load_b64 v[186:187], v183
	ds_load_b64 v[190:191], v183 offset: 8
	ds_load_b64 v[194:195], v183 offset: 64
	ds_load_b64 v[198:199], v183 offset: 72


 ;B on bank 0-1
	ds_load_b64 v[184:185], v202
	ds_load_b64 v[188:189], v202 offset: 8
	ds_load_b64 v[192:193], v202 offset: 128
	ds_load_b64 v[196:197], v202 offset: 136
	ds_load_b64 v[200:201], v202 offset: 256
	ds_load_b64 v[204:205], v202 offset: 264
	ds_load_b64 v[208:209], v202 offset: 384
	ds_load_b64 v[212:213], v202 offset: 392

	v_add_nc_u32_e32 v183, 0x210, v183
	v_add_nc_u32_e32 v202, 0x200, v202
	;s_cmpk_lg_i32 s14, 0x1000
	s_waitcnt lgkmcnt(0)

;	new vpgrs allocation

v_dual_fmac_f32 v5, v186, v184 :: v_dual_fmac_f32 v2, v187, v185
s_setprio 1
v_dual_fmac_f32 v3, v186, v185 :: v_dual_fmac_f32 v4, v187, v184
v_dual_fmac_f32 v9, v186, v188 :: v_dual_fmac_f32 v6, v187, v189
v_dual_fmac_f32 v7, v187, v188 :: v_dual_fmac_f32 v8, v186, v189
v_dual_fmac_f32 v13, v190, v188 :: v_dual_fmac_f32 v10, v191, v189
v_dual_fmac_f32 v11, v190, v189 :: v_dual_fmac_f32 v12, v191, v188
v_dual_fmac_f32 v17, v190, v184 :: v_dual_fmac_f32 v14, v191, v185
v_dual_fmac_f32 v15, v191, v184 :: v_dual_fmac_f32 v16, v190, v185
v_dual_fmac_f32 v21, v194, v184 :: v_dual_fmac_f32 v18, v195, v185
v_dual_fmac_f32 v19, v194, v185 :: v_dual_fmac_f32 v20, v195, v184
v_dual_fmac_f32 v25, v194, v188 :: v_dual_fmac_f32 v22, v195, v189
v_dual_fmac_f32 v23, v195, v188 :: v_dual_fmac_f32 v24, v194, v189
v_dual_fmac_f32 v29, v198, v188 :: v_dual_fmac_f32 v26, v199, v189
v_dual_fmac_f32 v27, v198, v189 :: v_dual_fmac_f32 v28, v199, v188
v_dual_fmac_f32 v33, v198, v192 :: v_dual_fmac_f32 v30, v199, v193
v_dual_fmac_f32 v31, v199, v192 :: v_dual_fmac_f32 v32, v198, v193
v_dual_fmac_f32 v37, v186, v192 :: v_dual_fmac_f32 v34, v187, v193
v_dual_fmac_f32 v35, v186, v193 :: v_dual_fmac_f32 v36, v187, v192
v_dual_fmac_f32 v41, v186, v196 :: v_dual_fmac_f32 v38, v187, v197
v_dual_fmac_f32 v39, v187, v196 :: v_dual_fmac_f32 v40, v186, v197
v_dual_fmac_f32 v45, v190, v196 :: v_dual_fmac_f32 v42, v191, v197
v_dual_fmac_f32 v43, v190, v197 :: v_dual_fmac_f32 v44, v191, v196
v_dual_fmac_f32 v49, v190, v192 :: v_dual_fmac_f32 v46, v191, v193
v_dual_fmac_f32 v47, v191, v192 :: v_dual_fmac_f32 v48, v190, v193
v_dual_fmac_f32 v53, v194, v192 :: v_dual_fmac_f32 v50, v195, v193
v_dual_fmac_f32 v51, v194, v193 :: v_dual_fmac_f32 v52, v195, v192
v_dual_fmac_f32 v57, v194, v196 :: v_dual_fmac_f32 v54, v195, v197
v_dual_fmac_f32 v55, v195, v196 :: v_dual_fmac_f32 v56, v194, v197
v_dual_fmac_f32 v61, v198, v196 :: v_dual_fmac_f32 v58, v199, v197
v_dual_fmac_f32 v59, v198, v197 :: v_dual_fmac_f32 v60, v199, v196
v_dual_fmac_f32 v65, v198, v200 :: v_dual_fmac_f32 v62, v199, v201
v_dual_fmac_f32 v63, v199, v200 :: v_dual_fmac_f32 v64, v198, v201
v_dual_fmac_f32 v69, v186, v200 :: v_dual_fmac_f32 v66, v187, v201
v_dual_fmac_f32 v67, v186, v201 :: v_dual_fmac_f32 v68, v187, v200
v_dual_fmac_f32 v73, v186, v204 :: v_dual_fmac_f32 v70, v187, v205
v_dual_fmac_f32 v71, v187, v204 :: v_dual_fmac_f32 v72, v186, v205
v_dual_fmac_f32 v77, v190, v204 :: v_dual_fmac_f32 v74, v191, v205
v_dual_fmac_f32 v75, v190, v205 :: v_dual_fmac_f32 v76, v191, v204
v_dual_fmac_f32 v81, v190, v200 :: v_dual_fmac_f32 v78, v191, v201
v_dual_fmac_f32 v79, v191, v200 :: v_dual_fmac_f32 v80, v190, v201
v_dual_fmac_f32 v85, v194, v200 :: v_dual_fmac_f32 v82, v195, v201
v_dual_fmac_f32 v83, v194, v201 :: v_dual_fmac_f32 v84, v195, v200
v_dual_fmac_f32 v89, v194, v204 :: v_dual_fmac_f32 v86, v195, v205
v_dual_fmac_f32 v87, v195, v204 :: v_dual_fmac_f32 v88, v194, v205
v_dual_fmac_f32 v93, v198, v204 :: v_dual_fmac_f32 v90, v199, v205
v_dual_fmac_f32 v91, v198, v205 :: v_dual_fmac_f32 v92, v199, v204
v_dual_fmac_f32 v97, v198, v208 :: v_dual_fmac_f32 v94, v199, v209
v_dual_fmac_f32 v95, v199, v208 :: v_dual_fmac_f32 v96, v198, v209
v_dual_fmac_f32 v101, v186, v208 :: v_dual_fmac_f32 v98, v187, v209
v_dual_fmac_f32 v99, v186, v209 :: v_dual_fmac_f32 v100, v187, v208
v_dual_fmac_f32 v105, v186, v212 :: v_dual_fmac_f32 v102, v187, v213
v_dual_fmac_f32 v103, v187, v212 :: v_dual_fmac_f32 v104, v186, v213
v_dual_fmac_f32 v109, v190, v212 :: v_dual_fmac_f32 v106, v191, v213
v_dual_fmac_f32 v107, v190, v213 :: v_dual_fmac_f32 v108, v191, v212
v_dual_fmac_f32 v113, v190, v208 :: v_dual_fmac_f32 v110, v191, v209
v_dual_fmac_f32 v111, v191, v208 :: v_dual_fmac_f32 v112, v190, v209
v_dual_fmac_f32 v117, v194, v208 :: v_dual_fmac_f32 v114, v195, v209
v_dual_fmac_f32 v115, v194, v209 :: v_dual_fmac_f32 v116, v195, v208
v_dual_fmac_f32 v121, v194, v212 :: v_dual_fmac_f32 v122, v195, v213
v_dual_fmac_f32 v123, v195, v212 :: v_dual_fmac_f32 v120, v194, v213
v_dual_fmac_f32 v129, v198, v212 :: v_dual_fmac_f32 v126, v199, v213
v_dual_fmac_f32 v127, v198, v213 :: v_dual_fmac_f32 v124, v199, v212
v_dual_fmac_f32 v133, v198, v184 :: v_dual_fmac_f32 v214, v199, v185
v_dual_fmac_f32 v131, v199, v184 :: v_dual_fmac_f32 v128, v198, v185

s_setprio 0

	s_clause 0x1
	global_load_b32	 v175, v215, s[40:41]
	global_load_b32	 v176, v215, s[42:43]

	s_clause 0xB;
 ;A on bank 2-3
	ds_load_b64 v[186:187], v183
	ds_load_b64 v[190:191], v183 offset: 8
	ds_load_b64 v[194:195], v183 offset: 64
	ds_load_b64 v[198:199], v183 offset: 72


 ;B on bank 0-1
	ds_load_b64 v[184:185], v202
	ds_load_b64 v[188:189], v202 offset: 8
	ds_load_b64 v[192:193], v202 offset: 128
	ds_load_b64 v[196:197], v202 offset: 136
	ds_load_b64 v[200:201], v202 offset: 256
	ds_load_b64 v[204:205], v202 offset: 264
	ds_load_b64 v[208:209], v202 offset: 384
	ds_load_b64 v[212:213], v202 offset: 392

	v_add_nc_u32_e32 v183, 0x210, v183
	v_add_nc_u32_e32 v202, 0x200, v202
	;s_cmpk_lg_i32 s14, 0x1000
	s_waitcnt lgkmcnt(0)

;	; new vpgrs allocation

v_dual_fmac_f32 v5, v186, v184 :: v_dual_fmac_f32 v2, v187, v185
s_setprio 1
v_dual_fmac_f32 v3, v186, v185 :: v_dual_fmac_f32 v4, v187, v184
v_dual_fmac_f32 v9, v186, v188 :: v_dual_fmac_f32 v6, v187, v189
v_dual_fmac_f32 v7, v187, v188 :: v_dual_fmac_f32 v8, v186, v189
v_dual_fmac_f32 v13, v190, v188 :: v_dual_fmac_f32 v10, v191, v189
v_dual_fmac_f32 v11, v190, v189 :: v_dual_fmac_f32 v12, v191, v188
v_dual_fmac_f32 v17, v190, v184 :: v_dual_fmac_f32 v14, v191, v185
v_dual_fmac_f32 v15, v191, v184 :: v_dual_fmac_f32 v16, v190, v185
v_dual_fmac_f32 v21, v194, v184 :: v_dual_fmac_f32 v18, v195, v185
v_dual_fmac_f32 v19, v194, v185 :: v_dual_fmac_f32 v20, v195, v184
v_dual_fmac_f32 v25, v194, v188 :: v_dual_fmac_f32 v22, v195, v189
v_dual_fmac_f32 v23, v195, v188 :: v_dual_fmac_f32 v24, v194, v189
v_dual_fmac_f32 v29, v198, v188 :: v_dual_fmac_f32 v26, v199, v189
v_dual_fmac_f32 v27, v198, v189 :: v_dual_fmac_f32 v28, v199, v188
v_dual_fmac_f32 v33, v198, v192 :: v_dual_fmac_f32 v30, v199, v193
v_dual_fmac_f32 v31, v199, v192 :: v_dual_fmac_f32 v32, v198, v193
v_dual_fmac_f32 v37, v186, v192 :: v_dual_fmac_f32 v34, v187, v193
v_dual_fmac_f32 v35, v186, v193 :: v_dual_fmac_f32 v36, v187, v192
v_dual_fmac_f32 v41, v186, v196 :: v_dual_fmac_f32 v38, v187, v197
v_dual_fmac_f32 v39, v187, v196 :: v_dual_fmac_f32 v40, v186, v197
v_dual_fmac_f32 v45, v190, v196 :: v_dual_fmac_f32 v42, v191, v197
v_dual_fmac_f32 v43, v190, v197 :: v_dual_fmac_f32 v44, v191, v196
v_dual_fmac_f32 v49, v190, v192 :: v_dual_fmac_f32 v46, v191, v193
v_dual_fmac_f32 v47, v191, v192 :: v_dual_fmac_f32 v48, v190, v193
v_dual_fmac_f32 v53, v194, v192 :: v_dual_fmac_f32 v50, v195, v193
v_dual_fmac_f32 v51, v194, v193 :: v_dual_fmac_f32 v52, v195, v192
v_dual_fmac_f32 v57, v194, v196 :: v_dual_fmac_f32 v54, v195, v197
v_dual_fmac_f32 v55, v195, v196 :: v_dual_fmac_f32 v56, v194, v197
v_dual_fmac_f32 v61, v198, v196 :: v_dual_fmac_f32 v58, v199, v197
v_dual_fmac_f32 v59, v198, v197 :: v_dual_fmac_f32 v60, v199, v196
v_dual_fmac_f32 v65, v198, v200 :: v_dual_fmac_f32 v62, v199, v201
v_dual_fmac_f32 v63, v199, v200 :: v_dual_fmac_f32 v64, v198, v201
v_dual_fmac_f32 v69, v186, v200 :: v_dual_fmac_f32 v66, v187, v201
v_dual_fmac_f32 v67, v186, v201 :: v_dual_fmac_f32 v68, v187, v200
v_dual_fmac_f32 v73, v186, v204 :: v_dual_fmac_f32 v70, v187, v205
v_dual_fmac_f32 v71, v187, v204 :: v_dual_fmac_f32 v72, v186, v205
v_dual_fmac_f32 v77, v190, v204 :: v_dual_fmac_f32 v74, v191, v205
v_dual_fmac_f32 v75, v190, v205 :: v_dual_fmac_f32 v76, v191, v204
v_dual_fmac_f32 v81, v190, v200 :: v_dual_fmac_f32 v78, v191, v201
v_dual_fmac_f32 v79, v191, v200 :: v_dual_fmac_f32 v80, v190, v201
v_dual_fmac_f32 v85, v194, v200 :: v_dual_fmac_f32 v82, v195, v201
v_dual_fmac_f32 v83, v194, v201 :: v_dual_fmac_f32 v84, v195, v200
v_dual_fmac_f32 v89, v194, v204 :: v_dual_fmac_f32 v86, v195, v205
v_dual_fmac_f32 v87, v195, v204 :: v_dual_fmac_f32 v88, v194, v205
v_dual_fmac_f32 v93, v198, v204 :: v_dual_fmac_f32 v90, v199, v205
v_dual_fmac_f32 v91, v198, v205 :: v_dual_fmac_f32 v92, v199, v204
v_dual_fmac_f32 v97, v198, v208 :: v_dual_fmac_f32 v94, v199, v209
v_dual_fmac_f32 v95, v199, v208 :: v_dual_fmac_f32 v96, v198, v209
v_dual_fmac_f32 v101, v186, v208 :: v_dual_fmac_f32 v98, v187, v209
v_dual_fmac_f32 v99, v186, v209 :: v_dual_fmac_f32 v100, v187, v208
v_dual_fmac_f32 v105, v186, v212 :: v_dual_fmac_f32 v102, v187, v213
v_dual_fmac_f32 v103, v187, v212 :: v_dual_fmac_f32 v104, v186, v213
v_dual_fmac_f32 v109, v190, v212 :: v_dual_fmac_f32 v106, v191, v213
v_dual_fmac_f32 v107, v190, v213 :: v_dual_fmac_f32 v108, v191, v212
v_dual_fmac_f32 v113, v190, v208 :: v_dual_fmac_f32 v110, v191, v209
v_dual_fmac_f32 v111, v191, v208 :: v_dual_fmac_f32 v112, v190, v209
v_dual_fmac_f32 v117, v194, v208 :: v_dual_fmac_f32 v114, v195, v209
v_dual_fmac_f32 v115, v194, v209 :: v_dual_fmac_f32 v116, v195, v208
v_dual_fmac_f32 v121, v194, v212 :: v_dual_fmac_f32 v122, v195, v213
v_dual_fmac_f32 v123, v195, v212 :: v_dual_fmac_f32 v120, v194, v213
v_dual_fmac_f32 v129, v198, v212 :: v_dual_fmac_f32 v126, v199, v213
v_dual_fmac_f32 v127, v198, v213 :: v_dual_fmac_f32 v124, v199, v212
v_dual_fmac_f32 v133, v198, v184 :: v_dual_fmac_f32 v214, v199, v185
v_dual_fmac_f32 v131, v199, v184 :: v_dual_fmac_f32 v128, v198, v185

s_setprio 0


	; Global memory read for matrix A
	s_clause 0x1
	global_load_b32	 v177, v215, s[44:45]
	global_load_b32	 v178, v215, s[46:47]

	s_clause 0xB;
 ;A on bank 2-3
	ds_load_b64 v[186:187], v183
	ds_load_b64 v[190:191], v183 offset: 8
	ds_load_b64 v[194:195], v183 offset: 64
	ds_load_b64 v[198:199], v183 offset: 72


 ;B on bank 0-1
	ds_load_b64 v[184:185], v202
	ds_load_b64 v[188:189], v202 offset: 8
	ds_load_b64 v[192:193], v202 offset: 128
	ds_load_b64 v[196:197], v202 offset: 136
	ds_load_b64 v[200:201], v202 offset: 256
	ds_load_b64 v[204:205], v202 offset: 264
	ds_load_b64 v[208:209], v202 offset: 384
	ds_load_b64 v[212:213], v202 offset: 392

	v_add_nc_u32_e32 v183, 0x210, v183
	v_add_nc_u32_e32 v202, 0x200, v202
	;s_cmpk_lg_i32 s14, 0x1000
	s_waitcnt lgkmcnt(0)

;	; new vpgrs allocation

v_dual_fmac_f32 v5, v186, v184 :: v_dual_fmac_f32 v2, v187, v185
s_setprio 1
v_dual_fmac_f32 v3, v186, v185 :: v_dual_fmac_f32 v4, v187, v184
v_dual_fmac_f32 v9, v186, v188 :: v_dual_fmac_f32 v6, v187, v189
v_dual_fmac_f32 v7, v187, v188 :: v_dual_fmac_f32 v8, v186, v189
v_dual_fmac_f32 v13, v190, v188 :: v_dual_fmac_f32 v10, v191, v189
v_dual_fmac_f32 v11, v190, v189 :: v_dual_fmac_f32 v12, v191, v188
v_dual_fmac_f32 v17, v190, v184 :: v_dual_fmac_f32 v14, v191, v185
v_dual_fmac_f32 v15, v191, v184 :: v_dual_fmac_f32 v16, v190, v185
v_dual_fmac_f32 v21, v194, v184 :: v_dual_fmac_f32 v18, v195, v185
v_dual_fmac_f32 v19, v194, v185 :: v_dual_fmac_f32 v20, v195, v184
v_dual_fmac_f32 v25, v194, v188 :: v_dual_fmac_f32 v22, v195, v189
v_dual_fmac_f32 v23, v195, v188 :: v_dual_fmac_f32 v24, v194, v189
v_dual_fmac_f32 v29, v198, v188 :: v_dual_fmac_f32 v26, v199, v189
v_dual_fmac_f32 v27, v198, v189 :: v_dual_fmac_f32 v28, v199, v188
v_dual_fmac_f32 v33, v198, v192 :: v_dual_fmac_f32 v30, v199, v193
v_dual_fmac_f32 v31, v199, v192 :: v_dual_fmac_f32 v32, v198, v193
v_dual_fmac_f32 v37, v186, v192 :: v_dual_fmac_f32 v34, v187, v193
v_dual_fmac_f32 v35, v186, v193 :: v_dual_fmac_f32 v36, v187, v192
v_dual_fmac_f32 v41, v186, v196 :: v_dual_fmac_f32 v38, v187, v197
v_dual_fmac_f32 v39, v187, v196 :: v_dual_fmac_f32 v40, v186, v197
v_dual_fmac_f32 v45, v190, v196 :: v_dual_fmac_f32 v42, v191, v197
v_dual_fmac_f32 v43, v190, v197 :: v_dual_fmac_f32 v44, v191, v196
v_dual_fmac_f32 v49, v190, v192 :: v_dual_fmac_f32 v46, v191, v193
v_dual_fmac_f32 v47, v191, v192 :: v_dual_fmac_f32 v48, v190, v193
v_dual_fmac_f32 v53, v194, v192 :: v_dual_fmac_f32 v50, v195, v193
v_dual_fmac_f32 v51, v194, v193 :: v_dual_fmac_f32 v52, v195, v192
v_dual_fmac_f32 v57, v194, v196 :: v_dual_fmac_f32 v54, v195, v197
v_dual_fmac_f32 v55, v195, v196 :: v_dual_fmac_f32 v56, v194, v197
v_dual_fmac_f32 v61, v198, v196 :: v_dual_fmac_f32 v58, v199, v197
v_dual_fmac_f32 v59, v198, v197 :: v_dual_fmac_f32 v60, v199, v196
v_dual_fmac_f32 v65, v198, v200 :: v_dual_fmac_f32 v62, v199, v201
v_dual_fmac_f32 v63, v199, v200 :: v_dual_fmac_f32 v64, v198, v201
v_dual_fmac_f32 v69, v186, v200 :: v_dual_fmac_f32 v66, v187, v201
v_dual_fmac_f32 v67, v186, v201 :: v_dual_fmac_f32 v68, v187, v200
v_dual_fmac_f32 v73, v186, v204 :: v_dual_fmac_f32 v70, v187, v205
v_dual_fmac_f32 v71, v187, v204 :: v_dual_fmac_f32 v72, v186, v205
v_dual_fmac_f32 v77, v190, v204 :: v_dual_fmac_f32 v74, v191, v205
v_dual_fmac_f32 v75, v190, v205 :: v_dual_fmac_f32 v76, v191, v204
v_dual_fmac_f32 v81, v190, v200 :: v_dual_fmac_f32 v78, v191, v201
v_dual_fmac_f32 v79, v191, v200 :: v_dual_fmac_f32 v80, v190, v201
v_dual_fmac_f32 v85, v194, v200 :: v_dual_fmac_f32 v82, v195, v201
v_dual_fmac_f32 v83, v194, v201 :: v_dual_fmac_f32 v84, v195, v200
v_dual_fmac_f32 v89, v194, v204 :: v_dual_fmac_f32 v86, v195, v205
v_dual_fmac_f32 v87, v195, v204 :: v_dual_fmac_f32 v88, v194, v205
v_dual_fmac_f32 v93, v198, v204 :: v_dual_fmac_f32 v90, v199, v205
v_dual_fmac_f32 v91, v198, v205 :: v_dual_fmac_f32 v92, v199, v204
v_dual_fmac_f32 v97, v198, v208 :: v_dual_fmac_f32 v94, v199, v209
v_dual_fmac_f32 v95, v199, v208 :: v_dual_fmac_f32 v96, v198, v209
v_dual_fmac_f32 v101, v186, v208 :: v_dual_fmac_f32 v98, v187, v209
v_dual_fmac_f32 v99, v186, v209 :: v_dual_fmac_f32 v100, v187, v208
v_dual_fmac_f32 v105, v186, v212 :: v_dual_fmac_f32 v102, v187, v213
v_dual_fmac_f32 v103, v187, v212 :: v_dual_fmac_f32 v104, v186, v213
v_dual_fmac_f32 v109, v190, v212 :: v_dual_fmac_f32 v106, v191, v213
v_dual_fmac_f32 v107, v190, v213 :: v_dual_fmac_f32 v108, v191, v212
v_dual_fmac_f32 v113, v190, v208 :: v_dual_fmac_f32 v110, v191, v209
v_dual_fmac_f32 v111, v191, v208 :: v_dual_fmac_f32 v112, v190, v209
v_dual_fmac_f32 v117, v194, v208 :: v_dual_fmac_f32 v114, v195, v209
v_dual_fmac_f32 v115, v194, v209 :: v_dual_fmac_f32 v116, v195, v208
v_dual_fmac_f32 v121, v194, v212 :: v_dual_fmac_f32 v122, v195, v213
v_dual_fmac_f32 v123, v195, v212 :: v_dual_fmac_f32 v120, v194, v213
v_dual_fmac_f32 v129, v198, v212 :: v_dual_fmac_f32 v126, v199, v213
v_dual_fmac_f32 v127, v198, v213 :: v_dual_fmac_f32 v124, v199, v212
v_dual_fmac_f32 v133, v198, v184 :: v_dual_fmac_f32 v214, v199, v185
v_dual_fmac_f32 v131, v199, v184 :: v_dual_fmac_f32 v128, v198, v185

s_setprio 0

	s_clause 0x1
	global_load_b32	 v179, v215, s[48:49]
	global_load_b32	 v180, v215, s[50:51]


	s_clause 0xB;
 ;A on bank 2-3
	ds_load_b64 v[186:187], v183
	ds_load_b64 v[190:191], v183 offset: 8
	ds_load_b64 v[194:195], v183 offset: 64
	ds_load_b64 v[198:199], v183 offset: 72


 ;B on bank 0-1
	ds_load_b64 v[184:185], v202
	ds_load_b64 v[188:189], v202 offset: 8
	ds_load_b64 v[192:193], v202 offset: 128
	ds_load_b64 v[196:197], v202 offset: 136
	ds_load_b64 v[200:201], v202 offset: 256
	ds_load_b64 v[204:205], v202 offset: 264
	ds_load_b64 v[208:209], v202 offset: 384
	ds_load_b64 v[212:213], v202 offset: 392

	v_add_nc_u32_e32 v183, 0x210, v183
	v_add_nc_u32_e32 v202, 0x200, v202
	;s_cmpk_lg_i32 s14, 0x1000
	s_waitcnt lgkmcnt(0)

;	; new vpgrs allocation

v_dual_fmac_f32 v5, v186, v184 :: v_dual_fmac_f32 v2, v187, v185
s_setprio 1
v_dual_fmac_f32 v3, v186, v185 :: v_dual_fmac_f32 v4, v187, v184
v_dual_fmac_f32 v9, v186, v188 :: v_dual_fmac_f32 v6, v187, v189
v_dual_fmac_f32 v7, v187, v188 :: v_dual_fmac_f32 v8, v186, v189
v_dual_fmac_f32 v13, v190, v188 :: v_dual_fmac_f32 v10, v191, v189
v_dual_fmac_f32 v11, v190, v189 :: v_dual_fmac_f32 v12, v191, v188
v_dual_fmac_f32 v17, v190, v184 :: v_dual_fmac_f32 v14, v191, v185
v_dual_fmac_f32 v15, v191, v184 :: v_dual_fmac_f32 v16, v190, v185
v_dual_fmac_f32 v21, v194, v184 :: v_dual_fmac_f32 v18, v195, v185
v_dual_fmac_f32 v19, v194, v185 :: v_dual_fmac_f32 v20, v195, v184
v_dual_fmac_f32 v25, v194, v188 :: v_dual_fmac_f32 v22, v195, v189
v_dual_fmac_f32 v23, v195, v188 :: v_dual_fmac_f32 v24, v194, v189
v_dual_fmac_f32 v29, v198, v188 :: v_dual_fmac_f32 v26, v199, v189
v_dual_fmac_f32 v27, v198, v189 :: v_dual_fmac_f32 v28, v199, v188
v_dual_fmac_f32 v33, v198, v192 :: v_dual_fmac_f32 v30, v199, v193
v_dual_fmac_f32 v31, v199, v192 :: v_dual_fmac_f32 v32, v198, v193
v_dual_fmac_f32 v37, v186, v192 :: v_dual_fmac_f32 v34, v187, v193
v_dual_fmac_f32 v35, v186, v193 :: v_dual_fmac_f32 v36, v187, v192
v_dual_fmac_f32 v41, v186, v196 :: v_dual_fmac_f32 v38, v187, v197
v_dual_fmac_f32 v39, v187, v196 :: v_dual_fmac_f32 v40, v186, v197
v_dual_fmac_f32 v45, v190, v196 :: v_dual_fmac_f32 v42, v191, v197
v_dual_fmac_f32 v43, v190, v197 :: v_dual_fmac_f32 v44, v191, v196
v_dual_fmac_f32 v49, v190, v192 :: v_dual_fmac_f32 v46, v191, v193
v_dual_fmac_f32 v47, v191, v192 :: v_dual_fmac_f32 v48, v190, v193
v_dual_fmac_f32 v53, v194, v192 :: v_dual_fmac_f32 v50, v195, v193
v_dual_fmac_f32 v51, v194, v193 :: v_dual_fmac_f32 v52, v195, v192
v_dual_fmac_f32 v57, v194, v196 :: v_dual_fmac_f32 v54, v195, v197
v_dual_fmac_f32 v55, v195, v196 :: v_dual_fmac_f32 v56, v194, v197
v_dual_fmac_f32 v61, v198, v196 :: v_dual_fmac_f32 v58, v199, v197
v_dual_fmac_f32 v59, v198, v197 :: v_dual_fmac_f32 v60, v199, v196
v_dual_fmac_f32 v65, v198, v200 :: v_dual_fmac_f32 v62, v199, v201
v_dual_fmac_f32 v63, v199, v200 :: v_dual_fmac_f32 v64, v198, v201
v_dual_fmac_f32 v69, v186, v200 :: v_dual_fmac_f32 v66, v187, v201
v_dual_fmac_f32 v67, v186, v201 :: v_dual_fmac_f32 v68, v187, v200
v_dual_fmac_f32 v73, v186, v204 :: v_dual_fmac_f32 v70, v187, v205
v_dual_fmac_f32 v71, v187, v204 :: v_dual_fmac_f32 v72, v186, v205
v_dual_fmac_f32 v77, v190, v204 :: v_dual_fmac_f32 v74, v191, v205
v_dual_fmac_f32 v75, v190, v205 :: v_dual_fmac_f32 v76, v191, v204
v_dual_fmac_f32 v81, v190, v200 :: v_dual_fmac_f32 v78, v191, v201
v_dual_fmac_f32 v79, v191, v200 :: v_dual_fmac_f32 v80, v190, v201
v_dual_fmac_f32 v85, v194, v200 :: v_dual_fmac_f32 v82, v195, v201
v_dual_fmac_f32 v83, v194, v201 :: v_dual_fmac_f32 v84, v195, v200
v_dual_fmac_f32 v89, v194, v204 :: v_dual_fmac_f32 v86, v195, v205
v_dual_fmac_f32 v87, v195, v204 :: v_dual_fmac_f32 v88, v194, v205
v_dual_fmac_f32 v93, v198, v204 :: v_dual_fmac_f32 v90, v199, v205
v_dual_fmac_f32 v91, v198, v205 :: v_dual_fmac_f32 v92, v199, v204
v_dual_fmac_f32 v97, v198, v208 :: v_dual_fmac_f32 v94, v199, v209
v_dual_fmac_f32 v95, v199, v208 :: v_dual_fmac_f32 v96, v198, v209
v_dual_fmac_f32 v101, v186, v208 :: v_dual_fmac_f32 v98, v187, v209
v_dual_fmac_f32 v99, v186, v209 :: v_dual_fmac_f32 v100, v187, v208
v_dual_fmac_f32 v105, v186, v212 :: v_dual_fmac_f32 v102, v187, v213
v_dual_fmac_f32 v103, v187, v212 :: v_dual_fmac_f32 v104, v186, v213
v_dual_fmac_f32 v109, v190, v212 :: v_dual_fmac_f32 v106, v191, v213
v_dual_fmac_f32 v107, v190, v213 :: v_dual_fmac_f32 v108, v191, v212
v_dual_fmac_f32 v113, v190, v208 :: v_dual_fmac_f32 v110, v191, v209
v_dual_fmac_f32 v111, v191, v208 :: v_dual_fmac_f32 v112, v190, v209
v_dual_fmac_f32 v117, v194, v208 :: v_dual_fmac_f32 v114, v195, v209
v_dual_fmac_f32 v115, v194, v209 :: v_dual_fmac_f32 v116, v195, v208
v_dual_fmac_f32 v121, v194, v212 :: v_dual_fmac_f32 v122, v195, v213
v_dual_fmac_f32 v123, v195, v212 :: v_dual_fmac_f32 v120, v194, v213
v_dual_fmac_f32 v129, v198, v212 :: v_dual_fmac_f32 v126, v199, v213
v_dual_fmac_f32 v127, v198, v213 :: v_dual_fmac_f32 v124, v199, v212
v_dual_fmac_f32 v133, v198, v184 :: v_dual_fmac_f32 v214, v199, v185
v_dual_fmac_f32 v131, v199, v184 :: v_dual_fmac_f32 v128, v198, v185

s_setprio 0
	s_clause 0x1
	global_load_b32	 v181, v215, s[52:53]
	global_load_b32	 v182, v215, s[54:55]
	s_clause 0xB;
 ;A on bank 2-3
	ds_load_b64 v[186:187], v183
	ds_load_b64 v[190:191], v183 offset: 8
	ds_load_b64 v[194:195], v183 offset: 64
	ds_load_b64 v[198:199], v183 offset: 72


 ;B on bank 0-1
	ds_load_b64 v[184:185], v202
	ds_load_b64 v[188:189], v202 offset: 8
	ds_load_b64 v[192:193], v202 offset: 128
	ds_load_b64 v[196:197], v202 offset: 136
	ds_load_b64 v[200:201], v202 offset: 256
	ds_load_b64 v[204:205], v202 offset: 264
	ds_load_b64 v[208:209], v202 offset: 384
	ds_load_b64 v[212:213], v202 offset: 392

	v_add_nc_u32_e32 v183, 0x210, v183
	v_add_nc_u32_e32 v202, 0x200, v202
	s_waitcnt lgkmcnt(0)

;	new vpgrs allocation

v_dual_fmac_f32 v5, v186, v184 :: v_dual_fmac_f32 v2, v187, v185
s_setprio 1
v_dual_fmac_f32 v3, v186, v185 :: v_dual_fmac_f32 v4, v187, v184
v_dual_fmac_f32 v9, v186, v188 :: v_dual_fmac_f32 v6, v187, v189
v_dual_fmac_f32 v7, v187, v188 :: v_dual_fmac_f32 v8, v186, v189
v_dual_fmac_f32 v13, v190, v188 :: v_dual_fmac_f32 v10, v191, v189
v_dual_fmac_f32 v11, v190, v189 :: v_dual_fmac_f32 v12, v191, v188
v_dual_fmac_f32 v17, v190, v184 :: v_dual_fmac_f32 v14, v191, v185
v_dual_fmac_f32 v15, v191, v184 :: v_dual_fmac_f32 v16, v190, v185
v_dual_fmac_f32 v21, v194, v184 :: v_dual_fmac_f32 v18, v195, v185
v_dual_fmac_f32 v19, v194, v185 :: v_dual_fmac_f32 v20, v195, v184
v_dual_fmac_f32 v25, v194, v188 :: v_dual_fmac_f32 v22, v195, v189
v_dual_fmac_f32 v23, v195, v188 :: v_dual_fmac_f32 v24, v194, v189
v_dual_fmac_f32 v29, v198, v188 :: v_dual_fmac_f32 v26, v199, v189
v_dual_fmac_f32 v27, v198, v189 :: v_dual_fmac_f32 v28, v199, v188
v_dual_fmac_f32 v33, v198, v192 :: v_dual_fmac_f32 v30, v199, v193
v_dual_fmac_f32 v31, v199, v192 :: v_dual_fmac_f32 v32, v198, v193
v_dual_fmac_f32 v37, v186, v192 :: v_dual_fmac_f32 v34, v187, v193
v_dual_fmac_f32 v35, v186, v193 :: v_dual_fmac_f32 v36, v187, v192
v_dual_fmac_f32 v41, v186, v196 :: v_dual_fmac_f32 v38, v187, v197
v_dual_fmac_f32 v39, v187, v196 :: v_dual_fmac_f32 v40, v186, v197
v_dual_fmac_f32 v45, v190, v196 :: v_dual_fmac_f32 v42, v191, v197
v_dual_fmac_f32 v43, v190, v197 :: v_dual_fmac_f32 v44, v191, v196
v_dual_fmac_f32 v49, v190, v192 :: v_dual_fmac_f32 v46, v191, v193
v_dual_fmac_f32 v47, v191, v192 :: v_dual_fmac_f32 v48, v190, v193
v_dual_fmac_f32 v53, v194, v192 :: v_dual_fmac_f32 v50, v195, v193
v_dual_fmac_f32 v51, v194, v193 :: v_dual_fmac_f32 v52, v195, v192
v_dual_fmac_f32 v57, v194, v196 :: v_dual_fmac_f32 v54, v195, v197
v_dual_fmac_f32 v55, v195, v196 :: v_dual_fmac_f32 v56, v194, v197
v_dual_fmac_f32 v61, v198, v196 :: v_dual_fmac_f32 v58, v199, v197
v_dual_fmac_f32 v59, v198, v197 :: v_dual_fmac_f32 v60, v199, v196
v_dual_fmac_f32 v65, v198, v200 :: v_dual_fmac_f32 v62, v199, v201
v_dual_fmac_f32 v63, v199, v200 :: v_dual_fmac_f32 v64, v198, v201
v_dual_fmac_f32 v69, v186, v200 :: v_dual_fmac_f32 v66, v187, v201
v_dual_fmac_f32 v67, v186, v201 :: v_dual_fmac_f32 v68, v187, v200
v_dual_fmac_f32 v73, v186, v204 :: v_dual_fmac_f32 v70, v187, v205
v_dual_fmac_f32 v71, v187, v204 :: v_dual_fmac_f32 v72, v186, v205
v_dual_fmac_f32 v77, v190, v204 :: v_dual_fmac_f32 v74, v191, v205
v_dual_fmac_f32 v75, v190, v205 :: v_dual_fmac_f32 v76, v191, v204
v_dual_fmac_f32 v81, v190, v200 :: v_dual_fmac_f32 v78, v191, v201
v_dual_fmac_f32 v79, v191, v200 :: v_dual_fmac_f32 v80, v190, v201
v_dual_fmac_f32 v85, v194, v200 :: v_dual_fmac_f32 v82, v195, v201
v_dual_fmac_f32 v83, v194, v201 :: v_dual_fmac_f32 v84, v195, v200
v_dual_fmac_f32 v89, v194, v204 :: v_dual_fmac_f32 v86, v195, v205
v_dual_fmac_f32 v87, v195, v204 :: v_dual_fmac_f32 v88, v194, v205
v_dual_fmac_f32 v93, v198, v204 :: v_dual_fmac_f32 v90, v199, v205
v_dual_fmac_f32 v91, v198, v205 :: v_dual_fmac_f32 v92, v199, v204
v_dual_fmac_f32 v97, v198, v208 :: v_dual_fmac_f32 v94, v199, v209
v_dual_fmac_f32 v95, v199, v208 :: v_dual_fmac_f32 v96, v198, v209
v_dual_fmac_f32 v101, v186, v208 :: v_dual_fmac_f32 v98, v187, v209
v_dual_fmac_f32 v99, v186, v209 :: v_dual_fmac_f32 v100, v187, v208
v_dual_fmac_f32 v105, v186, v212 :: v_dual_fmac_f32 v102, v187, v213
v_dual_fmac_f32 v103, v187, v212 :: v_dual_fmac_f32 v104, v186, v213
v_dual_fmac_f32 v109, v190, v212 :: v_dual_fmac_f32 v106, v191, v213
v_dual_fmac_f32 v107, v190, v213 :: v_dual_fmac_f32 v108, v191, v212
v_dual_fmac_f32 v113, v190, v208 :: v_dual_fmac_f32 v110, v191, v209
v_dual_fmac_f32 v111, v191, v208 :: v_dual_fmac_f32 v112, v190, v209
v_dual_fmac_f32 v117, v194, v208 :: v_dual_fmac_f32 v114, v195, v209
v_dual_fmac_f32 v115, v194, v209 :: v_dual_fmac_f32 v116, v195, v208
v_dual_fmac_f32 v121, v194, v212 :: v_dual_fmac_f32 v122, v195, v213
v_dual_fmac_f32 v123, v195, v212 :: v_dual_fmac_f32 v120, v194, v213
v_dual_fmac_f32 v129, v198, v212 :: v_dual_fmac_f32 v126, v199, v213
v_dual_fmac_f32 v127, v198, v213 :: v_dual_fmac_f32 v124, v199, v212
v_dual_fmac_f32 v133, v198, v184 :: v_dual_fmac_f32 v214, v199, v185
v_dual_fmac_f32 v131, v199, v184 :: v_dual_fmac_f32 v128, v198, v185

s_setprio 0





	s_clause 0xB;
 ;A on bank 2-3
	ds_load_b64 v[186:187], v183
	ds_load_b64 v[190:191], v183 offset: 8
	ds_load_b64 v[194:195], v183 offset: 64
	ds_load_b64 v[198:199], v183 offset: 72


 ;B on bank 0-1
	ds_load_b64 v[184:185], v202
	ds_load_b64 v[188:189], v202 offset: 8
	ds_load_b64 v[192:193], v202 offset: 128
	ds_load_b64 v[196:197], v202 offset: 136
	ds_load_b64 v[200:201], v202 offset: 256
	ds_load_b64 v[204:205], v202 offset: 264
	ds_load_b64 v[208:209], v202 offset: 384
	ds_load_b64 v[212:213], v202 offset: 392

	v_add_nc_u32_e32 v183, 0x210, v183
	v_add_nc_u32_e32 v202, 0x200, v202
	;s_cmpk_lg_i32 s14, 0x1000
	s_waitcnt lgkmcnt(0)

;	; new vpgrs allocation

v_dual_fmac_f32 v5, v186, v184 :: v_dual_fmac_f32 v2, v187, v185
s_setprio 1
v_dual_fmac_f32 v3, v186, v185 :: v_dual_fmac_f32 v4, v187, v184
v_dual_fmac_f32 v9, v186, v188 :: v_dual_fmac_f32 v6, v187, v189
v_dual_fmac_f32 v7, v187, v188 :: v_dual_fmac_f32 v8, v186, v189
v_dual_fmac_f32 v13, v190, v188 :: v_dual_fmac_f32 v10, v191, v189
v_dual_fmac_f32 v11, v190, v189 :: v_dual_fmac_f32 v12, v191, v188
v_dual_fmac_f32 v17, v190, v184 :: v_dual_fmac_f32 v14, v191, v185
v_dual_fmac_f32 v15, v191, v184 :: v_dual_fmac_f32 v16, v190, v185
v_dual_fmac_f32 v21, v194, v184 :: v_dual_fmac_f32 v18, v195, v185
v_dual_fmac_f32 v19, v194, v185 :: v_dual_fmac_f32 v20, v195, v184
v_dual_fmac_f32 v25, v194, v188 :: v_dual_fmac_f32 v22, v195, v189
v_dual_fmac_f32 v23, v195, v188 :: v_dual_fmac_f32 v24, v194, v189
v_dual_fmac_f32 v29, v198, v188 :: v_dual_fmac_f32 v26, v199, v189
v_dual_fmac_f32 v27, v198, v189 :: v_dual_fmac_f32 v28, v199, v188
v_dual_fmac_f32 v33, v198, v192 :: v_dual_fmac_f32 v30, v199, v193
v_dual_fmac_f32 v31, v199, v192 :: v_dual_fmac_f32 v32, v198, v193
v_dual_fmac_f32 v37, v186, v192 :: v_dual_fmac_f32 v34, v187, v193
v_dual_fmac_f32 v35, v186, v193 :: v_dual_fmac_f32 v36, v187, v192
v_dual_fmac_f32 v41, v186, v196 :: v_dual_fmac_f32 v38, v187, v197
v_dual_fmac_f32 v39, v187, v196 :: v_dual_fmac_f32 v40, v186, v197
v_dual_fmac_f32 v45, v190, v196 :: v_dual_fmac_f32 v42, v191, v197
v_dual_fmac_f32 v43, v190, v197 :: v_dual_fmac_f32 v44, v191, v196
v_dual_fmac_f32 v49, v190, v192 :: v_dual_fmac_f32 v46, v191, v193
v_dual_fmac_f32 v47, v191, v192 :: v_dual_fmac_f32 v48, v190, v193
v_dual_fmac_f32 v53, v194, v192 :: v_dual_fmac_f32 v50, v195, v193
v_dual_fmac_f32 v51, v194, v193 :: v_dual_fmac_f32 v52, v195, v192
v_dual_fmac_f32 v57, v194, v196 :: v_dual_fmac_f32 v54, v195, v197
v_dual_fmac_f32 v55, v195, v196 :: v_dual_fmac_f32 v56, v194, v197
v_dual_fmac_f32 v61, v198, v196 :: v_dual_fmac_f32 v58, v199, v197
v_dual_fmac_f32 v59, v198, v197 :: v_dual_fmac_f32 v60, v199, v196
v_dual_fmac_f32 v65, v198, v200 :: v_dual_fmac_f32 v62, v199, v201
v_dual_fmac_f32 v63, v199, v200 :: v_dual_fmac_f32 v64, v198, v201
v_dual_fmac_f32 v69, v186, v200 :: v_dual_fmac_f32 v66, v187, v201
v_dual_fmac_f32 v67, v186, v201 :: v_dual_fmac_f32 v68, v187, v200
v_dual_fmac_f32 v73, v186, v204 :: v_dual_fmac_f32 v70, v187, v205
v_dual_fmac_f32 v71, v187, v204 :: v_dual_fmac_f32 v72, v186, v205
v_dual_fmac_f32 v77, v190, v204 :: v_dual_fmac_f32 v74, v191, v205
v_dual_fmac_f32 v75, v190, v205 :: v_dual_fmac_f32 v76, v191, v204
v_dual_fmac_f32 v81, v190, v200 :: v_dual_fmac_f32 v78, v191, v201
v_dual_fmac_f32 v79, v191, v200 :: v_dual_fmac_f32 v80, v190, v201
v_dual_fmac_f32 v85, v194, v200 :: v_dual_fmac_f32 v82, v195, v201
v_dual_fmac_f32 v83, v194, v201 :: v_dual_fmac_f32 v84, v195, v200
v_dual_fmac_f32 v89, v194, v204 :: v_dual_fmac_f32 v86, v195, v205
v_dual_fmac_f32 v87, v195, v204 :: v_dual_fmac_f32 v88, v194, v205
v_dual_fmac_f32 v93, v198, v204 :: v_dual_fmac_f32 v90, v199, v205
v_dual_fmac_f32 v91, v198, v205 :: v_dual_fmac_f32 v92, v199, v204
v_dual_fmac_f32 v97, v198, v208 :: v_dual_fmac_f32 v94, v199, v209
v_dual_fmac_f32 v95, v199, v208 :: v_dual_fmac_f32 v96, v198, v209
v_dual_fmac_f32 v101, v186, v208 :: v_dual_fmac_f32 v98, v187, v209
v_dual_fmac_f32 v99, v186, v209 :: v_dual_fmac_f32 v100, v187, v208
v_dual_fmac_f32 v105, v186, v212 :: v_dual_fmac_f32 v102, v187, v213
v_dual_fmac_f32 v103, v187, v212 :: v_dual_fmac_f32 v104, v186, v213
v_dual_fmac_f32 v109, v190, v212 :: v_dual_fmac_f32 v106, v191, v213
v_dual_fmac_f32 v107, v190, v213 :: v_dual_fmac_f32 v108, v191, v212
v_dual_fmac_f32 v113, v190, v208 :: v_dual_fmac_f32 v110, v191, v209
v_dual_fmac_f32 v111, v191, v208 :: v_dual_fmac_f32 v112, v190, v209
v_dual_fmac_f32 v117, v194, v208 :: v_dual_fmac_f32 v114, v195, v209
v_dual_fmac_f32 v115, v194, v209 :: v_dual_fmac_f32 v116, v195, v208
v_dual_fmac_f32 v121, v194, v212 :: v_dual_fmac_f32 v122, v195, v213
v_dual_fmac_f32 v123, v195, v212 :: v_dual_fmac_f32 v120, v194, v213
v_dual_fmac_f32 v129, v198, v212 :: v_dual_fmac_f32 v126, v199, v213
v_dual_fmac_f32 v127, v198, v213 :: v_dual_fmac_f32 v124, v199, v212
v_dual_fmac_f32 v133, v198, v184 :: v_dual_fmac_f32 v214, v199, v185
v_dual_fmac_f32 v131, v199, v184 :: v_dual_fmac_f32 v128, v198, v185

s_setprio 0



	;s_cbranch_scc1 .LBB0_9
; %bb.10:                               ;   in Loop: Header=BB0_6 Depth=1
	s_and_not1_b32 vcc_lo, exec_lo, s13
	s_waitcnt vmcnt(0)
	s_barrier
	s_cbranch_vccnz .LBB0_5
; %bb.11:                               ; %.preheader190.preheader
                                        ;   in Loop: Header=BB0_6 Depth=1
	ds_store_b32 v155, v167
	ds_store_b32 v141, v175
	ds_store_b32 v158, v168
	ds_store_b32 v142, v176
	ds_store_b32 v159, v169
	ds_store_b32 v143, v177
	ds_store_b32 v160, v170
	ds_store_b32 v144, v178
	ds_store_b32 v161, v171
	ds_store_b32 v145, v179
	ds_store_b32 v162, v172
	ds_store_b32 v146, v180
	ds_store_b32 v163, v173
	ds_store_b32 v147, v181
	ds_store_b32 v164, v174
	ds_store_b32 v148, v182
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_branch .LBB0_5

.LBB0_12:                               ; %Flow
;Restoring VGPR original allocation
	; v2 -> v128 & v128 ->  v2
	v_mov_b32 v200, v128
	v_mov_b32 v128, v2
	v_mov_b32 v2, v200
	; v128 -> v56 & v56 ->  v128
	v_mov_b32 v200, v56
	v_mov_b32 v56, v2
	v_mov_b32 v2, v200
	; v56 -> v46 & v46 ->  v56
	v_mov_b32 v200, v46
	v_mov_b32 v46, v2
	v_mov_b32 v2, v200
	; v46 -> v100 & v100 ->  v46
	v_mov_b32 v200, v100
	v_mov_b32 v100, v2
	v_mov_b32 v2, v200
	; v100 -> v77 & v77 ->  v100
	v_mov_b32 v200, v77
	v_mov_b32 v77, v2
	v_mov_b32 v2, v200
	; v77 -> v87 & v87 ->  v77
	v_mov_b32 v200, v87
	v_mov_b32 v87, v2
	v_mov_b32 v2, v200
	; v87 -> v27 & v27 ->  v87
	v_mov_b32 v200, v27
	v_mov_b32 v27, v2
	v_mov_b32 v2, v200
	; v27 -> v54 & v54 ->  v27
	v_mov_b32 v200, v54
	v_mov_b32 v54, v2
	v_mov_b32 v2, v200
	; v54 -> v42 & v42 ->  v54
	v_mov_b32 v200, v42
	v_mov_b32 v42, v2
	v_mov_b32 v2, v200
	; v42 -> v98 & v98 ->  v42
	v_mov_b32 v200, v98
	v_mov_b32 v98, v2
	v_mov_b32 v2, v200
	; v98 -> v76 & v76 ->  v98
	v_mov_b32 v200, v76
	v_mov_b32 v76, v2
	v_mov_b32 v2, v200
	; v76 -> v83 & v83 ->  v76
	v_mov_b32 v200, v83
	v_mov_b32 v83, v2
	v_mov_b32 v2, v200
	; v83 -> v32 & v32 ->  v83
	v_mov_b32 v200, v32
	v_mov_b32 v32, v2
	v_mov_b32 v2, v200
	; v32 -> v40 & v40 ->  v32
	v_mov_b32 v200, v40
	v_mov_b32 v40, v2
	v_mov_b32 v2, v200
	; v40 -> v110 & v110 ->  v40
	v_mov_b32 v200, v110
	v_mov_b32 v110, v2
	v_mov_b32 v2, v200
	; v110 -> v68 & v68 ->  v110
	v_mov_b32 v200, v68
	v_mov_b32 v68, v2
	v_mov_b32 v2, v200
	; v68 -> v93 & v93 ->  v68
	v_mov_b32 v200, v93
	v_mov_b32 v93, v2
	v_mov_b32 v2, v200
	; v93 -> v23 & v23 ->  v93
	v_mov_b32 v200, v23
	v_mov_b32 v23, v2
	v_mov_b32 v2, v200
	; v23 -> v59 & v59 ->  v23
	v_mov_b32 v200, v59
	v_mov_b32 v59, v2
	v_mov_b32 v2, v200
	; v59 -> v38 & v38 ->  v59
	v_mov_b32 v200, v38
	v_mov_b32 v38, v2
	v_mov_b32 v2, v200
	; v38 -> v106 & v106 ->  v38
	v_mov_b32 v200, v106
	v_mov_b32 v106, v2
	v_mov_b32 v2, v200
	; v106 -> v66 & v66 ->  v106
	v_mov_b32 v200, v66
	v_mov_b32 v66, v2
	v_mov_b32 v2, v200
	; v66 -> v92 & v92 ->  v66
	v_mov_b32 v200, v92
	v_mov_b32 v92, v2
	v_mov_b32 v2, v200
	; v92 -> v19 & v19 ->  v92
	v_mov_b32 v200, v19
	v_mov_b32 v19, v2
	v_mov_b32 v2, v200
	; v19 -> v64 & v64 ->  v19
	v_mov_b32 v200, v64
	v_mov_b32 v64, v2
	v_mov_b32 v2, v200
	; v64 -> v24 & v24 ->  v64
	v_mov_b32 v200, v24
	v_mov_b32 v24, v2
	v_mov_b32 v2, v200
	; v24 -> v62 & v62 ->  v24
	v_mov_b32 v200, v62
	v_mov_b32 v62, v2
	v_mov_b32 v2, v200
	; v62 -> v20 & v20 ->  v62
	v_mov_b32 v200, v20
	v_mov_b32 v20, v2
	v_mov_b32 v2, v200
	; v20 -> v61 & v61 ->  v20
	v_mov_b32 v200, v61
	v_mov_b32 v61, v2
	v_mov_b32 v2, v200
	; v61 -> v39 & v39 ->  v61
	v_mov_b32 v200, v39
	v_mov_b32 v39, v2
	v_mov_b32 v2, v200
	; v39 -> v107 & v107 ->  v39
	v_mov_b32 v200, v107
	v_mov_b32 v107, v2
	v_mov_b32 v2, v200
	; v107 -> v70 & v70 ->  v107
	v_mov_b32 v200, v70
	v_mov_b32 v70, v2
	v_mov_b32 v2, v200
	; v70 -> v90 & v90 ->  v70
	v_mov_b32 v200, v90
	v_mov_b32 v90, v2
	v_mov_b32 v2, v200
	; v90 -> v18 & v18 ->  v90
	v_mov_b32 v200, v18
	v_mov_b32 v18, v2
	v_mov_b32 v2, v200
	; v18 -> v60 & v60 ->  v18
	v_mov_b32 v200, v60
	v_mov_b32 v60, v2
	v_mov_b32 v2, v200
	; v60 -> v35 & v35 ->  v60
	v_mov_b32 v200, v35
	v_mov_b32 v35, v2
	v_mov_b32 v2, v200
	; v35 -> v112 & v112 ->  v35
	v_mov_b32 v200, v112
	v_mov_b32 v112, v2
	v_mov_b32 v2, v200
	; v112 -> v72 & v72 ->  v112
	v_mov_b32 v200, v72
	v_mov_b32 v72, v2
	v_mov_b32 v2, v200
	; v72 -> v94 & v94 ->  v72
	v_mov_b32 v200, v94
	v_mov_b32 v94, v2
	v_mov_b32 v2, v200
	; v94 -> v4 & v4 ->  v94
	v_mov_b32 v200, v4
	v_mov_b32 v4, v2
	v_mov_b32 v2, v200
	; v4 -> v129 & v129 ->  v4
	v_mov_b32 v200, v129
	v_mov_b32 v129, v2
	v_mov_b32 v2, v200
	; v129 -> v7 & v7 ->  v129
	v_mov_b32 v200, v7
	v_mov_b32 v7, v2
	v_mov_b32 v2, v200
	; v7 -> v127 & v127 ->  v7
	v_mov_b32 v200, v127
	v_mov_b32 v127, v2
	v_mov_b32 v2, v200
	; v127 -> v6 & v6 ->  v127
	v_mov_b32 v200, v6
	v_mov_b32 v6, v2
	v_mov_b32 v2, v200
	; v6 -> v126 & v126 ->  v6
	v_mov_b32 v200, v126
	v_mov_b32 v126, v2
	v_mov_b32 v2, v200
	;NOP 126(2) -> 2
	; v3 -> v133 & v133 ->  v3
	v_mov_b32 v200, v133
	v_mov_b32 v133, v3
	v_mov_b32 v3, v200
	; v133 -> v57 & v57 ->  v133
	v_mov_b32 v200, v57
	v_mov_b32 v57, v3
	v_mov_b32 v3, v200
	; v57 -> v47 & v47 ->  v57
	v_mov_b32 v200, v47
	v_mov_b32 v47, v3
	v_mov_b32 v3, v200
	; v47 -> v101 & v101 ->  v47
	v_mov_b32 v200, v101
	v_mov_b32 v101, v3
	v_mov_b32 v3, v200
	; v101 -> v81 & v81 ->  v101
	v_mov_b32 v200, v81
	v_mov_b32 v81, v3
	v_mov_b32 v3, v200
	; v81 -> v89 & v89 ->  v81
	v_mov_b32 v200, v89
	v_mov_b32 v89, v3
	v_mov_b32 v3, v200
	; v89 -> v31 & v31 ->  v89
	v_mov_b32 v200, v31
	v_mov_b32 v31, v3
	v_mov_b32 v3, v200
	; v31 -> v37 & v37 ->  v31
	v_mov_b32 v200, v37
	v_mov_b32 v37, v3
	v_mov_b32 v3, v200
	; v37 -> v113 & v113 ->  v37
	v_mov_b32 v200, v113
	v_mov_b32 v113, v3
	v_mov_b32 v3, v200
	; v113 -> v73 & v73 ->  v113
	v_mov_b32 v200, v73
	v_mov_b32 v73, v3
	v_mov_b32 v3, v200
	; v73 -> v95 & v95 ->  v73
	v_mov_b32 v200, v95
	v_mov_b32 v95, v3
	v_mov_b32 v3, v200
	; v95 -> v5 & v5 ->  v95
	v_mov_b32 v200, v5
	v_mov_b32 v5, v3
	v_mov_b32 v3, v200
	; v5 -> v124 & v124 ->  v5
	v_mov_b32 v200, v124
	v_mov_b32 v124, v3
	v_mov_b32 v3, v200
	;NOP 124(3) -> 3
	; v8 -> v131 & v131 ->  v8
	v_mov_b32 v200, v131
	v_mov_b32 v131, v8
	v_mov_b32 v8, v200
	; v131 -> v53 & v53 ->  v131
	v_mov_b32 v200, v53
	v_mov_b32 v53, v8
	v_mov_b32 v8, v200
	; v53 -> v49 & v49 ->  v53
	v_mov_b32 v200, v49
	v_mov_b32 v49, v8
	v_mov_b32 v8, v200
	; v49 -> v105 & v105 ->  v49
	v_mov_b32 v200, v105
	v_mov_b32 v105, v8
	v_mov_b32 v8, v200
	; v105 -> v79 & v79 ->  v105
	v_mov_b32 v200, v79
	v_mov_b32 v79, v8
	v_mov_b32 v8, v200
	; v79 -> v85 & v85 ->  v79
	v_mov_b32 v200, v85
	v_mov_b32 v85, v8
	v_mov_b32 v8, v200
	; v85 -> v33 & v33 ->  v85
	v_mov_b32 v200, v33
	v_mov_b32 v33, v8
	v_mov_b32 v8, v200
	; v33 -> v41 & v41 ->  v33
	v_mov_b32 v200, v41
	v_mov_b32 v41, v8
	v_mov_b32 v8, v200
	; v41 -> v111 & v111 ->  v41
	v_mov_b32 v200, v111
	v_mov_b32 v111, v8
	v_mov_b32 v8, v200
	; v111 -> v69 & v69 ->  v111
	v_mov_b32 v200, v69
	v_mov_b32 v69, v8
	v_mov_b32 v8, v200
	; v69 -> v97 & v97 ->  v69
	v_mov_b32 v200, v97
	v_mov_b32 v97, v8
	v_mov_b32 v8, v200
	; v97 -> v9 & v9 ->  v97
	v_mov_b32 v200, v9
	v_mov_b32 v9, v8
	v_mov_b32 v8, v200
	; v9 -> v132 & v132 ->  v9
	v_mov_b32 v200, v132
	v_mov_b32 v132, v8
	v_mov_b32 v8, v200
	; v10 -> v114 & v114 ->  v10
	v_mov_b32 v200, v114
	v_mov_b32 v114, v10
	v_mov_b32 v10, v200
	; v114 -> v12 & v12 ->  v114
	v_mov_b32 v200, v12
	v_mov_b32 v12, v10
	v_mov_b32 v10, v200
	; v12 -> v115 & v115 ->  v12
	v_mov_b32 v200, v115
	v_mov_b32 v115, v10
	v_mov_b32 v10, v200
	; v115 -> v16 & v16 ->  v115
	v_mov_b32 v200, v16
	v_mov_b32 v16, v10
	v_mov_b32 v10, v200
	; v16 -> v122 & v122 ->  v16
	v_mov_b32 v200, v122
	v_mov_b32 v122, v10
	v_mov_b32 v10, v200
	;NOP 122(10) -> 10
	; v11 -> v120 & v120 ->  v11
	v_mov_b32 v200, v120
	v_mov_b32 v120, v11
	v_mov_b32 v11, v200
	; v120 -> v14 & v14 ->  v120
	v_mov_b32 v200, v14
	v_mov_b32 v14, v11
	v_mov_b32 v11, v200
	; v14 -> v116 & v116 ->  v14
	v_mov_b32 v200, v116
	v_mov_b32 v116, v11
	v_mov_b32 v11, v200
	; v116 -> v13 & v13 ->  v116
	v_mov_b32 v200, v13
	v_mov_b32 v13, v11
	v_mov_b32 v11, v200
	; v13 -> v121 & v121 ->  v13
	v_mov_b32 v200, v121
	v_mov_b32 v121, v11
	v_mov_b32 v11, v200
	; v121 -> v15 & v15 ->  v121
	v_mov_b32 v200, v15
	v_mov_b32 v15, v11
	v_mov_b32 v11, v200
	; v15 -> v117 & v117 ->  v15
	v_mov_b32 v200, v117
	v_mov_b32 v117, v11
	v_mov_b32 v11, v200
	; v117 -> v17 & v17 ->  v117
	v_mov_b32 v200, v17
	v_mov_b32 v17, v11
	v_mov_b32 v11, v200
	; v17 -> v123 & v123 ->  v17
	v_mov_b32 v200, v123
	v_mov_b32 v123, v11
	v_mov_b32 v11, v200
	;NOP 123(11) -> 11
	; v21 -> v65 & v65 ->  v21
	v_mov_b32 v200, v65
	v_mov_b32 v65, v21
	v_mov_b32 v21, v200
	; v65 -> v25 & v25 ->  v65
	v_mov_b32 v200, v25
	v_mov_b32 v25, v21
	v_mov_b32 v21, v200
	; v25 -> v63 & v63 ->  v25
	v_mov_b32 v200, v63
	v_mov_b32 v63, v21
	v_mov_b32 v21, v200
	;NOP 63(21) -> 21
	; v22 -> v58 & v58 ->  v22
	v_mov_b32 v200, v58
	v_mov_b32 v58, v22
	v_mov_b32 v22, v200
	; v58 -> v34 & v34 ->  v58
	v_mov_b32 v200, v34
	v_mov_b32 v34, v22
	v_mov_b32 v22, v200
	; v34 -> v108 & v108 ->  v34
	v_mov_b32 v200, v108
	v_mov_b32 v108, v22
	v_mov_b32 v22, v200
	; v108 -> v67 & v67 ->  v108
	v_mov_b32 v200, v67
	v_mov_b32 v67, v22
	v_mov_b32 v22, v200
	; v67 -> v96 & v96 ->  v67
	v_mov_b32 v200, v96
	v_mov_b32 v96, v22
	v_mov_b32 v22, v200
	; v96 -> v8 & v8 ->  v96
	v_mov_b32 v200, v8
	v_mov_b32 v8, v22
	v_mov_b32 v22, v200
	; v26 -> v50 & v50 ->  v26
	v_mov_b32 v200, v50
	v_mov_b32 v50, v26
	v_mov_b32 v26, v200
	; v50 -> v44 & v44 ->  v50
	v_mov_b32 v200, v44
	v_mov_b32 v44, v26
	v_mov_b32 v26, v200
	; v44 -> v99 & v99 ->  v44
	v_mov_b32 v200, v99
	v_mov_b32 v99, v26
	v_mov_b32 v26, v200
	; v99 -> v80 & v80 ->  v99
	v_mov_b32 v200, v80
	v_mov_b32 v80, v26
	v_mov_b32 v26, v200
	; v80 -> v88 & v88 ->  v80
	v_mov_b32 v200, v88
	v_mov_b32 v88, v26
	v_mov_b32 v26, v200
	; v88 -> v30 & v30 ->  v88
	v_mov_b32 v200, v30
	v_mov_b32 v30, v26
	v_mov_b32 v26, v200
	; v30 -> v36 & v36 ->  v30
	v_mov_b32 v200, v36
	v_mov_b32 v36, v26
	v_mov_b32 v26, v200
	; v36 -> v109 & v109 ->  v36
	v_mov_b32 v200, v109
	v_mov_b32 v109, v26
	v_mov_b32 v26, v200
	; v109 -> v71 & v71 ->  v109
	v_mov_b32 v200, v71
	v_mov_b32 v71, v26
	v_mov_b32 v26, v200
	; v71 -> v91 & v91 ->  v71
	v_mov_b32 v200, v91
	v_mov_b32 v91, v26
	v_mov_b32 v26, v200
	; v91 -> v22 & v22 ->  v91
	v_mov_b32 v200, v22
	v_mov_b32 v22, v26
	v_mov_b32 v26, v200
	; v28 -> v51 & v51 ->  v28
	v_mov_b32 v200, v51
	v_mov_b32 v51, v28
	v_mov_b32 v28, v200
	; v51 -> v48 & v48 ->  v51
	v_mov_b32 v200, v48
	v_mov_b32 v48, v28
	v_mov_b32 v28, v200
	; v48 -> v104 & v104 ->  v48
	v_mov_b32 v200, v104
	v_mov_b32 v104, v28
	v_mov_b32 v28, v200
	; v104 -> v78 & v78 ->  v104
	v_mov_b32 v200, v78
	v_mov_b32 v78, v28
	v_mov_b32 v28, v200
	; v78 -> v84 & v84 ->  v78
	v_mov_b32 v200, v84
	v_mov_b32 v84, v28
	v_mov_b32 v28, v200
	; v84 -> v29 & v29 ->  v84
	v_mov_b32 v200, v29
	v_mov_b32 v29, v28
	v_mov_b32 v28, v200
	; v29 -> v55 & v55 ->  v29
	v_mov_b32 v200, v55
	v_mov_b32 v55, v28
	v_mov_b32 v28, v200
	; v55 -> v43 & v43 ->  v55
	v_mov_b32 v200, v43
	v_mov_b32 v43, v28
	v_mov_b32 v28, v200
	; v43 -> v102 & v102 ->  v43
	v_mov_b32 v200, v102
	v_mov_b32 v102, v28
	v_mov_b32 v28, v200
	; v102 -> v74 & v74 ->  v102
	v_mov_b32 v200, v74
	v_mov_b32 v74, v28
	v_mov_b32 v28, v200
	; v74 -> v82 & v82 ->  v74
	v_mov_b32 v200, v82
	v_mov_b32 v82, v28
	v_mov_b32 v28, v200
	;NOP 82(28) -> 28
	; v45 -> v103 & v103 ->  v45
	v_mov_b32 v200, v103
	v_mov_b32 v103, v45
	v_mov_b32 v45, v200
	; v103 -> v75 & v75 ->  v103
	v_mov_b32 v200, v75
	v_mov_b32 v75, v45
	v_mov_b32 v45, v200
	; v75 -> v86 & v86 ->  v75
	v_mov_b32 v200, v86
	v_mov_b32 v86, v45
	v_mov_b32 v45, v200
	; v86 -> v26 & v26 ->  v86
	v_mov_b32 v200, v26
	v_mov_b32 v26, v45
	v_mov_b32 v45, v200
	; v52 -> v45 & v45 ->  v52
	v_mov_b32 v200, v45
	v_mov_b32 v45, v52
	v_mov_b32 v52, v200
	; v214 -> v52 & v52 ->  v214
	v_mov_b32 v200, v52
	v_mov_b32 v52, v214
	v_mov_b32 v214, v200


	v_dual_mov_b32 v149, v135 :: v_dual_mov_b32 v150, v136
.LBB0_13:                               ; %Flow1143

	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_mul_f32 v139, s5, v133 :: v_dual_and_b32 v0, 0x60, v0
	v_or_b32_e32 v118, s2, v149
	v_dual_mul_f32 v140, s5, v132 :: v_dual_mul_f32 v141, s5, v131
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_nc_u32_e32 v0, s3, v0
	v_or_b32_e32 v119, v0, v150
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v144, v119, s4
	v_add_nc_u32_e32 v0, v118, v144
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v1, 31, v0
	v_lshlrev_b64 v[0:1], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_u32 v0, vcc_lo, s0, v0
	v_add_co_ci_u32_e32 v1, vcc_lo, s1, v1, vcc_lo
	global_load_b128 v[134:137], v[0:1], off
	s_waitcnt vmcnt(0)
	v_dual_mul_f32 v138, s5, v124 :: v_dual_fmac_f32 v141, s6, v137
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v138, s6, v134 :: v_dual_add_nc_u32 v145, s4, v144
	v_fmac_f32_e32 v139, s6, v135
	v_mul_f32_e32 v137, s5, v126
	v_dual_mul_f32 v135, s5, v128 :: v_dual_add_nc_u32 v142, v118, v145
	v_fmac_f32_e32 v140, s6, v136
	v_mul_f32_e32 v134, s5, v129
	v_dual_mul_f32 v128, s5, v123 :: v_dual_mul_f32 v129, s5, v122
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v143, 31, v142
	global_store_b128 v[0:1], v[138:141], off
	v_add_nc_u32_e32 v138, s4, v145
	v_mul_f32_e32 v136, s5, v127
	v_lshlrev_b64 v[124:125], 2, v[142:143]
	v_add_co_u32 v124, vcc_lo, s0, v124
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_add_co_ci_u32_e32 v125, vcc_lo, s1, v125, vcc_lo
	global_load_b128 v[130:133], v[124:125], off
	s_waitcnt vmcnt(0)
	v_dual_fmac_f32 v137, s6, v133 :: v_dual_add_nc_u32 v0, v118, v138
	v_ashrrev_i32_e32 v1, 31, v0
	v_dual_fmac_f32 v135, s6, v131 :: v_dual_fmac_f32 v136, s6, v132
	v_dual_mul_f32 v131, s5, v120 :: v_dual_fmac_f32 v134, s6, v130
	s_delay_alu instid0(VALU_DEP_3)
	v_lshlrev_b64 v[0:1], 2, v[0:1]
	global_store_b128 v[124:125], v[134:137], off
	v_add_nc_u32_e32 v134, s4, v138
	v_add_co_u32 v0, vcc_lo, s0, v0
	v_add_co_ci_u32_e32 v1, vcc_lo, s1, v1, vcc_lo
	global_load_b128 v[124:127], v[0:1], off
	s_waitcnt vmcnt(0)
	v_fmac_f32_e32 v129, s6, v125
	v_dual_mul_f32 v125, s5, v116 :: v_dual_add_nc_u32 v132, v118, v134
	v_fmac_f32_e32 v131, s6, v127
	v_dual_mul_f32 v127, s5, v114 :: v_dual_mul_f32 v130, s5, v121
	v_fmac_f32_e32 v128, s6, v124
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_ashrrev_i32_e32 v133, 31, v132
	v_mul_f32_e32 v124, s5, v117
	v_fmac_f32_e32 v130, s6, v126
	s_delay_alu instid0(VALU_DEP_3)
	v_lshlrev_b64 v[120:121], 2, v[132:133]
	global_store_b128 v[0:1], v[128:131], off
	v_add_nc_u32_e32 v1, 32, v118
	v_add_nc_u32_e32 v0, 0x60, v118
	v_add_co_u32 v132, vcc_lo, s0, v120
	v_add_co_ci_u32_e32 v133, vcc_lo, s1, v121, vcc_lo
	global_load_b128 v[120:123], v[132:133], off
	s_waitcnt vmcnt(0)
	v_fmac_f32_e32 v125, s6, v121
	v_mul_f32_e32 v121, s5, v112
	v_dual_fmac_f32 v127, s6, v123 :: v_dual_add_nc_u32 v128, v1, v144
	v_dual_mul_f32 v123, s5, v110 :: v_dual_mul_f32 v126, s5, v115
	v_fmac_f32_e32 v124, s6, v120
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_ashrrev_i32_e32 v129, 31, v128
	v_mul_f32_e32 v120, s5, v113
	v_fmac_f32_e32 v126, s6, v122
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64 v[114:115], 2, v[128:129]
	global_store_b128 v[132:133], v[124:127], off
	v_add_nc_u32_e32 v124, v1, v145
	v_add_co_u32 v128, vcc_lo, s0, v114
	v_add_co_ci_u32_e32 v129, vcc_lo, s1, v115, vcc_lo
	v_ashrrev_i32_e32 v125, 31, v124
	global_load_b128 v[114:117], v[128:129], off
	s_waitcnt vmcnt(0)
	v_fmac_f32_e32 v123, s6, v117
	v_dual_mul_f32 v117, s5, v106 :: v_dual_mul_f32 v122, s5, v111
	v_lshlrev_b64 v[110:111], 2, v[124:125]
	v_dual_fmac_f32 v120, s6, v114 :: v_dual_fmac_f32 v121, s6, v115
	v_mul_f32_e32 v114, s5, v109
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v122, s6, v116
	v_mul_f32_e32 v116, s5, v107
	v_add_co_u32 v124, vcc_lo, s0, v110
	v_add_co_ci_u32_e32 v125, vcc_lo, s1, v111, vcc_lo
	global_store_b128 v[128:129], v[120:123], off
	v_dual_mul_f32 v115, s5, v108 :: v_dual_add_nc_u32 v120, v1, v138
	global_load_b128 v[110:113], v[124:125], off
	v_ashrrev_i32_e32 v121, 31, v120
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64 v[106:107], 2, v[120:121]
	v_add_co_u32 v120, vcc_lo, s0, v106
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e32 v121, vcc_lo, s1, v107, vcc_lo
	s_waitcnt vmcnt(0)
	v_dual_fmac_f32 v114, s6, v110 :: v_dual_fmac_f32 v115, s6, v111
	v_dual_fmac_f32 v116, s6, v112 :: v_dual_fmac_f32 v117, s6, v113
	v_dual_mul_f32 v112, s5, v103 :: v_dual_mul_f32 v113, s5, v102
	v_dual_mul_f32 v110, s5, v105 :: v_dual_mul_f32 v111, s5, v104
	global_store_b128 v[124:125], v[114:117], off
	global_load_b128 v[106:109], v[120:121], off
	v_add_nc_u32_e32 v114, v1, v134
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v115, 31, v114
	v_lshlrev_b64 v[102:103], 2, v[114:115]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_u32 v114, vcc_lo, s0, v102
	v_add_co_ci_u32_e32 v115, vcc_lo, s1, v103, vcc_lo
	v_add_nc_u32_e32 v102, 64, v118
	s_waitcnt vmcnt(0)
	v_dual_fmac_f32 v110, s6, v106 :: v_dual_fmac_f32 v111, s6, v107
	v_dual_fmac_f32 v112, s6, v108 :: v_dual_fmac_f32 v113, s6, v109
	v_mul_f32_e32 v109, s5, v99
	v_dual_mul_f32 v107, s5, v101 :: v_dual_mul_f32 v108, s5, v100
	global_store_b128 v[120:121], v[110:113], off
	global_load_b128 v[103:106], v[114:115], off
	v_dual_mul_f32 v110, s5, v98 :: v_dual_add_nc_u32 v111, v102, v144
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v112, 31, v111
	v_lshlrev_b64 v[98:99], 2, v[111:112]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_u32 v111, vcc_lo, s0, v98
	v_add_co_ci_u32_e32 v112, vcc_lo, s1, v99, vcc_lo
	s_waitcnt vmcnt(0)
	v_dual_fmac_f32 v107, s6, v103 :: v_dual_fmac_f32 v108, s6, v104
	v_dual_fmac_f32 v109, s6, v105 :: v_dual_fmac_f32 v110, s6, v106
	v_dual_mul_f32 v104, s5, v96 :: v_dual_mul_f32 v105, s5, v95
	v_dual_mul_f32 v106, s5, v94 :: v_dual_mul_f32 v103, s5, v97
	global_store_b128 v[114:115], v[107:110], off
	global_load_b128 v[98:101], v[111:112], off
	s_waitcnt vmcnt(0)
	v_dual_fmac_f32 v104, s6, v99 :: v_dual_mul_f32 v99, s5, v92
	v_add_nc_u32_e32 v107, v102, v145
	v_fmac_f32_e32 v103, s6, v98
	v_dual_fmac_f32 v105, s6, v100 :: v_dual_fmac_f32 v106, s6, v101
	v_mul_f32_e32 v100, s5, v91
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_ashrrev_i32_e32 v108, 31, v107
	v_dual_mul_f32 v101, s5, v90 :: v_dual_mul_f32 v98, s5, v93
	global_store_b128 v[111:112], v[103:106], off
	v_add_nc_u32_e32 v103, v102, v138
	v_lshlrev_b64 v[94:95], 2, v[107:108]
	v_ashrrev_i32_e32 v104, 31, v103
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_co_u32 v107, vcc_lo, s0, v94
	v_add_co_ci_u32_e32 v108, vcc_lo, s1, v95, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_lshlrev_b64 v[90:91], 2, v[103:104]
	global_load_b128 v[94:97], v[107:108], off
	v_add_co_u32 v103, vcc_lo, s0, v90
	v_add_co_ci_u32_e32 v104, vcc_lo, s1, v91, vcc_lo
	s_waitcnt vmcnt(0)
	v_dual_fmac_f32 v98, s6, v94 :: v_dual_fmac_f32 v99, s6, v95
	v_dual_fmac_f32 v100, s6, v96 :: v_dual_fmac_f32 v101, s6, v97
	v_dual_mul_f32 v96, s5, v87 :: v_dual_mul_f32 v97, s5, v86
	v_dual_mul_f32 v94, s5, v89 :: v_dual_mul_f32 v95, s5, v88
	global_store_b128 v[107:108], v[98:101], off
	global_load_b128 v[90:93], v[103:104], off
	v_add_nc_u32_e32 v98, v102, v134
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v99, 31, v98
	v_lshlrev_b64 v[86:87], 2, v[98:99]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_u32 v98, vcc_lo, s0, v86
	v_add_co_ci_u32_e32 v99, vcc_lo, s1, v87, vcc_lo
	s_waitcnt vmcnt(0)
	v_dual_fmac_f32 v94, s6, v90 :: v_dual_fmac_f32 v95, s6, v91
	v_dual_fmac_f32 v96, s6, v92 :: v_dual_fmac_f32 v97, s6, v93
	v_dual_mul_f32 v92, s5, v83 :: v_dual_mul_f32 v93, s5, v82
	v_dual_mul_f32 v90, s5, v85 :: v_dual_mul_f32 v91, s5, v84
	global_store_b128 v[103:104], v[94:97], off
	global_load_b128 v[86:89], v[98:99], off
	v_add_nc_u32_e32 v94, v0, v144
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v95, 31, v94
	v_lshlrev_b64 v[82:83], 2, v[94:95]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_u32 v94, vcc_lo, s0, v82
	v_add_co_ci_u32_e32 v95, vcc_lo, s1, v83, vcc_lo
	s_waitcnt vmcnt(0)
	v_dual_fmac_f32 v90, s6, v86 :: v_dual_fmac_f32 v91, s6, v87
	v_dual_fmac_f32 v92, s6, v88 :: v_dual_fmac_f32 v93, s6, v89
	v_dual_mul_f32 v87, s5, v80 :: v_dual_mul_f32 v88, s5, v79
	v_dual_mul_f32 v89, s5, v78 :: v_dual_mul_f32 v86, s5, v81
	global_store_b128 v[98:99], v[90:93], off
	global_load_b128 v[82:85], v[94:95], off
	s_waitcnt vmcnt(0)
	v_fmac_f32_e32 v87, s6, v83
	v_dual_mul_f32 v83, s5, v76 :: v_dual_add_nc_u32 v90, v0, v145
	v_fmac_f32_e32 v86, s6, v82
	v_dual_fmac_f32 v88, s6, v84 :: v_dual_fmac_f32 v89, s6, v85
	v_mul_f32_e32 v84, s5, v75
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_ashrrev_i32_e32 v91, 31, v90
	v_dual_mul_f32 v85, s5, v74 :: v_dual_mul_f32 v82, s5, v77
	global_store_b128 v[94:95], v[86:89], off
	v_add_nc_u32_e32 v86, v0, v138
	v_lshlrev_b64 v[78:79], 2, v[90:91]
	v_ashrrev_i32_e32 v87, 31, v86
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_co_u32 v90, vcc_lo, s0, v78
	v_add_co_ci_u32_e32 v91, vcc_lo, s1, v79, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_lshlrev_b64 v[74:75], 2, v[86:87]
	global_load_b128 v[78:81], v[90:91], off
	v_add_co_u32 v86, vcc_lo, s0, v74
	v_add_co_ci_u32_e32 v87, vcc_lo, s1, v75, vcc_lo
	s_waitcnt vmcnt(0)
	v_dual_fmac_f32 v82, s6, v78 :: v_dual_fmac_f32 v83, s6, v79
	v_dual_fmac_f32 v84, s6, v80 :: v_dual_fmac_f32 v85, s6, v81
	v_dual_mul_f32 v80, s5, v71 :: v_dual_mul_f32 v81, s5, v70
	v_dual_mul_f32 v78, s5, v73 :: v_dual_mul_f32 v79, s5, v72
	global_store_b128 v[90:91], v[82:85], off
	global_load_b128 v[74:77], v[86:87], off
	v_add_nc_u32_e32 v82, v0, v134
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v83, 31, v82
	v_lshlrev_b64 v[70:71], 2, v[82:83]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_u32 v82, vcc_lo, s0, v70
	v_add_co_ci_u32_e32 v83, vcc_lo, s1, v71, vcc_lo
	s_waitcnt vmcnt(0)
	v_dual_fmac_f32 v78, s6, v74 :: v_dual_fmac_f32 v79, s6, v75
	v_dual_fmac_f32 v80, s6, v76 :: v_dual_fmac_f32 v81, s6, v77
	v_or_b32_e32 v74, 16, v119
	v_dual_mul_f32 v76, s5, v67 :: v_dual_mul_f32 v77, s5, v66
	v_mul_f32_e32 v75, s5, v68
	global_store_b128 v[86:87], v[78:81], off
	global_load_b128 v[70:73], v[82:83], off
	v_mul_lo_u32 v80, v74, s4
	v_mul_f32_e32 v74, s5, v69
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_nc_u32_e32 v78, v118, v80
	v_ashrrev_i32_e32 v79, 31, v78
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64 v[66:67], 2, v[78:79]
	v_add_co_u32 v78, vcc_lo, s0, v66
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e32 v79, vcc_lo, s1, v67, vcc_lo
	s_waitcnt vmcnt(0)
	v_dual_fmac_f32 v74, s6, v70 :: v_dual_fmac_f32 v75, s6, v71
	v_dual_fmac_f32 v76, s6, v72 :: v_dual_fmac_f32 v77, s6, v73
	v_or_b32_e32 v70, 17, v119
	v_dual_mul_f32 v72, s5, v63 :: v_dual_mul_f32 v73, s5, v62
	v_mul_f32_e32 v71, s5, v64
	global_store_b128 v[82:83], v[74:77], off
	global_load_b128 v[66:69], v[78:79], off
	v_mul_lo_u32 v76, v70, s4
	v_mul_f32_e32 v70, s5, v65
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_nc_u32_e32 v74, v118, v76
	v_ashrrev_i32_e32 v75, 31, v74
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64 v[62:63], 2, v[74:75]
	v_add_co_u32 v74, vcc_lo, s0, v62
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e32 v75, vcc_lo, s1, v63, vcc_lo
	s_waitcnt vmcnt(0)
	v_dual_fmac_f32 v70, s6, v66 :: v_dual_fmac_f32 v71, s6, v67
	v_dual_fmac_f32 v72, s6, v68 :: v_dual_fmac_f32 v73, s6, v69
	v_or_b32_e32 v66, 18, v119
	v_dual_mul_f32 v68, s5, v59 :: v_dual_mul_f32 v69, s5, v58
	v_mul_f32_e32 v67, s5, v60
	global_store_b128 v[78:79], v[70:73], off
	global_load_b128 v[62:65], v[74:75], off
	v_mul_lo_u32 v72, v66, s4
	v_mul_f32_e32 v66, s5, v61
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_nc_u32_e32 v70, v118, v72
	v_ashrrev_i32_e32 v71, 31, v70
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64 v[58:59], 2, v[70:71]
	v_add_co_u32 v70, vcc_lo, s0, v58
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e32 v71, vcc_lo, s1, v59, vcc_lo
	s_waitcnt vmcnt(0)
	v_dual_fmac_f32 v66, s6, v62 :: v_dual_fmac_f32 v67, s6, v63
	v_dual_fmac_f32 v68, s6, v64 :: v_dual_fmac_f32 v69, s6, v65
	v_or_b32_e32 v62, 19, v119
	v_dual_mul_f32 v64, s5, v55 :: v_dual_mul_f32 v65, s5, v54
	v_mul_f32_e32 v63, s5, v56
	global_store_b128 v[74:75], v[66:69], off
	global_load_b128 v[58:61], v[70:71], off
	v_mul_lo_u32 v68, v62, s4
	v_mul_f32_e32 v62, s5, v57
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_nc_u32_e32 v66, v118, v68
	v_ashrrev_i32_e32 v67, 31, v66
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64 v[54:55], 2, v[66:67]
	v_add_co_u32 v66, vcc_lo, s0, v54
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e32 v67, vcc_lo, s1, v55, vcc_lo
	s_waitcnt vmcnt(0)
	v_dual_fmac_f32 v62, s6, v58 :: v_dual_fmac_f32 v63, s6, v59
	v_dual_fmac_f32 v64, s6, v60 :: v_dual_fmac_f32 v65, s6, v61
	v_dual_mul_f32 v60, s5, v51 :: v_dual_mul_f32 v61, s5, v50
	v_dual_mul_f32 v58, s5, v53 :: v_dual_mul_f32 v59, s5, v52
	global_store_b128 v[70:71], v[62:65], off
	global_load_b128 v[54:57], v[66:67], off
	v_add_nc_u32_e32 v62, v1, v80
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v63, 31, v62
	v_lshlrev_b64 v[50:51], 2, v[62:63]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_u32 v62, vcc_lo, s0, v50
	v_add_co_ci_u32_e32 v63, vcc_lo, s1, v51, vcc_lo
	s_waitcnt vmcnt(0)
	v_dual_fmac_f32 v58, s6, v54 :: v_dual_fmac_f32 v59, s6, v55
	v_dual_fmac_f32 v60, s6, v56 :: v_dual_fmac_f32 v61, s6, v57
	v_dual_mul_f32 v56, s5, v47 :: v_dual_mul_f32 v57, s5, v46
	v_dual_mul_f32 v54, s5, v49 :: v_dual_mul_f32 v55, s5, v48
	global_store_b128 v[66:67], v[58:61], off
	global_load_b128 v[50:53], v[62:63], off
	v_add_nc_u32_e32 v58, v1, v76
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v59, 31, v58
	v_lshlrev_b64 v[46:47], 2, v[58:59]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_u32 v58, vcc_lo, s0, v46
	v_add_co_ci_u32_e32 v59, vcc_lo, s1, v47, vcc_lo
	s_waitcnt vmcnt(0)
	v_dual_fmac_f32 v54, s6, v50 :: v_dual_fmac_f32 v55, s6, v51
	v_dual_fmac_f32 v56, s6, v52 :: v_dual_fmac_f32 v57, s6, v53
	v_dual_mul_f32 v52, s5, v43 :: v_dual_mul_f32 v53, s5, v42
	v_dual_mul_f32 v50, s5, v45 :: v_dual_mul_f32 v51, s5, v44
	global_store_b128 v[62:63], v[54:57], off
	global_load_b128 v[46:49], v[58:59], off
	v_add_nc_u32_e32 v54, v1, v72
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v55, 31, v54
	v_lshlrev_b64 v[42:43], 2, v[54:55]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_u32 v54, vcc_lo, s0, v42
	v_add_co_ci_u32_e32 v55, vcc_lo, s1, v43, vcc_lo
	s_waitcnt vmcnt(0)
	v_dual_fmac_f32 v50, s6, v46 :: v_dual_fmac_f32 v51, s6, v47
	v_dual_fmac_f32 v52, s6, v48 :: v_dual_fmac_f32 v53, s6, v49
	v_dual_mul_f32 v48, s5, v39 :: v_dual_mul_f32 v49, s5, v38
	v_dual_mul_f32 v46, s5, v41 :: v_dual_mul_f32 v47, s5, v40
	global_store_b128 v[58:59], v[50:53], off
	global_load_b128 v[42:45], v[54:55], off
	v_add_nc_u32_e32 v50, v1, v68
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v51, 31, v50
	v_lshlrev_b64 v[38:39], 2, v[50:51]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_u32 v50, vcc_lo, s0, v38
	v_add_co_ci_u32_e32 v51, vcc_lo, s1, v39, vcc_lo
	s_waitcnt vmcnt(0)
	v_dual_fmac_f32 v46, s6, v42 :: v_dual_fmac_f32 v47, s6, v43
	v_dual_fmac_f32 v48, s6, v44 :: v_dual_fmac_f32 v49, s6, v45
	v_dual_mul_f32 v44, s5, v35 :: v_dual_mul_f32 v45, s5, v34
	v_dual_mul_f32 v42, s5, v37 :: v_dual_mul_f32 v43, s5, v36
	global_store_b128 v[54:55], v[46:49], off
	global_load_b128 v[38:41], v[50:51], off
	v_add_nc_u32_e32 v46, v102, v80
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v47, 31, v46
	v_lshlrev_b64 v[34:35], 2, v[46:47]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_u32 v46, vcc_lo, s0, v34
	v_add_co_ci_u32_e32 v47, vcc_lo, s1, v35, vcc_lo
	s_waitcnt vmcnt(0)
	v_dual_fmac_f32 v42, s6, v38 :: v_dual_fmac_f32 v43, s6, v39
	v_dual_fmac_f32 v44, s6, v40 :: v_dual_fmac_f32 v45, s6, v41
	v_dual_mul_f32 v40, s5, v31 :: v_dual_mul_f32 v41, s5, v30
	v_dual_mul_f32 v38, s5, v33 :: v_dual_mul_f32 v39, s5, v32
	global_store_b128 v[50:51], v[42:45], off
	global_load_b128 v[34:37], v[46:47], off
	v_add_nc_u32_e32 v42, v102, v76
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v43, 31, v42
	v_lshlrev_b64 v[30:31], 2, v[42:43]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_u32 v42, vcc_lo, s0, v30
	v_add_co_ci_u32_e32 v43, vcc_lo, s1, v31, vcc_lo
	s_waitcnt vmcnt(0)
	v_dual_fmac_f32 v38, s6, v34 :: v_dual_fmac_f32 v39, s6, v35
	v_dual_fmac_f32 v40, s6, v36 :: v_dual_fmac_f32 v41, s6, v37
	v_dual_mul_f32 v36, s5, v27 :: v_dual_mul_f32 v37, s5, v26
	v_dual_mul_f32 v34, s5, v29 :: v_dual_mul_f32 v35, s5, v28
	global_store_b128 v[46:47], v[38:41], off
	global_load_b128 v[30:33], v[42:43], off
	v_add_nc_u32_e32 v38, v102, v72
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v39, 31, v38
	v_lshlrev_b64 v[26:27], 2, v[38:39]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_u32 v38, vcc_lo, s0, v26
	v_add_co_ci_u32_e32 v39, vcc_lo, s1, v27, vcc_lo
	s_waitcnt vmcnt(0)
	v_dual_fmac_f32 v34, s6, v30 :: v_dual_fmac_f32 v35, s6, v31
	v_dual_fmac_f32 v36, s6, v32 :: v_dual_fmac_f32 v37, s6, v33
	v_dual_mul_f32 v32, s5, v23 :: v_dual_mul_f32 v33, s5, v22
	v_dual_mul_f32 v30, s5, v25 :: v_dual_mul_f32 v31, s5, v24
	global_store_b128 v[42:43], v[34:37], off
	global_load_b128 v[26:29], v[38:39], off
	v_add_nc_u32_e32 v34, v102, v68
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v35, 31, v34
	v_lshlrev_b64 v[22:23], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_u32 v34, vcc_lo, s0, v22
	v_add_co_ci_u32_e32 v35, vcc_lo, s1, v23, vcc_lo
	s_waitcnt vmcnt(0)
	v_dual_fmac_f32 v30, s6, v26 :: v_dual_fmac_f32 v31, s6, v27
	v_dual_fmac_f32 v32, s6, v28 :: v_dual_fmac_f32 v33, s6, v29
	v_dual_mul_f32 v28, s5, v19 :: v_dual_mul_f32 v29, s5, v18
	v_dual_mul_f32 v26, s5, v21 :: v_dual_mul_f32 v27, s5, v20
	global_store_b128 v[38:39], v[30:33], off
	global_load_b128 v[22:25], v[34:35], off
	v_add_nc_u32_e32 v30, v0, v80
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v31, 31, v30
	v_lshlrev_b64 v[18:19], 2, v[30:31]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_u32 v30, vcc_lo, s0, v18
	v_add_co_ci_u32_e32 v31, vcc_lo, s1, v19, vcc_lo
	s_waitcnt vmcnt(0)
	v_dual_fmac_f32 v26, s6, v22 :: v_dual_fmac_f32 v27, s6, v23
	v_dual_fmac_f32 v28, s6, v24 :: v_dual_fmac_f32 v29, s6, v25
	v_dual_mul_f32 v24, s5, v15 :: v_dual_mul_f32 v25, s5, v14
	v_dual_mul_f32 v22, s5, v17 :: v_dual_mul_f32 v23, s5, v16
	global_store_b128 v[34:35], v[26:29], off
	global_load_b128 v[18:21], v[30:31], off
	v_add_nc_u32_e32 v26, v0, v76
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v27, 31, v26
	v_lshlrev_b64 v[14:15], 2, v[26:27]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_u32 v26, vcc_lo, s0, v14
	v_add_co_ci_u32_e32 v27, vcc_lo, s1, v15, vcc_lo
	s_waitcnt vmcnt(0)
	v_dual_fmac_f32 v22, s6, v18 :: v_dual_fmac_f32 v23, s6, v19
	v_dual_fmac_f32 v24, s6, v20 :: v_dual_fmac_f32 v25, s6, v21
	v_dual_mul_f32 v20, s5, v11 :: v_dual_mul_f32 v21, s5, v10
	v_dual_mul_f32 v18, s5, v13 :: v_dual_mul_f32 v19, s5, v12
	global_store_b128 v[30:31], v[22:25], off
	global_load_b128 v[14:17], v[26:27], off
	v_add_nc_u32_e32 v22, v0, v72
	v_add_nc_u32_e32 v0, v0, v68
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_ashrrev_i32_e32 v23, 31, v22
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64 v[10:11], 2, v[22:23]
	v_lshlrev_b64 v[0:1], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_co_u32 v22, vcc_lo, s0, v10
	v_add_co_ci_u32_e32 v23, vcc_lo, s1, v11, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_add_co_u32 v0, vcc_lo, s0, v0
	v_add_co_ci_u32_e32 v1, vcc_lo, s1, v1, vcc_lo
	s_waitcnt vmcnt(0)
	v_dual_fmac_f32 v18, s6, v14 :: v_dual_fmac_f32 v19, s6, v15
	v_dual_fmac_f32 v20, s6, v16 :: v_dual_fmac_f32 v21, s6, v17
	v_dual_mul_f32 v14, s5, v9 :: v_dual_mul_f32 v15, s5, v8
	v_dual_mul_f32 v16, s5, v7 :: v_dual_mul_f32 v17, s5, v6
	global_store_b128 v[26:27], v[18:21], off
	global_load_b128 v[10:13], v[22:23], off
	s_waitcnt vmcnt(0)
	v_dual_fmac_f32 v14, s6, v10 :: v_dual_fmac_f32 v15, s6, v11
	v_dual_fmac_f32 v16, s6, v12 :: v_dual_fmac_f32 v17, s6, v13
	v_dual_mul_f32 v10, s5, v5 :: v_dual_mul_f32 v11, s5, v4
	v_dual_mul_f32 v12, s5, v3 :: v_dual_mul_f32 v13, s5, v2
	global_store_b128 v[22:23], v[14:17], off
	global_load_b128 v[6:9], v[0:1], off
	s_waitcnt vmcnt(0)
	v_dual_fmac_f32 v11, s6, v7 :: v_dual_fmac_f32 v10, s6, v6
	v_dual_fmac_f32 v12, s6, v8 :: v_dual_fmac_f32 v13, s6, v9
	global_store_b128 v[0:1], v[10:13], off
	s_nop 0
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel kernel
		.amdhsa_group_segment_fixed_size 8320
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 36
		.amdhsa_user_sgpr_count 14
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 216
		.amdhsa_next_free_sgpr 16
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 0
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 0
		.amdhsa_shared_vgpr_count 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end0:
	.size	kernel, .Lfunc_end0-kernel
                                        ; -- End function
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 6972
; NumSgprs: 18
; NumVgprs: 208
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 8320 bytes/workgroup (compile time only)
; SGPRBlocks: 2
; VGPRBlocks: 25
; NumSGPRsForWavesPerEU: 18
; NumVGPRsForWavesPerEU: 248
; Occupancy: 7
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 14
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.p2alignl 7, 3214868480
	.fill 96, 4, 3214868480
	.ident	"clang version 19.0.0git (git@github.amd.com:Compute-Mirrors/llvm-project b3dbdf4f03718d63a3292f784216fddb3e73d521)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 8320
    .kernarg_segment_align: 8
    .kernarg_segment_size: 24
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 128
    .name:           kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     60
    .sgpr_spill_count: 0
    .symbol:         kernel.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     216
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 0
amdhsa.target:   amdgcn-amd-amdhsa--gfx1100
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
