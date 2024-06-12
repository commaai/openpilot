#ifndef MSM_CAM_ISPIF_H
#define MSM_CAM_ISPIF_H

#define CSID_VERSION_V20                      0x02000011
#define CSID_VERSION_V22                      0x02001000
#define CSID_VERSION_V30                      0x30000000
#define CSID_VERSION_V3                      0x30000000

enum msm_ispif_vfe_intf {
	VFE0,
	VFE1,
	VFE_MAX
};
#define VFE0_MASK    (1 << VFE0)
#define VFE1_MASK    (1 << VFE1)

enum msm_ispif_intftype {
	PIX0,
	RDI0,
	PIX1,
	RDI1,
	RDI2,
	INTF_MAX
};
#define MAX_PARAM_ENTRIES (INTF_MAX * 2)
#define MAX_CID_CH	8

#define PIX0_MASK (1 << PIX0)
#define PIX1_MASK (1 << PIX1)
#define RDI0_MASK (1 << RDI0)
#define RDI1_MASK (1 << RDI1)
#define RDI2_MASK (1 << RDI2)


enum msm_ispif_vc {
	VC0,
	VC1,
	VC2,
	VC3,
	VC_MAX
};

enum msm_ispif_cid {
	CID0,
	CID1,
	CID2,
	CID3,
	CID4,
	CID5,
	CID6,
	CID7,
	CID8,
	CID9,
	CID10,
	CID11,
	CID12,
	CID13,
	CID14,
	CID15,
	CID_MAX
};

enum msm_ispif_csid {
	CSID0,
	CSID1,
	CSID2,
	CSID3,
	CSID_MAX
};

struct msm_ispif_params_entry {
	enum msm_ispif_vfe_intf vfe_intf;
	enum msm_ispif_intftype intftype;
	int num_cids;
	enum msm_ispif_cid cids[3];
	enum msm_ispif_csid csid;
	int crop_enable;
	uint16_t crop_start_pixel;
	uint16_t crop_end_pixel;
};

struct msm_ispif_param_data {
	uint32_t num;
	struct msm_ispif_params_entry entries[MAX_PARAM_ENTRIES];
};

struct msm_isp_info {
	uint32_t max_resolution;
	uint32_t id;
	uint32_t ver;
};

struct msm_ispif_vfe_info {
	int num_vfe;
	struct msm_isp_info info[VFE_MAX];
};

enum ispif_cfg_type_t {
	ISPIF_CLK_ENABLE,
	ISPIF_CLK_DISABLE,
	ISPIF_INIT,
	ISPIF_CFG,
	ISPIF_START_FRAME_BOUNDARY,
	ISPIF_RESTART_FRAME_BOUNDARY,
	ISPIF_STOP_FRAME_BOUNDARY,
	ISPIF_STOP_IMMEDIATELY,
	ISPIF_RELEASE,
	ISPIF_ENABLE_REG_DUMP,
	ISPIF_SET_VFE_INFO,
};

struct ispif_cfg_data {
	enum ispif_cfg_type_t cfg_type;
	union {
		int reg_dump;                        /* ISPIF_ENABLE_REG_DUMP */
		uint32_t csid_version;               /* ISPIF_INIT */
		struct msm_ispif_vfe_info vfe_info;  /* ISPIF_SET_VFE_INFO */
		struct msm_ispif_param_data params;  /* CFG, START, STOP */
	};
};

#define VIDIOC_MSM_ISPIF_CFG \
	_IOWR('V', BASE_VIDIOC_PRIVATE, struct ispif_cfg_data)

#endif /* MSM_CAM_ISPIF_H */
