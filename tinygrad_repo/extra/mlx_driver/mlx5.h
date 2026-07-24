// MLX5 autogen header — kernel struct layouts and constants
typedef unsigned char __u8;
typedef unsigned short __be16;
typedef unsigned int __be32;
typedef unsigned long long __be64;

// --- device.h structs ---

struct mlx5_cmd_layout {
    __u8    type;
    __u8    rsvd0[3];
    __be32  inlen;
    __be64  in_ptr;
    __be32  in[4];
    __be32  out[4];
    __be64  out_ptr;
    __be32  outlen;
    __u8    token;
    __u8    sig;
    __u8    rsvd1;
    __u8    status_own;
};

struct mlx5_cmd_prot_block {
    __u8    data[512];
    __u8    rsvd0[48];
    __be64  next;
    __be32  block_num;
    __u8    rsvd1;
    __u8    token;
    __u8    ctrl_sig;
    __u8    sig;
};

struct mlx5_init_seg {
    __be32  fw_rev;
    __be32  cmdif_rev_fw_sub;
    __be32  rsvd0[2];
    __be32  cmdq_addr_h;
    __be32  cmdq_addr_l_sz;
    __be32  cmd_dbell;
    __be32  rsvd1[120];
    __be32  initializing;
};

// --- Command opcodes (mlx5_ifc.h) ---
#define MLX5_CMD_OP_QUERY_HCA_CAP            0x100
#define MLX5_CMD_OP_QUERY_ADAPTER            0x101
#define MLX5_CMD_OP_INIT_HCA                 0x102
#define MLX5_CMD_OP_TEARDOWN_HCA             0x103
#define MLX5_CMD_OP_ENABLE_HCA               0x104
#define MLX5_CMD_OP_DISABLE_HCA              0x105
#define MLX5_CMD_OP_QUERY_PAGES              0x107
#define MLX5_CMD_OP_MANAGE_PAGES             0x108
#define MLX5_CMD_OP_SET_HCA_CAP              0x109
#define MLX5_CMD_OP_QUERY_ISSI               0x10a
#define MLX5_CMD_OP_SET_ISSI                 0x10b
#define MLX5_CMD_OP_SET_DRIVER_VERSION       0x10d
#define MLX5_CMD_OP_CREATE_MKEY              0x200
#define MLX5_CMD_OP_QUERY_SPECIAL_CONTEXTS   0x203
#define MLX5_CMD_OP_CREATE_EQ                0x301
#define MLX5_CMD_OP_DESTROY_EQ               0x302
#define MLX5_CMD_OP_CREATE_CQ                0x400
#define MLX5_CMD_OP_DESTROY_CQ               0x401
#define MLX5_CMD_OP_CREATE_QP                0x500
#define MLX5_CMD_OP_DESTROY_QP               0x501
#define MLX5_CMD_OP_RST2INIT_QP              0x502
#define MLX5_CMD_OP_INIT2RTR_QP              0x503
#define MLX5_CMD_OP_RTR2RTS_QP              0x504
#define MLX5_CMD_OP_QUERY_NIC_VPORT_CONTEXT  0x754
#define MLX5_CMD_OP_MODIFY_NIC_VPORT_CONTEXT 0x755
#define MLX5_CMD_OP_SET_ROCE_ADDRESS         0x761
#define MLX5_CMD_OP_ALLOC_PD                 0x800
#define MLX5_CMD_OP_ALLOC_UAR                0x802
#define MLX5_CMD_OP_ACCESS_REG               0x805
#define MLX5_CMD_OP_ALLOC_TRANSPORT_DOMAIN   0x816

// --- Command status (device.h) ---
#define MLX5_CMD_STAT_OK                     0x0
#define MLX5_CMD_STAT_INT_ERR                0x1
#define MLX5_CMD_STAT_BAD_OP_ERR             0x2
#define MLX5_CMD_STAT_BAD_PARAM_ERR          0x3
#define MLX5_CMD_STAT_BAD_SYS_STATE_ERR      0x4
#define MLX5_CMD_STAT_BAD_RES_ERR            0x5
#define MLX5_CMD_STAT_RES_BUSY               0x6
#define MLX5_CMD_STAT_LIM_ERR                0x8
#define MLX5_CMD_STAT_BAD_RES_STATE_ERR      0x9
#define MLX5_CMD_STAT_NO_RES_ERR             0xf
#define MLX5_CMD_STAT_BAD_INP_LEN_ERR        0x50
#define MLX5_CMD_STAT_BAD_OUTP_LEN_ERR       0x51

// --- HCA cap types ---
#define MLX5_CAP_GENERAL          0x0
#define MLX5_CAP_ODP              0x2
#define MLX5_CAP_ATOMIC           0x3
#define MLX5_CAP_ROCE             0x4
#define HCA_CAP_OPMOD_GET_MAX     0
#define HCA_CAP_OPMOD_GET_CUR     1

// --- Pages ---
#define MLX5_PAGES_GIVE           1
#define MLX5_PAGES_TAKE           2
#define MLX5_BOOT_PAGES           1
#define MLX5_INIT_PAGES           2

// --- Registers ---
#define MLX5_REG_HOST_ENDIANNESS  0x7004
#define MLX5_REG_DTOR             0xC00E

// --- Misc ---
#define MLX5_PCI_CMD_XPORT        0x07
#define MLX5_CMD_DATA_BLOCK_SIZE  512
#define CMD_OWNER_HW              0x01

// --- IFC cmd_hca_cap bit offsets ---
#define CAP_GEN_ABS_NATIVE_PORT_NUM                       0x007
#define CAP_GEN_HCA_CAP_2                                 0x020
#define CAP_GEN_EVENT_ON_VHCA_STATE_ALLOCATED              0x023
#define CAP_GEN_EVENT_ON_VHCA_STATE_ACTIVE                 0x024
#define CAP_GEN_EVENT_ON_VHCA_STATE_IN_USE                 0x025
#define CAP_GEN_EVENT_ON_VHCA_STATE_TEARDOWN_REQUEST       0x026
#define CAP_GEN_LOG_MAX_QP                                 0x09B
#define CAP_GEN_LOG_MAX_CQ                                 0x0DB
#define CAP_GEN_RELEASE_ALL_PAGES                          0x145
#define CAP_GEN_CACHE_LINE_128BYTE                         0x164
#define CAP_GEN_NUM_PORTS                                  0x1B8
#define CAP_GEN_PKEY_TABLE_SIZE                            0x190
#define CAP_GEN_PCI_SYNC_FOR_FW_UPDATE_EVENT               0x1F1
#define CAP_GEN_CMDIF_CHECKSUM                             0x210
#define CAP_GEN_DCT                                        0x21A
#define CAP_GEN_ROCE                                       0x21D
#define CAP_GEN_ATOMIC                                     0x21E
#define CAP_GEN_ODP                                        0x227
#define CAP_GEN_MKEY_BY_NAME                               0x266
#define CAP_GEN_LOG_MAX_PD                                 0x32B
#define CAP_GEN_PCIE_RESET_USING_HOTRESET                  0x335
#define CAP_GEN_PCI_SYNC_FOR_FW_UPDATE_WITH_DRIVER_UNLOAD  0x336
#define CAP_GEN_VHCA_STATE                                 0x3EA
#define CAP_GEN_ROCE_RW_SUPPORTED                          0x3A1
#define CAP_GEN_LOG_MAX_CURRENT_UC_LIST                    0x3FB
#define CAP_GEN_LOG_UAR_PAGE_SZ                            0x490
#define CAP_GEN_NUM_VHCA_PORTS                             0x610
#define CAP_GEN_SW_OWNER_ID                                0x61E
#define CAP_GEN_NUM_TOTAL_DYNAMIC_VF_MSIX                  0x708
