#ifndef _UAPI_MSM_ION_H
#define _UAPI_MSM_ION_H

#include "ion.h"

enum msm_ion_heap_types {
	ION_HEAP_TYPE_MSM_START = ION_HEAP_TYPE_CUSTOM + 1,
	ION_HEAP_TYPE_SECURE_DMA = ION_HEAP_TYPE_MSM_START,
	ION_HEAP_TYPE_SYSTEM_SECURE,
	ION_HEAP_TYPE_HYP_CMA,
	/*
	 * if you add a heap type here you should also add it to
	 * heap_types_info[] in msm_ion.c
	 */
};

/**
 * These are the only ids that should be used for Ion heap ids.
 * The ids listed are the order in which allocation will be attempted
 * if specified. Don't swap the order of heap ids unless you know what
 * you are doing!
 * Id's are spaced by purpose to allow new Id's to be inserted in-between (for
 * possible fallbacks)
 */

enum ion_heap_ids {
	INVALID_HEAP_ID = -1,
	ION_CP_MM_HEAP_ID = 8,
	ION_SECURE_HEAP_ID = 9,
	ION_SECURE_DISPLAY_HEAP_ID = 10,
	ION_CP_MFC_HEAP_ID = 12,
	ION_CP_WB_HEAP_ID = 16, /* 8660 only */
	ION_CAMERA_HEAP_ID = 20, /* 8660 only */
	ION_SYSTEM_CONTIG_HEAP_ID = 21,
	ION_ADSP_HEAP_ID = 22,
	ION_PIL1_HEAP_ID = 23, /* Currently used for other PIL images */
	ION_SF_HEAP_ID = 24,
	ION_SYSTEM_HEAP_ID = 25,
	ION_PIL2_HEAP_ID = 26, /* Currently used for modem firmware images */
	ION_QSECOM_HEAP_ID = 27,
	ION_AUDIO_HEAP_ID = 28,

	ION_MM_FIRMWARE_HEAP_ID = 29,

	ION_HEAP_ID_RESERVED = 31 /** Bit reserved for ION_FLAG_SECURE flag */
};

/*
 * The IOMMU heap is deprecated! Here are some aliases for backwards
 * compatibility:
 */
#define ION_IOMMU_HEAP_ID ION_SYSTEM_HEAP_ID
#define ION_HEAP_TYPE_IOMMU ION_HEAP_TYPE_SYSTEM

enum ion_fixed_position {
	NOT_FIXED,
	FIXED_LOW,
	FIXED_MIDDLE,
	FIXED_HIGH,
};

enum cp_mem_usage {
	VIDEO_BITSTREAM = 0x1,
	VIDEO_PIXEL = 0x2,
	VIDEO_NONPIXEL = 0x3,
	DISPLAY_SECURE_CP_USAGE = 0x4,
	CAMERA_SECURE_CP_USAGE = 0x5,
	MAX_USAGE = 0x6,
	UNKNOWN = 0x7FFFFFFF,
};

/**
 * Flags to be used when allocating from the secure heap for
 * content protection
 */
#define ION_FLAG_CP_TOUCH (1 << 17)
#define ION_FLAG_CP_BITSTREAM (1 << 18)
#define ION_FLAG_CP_PIXEL  (1 << 19)
#define ION_FLAG_CP_NON_PIXEL (1 << 20)
#define ION_FLAG_CP_CAMERA (1 << 21)
#define ION_FLAG_CP_HLOS (1 << 22)
#define ION_FLAG_CP_HLOS_FREE (1 << 23)
#define ION_FLAG_CP_SEC_DISPLAY (1 << 25)
#define ION_FLAG_CP_APP (1 << 26)

/**
 * Flag to allow non continguous allocation of memory from secure
 * heap
 */
#define ION_FLAG_ALLOW_NON_CONTIG (1 << 24)

/**
 * Flag to use when allocating to indicate that a heap is secure.
 */
#define ION_FLAG_SECURE (1 << ION_HEAP_ID_RESERVED)

/**
 * Flag for clients to force contiguous memort allocation
 *
 * Use of this flag is carefully monitored!
 */
#define ION_FLAG_FORCE_CONTIGUOUS (1 << 30)

/*
 * Used in conjunction with heap which pool memory to force an allocation
 * to come from the page allocator directly instead of from the pool allocation
 */
#define ION_FLAG_POOL_FORCE_ALLOC (1 << 16)


#define ION_FLAG_POOL_PREFETCH (1 << 27)

/**
* Deprecated! Please use the corresponding ION_FLAG_*
*/
#define ION_SECURE ION_FLAG_SECURE
#define ION_FORCE_CONTIGUOUS ION_FLAG_FORCE_CONTIGUOUS

/**
 * Macro should be used with ion_heap_ids defined above.
 */
#define ION_HEAP(bit) (1 << (bit))

#define ION_ADSP_HEAP_NAME	"adsp"
#define ION_SYSTEM_HEAP_NAME	"system"
#define ION_VMALLOC_HEAP_NAME	ION_SYSTEM_HEAP_NAME
#define ION_KMALLOC_HEAP_NAME	"kmalloc"
#define ION_AUDIO_HEAP_NAME	"audio"
#define ION_SF_HEAP_NAME	"sf"
#define ION_MM_HEAP_NAME	"mm"
#define ION_CAMERA_HEAP_NAME	"camera_preview"
#define ION_IOMMU_HEAP_NAME	"iommu"
#define ION_MFC_HEAP_NAME	"mfc"
#define ION_WB_HEAP_NAME	"wb"
#define ION_MM_FIRMWARE_HEAP_NAME	"mm_fw"
#define ION_PIL1_HEAP_NAME  "pil_1"
#define ION_PIL2_HEAP_NAME  "pil_2"
#define ION_QSECOM_HEAP_NAME	"qsecom"
#define ION_SECURE_HEAP_NAME	"secure_heap"
#define ION_SECURE_DISPLAY_HEAP_NAME "secure_display"

#define ION_SET_CACHED(__cache)		(__cache | ION_FLAG_CACHED)
#define ION_SET_UNCACHED(__cache)	(__cache & ~ION_FLAG_CACHED)

#define ION_IS_CACHED(__flags)	((__flags) & ION_FLAG_CACHED)

/* struct ion_flush_data - data passed to ion for flushing caches
 *
 * @handle:	handle with data to flush
 * @fd:		fd to flush
 * @vaddr:	userspace virtual address mapped with mmap
 * @offset:	offset into the handle to flush
 * @length:	length of handle to flush
 *
 * Performs cache operations on the handle. If p is the start address
 * of the handle, p + offset through p + offset + length will have
 * the cache operations performed
 */
struct ion_flush_data {
	ion_user_handle_t handle;
	int fd;
	void *vaddr;
	unsigned int offset;
	unsigned int length;
};

struct ion_prefetch_regions {
	unsigned int vmid;
	size_t *sizes;
	unsigned int nr_sizes;
};

struct ion_prefetch_data {
	int heap_id;
	unsigned long len;
	/* Is unsigned long bad? 32bit compiler vs 64 bit compiler*/
	struct ion_prefetch_regions *regions;
	unsigned int nr_regions;
};

#define ION_IOC_MSM_MAGIC 'M'

/**
 * DOC: ION_IOC_CLEAN_CACHES - clean the caches
 *
 * Clean the caches of the handle specified.
 */
#define ION_IOC_CLEAN_CACHES	_IOWR(ION_IOC_MSM_MAGIC, 0, \
						struct ion_flush_data)
/**
 * DOC: ION_IOC_INV_CACHES - invalidate the caches
 *
 * Invalidate the caches of the handle specified.
 */
#define ION_IOC_INV_CACHES	_IOWR(ION_IOC_MSM_MAGIC, 1, \
						struct ion_flush_data)
/**
 * DOC: ION_IOC_CLEAN_INV_CACHES - clean and invalidate the caches
 *
 * Clean and invalidate the caches of the handle specified.
 */
#define ION_IOC_CLEAN_INV_CACHES	_IOWR(ION_IOC_MSM_MAGIC, 2, \
						struct ion_flush_data)

#define ION_IOC_PREFETCH		_IOWR(ION_IOC_MSM_MAGIC, 3, \
						struct ion_prefetch_data)

#define ION_IOC_DRAIN			_IOWR(ION_IOC_MSM_MAGIC, 4, \
						struct ion_prefetch_data)

#endif
