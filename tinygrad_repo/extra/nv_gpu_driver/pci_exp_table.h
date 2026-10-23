/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#ifndef PCIEXPTBL_H
#define PCIEXPTBL_H

#define NV_BCRT_HASH_INFO_BASE_CODE_TYPE_VBIOS_BASE   0x00
#define NV_BCRT_HASH_INFO_BASE_CODE_TYPE_VBIOS_EXT    0xE0

//
// The VBIOS object comes from walking the PCI expansion code block
// The following structure holds the expansion code format.
//
#define PCI_EXP_ROM_SIGNATURE     0xaa55
#define PCI_EXP_ROM_SIGNATURE_NV  0x4e56 // "VN" in word format
#define PCI_EXP_ROM_SIGNATURE_NV2 0xbb77
#define IS_VALID_PCI_ROM_SIG(sig) ((sig == PCI_EXP_ROM_SIGNATURE) ||    \
                                   (sig == PCI_EXP_ROM_SIGNATURE_NV) || \
                                   (sig == PCI_EXP_ROM_SIGNATURE_NV2))

#define OFFSETOF_PCI_EXP_ROM_SIG                    0x0
#define OFFSETOF_PCI_EXP_ROM_NBSI_DATA_OFFSET       0x16
#define OFFSETOF_PCI_EXP_ROM_PCI_DATA_STRUCT_PTR    0x18

#pragma pack(1)
typedef struct _PCI_EXP_ROM_STANDARD
{
    NvU16       sig;                //  00h: ROM Signature 0xaa55
    NvU8        reserved [0x16];    //  02h: Reserved (processor architecture unique data)
    NvU16       pciDataStrucPtr;    //  18h: Pointer to PCI Data Structure
    NvU32       sizeOfBlock;        //  1Ah: <NBSI-specific appendage>
} PCI_EXP_ROM_STANDARD, *PPCI_EXP_ROM_STANDARD;
#pragma pack()

#pragma pack(1)
typedef struct _PCI_EXP_ROM_NBSI
{
    NvU16       sig;                //  00h: ROM Signature 0xaa55
    NvU8        reserved [0x14];    //  02h: Reserved (processor architecture unique data)
    NvU16       nbsiDataOffset;     //  16h: Offset from header to NBSI image
    NvU16       pciDataStrucPtr;    //  18h: Pointer to PCI Data Structure
    NvU32       sizeOfBlock;        //  1Ah: <NBSI-specific appendage>
} PCI_EXP_ROM_NBSI, *PPCI_EXP_ROM_NBSI;
#pragma pack()

typedef union _PCI_EXP_ROM {
    PCI_EXP_ROM_STANDARD standard;
    PCI_EXP_ROM_NBSI nbsi;
} PCI_EXP_ROM, *PPCI_EXP_ROM;

#define PCI_DATA_STRUCT_SIGNATURE     0x52494350 // "PCIR" in dword format
#define PCI_DATA_STRUCT_SIGNATURE_NV  0x5344504E // "NPDS" in dword format
#define PCI_DATA_STRUCT_SIGNATURE_NV2 0x53494752 // "RGIS" in dword format
#define IS_VALID_PCI_DATA_SIG(sig) ((sig == PCI_DATA_STRUCT_SIGNATURE) ||    \
                                    (sig == PCI_DATA_STRUCT_SIGNATURE_NV) || \
                                    (sig == PCI_DATA_STRUCT_SIGNATURE_NV2))

#define PCI_LAST_IMAGE NVBIT(7)
#define PCI_ROM_IMAGE_BLOCK_SIZE 512U

#define OFFSETOF_PCI_DATA_STRUCT_SIG        0x0
#define OFFSETOF_PCI_DATA_STRUCT_VENDOR_ID  0x4
#define OFFSETOF_PCI_DATA_STRUCT_LEN        0xa
#define OFFSETOF_PCI_DATA_STRUCT_CLASS_CODE 0xd
#define OFFSETOF_PCI_DATA_STRUCT_CODE_TYPE  0x14
#define OFFSETOF_PCI_DATA_STRUCT_IMAGE_LEN  0x10
#define OFFSETOF_PCI_DATA_STRUCT_LAST_IMAGE 0x15

#pragma pack(1)
typedef struct _PCI_DATA_STRUCT
{
    NvU32       sig;                //  00h: Signature, the string "PCIR" or NVIDIA's alternate "NPDS"
    NvU16       vendorID;           //  04h: Vendor Identification
    NvU16       deviceID;           //  06h: Device Identification
    NvU16       deviceListPtr;      //  08h: Device List Pointer
    NvU16       pciDataStructLen;   //  0Ah: PCI Data Structure Length
    NvU8        pciDataStructRev;   //  0Ch: PCI Data Structure Revision
    NvU8        classCode[3];       //  0Dh: Class Code
    NvU16       imageLen;           //  10h: Image Length (units of 512 bytes)
    NvU16       vendorRomRev;       //  12h: Revision Level of the Vendor's ROM
    NvU8        codeType;           //  14h: holds NBSI_OBJ_CODE_TYPE (0x70) and others
    NvU8        lastImage;          //  15h: Last Image Indicator: bit7=1 is lastImage
    NvU16       maxRunTimeImageLen; //  16h: Maximum Run-time Image Length (units of 512 bytes)
} PCI_DATA_STRUCT, *PPCI_DATA_STRUCT;
#pragma pack()

#define NV_PCI_DATA_EXT_SIG 0x4544504E // "NPDE" in dword format
#define NV_PCI_DATA_EXT_REV_10 0x100      // 1.0
#define NV_PCI_DATA_EXT_REV_11 0x101      // 1.1

#define OFFSETOF_PCI_DATA_EXT_STRUCT_SIG            0x0
#define OFFSETOF_PCI_DATA_EXT_STRUCT_LEN            0x6
#define OFFSETOF_PCI_DATA_EXT_STRUCT_REV            0x4
#define OFFSETOF_PCI_DATA_EXT_STRUCT_SUBIMAGE_LEN   0x8
#define OFFSETOF_PCI_DATA_EXT_STRUCT_LAST_IMAGE     0xa
#define OFFSETOF_PCI_DATA_EXT_STRUCT_FLAGS          0xb

#define PCI_DATA_EXT_STRUCT_FLAGS_CHECKSUM_DISABLED 0x04

#pragma pack(1)
typedef struct _NV_PCI_DATA_EXT_STRUCT
{
    NvU32   signature;          //  00h: Signature, the string "NPDE"
    NvU16   nvPciDataExtRev;    //  04h: NVIDIA PCI Data Extension Revision
    NvU16   nvPciDataExtLen;    //  06h: NVIDIA PCI Data Extension Length
    NvU16   subimageLen;        //  08h: Sub-image Length
    NvU8    privLastImage;      //  0Ah: Private Last Image Indicator
    NvU8    flags;              //  0Bh: Private images enabled if bit0=1
} NV_PCI_DATA_EXT_STRUCT, *PNV_PCI_DATA_EXT_STRUCT;
#pragma pack()

#endif // PCIEXPTBL_H


