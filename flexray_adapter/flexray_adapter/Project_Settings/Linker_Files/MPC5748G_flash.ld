/*
** ###################################################################
**     Processor:           MPC5748G with 256 KB SRAM
**     Compiler:            GNU C Compiler
**
**     Abstract:
**         Linker file for the GNU C Compiler
**
**     Copyright 2017 NXP
**     All rights reserved.
**
**     THIS SOFTWARE IS PROVIDED BY NXP "AS IS" AND ANY EXPRESSED OR
**     IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
**     OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
**     IN NO EVENT SHALL NXP OR ITS CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
**     INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
**     (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
**     SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
**     HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
**     STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
**     IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
**     THE POSSIBILITY OF SUCH DAMAGE.
**
**     http:                 www.nxp.com
**
** ###################################################################
*/

/* Entry Point */
ENTRY(_start)

/* define heap and stack size */
__HEAP_SIZE = DEFINED(__heap_size__)  ? __heap_size__  : 0x00000000;
__STACK_SIZE = DEFINED(__stack_size__) ? __stack_size__ : 0x00001000;

SRAM_SIZE =  512K;
/* Define SRAM Base Address */
SRAM_BASE_ADDR = 0x40000000;

MEMORY
{

    flash_rchw : org = 0x00FA0000,          len = 0x4
    cpu0_reset_vec : org = 0x00FA0000+0x10, len = 0x4
    cpu1_reset_vec : org = 0x00FA0000+0x14, len = 0x4
    cpu2_reset_vec : org = 0x00FA0000+0x04, len = 0x4

    m_text : org = 0x01000000,              len = 5568K
    m_vectors_ram : org = 0x40000000,       len = 0xC00
    m_data : org = 0x40000000+0xC00,        len = 512K-0xC00
}


SECTIONS
{
    .rchw   :
    {
        KEEP(*(.rchw))
    } > flash_rchw

    .cpu0_reset_vector  :
    {
        KEEP(*(.cpu0_reset_vector))
    } > cpu0_reset_vec

    .cpu1_reset_vector  :
    {
        KEEP(*(.cpu1_reset_vector))
    } > cpu1_reset_vec

    .cpu2_reset_vector  :
    {
        KEEP(*(.cpu2_reset_vector))
    } > cpu2_reset_vec

    /* Note: if you move the 'startup' section shall modify the RCHW2_2 value for the corresponding core in the flashrchw.c file. */
    .startup : ALIGN(0x400)
    {
        __start = . ;
        *(.startup)
    } > m_text

    .core_exceptions_table   : ALIGN(4096)
    {
        __IVPR_VALUE = . ;
        *(.core_exceptions_table)
    } > m_text

    .intc_vector_table   : ALIGN(4096)
    {
        __VECTOR_TABLE = .;
        . = ALIGN(4);
        __interrupts_start__ = .;
        KEEP(*(.intc_vector_table))     /* Startup code */
        __interrupts_end__ = .;
    } > m_text
    __RAM_VECTOR_TABLE_SIZE = 0xC00;

    .text :
    {
        *(.text.startup)
        *(.text)
        *(.text.*)
        KEEP(*(.init))
        KEEP(*(.fini))
        . = ALIGN(16);
    } > m_text
    .got2 :
    {
        *(.got2*)
    } > m_text
/* migration to version v1.2
   define section PREINIT_ARRAY */

.preinit_array     :
  {
    PROVIDE_HIDDEN (__preinit_array_start = .);
    KEEP (*(.preinit_array))
    PROVIDE_HIDDEN (__preinit_array_end = .);
  } > m_text


/* end section PREINIT_ARRAY */
/*  migration to version v1.2
    define section INIT_ARRAY*/

  .init_array     :
  {
    PROVIDE_HIDDEN (__init_array_start = .);
    KEEP (*(SORT(.init_array.*)))
    KEEP (*(.init_array ))
    PROVIDE_HIDDEN (__init_array_end = .);
  } > m_text


/* end section INIT_ARRAY */

/* migration to version v1.2
   define section DTORS */

  .dtors          :
  {
    KEEP (*crtbegin.o(.dtors))
    KEEP (*crtbegin?.o(.dtors))
    KEEP (*(EXCLUDE_FILE (*crtend.o *crtend?.o ) .dtors))
    KEEP (*(SORT(.dtors.*)))
    KEEP (*(.dtors))
  } > m_text

/* end section DTORS */
/* migration to version v1.2
   define section CTORS */

  .ctors          :
  {
    /* gcc uses crtbegin.o to find the start of
       the constructors, so we make sure it is
       first.  Because this is a wildcard, it
       doesn't matter if the user does not
       actually link against crtbegin.o; the
       linker won't look for a file to match a
       wildcard.  The wildcard also means that it
       doesn't matter which directory crtbegin.o
       is in.  */
    KEEP (*crtbegin.o(.ctors))
    KEEP (*crtbegin?.o(.ctors))
    /* We don't want to include the .ctor section from
       the crtend.o file until after the sorted ctors.
       The .ctor section from the crtend file contains the
       end of ctors marker and it must be last */
    KEEP (*(EXCLUDE_FILE (*crtend.o *crtend?.o ) .ctors))
    KEEP (*(SORT(.ctors.*)))
    KEEP (*(.ctors))
  } > m_text

/* end section CTORS */

/* migration to version v1.2
   define section FINI_ARRAY */


  .fini_array     :
  {
    PROVIDE_HIDDEN (__fini_array_start = .);
    KEEP (*(SORT(.fini_array.*)))
    KEEP (*(.fini_array ))
    PROVIDE_HIDDEN (__fini_array_end = .);
  } > m_text



/* end  section FINI_ARRAY */

    .rodata :
    {
        *(.rodata)
        *(.rodata.*)
    } > m_text

    .eh_frame_hdr : { *(.eh_frame_hdr) } > m_text
    .eh_frame     : { KEEP (*(.eh_frame)) } > m_text
    __TEXT_END = .;               /* Define a global symbol at end of code. */
    __CUSTOM_ROM = .;

    .interrupts_ram :
    {
        __VECTOR_RAM = .;
        *(.m_interrupts_ram)
        . += __RAM_VECTOR_TABLE_SIZE;
    } > m_vectors_ram
    .customSectionBlock : AT (__CUSTOM_ROM)
    {
        . = ALIGN(4);
        __customSectionStart = .;
        __customSection_start__ = .;
        KEEP(*(.customSection))  /* Keep section even if not referenced. */
        __customSection_end__ = .;
        __customSectionEnd = .;
     } > m_data
     __CUSTOM_END = __CUSTOM_ROM + (__customSection_end__ - __customSection_start__);
     __DATA_ROM = __CUSTOM_END;               /* Symbol is used by startup for data initialization. */
    .data   :AT (__DATA_ROM)
    {
        . = ALIGN(4);
        __DATA_RAM = .;
        __data_start__ = .;       /* Create a global symbol at data start. */
        *(.data)
        *(.data.*)
        *(.sdata)
        *(.sdata.*)
        *(.sdata2)
        *(.sdata2.*)
        . = ALIGN(4);
        __data_end__ = .;         /* Define a global symbol at data end. */
    }  > m_data

    __DATA_END = __DATA_ROM + (__data_end__ - __data_start__);
    __CODE_ROM = __DATA_END;      /* Symbol is used by code initialization. */

    .code : AT(__CODE_ROM)
    {
        . = ALIGN(4);
        __CODE_RAM = .;
        __code_start__ = .;       /* Create a global symbol at code start. */
        __code_ram_start__ = .;
        KEEP(*(.code_ram))            /* Custom section for storing code in RAM */
        __code_ram_end__ = .;
        . = ALIGN(4);
        __code_end__ = .;         /* Define a global symbol at code end. */
    } > m_data
    __CODE_END = __CODE_ROM + (__code_end__ - __code_start__);

    .bss   (NOLOAD)  :
    {
        . = ALIGN(4);
        __BSS_START = .;
        __bss_start__ = .;
        *(.sbss)
        *(.sbss.*)
        *(.bss)
        *(.bss.*)
        *(.sbss2)
        *(.sbss2.*)
        *(COMMON)
        __bss_end__ = .;
        . = ALIGN(4);
        __BSS_END = .;
    } > m_data

    .stack (NOLOAD) : ALIGN(16)
    {
        __HEAP = . ;
        PROVIDE (_end = . );
        PROVIDE (end = . );
        . += __HEAP_SIZE ;
        __HEAP_END = . ;
        _stack_end = . ;
        . +=  __STACK_SIZE ;
        _stack_addr = . ;
        __SP_INIT = . ;
        . += 4;
    } > m_data

/*-------- LABELS USED IN CODE -------------------------------*/

/* Labels Used for Initialising SRAM ECC */
__SRAM_SIZE = SRAM_SIZE;
__SRAM_BASE_ADDR = SRAM_BASE_ADDR;

__BSS_SIZE    = __BSS_END - __BSS_START;

}

