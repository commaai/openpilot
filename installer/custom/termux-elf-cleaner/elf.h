#ifndef ELF_H_INCLUDED
#define ELF_H_INCLUDED

#include <stdint.h>

/* Type for a 16-bit quantity.  */
typedef uint16_t Elf32_Half;
typedef uint16_t Elf64_Half;

/* Types for signed and unsigned 32-bit quantities.  */
typedef uint32_t Elf32_Word;
typedef int32_t  Elf32_Sword;
typedef uint32_t Elf64_Word;
typedef int32_t  Elf64_Sword;

/* Types for signed and unsigned 64-bit quantities.  */
typedef uint64_t Elf32_Xword;
typedef int64_t  Elf32_Sxword;
typedef uint64_t Elf64_Xword;
typedef int64_t  Elf64_Sxword;

/* Type of addresses.  */
typedef uint32_t Elf32_Addr;
typedef uint64_t Elf64_Addr;

/* Type of file offsets.  */
typedef uint32_t Elf32_Off;
typedef uint64_t Elf64_Off;

/* Type for section indices, which are 16-bit quantities.  */
typedef uint16_t Elf32_Section;
typedef uint16_t Elf64_Section;

/* Type for version symbol information.  */
typedef Elf32_Half Elf32_Versym;
typedef Elf64_Half Elf64_Versym;


/* The ELF file header.  This appears at the start of every ELF file.  */
typedef struct {
	unsigned char e_ident[16];     /* Magic number and other info */
	Elf32_Half    e_type;                 /* Object file type */
	Elf32_Half    e_machine;              /* Architecture */
	Elf32_Word    e_version;              /* Object file version */
	Elf32_Addr    e_entry;                /* Entry point virtual address */
	Elf32_Off     e_phoff;                /* Program header table (usually follows elf header directly) file offset */
	Elf32_Off     e_shoff;                /* Section header table (at end of file) file offset */
	Elf32_Word    e_flags;                /* Processor-specific flags */
	Elf32_Half    e_ehsize;               /* ELF header size in bytes */
	Elf32_Half    e_phentsize;            /* Program header table entry size */
	Elf32_Half    e_phnum;                /* Program header table entry count */
	Elf32_Half    e_shentsize;            /* Section header table entry size */
	Elf32_Half    e_shnum;                /* Section header table entry count */
	Elf32_Half    e_shstrndx;             /* Section header string table index */
} Elf32_Ehdr;
typedef struct {
	unsigned char e_ident[16];     /* Magic number and other info */
	Elf64_Half    e_type;                 /* Object file type */
	Elf64_Half    e_machine;              /* Architecture */
	Elf64_Word    e_version;              /* Object file version */
	Elf64_Addr    e_entry;                /* Entry point virtual address */
	Elf64_Off     e_phoff;                /* Program header table file offset */
	Elf64_Off     e_shoff;                /* Section header table file offset */
	Elf64_Word    e_flags;                /* Processor-specific flags */
	Elf64_Half    e_ehsize;               /* ELF header size in bytes */
	Elf64_Half    e_phentsize;            /* Program header table entry size */
	Elf64_Half    e_phnum;                /* Program header table entry count */
	Elf64_Half    e_shentsize;            /* Section header table entry size */
	Elf64_Half    e_shnum;                /* Section header table entry count */
	Elf64_Half    e_shstrndx;             /* Section header string table index */
} Elf64_Ehdr;

/* Section header entry. The number of section entries in the file are determined by the "e_shnum" field of the ELF header.*/
typedef struct {
	Elf32_Word    sh_name;                /* Section name (string tbl index) */
	Elf32_Word    sh_type;                /* Section type */
	Elf32_Word    sh_flags;               /* Section flags */
	Elf32_Addr    sh_addr;                /* Section virtual addr at execution */
	Elf32_Off     sh_offset;              /* Section file offset */
	Elf32_Word    sh_size;                /* Section size in bytes */
	Elf32_Word    sh_link;                /* Link to another section */
	Elf32_Word    sh_info;                /* Additional section information */
	Elf32_Word    sh_addralign;           /* Section alignment */
	Elf32_Word    sh_entsize;             /* Entry size if section holds table */
} Elf32_Shdr;
typedef struct {
	Elf64_Word    sh_name;                /* Section name (string tbl index) */
	Elf64_Word    sh_type;                /* Section type */
	Elf64_Xword   sh_flags;               /* Section flags */
	Elf64_Addr    sh_addr;                /* Section virtual addr at execution */
	Elf64_Off     sh_offset;              /* Section file offset */
	Elf64_Xword   sh_size;                /* Section size in bytes */
	Elf64_Word    sh_link;                /* Link to another section */
	Elf64_Word    sh_info;                /* Additional section information */
	Elf64_Xword   sh_addralign;           /* Section alignment */
	Elf64_Xword   sh_entsize;             /* Entry size if section holds table */
} Elf64_Shdr;

/* Legal values for sh_type (section type).  */
#define SHT_NULL          0             /* Section header table entry unused */
#define SHT_PROGBITS      1             /* Program data */
#define SHT_SYMTAB        2             /* Symbol table */
#define SHT_STRTAB        3             /* String table */
#define SHT_RELA          4             /* Relocation entries with addends */
#define SHT_HASH          5             /* Symbol hash table */
#define SHT_DYNAMIC       6             /* Dynamic linking information. Contains Elf32_Dyn/Elf64_Dyn entries. */
#define SHT_NOTE          7             /* Notes */
#define SHT_NOBITS        8             /* Program space with no data (bss) */
#define SHT_REL           9             /* Relocation entries, no addends */
#define SHT_SHLIB         10            /* Reserved */
#define SHT_DYNSYM        11            /* Dynamic linker symbol table */
#define SHT_INIT_ARRAY    14            /* Array of constructors */
#define SHT_FINI_ARRAY    15            /* Array of destructors */
#define SHT_PREINIT_ARRAY 16            /* Array of pre-constructors */
#define SHT_GROUP         17            /* Section group */
#define SHT_SYMTAB_SHNDX  18            /* Extended section indeces */
#define SHT_NUM           19            /* Number of defined types.  */
#define SHT_LOOS          0x60000000    /* Start OS-specific.  */
#define SHT_GNU_ATTRIBUTES 0x6ffffff5   /* Object attributes.  */
#define SHT_GNU_HASH      0x6ffffff6    /* GNU-style hash table.  */
#define SHT_GNU_LIBLIST   0x6ffffff7    /* Prelink library list */
#define SHT_CHECKSUM      0x6ffffff8    /* Checksum for DSO content.  */
#define SHT_LOSUNW        0x6ffffffa    /* Sun-specific low bound.  */
#define SHT_SUNW_move     0x6ffffffa
#define SHT_SUNW_COMDAT   0x6ffffffb
#define SHT_SUNW_syminfo  0x6ffffffc
#define SHT_GNU_verdef    0x6ffffffd    /* Version definition section.  */
#define SHT_GNU_verneed   0x6ffffffe    /* Version needs section.  */
#define SHT_GNU_versym    0x6fffffff    /* Version symbol table.  */
#define SHT_HISUNW        0x6fffffff    /* Sun-specific high bound.  */
#define SHT_HIOS          0x6fffffff    /* End OS-specific type */
#define SHT_LOPROC        0x70000000    /* Start of processor-specific */
#define SHT_HIPROC        0x7fffffff    /* End of processor-specific */
#define SHT_LOUSER        0x80000000    /* Start of application-specific */
#define SHT_HIUSER        0x8fffffff    /* End of application-specific */

/* Dynamic section entry.  */
typedef struct {
	Elf32_Sword d_tag;                  			/* Dynamic entry type */
	union { Elf32_Word d_val; Elf32_Addr d_ptr; } d_un; 	/* Integer or address value */
} Elf32_Dyn;
typedef struct {
	Elf64_Sxword d_tag;                  			/* Dynamic entry type */
	union { Elf64_Xword d_val; Elf64_Addr d_ptr; } d_un;	/* Integer or address value */
} Elf64_Dyn;

/* Legal values for d_tag (dynamic entry type).  */
#define DT_NULL         0               /* Marks end of dynamic section */
#define DT_NEEDED       1               /* Name of needed library */
#define DT_PLTRELSZ     2               /* Size in bytes of PLT relocs */
#define DT_PLTGOT       3               /* Processor defined value */
#define DT_HASH         4               /* Address of symbol hash table */
#define DT_STRTAB       5               /* Address of string table */
#define DT_SYMTAB       6               /* Address of symbol table */
#define DT_RELA         7               /* Address of Rela relocs */
#define DT_RELASZ       8               /* Total size of Rela relocs */
#define DT_RELAENT      9               /* Size of one Rela reloc */
#define DT_STRSZ        10              /* Size of string table */
#define DT_SYMENT       11              /* Size of one symbol table entry */
#define DT_INIT         12              /* Address of init function */
#define DT_FINI         13              /* Address of termination function */
#define DT_SONAME       14              /* Name of shared object */
#define DT_RPATH        15              /* Library search path (deprecated) */
#define DT_SYMBOLIC     16              /* Start symbol search here */
#define DT_REL          17              /* Address of Rel relocs */
#define DT_RELSZ        18              /* Total size of Rel relocs */
#define DT_RELENT       19              /* Size of one Rel reloc */
#define DT_PLTREL       20              /* Type of reloc in PLT */
#define DT_DEBUG        21              /* For debugging; unspecified */
#define DT_TEXTREL      22              /* Reloc might modify .text */
#define DT_JMPREL       23              /* Address of PLT relocs */
#define DT_BIND_NOW     24              /* Process relocations of object */
#define DT_INIT_ARRAY   25              /* Array with addresses of init fct */
#define DT_FINI_ARRAY   26              /* Array with addresses of fini fct */
#define DT_INIT_ARRAYSZ 27              /* Size in bytes of DT_INIT_ARRAY */
#define DT_FINI_ARRAYSZ 28              /* Size in bytes of DT_FINI_ARRAY */
#define DT_RUNPATH      29              /* Library search path */
#define DT_FLAGS        30              /* Flags for the object being loaded */
#define DT_ENCODING     32              /* Start of encoded range */
#define DT_PREINIT_ARRAY 32             /* Array with addresses of preinit fct*/
#define DT_PREINIT_ARRAYSZ 33           /* size in bytes of DT_PREINIT_ARRAY */
#define DT_NUM          34              /* Number used */
#define DT_LOOS         0x6000000d      /* Start of OS-specific */
#define DT_HIOS         0x6ffff000      /* End of OS-specific */
#define DT_VERDEF       0x6ffffffc
#define DT_VERDEFNUM    0x6ffffffd
#define DT_LOPROC       0x70000000      /* Start of processor-specific */
#define DT_HIPROC       0x7fffffff      /* End of processor-specific */


/* Symbol table entry.  */
typedef struct {
	Elf32_Word    st_name;                /* Symbol name (string tbl index) */
	Elf32_Addr    st_value;               /* Symbol value */
	Elf32_Word    st_size;                /* Symbol size */
	unsigned char st_info;                /* Symbol type and binding */
	unsigned char st_other;               /* Symbol visibility */
	Elf32_Section st_shndx;               /* Section index */
} Elf32_Sym;

typedef struct {
	Elf64_Word    st_name;                /* Symbol name (string tbl index) */
	unsigned char st_info;                /* Symbol type and binding */
	unsigned char st_other;               /* Symbol visibility */
	Elf64_Section st_shndx;               /* Section index */
	Elf64_Addr    st_value;               /* Symbol value */
	Elf64_Xword   st_size;                /* Symbol size */
} Elf64_Sym;


#endif
