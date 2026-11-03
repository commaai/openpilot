# Corolla TSS3 signer kit

This directory vendors the runtime deployment files from the exact Corolla kit
previously distributed as `tss3-oracle-span-f612bdc.zip`, so the openpilot
checkout is the complete deployment unit. Build-only ELF/disassembly diagnostics
are intentionally omitted; they are not read by the launcher or installer.

- Kit source commit: `f612bdc22cf08cee985ac6a631e69fef9c3eff25`
- Target: `corolla-8965F1208000`
- Payload SHA-256: `6234bb5bd496a82fabb32c27b56b3db0f1907b5bf6482e4c19f8835153ff69fc`
- Payload size: `0x1000` bytes

The payload is volatile RAM firmware. The launcher does not perform a CodeFlash
write. `TSS3_ORACLE_TOOL` remains available as an explicit development override;
normal vehicle operation uses the bundled launcher beside this file.
