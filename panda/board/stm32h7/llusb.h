typedef struct
{
  __IO uint32_t HPRT;
}
USB_OTG_HostPortTypeDef;

USB_OTG_GlobalTypeDef *USBx = USB_OTG_HS;

#define USBx_HOST       ((USB_OTG_HostTypeDef *)((uint32_t)USBx + USB_OTG_HOST_BASE))
#define USBx_HOST_PORT  ((USB_OTG_HostPortTypeDef *)((uint32_t)USBx + USB_OTG_HOST_PORT_BASE))
#define USBx_DEVICE     ((USB_OTG_DeviceTypeDef *)((uint32_t)USBx + USB_OTG_DEVICE_BASE))
#define USBx_INEP(i)    ((USB_OTG_INEndpointTypeDef *)((uint32_t)USBx + USB_OTG_IN_ENDPOINT_BASE + ((i) * USB_OTG_EP_REG_SIZE)))
#define USBx_OUTEP(i)   ((USB_OTG_OUTEndpointTypeDef *)((uint32_t)USBx + USB_OTG_OUT_ENDPOINT_BASE + ((i) * USB_OTG_EP_REG_SIZE)))
#define USBx_DFIFO(i)   *(__IO uint32_t *)((uint32_t)USBx + USB_OTG_FIFO_BASE + ((i) * USB_OTG_FIFO_SIZE))
#define USBx_PCGCCTL    *(__IO uint32_t *)((uint32_t)USBx + USB_OTG_PCGCCTL_BASE)

#define USBD_FS_TRDT_VALUE        6U
#define USB_OTG_SPEED_FULL        3U
#define DCFG_FRAME_INTERVAL_80    0U


void usb_irqhandler(void);

void OTG_HS_IRQ_Handler(void) {
  NVIC_DisableIRQ(OTG_HS_IRQn);
  usb_irqhandler();
  NVIC_EnableIRQ(OTG_HS_IRQn);
}

void usb_init(void) {
  REGISTER_INTERRUPT(OTG_HS_IRQn, OTG_HS_IRQ_Handler, 1500000U, FAULT_INTERRUPT_RATE_USB) // TODO: Find out a better rate limit for USB. Now it's the 1.5MB/s rate

  // Disable global interrupt
  USBx->GAHBCFG &= ~(USB_OTG_GAHBCFG_GINT);
  // Select FS Embedded PHY
  USBx->GUSBCFG |= USB_OTG_GUSBCFG_PHYSEL;
  // Force device mode
  USBx->GUSBCFG &= ~(USB_OTG_GUSBCFG_FHMOD | USB_OTG_GUSBCFG_FDMOD);
  USBx->GUSBCFG |= USB_OTG_GUSBCFG_FDMOD;
  delay(250000); // Wait for about 25ms (explicitly stated in H7 ref manual)
  // Wait for AHB master IDLE state.
  while ((USBx->GRSTCTL & USB_OTG_GRSTCTL_AHBIDL) == 0);
  // Core Soft Reset
  USBx->GRSTCTL |= USB_OTG_GRSTCTL_CSRST;
  while ((USBx->GRSTCTL & USB_OTG_GRSTCTL_CSRST) == USB_OTG_GRSTCTL_CSRST);
  // Activate the USB Transceiver
  USBx->GCCFG |= USB_OTG_GCCFG_PWRDWN;

  for (uint8_t i = 0U; i < 15U; i++) {
    USBx->DIEPTXF[i] = 0U;
  }

  // VBUS Sensing setup
  USBx_DEVICE->DCTL |= USB_OTG_DCTL_SDIS;
  // Deactivate VBUS Sensing B
  USBx->GCCFG &= ~(USB_OTG_GCCFG_VBDEN);
  // B-peripheral session valid override enable
  USBx->GOTGCTL |= USB_OTG_GOTGCTL_BVALOEN;
  USBx->GOTGCTL |= USB_OTG_GOTGCTL_BVALOVAL;
  // Restart the Phy Clock
  USBx_PCGCCTL = 0U;
  // Device mode configuration
  USBx_DEVICE->DCFG |= DCFG_FRAME_INTERVAL_80;
  USBx_DEVICE->DCFG |= USB_OTG_SPEED_FULL | USB_OTG_DCFG_NZLSOHSK;

  // Flush FIFOs
  USBx->GRSTCTL = (USB_OTG_GRSTCTL_TXFFLSH | (0x10U << 6));
  while ((USBx->GRSTCTL & USB_OTG_GRSTCTL_TXFFLSH) == USB_OTG_GRSTCTL_TXFFLSH);

  USBx->GRSTCTL = USB_OTG_GRSTCTL_RXFFLSH;
  while ((USBx->GRSTCTL & USB_OTG_GRSTCTL_RXFFLSH) == USB_OTG_GRSTCTL_RXFFLSH);

  // Clear all pending Device Interrupts
  USBx_DEVICE->DIEPMSK = 0U;
  USBx_DEVICE->DOEPMSK = 0U;
  USBx_DEVICE->DAINTMSK = 0U;
  USBx_DEVICE->DIEPMSK &= ~(USB_OTG_DIEPMSK_TXFURM);

  // Disable all interrupts.
  USBx->GINTMSK = 0U;
  // Clear any pending interrupts
  USBx->GINTSTS = 0xBFFFFFFFU;
  // Enable interrupts matching to the Device mode ONLY
  USBx->GINTMSK = USB_OTG_GINTMSK_USBRST | USB_OTG_GINTMSK_ENUMDNEM | USB_OTG_GINTMSK_OTGINT |
                  USB_OTG_GINTMSK_RXFLVLM | USB_OTG_GINTMSK_GONAKEFFM | USB_OTG_GINTMSK_GINAKEFFM |
                  USB_OTG_GINTMSK_OEPINT | USB_OTG_GINTMSK_IEPINT |
                  USB_OTG_GINTMSK_CIDSCHGM | USB_OTG_GINTMSK_SRQIM | USB_OTG_GINTMSK_MMISM;

  // Set USB Turnaround time
  USBx->GUSBCFG |= ((USBD_FS_TRDT_VALUE << 10) & USB_OTG_GUSBCFG_TRDT);
  // Enables the controller's Global Int in the AHB Config reg
  USBx->GAHBCFG |= USB_OTG_GAHBCFG_GINT;
  // Soft disconnect disable:
  USBx_DEVICE->DCTL &= ~(USB_OTG_DCTL_SDIS);

  // enable the IRQ
  NVIC_EnableIRQ(OTG_HS_IRQn);
}
