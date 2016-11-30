// **** supporting defines ****

typedef struct
{
  __IO uint32_t HPRT;
}
USB_OTG_HostPortTypeDef;

#define USBx_HOST       ((USB_OTG_HostTypeDef *)((uint32_t)USBx + USB_OTG_HOST_BASE))
#define USBx_HOST_PORT  ((USB_OTG_HostPortTypeDef *)((uint32_t)USBx + USB_OTG_HOST_PORT_BASE))
#define USBx_DEVICE     ((USB_OTG_DeviceTypeDef *)((uint32_t)USBx + USB_OTG_DEVICE_BASE))
#define USBx_INEP(i)    ((USB_OTG_INEndpointTypeDef *)((uint32_t)USBx + USB_OTG_IN_ENDPOINT_BASE + (i)*USB_OTG_EP_REG_SIZE))        
#define USBx_OUTEP(i)   ((USB_OTG_OUTEndpointTypeDef *)((uint32_t)USBx + USB_OTG_OUT_ENDPOINT_BASE + (i)*USB_OTG_EP_REG_SIZE))        
#define USBx_DFIFO(i)   *(__IO uint32_t *)((uint32_t)USBx + USB_OTG_FIFO_BASE + (i) * USB_OTG_FIFO_SIZE)
#define USBx_PCGCCTL    *(__IO uint32_t *)((uint32_t)USBx + USB_OTG_PCGCCTL_BASE)

#define  USB_REQ_GET_STATUS                             0x00
#define  USB_REQ_CLEAR_FEATURE                          0x01
#define  USB_REQ_SET_FEATURE                            0x03
#define  USB_REQ_SET_ADDRESS                            0x05
#define  USB_REQ_GET_DESCRIPTOR                         0x06
#define  USB_REQ_SET_DESCRIPTOR                         0x07
#define  USB_REQ_GET_CONFIGURATION                      0x08
#define  USB_REQ_SET_CONFIGURATION                      0x09
#define  USB_REQ_GET_INTERFACE                          0x0A
#define  USB_REQ_SET_INTERFACE                          0x0B
#define  USB_REQ_SYNCH_FRAME                            0x0C

#define  USB_DESC_TYPE_DEVICE                              1
#define  USB_DESC_TYPE_CONFIGURATION                       2
#define  USB_DESC_TYPE_STRING                              3
#define  USB_DESC_TYPE_INTERFACE                           4
#define  USB_DESC_TYPE_ENDPOINT                            5
#define  USB_DESC_TYPE_DEVICE_QUALIFIER                    6
#define  USB_DESC_TYPE_OTHER_SPEED_CONFIGURATION           7

#define STS_GOUT_NAK                           1
#define STS_DATA_UPDT                          2
#define STS_XFER_COMP                          3
#define STS_SETUP_COMP                         4
#define STS_SETUP_UPDT                         6

#define USBD_FS_TRDT_VALUE           5

// interfaces
void usb_cb_control_msg();
void usb_cb_ep1_in(int len);
void usb_cb_ep2_out(uint8_t *usbdata, int len);
void usb_cb_ep3_out(uint8_t *usbdata, int len);

uint8_t device_desc[] = {
  0x12,0x01,0x00,0x01,
  0xFF,0xFF,0xFF,0x40,
  (USB_VID>>0)&0xFF,(USB_VID>>8)&0xFF,
  (USB_PID>>0)&0xFF,(USB_PID>>8)&0xFF,
  0x00,0x22,0x00,0x00,
  0x00,0x01};

uint8_t configuration_desc[] = {
  0x09, 0x02, 0x27, 0x00,
  0x01, 0x01, 0x00, 0xc0,
  0x32,
  // interface 0
  0x09, 0x04, 0x00, 0x00,
  0x03, 0xff, 0xFF, 0xFF,
  0x00,
  // endpoint 1, read CAN
  0x07, 0x05, 0x81, 0x02, 0x40, 0x00, 0x00,
  // endpoint 2, AES load
  0x07, 0x05, 0x02, 0x02, 0x10, 0x00, 0x00,
  // endpoint 3, send CAN
  0x07, 0x05, 0x03, 0x02, 0x40, 0x00, 0x00,
};

typedef union
{
  uint16_t w;
  struct BW
  {
    uint8_t msb;
    uint8_t lsb;
  }
  bw; 
}
uint16_t_uint8_t;


typedef union _USB_Setup
{
  uint32_t d8[2];
  
  struct _SetupPkt_Struc
  {
    uint8_t           bmRequestType;
    uint8_t           bRequest;
    uint16_t_uint8_t  wValue;
    uint16_t_uint8_t  wIndex;
    uint16_t_uint8_t  wLength;
  } b;
} 
USB_Setup_TypeDef; 

// current packet
USB_Setup_TypeDef setup;
uint8_t usbdata[0x100];

// packet read and write

void *USB_ReadPacket(void *dest, uint16_t len) {
  uint32_t i=0; 
  uint32_t count32b = (len + 3) / 4; 
  
  for ( i = 0; i < count32b; i++, dest += 4 ) {
    // packed?
    *(__attribute__((__packed__)) uint32_t *)dest = USBx_DFIFO(0);
  }
  return ((void *)dest);
}

void USB_WritePacket(const uint8_t *src, uint16_t len, uint32_t ep) {
  #ifdef DEBUG
    puts("writing ");
    hexdump(src, len);
  #endif
  uint32_t count32b = 0, i = 0;
  count32b = (len + 3) / 4; 

  // bullshit
  USBx_INEP(ep)->DIEPTSIZ = (USB_OTG_DIEPTSIZ_PKTCNT & (1 << 19)) | (len & USB_OTG_DIEPTSIZ_XFRSIZ);
  USBx_INEP(ep)->DIEPCTL |= (USB_OTG_DIEPCTL_CNAK | USB_OTG_DIEPCTL_EPENA);

  // load the FIFO
  for (i = 0; i < count32b; i++, src += 4) {
    USBx_DFIFO(ep) = *((__attribute__((__packed__)) uint32_t *)src);
  }
}

void usb_reset() {
  // unmask endpoint interrupts, so many sets
  USBx_DEVICE->DAINT = 0xFFFFFFFF;
  USBx_DEVICE->DAINTMSK = 0xFFFFFFFF;
  //USBx_DEVICE->DOEPMSK = (USB_OTG_DOEPMSK_STUPM | USB_OTG_DOEPMSK_XFRCM | USB_OTG_DOEPMSK_EPDM);
  //USBx_DEVICE->DIEPMSK = (USB_OTG_DIEPMSK_TOM | USB_OTG_DIEPMSK_XFRCM | USB_OTG_DIEPMSK_EPDM | USB_OTG_DIEPMSK_ITTXFEMSK);
  //USBx_DEVICE->DIEPMSK = (USB_OTG_DIEPMSK_TOM | USB_OTG_DIEPMSK_XFRCM | USB_OTG_DIEPMSK_EPDM);

  // all interrupts for debugging
  USBx_DEVICE->DIEPMSK = 0xFFFFFFFF;
  USBx_DEVICE->DOEPMSK = 0xFFFFFFFF;

  // clear interrupts
  USBx_INEP(0)->DIEPINT = 0xFF;
  USBx_OUTEP(0)->DOEPINT = 0xFF;

  // unset the address
  USBx_DEVICE->DCFG &= ~USB_OTG_DCFG_DAD;

  // set up USB FIFOs
  // RX start address is fixed to 0
  USBx->GRXFSIZ = 0x40;

  // 0x100 to offset past GRXFSIZ
  USBx->DIEPTXF0_HNPTXFSIZ = (0x40 << 16) | 0x40;

  // EP1, massive
  USBx->DIEPTXF[0] = (0x40 << 16) | 0x80;

  // flush TX fifo
  USBx->GRSTCTL = USB_OTG_GRSTCTL_TXFFLSH | USB_OTG_GRSTCTL_TXFNUM_4;
  while ((USBx->GRSTCTL & USB_OTG_GRSTCTL_TXFFLSH) == USB_OTG_GRSTCTL_TXFFLSH);
  // flush RX FIFO
  USBx->GRSTCTL = USB_OTG_GRSTCTL_RXFFLSH;
  while ((USBx->GRSTCTL & USB_OTG_GRSTCTL_RXFFLSH) == USB_OTG_GRSTCTL_RXFFLSH);

  // no global NAK
  USBx_DEVICE->DCTL |= USB_OTG_DCTL_CGINAK;

  // ready to receive setup packets
  USBx_OUTEP(0)->DOEPTSIZ = USB_OTG_DOEPTSIZ_STUPCNT | (USB_OTG_DOEPTSIZ_PKTCNT & (1 << 19)) | (3 * 8);
}

void usb_setup() {
  uint8_t resp[0x20];
  // setup packet is ready
  switch (setup.b.bRequest) {
    case USB_REQ_SET_CONFIGURATION:
      // enable other endpoints, has to be here?
      USBx_INEP(1)->DIEPCTL = (0x40 & USB_OTG_DIEPCTL_MPSIZ) | (2 << 18) | (1 << 22) |
                              USB_OTG_DIEPCTL_SD0PID_SEVNFRM | USB_OTG_DIEPCTL_USBAEP;
      USBx_INEP(1)->DIEPINT = 0xFF;

      USBx_OUTEP(2)->DOEPTSIZ = (1 << 19) | 0x10;
      USBx_OUTEP(2)->DOEPCTL = (0x10 & USB_OTG_DOEPCTL_MPSIZ) | (2 << 18) |
                               USB_OTG_DOEPCTL_SD0PID_SEVNFRM | USB_OTG_DOEPCTL_USBAEP;
      USBx_OUTEP(2)->DOEPINT = 0xFF;

      USBx_OUTEP(3)->DOEPTSIZ = (1 << 19) | 0x40;
      USBx_OUTEP(3)->DOEPCTL = (0x40 & USB_OTG_DOEPCTL_MPSIZ) | (2 << 18) |
                               USB_OTG_DOEPCTL_SD0PID_SEVNFRM | USB_OTG_DOEPCTL_USBAEP;
      USBx_OUTEP(3)->DOEPINT = 0xFF;

      // mark ready to receive
      USBx_OUTEP(2)->DOEPCTL |= USB_OTG_DOEPCTL_EPENA | USB_OTG_DOEPCTL_CNAK;
      USBx_OUTEP(3)->DOEPCTL |= USB_OTG_DOEPCTL_EPENA | USB_OTG_DOEPCTL_CNAK;

      USB_WritePacket(0, 0, 0);
      USBx_OUTEP(0)->DOEPCTL |= USB_OTG_DOEPCTL_CNAK;
      break;
    case USB_REQ_SET_ADDRESS:
      // set now?
      USBx_DEVICE->DCFG |= ((setup.b.wValue.w & 0x7f) << 4);

      #ifdef DEBUG
        puts(" set address\n");
      #endif


      USB_WritePacket(0, 0, 0);
      USBx_OUTEP(0)->DOEPCTL |= USB_OTG_DOEPCTL_CNAK;

      break;
    case USB_REQ_GET_DESCRIPTOR:
      switch (setup.b.wValue.bw.lsb) {
        case USB_DESC_TYPE_DEVICE:
          //puts("    writing device descriptor\n");

          // setup transfer
          USB_WritePacket(device_desc, min(sizeof(device_desc), setup.b.wLength.w), 0);
          USBx_OUTEP(0)->DOEPCTL |= USB_OTG_DOEPCTL_CNAK;

          //puts("D");
          break;
        case USB_DESC_TYPE_CONFIGURATION:
          USB_WritePacket(configuration_desc, min(sizeof(configuration_desc), setup.b.wLength.w), 0);
          USBx_OUTEP(0)->DOEPCTL |= USB_OTG_DOEPCTL_CNAK;
          break;
        default:
          // nothing here?
          USB_WritePacket(0, 0, 0);
          USBx_OUTEP(0)->DOEPCTL |= USB_OTG_DOEPCTL_CNAK;
          break;
      }
      break;
    case USB_REQ_GET_STATUS:
      // empty resp?
      resp[0] = 0;
      resp[1] = 0;
      USB_WritePacket((void*)&resp, 2, 0);
      USBx_OUTEP(0)->DOEPCTL |= USB_OTG_DOEPCTL_CNAK;
      break;
    default:
      usb_cb_control_msg();
  }
}

void usb_init() {
  // internal PHY set before reset
  USBx->GUSBCFG |= USB_OTG_GUSBCFG_PHYSEL;

  // full speed PHY, do reset and remove power down
  puth(USBx->GRSTCTL);
  puts(" resetting PHY\n");
  while ((USBx->GRSTCTL & USB_OTG_GRSTCTL_AHBIDL) == 0);
  puts("AHB idle\n");

  // reset PHY here?
  USBx->GRSTCTL |= USB_OTG_GRSTCTL_CSRST;
  while ((USBx->GRSTCTL & USB_OTG_GRSTCTL_CSRST) == USB_OTG_GRSTCTL_CSRST);
  puts("reset done\n");

  // power up the PHY
  USBx->GCCFG = USB_OTG_GCCFG_PWRDWN | USB_OTG_GCCFG_NOVBUSSENS;

  // be a device, slowest timings
  //USBx->GUSBCFG = USB_OTG_GUSBCFG_FDMOD | USB_OTG_GUSBCFG_PHYSEL | USB_OTG_GUSBCFG_TRDT | USB_OTG_GUSBCFG_TOCAL;
  USBx->GUSBCFG = USB_OTG_GUSBCFG_FDMOD | USB_OTG_GUSBCFG_PHYSEL;
  USBx->GUSBCFG |= (uint32_t)((USBD_FS_TRDT_VALUE << 10) & USB_OTG_GUSBCFG_TRDT);
  //USBx->GUSBCFG = USB_OTG_GUSBCFG_PHYSEL | USB_OTG_GUSBCFG_TRDT | USB_OTG_GUSBCFG_TOCAL;

  // **** for debugging, doesn't seem to work ****
  //USBx->GUSBCFG |= USB_OTG_GUSBCFG_CTXPKT;

  // reset PHY clock
  USBx_PCGCCTL = 0;

  // enable the fancy OTG things
  USBx->GUSBCFG |= USB_OTG_GUSBCFG_HNPCAP | USB_OTG_GUSBCFG_SRPCAP;

  USBx_DEVICE->DCFG = USB_OTG_DCFG_NZLSOHSK | USB_OTG_DCFG_DSPD;
  //USBx_DEVICE->DCFG = USB_OTG_DCFG_DSPD;

  // setup USB interrupts
  // all interrupts except TXFIFO EMPTY
  //USBx->GINTMSK = 0xFFFFFFFF & ~(USB_OTG_GINTMSK_NPTXFEM | USB_OTG_GINTMSK_PTXFEM | USB_OTG_GINTSTS_SOF | USB_OTG_GINTSTS_EOPF);
  USBx->GINTMSK = 0xFFFFFFFF & ~(USB_OTG_GINTMSK_NPTXFEM | USB_OTG_GINTMSK_PTXFEM);

  USBx->GAHBCFG = USB_OTG_GAHBCFG_GINT;
  USBx->GINTSTS = 0;
}

// ***************************** USB port *****************************

void usb_irqhandler(void) {
  USBx->GINTMSK = 0;

  unsigned int gintsts = USBx->GINTSTS;

  // gintsts SUSPEND? 04008428
  #ifdef DEBUG
    unsigned int daint = USBx_DEVICE->DAINT;
    puth(gintsts);
    puts(" ep ");
    puth(daint);
    puts(" USB interrupt!\n");
  #endif

  if (gintsts & USB_OTG_GINTSTS_ESUSP) {
    puts("ESUSP detected\n");
  }

  if (gintsts & USB_OTG_GINTSTS_USBRST) {
    puts("USB reset\n");
    usb_reset();
  }

  if (gintsts & USB_OTG_GINTSTS_ENUMDNE) {
    puts("enumeration done ");
    // Full speed, ENUMSPD
    puth(USBx_DEVICE->DSTS);
    puts("\n");
  }

  if (gintsts & USB_OTG_GINTSTS_OTGINT) {
    puts("OTG int:");
    puth(USBx->GOTGINT);
    puts("\n");

    // getting ADTOCHG
    USBx->GOTGINT = USBx->GOTGINT;
  }

  // RX FIFO first
  if (gintsts & USB_OTG_GINTSTS_RXFLVL) {
    // 1. Read the Receive status pop register
    volatile unsigned int rxst = USBx->GRXSTSP;

    #ifdef DEBUG
      puts(" RX FIFO:");
      puth(rxst);
      puts(" status: ");
      puth((rxst & USB_OTG_GRXSTSP_PKTSTS) >> 17);
      puts(" len: ");
      puth((rxst & USB_OTG_GRXSTSP_BCNT) >> 4);
      puts("\n");
    #endif


    if (((rxst & USB_OTG_GRXSTSP_PKTSTS) >> 17) == STS_DATA_UPDT) {
      int endpoint = (rxst & USB_OTG_GRXSTSP_EPNUM);
      int len = (rxst & USB_OTG_GRXSTSP_BCNT) >> 4;
      USB_ReadPacket(&usbdata, len);
      #ifdef DEBUG
        puts("  data ");
        puth(len);
        puts("\n");
        hexdump(&usbdata, len);
      #endif

      if (endpoint == 2) {
        usb_cb_ep2_out(usbdata, len);
      }

      if (endpoint == 3) {
        usb_cb_ep3_out(usbdata, len);
      }
    } else if (((rxst & USB_OTG_GRXSTSP_PKTSTS) >> 17) == STS_SETUP_UPDT) {
      USB_ReadPacket(&setup, 8);
      #ifdef DEBUG
        puts("  setup ");
        hexdump(&setup, 8);
        puts("\n");
      #endif
    }
  }

  if (gintsts & USB_OTG_GINTSTS_HPRTINT) {
    // host
    puts("HPRT:");
    puth(USBx_HOST_PORT->HPRT);
    puts("\n");
    if (USBx_HOST_PORT->HPRT & USB_OTG_HPRT_PCDET) {
      USBx_HOST_PORT->HPRT |= USB_OTG_HPRT_PRST;
      USBx_HOST_PORT->HPRT |= USB_OTG_HPRT_PCDET;
    }

  }

  if (gintsts & USB_OTG_GINTSTS_BOUTNAKEFF) {
    // no global NAK, why is this getting set?
    #ifdef DEBUG
      puts("GLOBAL NAK\n");
    #endif
    USBx_DEVICE->DCTL |= USB_OTG_DCTL_CGONAK | USB_OTG_DCTL_CGINAK;
  }

  if (gintsts & USB_OTG_GINTSTS_SRQINT) {
    // we want to do "A-device host negotiation protocol" since we are the A-device
    puts("start request\n");
    //USBx->GUSBCFG |= USB_OTG_GUSBCFG_FDMOD;
    //USBx_HOST_PORT->HPRT = USB_OTG_HPRT_PPWR | USB_OTG_HPRT_PENA;
  }

  // out endpoint hit
  if (gintsts & USB_OTG_GINTSTS_OEPINT) {
    #ifdef DEBUG
      puts("  0:");
      puth(USBx_OUTEP(0)->DOEPINT);
      puts(" 2:");
      puth(USBx_OUTEP(2)->DOEPINT);
      puts(" 3:");
      puth(USBx_OUTEP(3)->DOEPINT);
      puts(" ");
      puth(USBx_OUTEP(3)->DOEPCTL);
      puts(" 4:");
      puth(USBx_OUTEP(4)->DOEPINT);
      puts(" OUT ENDPOINT\n");
    #endif

    if (USBx_OUTEP(2)->DOEPINT & USB_OTG_DOEPINT_XFRC) {
      #ifdef DEBUG
        puts("  OUT2 PACKET XFRC\n"); 
      #endif
      USBx_OUTEP(2)->DOEPTSIZ = (1 << 19) | 0x10;
      USBx_OUTEP(2)->DOEPCTL |= USB_OTG_DOEPCTL_EPENA | USB_OTG_DOEPCTL_CNAK;
    }

    if (USBx_OUTEP(3)->DOEPINT & USB_OTG_DOEPINT_XFRC) {
      #ifdef DEBUG
        puts("  OUT3 PACKET XFRC\n"); 
      #endif
      USBx_OUTEP(3)->DOEPTSIZ = (1 << 19) | 0x40;
      USBx_OUTEP(3)->DOEPCTL |= USB_OTG_DOEPCTL_EPENA | USB_OTG_DOEPCTL_CNAK;
    } else if (USBx_OUTEP(3)->DOEPINT & 0x2000) {
      #ifdef DEBUG
        puts("  OUT3 PACKET WTF\n"); 
      #endif
      // if NAK was set trigger this, unknown interrupt
      USBx_OUTEP(3)->DOEPTSIZ = (1 << 19) | 0x40;
      USBx_OUTEP(3)->DOEPCTL |= USB_OTG_DOEPCTL_CNAK;
    } else if (USBx_OUTEP(3)->DOEPINT) {
      puts("OUTEP3 error ");
      puth(USBx_OUTEP(3)->DOEPINT);
      puts("\n");
    }

    if (USBx_OUTEP(0)->DOEPINT & USB_OTG_DIEPINT_XFRC) {
      // ready for next packet
      USBx_OUTEP(0)->DOEPTSIZ = USB_OTG_DOEPTSIZ_STUPCNT | (USB_OTG_DOEPTSIZ_PKTCNT & (1 << 19)) | (1 * 8);
    }

    // respond to setup packets
    if (USBx_OUTEP(0)->DOEPINT & USB_OTG_DOEPINT_STUP) {
      usb_setup();
    }

    USBx_OUTEP(0)->DOEPINT = USBx_OUTEP(0)->DOEPINT;
    USBx_OUTEP(2)->DOEPINT = USBx_OUTEP(2)->DOEPINT;
    USBx_OUTEP(3)->DOEPINT = USBx_OUTEP(3)->DOEPINT;
  }


  // in endpoint hit
  if (gintsts & USB_OTG_GINTSTS_IEPINT) {
    #ifdef DEBUG
      puts("  ");
      puth(USBx_INEP(0)->DIEPINT);
      puts(" ");
      puth(USBx_INEP(1)->DIEPINT);
      puts(" IN ENDPOINT\n");
    #endif

    // this happens first
    if (USBx_INEP(1)->DIEPINT & USB_OTG_DIEPINT_XFRC) {
      #ifdef DEBUG
        puts("  IN PACKET SEND\n"); 
      #endif
      //USBx_DEVICE->DIEPEMPMSK = ~(1 << 1);
    }
    
    // *** IN token received when TxFIFO is empty
    if (USBx_INEP(1)->DIEPINT & USB_OTG_DIEPMSK_ITTXFEMSK) {
      #ifdef DEBUG
        puts("  IN PACKET QUEUE\n"); 
      #endif
      // TODO: always assuming max len, can we get the length?
      usb_cb_ep1_in(0x40);
    }

    // clear interrupts
    USBx_INEP(0)->DIEPINT = USBx_INEP(0)->DIEPINT;
    USBx_INEP(1)->DIEPINT = USBx_INEP(1)->DIEPINT;
  }


  // clear all interrupts
  USBx_DEVICE->DAINT = USBx_DEVICE->DAINT;
  USBx->GINTSTS = USBx->GINTSTS;

  USBx->GINTMSK = 0xFFFFFFFF & ~(USB_OTG_GINTMSK_NPTXFEM | USB_OTG_GINTMSK_PTXFEM | USB_OTG_GINTSTS_SOF | USB_OTG_GINTSTS_EOPF);
}

