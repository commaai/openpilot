/**
 * @file    panda.c
 * @author  Jessy Diamond Exum
 * @date    16 June 2017
 * @version 0.1
 * @brief   Driver for the Comma.ai Panda CAN adapter to allow it to be controlled via
 * the Linux SocketCAN interface.
 * @see https://github.com/commaai/panda for the full project.
 * @see Inspired by net/can/usb/mcba_usb.c from Linux Kernel 4.12-rc4.
 */

#include <linux/can.h>
#include <linux/can/dev.h>
#include <linux/can/error.h>
#include <linux/init.h>             // Macros used to mark up functions e.g., __init __exit
#include <linux/kernel.h>           // Contains types, macros, functions for the kernel
#include <linux/module.h>           // Core header for loading LKMs into the kernel
#include <linux/netdevice.h>
#include <linux/usb.h>

/* vendor and product id */
#define PANDA_MODULE_NAME "panda"
#define PANDA_VENDOR_ID 0XBBAA
#define PANDA_PRODUCT_ID 0XDDCC

#define PANDA_MAX_TX_URBS 20
#define PANDA_CTX_FREE PANDA_MAX_TX_URBS

#define PANDA_USB_RX_BUFF_SIZE 0x40
#define PANDA_USB_TX_BUFF_SIZE (sizeof(struct panda_usb_can_msg))

#define PANDA_NUM_CAN_INTERFACES 3

#define PANDA_CAN_TRANSMIT 1
#define PANDA_CAN_EXTENDED 4

#define PANDA_BITRATE 500000

#define PANDA_DLC_MASK  0x0F

#define SAFETY_ALLOUTPUT 17
#define SAFETY_SILENT 0

struct panda_usb_ctx {
  struct panda_inf_priv *priv;
  u32 ndx;
  u8 dlc;
};

struct panda_dev_priv;

struct panda_inf_priv {
  struct can_priv can;
  struct panda_usb_ctx tx_context[PANDA_MAX_TX_URBS];
  struct net_device *netdev;
  struct usb_anchor tx_submitted;
  atomic_t free_ctx_cnt;
  u8 interface_num;
  u8 mcu_can_ifnum;
  struct panda_dev_priv *priv_dev;
};

struct panda_dev_priv {
  struct usb_device *udev;
  struct device *dev;
  struct usb_anchor rx_submitted;
  struct panda_inf_priv *interfaces[PANDA_NUM_CAN_INTERFACES];
};

struct __packed panda_usb_can_msg {
  u32 rir;
  u32 bus_dat_len;
  u8 data[8];
};

static const struct usb_device_id panda_usb_table[] = {
  { USB_DEVICE(PANDA_VENDOR_ID, PANDA_PRODUCT_ID) },
  {} /* Terminating entry */
};

MODULE_DEVICE_TABLE(usb, panda_usb_table);


// panda:       CAN1 = 0   CAN2 = 1   CAN3 = 4
const int can_numbering[] = {0,1,4};

struct panda_inf_priv *
panda_get_inf_from_bus_id(struct panda_dev_priv *priv_dev, int bus_id){
  int inf_num;
  for(inf_num = 0; inf_num < PANDA_NUM_CAN_INTERFACES; inf_num++)
    if(can_numbering[inf_num] == bus_id)
      return priv_dev->interfaces[inf_num];
  return NULL;
}

// CTX handling shamlessly ripped from mcba_usb.c linux driver
static inline void panda_init_ctx(struct panda_inf_priv *priv)
{
  int i = 0;

  for (i = 0; i < PANDA_MAX_TX_URBS; i++) {
    priv->tx_context[i].ndx = PANDA_CTX_FREE;
    priv->tx_context[i].priv = priv;
  }

  atomic_set(&priv->free_ctx_cnt, ARRAY_SIZE(priv->tx_context));
}

static inline struct panda_usb_ctx *panda_usb_get_free_ctx(struct panda_inf_priv *priv,
							 struct can_frame *cf)
{
  int i = 0;
  struct panda_usb_ctx *ctx = NULL;

  for (i = 0; i < PANDA_MAX_TX_URBS; i++) {
    if (priv->tx_context[i].ndx == PANDA_CTX_FREE) {
      ctx = &priv->tx_context[i];
      ctx->ndx = i;
      ctx->dlc = cf->can_dlc;

      atomic_dec(&priv->free_ctx_cnt);
      break;
    }
  }

  printk("CTX num %d\n", atomic_read(&priv->free_ctx_cnt));
  if (!atomic_read(&priv->free_ctx_cnt)){
    /* That was the last free ctx. Slow down tx path */
    printk("SENDING TOO FAST\n");
    netif_stop_queue(priv->netdev);
  }

  return ctx;
}

/* panda_usb_free_ctx and panda_usb_get_free_ctx are executed by different
 * threads. The order of execution in below function is important.
 */
static inline void panda_usb_free_ctx(struct panda_usb_ctx *ctx)
{
  /* Increase number of free ctxs before freeing ctx */
  atomic_inc(&ctx->priv->free_ctx_cnt);

  ctx->ndx = PANDA_CTX_FREE;

  /* Wake up the queue once ctx is marked free */
  netif_wake_queue(ctx->priv->netdev);
}



static void panda_urb_unlink(struct panda_inf_priv *priv)
{
  usb_kill_anchored_urbs(&priv->priv_dev->rx_submitted);
  usb_kill_anchored_urbs(&priv->tx_submitted);
}

static int panda_set_output_enable(struct panda_inf_priv* priv, bool enable){
  return usb_control_msg(priv->priv_dev->udev,
			 usb_sndctrlpipe(priv->priv_dev->udev, 0),
			 0xDC, USB_TYPE_VENDOR | USB_RECIP_DEVICE,
			 enable ? SAFETY_ALLOUTPUT : SAFETY_SILENT, 0, NULL, 0, USB_CTRL_SET_TIMEOUT);
}

static void panda_usb_write_bulk_callback(struct urb *urb)
{
  struct panda_usb_ctx *ctx = urb->context;
  struct net_device *netdev;

  WARN_ON(!ctx);

  netdev = ctx->priv->netdev;

  /* free up our allocated buffer */
  usb_free_coherent(urb->dev, urb->transfer_buffer_length,
		    urb->transfer_buffer, urb->transfer_dma);

  if (!netif_device_present(netdev))
    return;

  netdev->stats.tx_packets++;
  netdev->stats.tx_bytes += ctx->dlc;

  can_get_echo_skb(netdev, ctx->ndx);

  if (urb->status)
    netdev_info(netdev, "Tx URB aborted (%d)\n", urb->status);

  /* Release the context */
  panda_usb_free_ctx(ctx);
}


static netdev_tx_t panda_usb_xmit(struct panda_inf_priv *priv,
				  struct panda_usb_can_msg *usb_msg,
				  struct panda_usb_ctx *ctx)
{
  struct urb *urb;
  u8 *buf;
  int err;

  /* create a URB, and a buffer for it, and copy the data to the URB */
  urb = usb_alloc_urb(0, GFP_ATOMIC);
  if (!urb)
    return -ENOMEM;

  buf = usb_alloc_coherent(priv->priv_dev->udev,
			   PANDA_USB_TX_BUFF_SIZE, GFP_ATOMIC,
			   &urb->transfer_dma);
  if (!buf) {
    err = -ENOMEM;
    goto nomembuf;
  }

  memcpy(buf, usb_msg, PANDA_USB_TX_BUFF_SIZE);

  usb_fill_bulk_urb(urb, priv->priv_dev->udev,
		    usb_sndbulkpipe(priv->priv_dev->udev, 3), buf,
		    PANDA_USB_TX_BUFF_SIZE, panda_usb_write_bulk_callback,
		    ctx);

  urb->transfer_flags |= URB_NO_TRANSFER_DMA_MAP;
  usb_anchor_urb(urb, &priv->tx_submitted);

  err = usb_submit_urb(urb, GFP_ATOMIC);
  if (unlikely(err))
    goto failed;

  /* Release our reference to this URB, the USB core will eventually free it entirely. */
  usb_free_urb(urb);

  return 0;

 failed:
  usb_unanchor_urb(urb);
  usb_free_coherent(priv->priv_dev->udev, PANDA_USB_TX_BUFF_SIZE, buf, urb->transfer_dma);

  if (err == -ENODEV)
    netif_device_detach(priv->netdev);
  else
    netdev_warn(priv->netdev, "failed tx_urb %d\n", err);

 nomembuf:
  usb_free_urb(urb);

  return err;
}

static void panda_usb_process_can_rx(struct panda_dev_priv *priv_dev,
				     struct panda_usb_can_msg *msg)
{
  struct can_frame *cf;
  struct sk_buff *skb;
  int bus_num;
  struct panda_inf_priv *priv_inf;
  struct net_device_stats *stats;

  bus_num = (msg->bus_dat_len >> 4) & 0xf;
  priv_inf = panda_get_inf_from_bus_id(priv_dev, bus_num);
  if(!priv_inf){
    printk("Got something on an unused interface %d\n", bus_num);
    return;
  }
  printk("Recv bus %d\n", bus_num);

  stats = &priv_inf->netdev->stats;
  //u16 sid;

  if (!netif_device_present(priv_inf->netdev))
    return;

  skb = alloc_can_skb(priv_inf->netdev, &cf);
  if (!skb)
    return;

  if(msg->rir & PANDA_CAN_EXTENDED){
    cf->can_id = (msg->rir >> 3) | CAN_EFF_FLAG;
  }else{
    cf->can_id = (msg->rir >> 21);
  }

  // TODO: Handle Remote Frames
  //if (msg->dlc & MCBA_DLC_RTR_MASK)
  //  cf->can_id |= CAN_RTR_FLAG;

  cf->can_dlc = get_can_dlc(msg->bus_dat_len & PANDA_DLC_MASK);

  memcpy(cf->data, msg->data, cf->can_dlc);

  stats->rx_packets++;
  stats->rx_bytes += cf->can_dlc;

  netif_rx(skb);
}

static void panda_usb_read_int_callback(struct urb *urb)
{
  struct panda_dev_priv *priv_dev = urb->context;
  int retval;
  int pos = 0;
  int inf_num;

  switch (urb->status) {
  case 0: /* success */
    break;
  case -ENOENT:
  case -ESHUTDOWN:
    return;
  default:
    dev_info(priv_dev->dev, "Rx URB aborted (%d)\n", urb->status);
    goto resubmit_urb;
  }

  while (pos < urb->actual_length) {
    struct panda_usb_can_msg *msg;

    if (pos + sizeof(struct panda_usb_can_msg) > urb->actual_length) {
      dev_err(priv_dev->dev, "format error\n");
      break;
    }

    msg = (struct panda_usb_can_msg *)(urb->transfer_buffer + pos);

    panda_usb_process_can_rx(priv_dev, msg);

    pos += sizeof(struct panda_usb_can_msg);
  }

 resubmit_urb:
  usb_fill_int_urb(urb, priv_dev->udev,
		    usb_rcvintpipe(priv_dev->udev, 1),
		    urb->transfer_buffer, PANDA_USB_RX_BUFF_SIZE,
		    panda_usb_read_int_callback, priv_dev, 5);

  retval = usb_submit_urb(urb, GFP_ATOMIC);

  if (retval == -ENODEV){
    for(inf_num = 0; inf_num < PANDA_NUM_CAN_INTERFACES; inf_num++)
      if(priv_dev->interfaces[inf_num])
	netif_device_detach(priv_dev->interfaces[inf_num]->netdev);
  }else if (retval)
    dev_err(priv_dev->dev, "failed resubmitting read bulk urb: %d\n", retval);
}


static int panda_usb_start(struct panda_dev_priv *priv_dev)
{
  int err;
  struct urb *urb = NULL;
  u8 *buf;
  int inf_num;

  for(inf_num = 0; inf_num < PANDA_NUM_CAN_INTERFACES; inf_num++)
    panda_init_ctx(priv_dev->interfaces[inf_num]);

  err = usb_set_interface(priv_dev->udev, 0, 1);
  if (err) {
    dev_err(priv_dev->dev, "Can not set alternate setting to 1, error: %i", err);
    return err;
  }

  /* create a URB, and a buffer for it */
  urb = usb_alloc_urb(0, GFP_KERNEL);
  if (!urb) {
    return -ENOMEM;
  }

  buf = usb_alloc_coherent(priv_dev->udev, PANDA_USB_RX_BUFF_SIZE,
			   GFP_KERNEL, &urb->transfer_dma);
  if (!buf) {
    dev_err(priv_dev->dev, "No memory left for USB buffer\n");
    usb_free_urb(urb);
    return -ENOMEM;
  }

  usb_fill_int_urb(urb, priv_dev->udev,
                   usb_rcvintpipe(priv_dev->udev, 1),
                   buf, PANDA_USB_RX_BUFF_SIZE,
                   panda_usb_read_int_callback, priv_dev, 5);
  urb->transfer_flags |= URB_NO_TRANSFER_DMA_MAP;

  usb_anchor_urb(urb, &priv_dev->rx_submitted);

  err = usb_submit_urb(urb, GFP_KERNEL);
  if (err) {
  usb_unanchor_urb(urb);
    usb_free_coherent(priv_dev->udev, PANDA_USB_RX_BUFF_SIZE,
		      buf, urb->transfer_dma);
    usb_free_urb(urb);
    dev_err(priv_dev->dev, "Failed in start, while submitting urb.\n");
    return err;
  }

  /* Drop reference, USB core will take care of freeing it */
  usb_free_urb(urb);


  return 0;
}

/* Open USB device */
static int panda_usb_open(struct net_device *netdev)
{
  struct panda_inf_priv *priv = netdev_priv(netdev);
  int err;

  /* common open */
  err = open_candev(netdev);
  if (err)
    return err;

  //priv->can_speed_check = true;
  priv->can.state = CAN_STATE_ERROR_ACTIVE;

  netif_start_queue(netdev);

  return 0;
}

/* Close USB device */
static int panda_usb_close(struct net_device *netdev)
{
  struct panda_inf_priv *priv = netdev_priv(netdev);

  priv->can.state = CAN_STATE_STOPPED;

  netif_stop_queue(netdev);

  /* Stop polling */
  panda_urb_unlink(priv);

  close_candev(netdev);

  return 0;
}

static netdev_tx_t panda_usb_start_xmit(struct sk_buff *skb,
					struct net_device *netdev)
{
  struct panda_inf_priv *priv_inf = netdev_priv(netdev);
  struct can_frame *cf = (struct can_frame *)skb->data;
  struct panda_usb_ctx *ctx = NULL;
  struct net_device_stats *stats = &priv_inf->netdev->stats;
  int err;
  struct panda_usb_can_msg usb_msg = {};
  int bus = priv_inf->mcu_can_ifnum;

  if (can_dropped_invalid_skb(netdev, skb)){
    printk("Invalid CAN packet");
    return NETDEV_TX_OK;
  }

  ctx = panda_usb_get_free_ctx(priv_inf, cf);

  //Warning: cargo cult. Can't tell what this is for, but it is
  //everywhere and encouraged in the documentation.
  can_put_echo_skb(skb, priv_inf->netdev, ctx->ndx);

  if(cf->can_id & CAN_EFF_FLAG){
    usb_msg.rir = cpu_to_le32(((cf->can_id & 0x1FFFFFFF) << 3) |
			      PANDA_CAN_TRANSMIT | PANDA_CAN_EXTENDED);
  }else{
    usb_msg.rir = cpu_to_le32(((cf->can_id & 0x7FF) << 21) | PANDA_CAN_TRANSMIT);
  }
  usb_msg.bus_dat_len = cpu_to_le32((cf->can_dlc & 0x0F) | (bus << 4));

  memcpy(usb_msg.data, cf->data, cf->can_dlc);

  //TODO Handle Remote Frames
  //if (cf->can_id & CAN_RTR_FLAG)
  //  usb_msg.dlc |= PANDA_DLC_RTR_MASK;

  netdev_err(netdev, "Received data from socket. canid: %x; len: %d\n", cf->can_id, cf->can_dlc);

  err = panda_usb_xmit(priv_inf, &usb_msg, ctx);
  if (err)
    goto xmit_failed;

  return NETDEV_TX_OK;

 xmit_failed:
  can_free_echo_skb(priv_inf->netdev, ctx->ndx);
  panda_usb_free_ctx(ctx);
  dev_kfree_skb(skb);
  stats->tx_dropped++;

  return NETDEV_TX_OK;
}

static const struct net_device_ops panda_netdev_ops = {
  .ndo_open = panda_usb_open,
  .ndo_stop = panda_usb_close,
  .ndo_start_xmit = panda_usb_start_xmit,
};

static int panda_usb_probe(struct usb_interface *intf,
			   const struct usb_device_id *id)
{
  struct net_device *netdev;
  struct panda_inf_priv *priv_inf;
  int err = -ENOMEM;
  int inf_num;
  struct panda_dev_priv *priv_dev;
  struct usb_device *usbdev = interface_to_usbdev(intf);

  priv_dev = kzalloc(sizeof(struct panda_dev_priv), GFP_KERNEL);
  if (!priv_dev) {
    dev_err(&intf->dev, "Couldn't alloc priv_dev\n");
    return -ENOMEM;
  }
  priv_dev->udev = usbdev;
  priv_dev->dev = &intf->dev;
  usb_set_intfdata(intf, priv_dev);

  ////// Interface privs
  for(inf_num = 0; inf_num < PANDA_NUM_CAN_INTERFACES; inf_num++){
    netdev = alloc_candev(sizeof(struct panda_inf_priv), PANDA_MAX_TX_URBS);
    if (!netdev) {
      dev_err(&intf->dev, "Couldn't alloc candev\n");
      goto cleanup_candev;
    }
    netdev->netdev_ops = &panda_netdev_ops;
    netdev->flags |= IFF_ECHO; /* we support local echo */

    priv_inf = netdev_priv(netdev);
    priv_inf->netdev = netdev;
    priv_inf->priv_dev = priv_dev;
    priv_inf->interface_num = inf_num;
    priv_inf->mcu_can_ifnum = can_numbering[inf_num];

    init_usb_anchor(&priv_dev->rx_submitted);
    init_usb_anchor(&priv_inf->tx_submitted);

    /* Init CAN device */
    priv_inf->can.state = CAN_STATE_STOPPED;
    priv_inf->can.bittiming.bitrate = PANDA_BITRATE;

    SET_NETDEV_DEV(netdev, &intf->dev);

    err = register_candev(netdev);
    if (err) {
      netdev_err(netdev, "couldn't register PANDA CAN device: %d\n", err);
      free_candev(priv_inf->netdev);
      goto cleanup_candev;
    }

    priv_dev->interfaces[inf_num] = priv_inf;
  }

  err = panda_usb_start(priv_dev);
  if (err) {
    dev_err(&intf->dev, "Failed to initialize Comma.ai Panda CAN controller\n");
    goto cleanup_candev;
  }

  err = panda_set_output_enable(priv_inf, true);
  if (err) {
    dev_info(&intf->dev, "Failed to initialize send enable message to Panda.\n");
    goto cleanup_candev;
  }

  dev_info(&intf->dev, "Comma.ai Panda CAN controller connected\n");

  return 0;

 cleanup_candev:
  for(inf_num = 0; inf_num < PANDA_NUM_CAN_INTERFACES; inf_num++){
    priv_inf = priv_dev->interfaces[inf_num];
    if(priv_inf){
      unregister_candev(priv_inf->netdev);
      free_candev(priv_inf->netdev);
    }else
      break;
  }

  kfree(priv_dev);

  return err;
}

/* Called by the usb core when driver is unloaded or device is removed */
static void panda_usb_disconnect(struct usb_interface *intf)
{
  struct panda_dev_priv *priv_dev = usb_get_intfdata(intf);
  struct panda_inf_priv *priv_inf;
  int inf_num;

  usb_set_intfdata(intf, NULL);

  for(inf_num = 0; inf_num < PANDA_NUM_CAN_INTERFACES; inf_num++){
    priv_inf = priv_dev->interfaces[inf_num];
    if(priv_inf){
      netdev_info(priv_inf->netdev, "device disconnected\n");
      unregister_candev(priv_inf->netdev);
      free_candev(priv_inf->netdev);
    }else
      break;
  }

  panda_urb_unlink(priv_inf);
  kfree(priv_dev);
}

static struct usb_driver panda_usb_driver = {
  .name = PANDA_MODULE_NAME,
  .probe = panda_usb_probe,
  .disconnect = panda_usb_disconnect,
  .id_table = panda_usb_table,
};

module_usb_driver(panda_usb_driver);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Jessy Diamond Exum <jessy.diamondman@gmail.com>");
MODULE_DESCRIPTION("SocketCAN driver for Comma.ai's Panda Adapter.");
MODULE_VERSION("0.1");
