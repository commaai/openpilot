#!/usr/bin/env bash
# adapted from https://github.com/mlcommons/training/blob/4bdf5c8ed218ad76565a2ba1ac27c919ccc6d689/stable_diffusion/README.md

# setup dirs

DATA=/raid/datasets/stable_diffusion

LAION=$DATA/laion-400m/webdataset-moments-filtered 
COCO=$DATA/coco2014
mkdir -p $LAION $COCO

CKPT=/raid/weights/stable_diffusion
mkdir -p $CKPT/clip $CKPT/sd $CKPT/inception

# download data

# if rclone isn't installed system-wide / in your PATH, put the executable path in quotes below
#RCLONE=""
RCLONE="rclone"

## VAE-encoded image latents, from 6.1M image subset of laion-400m
## about 1 TB for whole download
$RCLONE config create mlc-training s3 provider=Cloudflare access_key_id=76ea42eadb867e854061a1806220ee1e secret_access_key=a53625c4d45e3ca8ac0df8a353ea3a41ffc3292aa25259addd8b7dc5a6ce2936 endpoint=c2686074cb2caf5cbaf6d134bdba8b47.r2.cloudflarestorage.com
$RCLONE copy mlc-training:mlcommons-training-wg-public/stable_diffusion/datasets/laion-400m/moments-webdataset-filtered/ ${LAION} --include="*.tar" -P
$RCLONE copy mlc-training:mlcommons-training-wg-public/stable_diffusion/datasets/laion-400m/moments-webdataset-filtered/sha512sums.txt ${LAION} -P
cd $LAION && grep -E '\.tar$' sha512sums.txt | sha512sum -c --quiet - && \
  echo "All .tar files verified" || { echo "Checksum failure when validating downloaded Laion moments"; exit 1; }

## prompts and FID statistics from 30k image subset of coco2014
## 33 MB
$RCLONE config create mlc-training s3 provider=Cloudflare access_key_id=76ea42eadb867e854061a1806220ee1e secret_access_key=a53625c4d45e3ca8ac0df8a353ea3a41ffc3292aa25259addd8b7dc5a6ce2936 endpoint=c2686074cb2caf5cbaf6d134bdba8b47.r2.cloudflarestorage.com
$RCLONE copy mlc-training:mlcommons-training-wg-public/stable_diffusion/datasets/coco2014/val2014_30k.tsv ${COCO} -P

$RCLONE config create mlc-training s3 provider=Cloudflare access_key_id=76ea42eadb867e854061a1806220ee1e secret_access_key=a53625c4d45e3ca8ac0df8a353ea3a41ffc3292aa25259addd8b7dc5a6ce2936 endpoint=c2686074cb2caf5cbaf6d134bdba8b47.r2.cloudflarestorage.com
$RCLONE copy mlc-training:mlcommons-training-wg-public/stable_diffusion/datasets/coco2014/val2014_30k_stats.npz ${COCO} -P

# download checkpoints

## clip (needed for text and vision encoders for validation)
CLIP_WEIGHTS_URL="https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/resolve/main/open_clip_pytorch_model.bin"
CLIP_WEIGHTS_SHA256="9a78ef8e8c73fd0df621682e7a8e8eb36c6916cb3c16b291a082ecd52ab79cc4"
CLIP_CONFIG_URL="https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/raw/main/open_clip_config.json"
wget -N -P ${CKPT}/clip ${CLIP_WEIGHTS_URL}
wget -N -P ${CKPT}/clip ${CLIP_CONFIG_URL}
echo "${CLIP_WEIGHTS_SHA256}  ${CKPT}/clip/open_clip_pytorch_model.bin"                    | sha256sum -c

## sd (needed for latent->image decoder for validation, also has clip text encoder for training)
SD_WEIGHTS_URL='https://huggingface.co/stabilityai/stable-diffusion-2-base/resolve/main/512-base-ema.ckpt'
SD_WEIGHTS_SHA256="d635794c1fedfdfa261e065370bea59c651fc9bfa65dc6d67ad29e11869a1824"
wget -N -P ${CKPT}/sd ${SD_WEIGHTS_URL}
echo "${SD_WEIGHTS_SHA256}  ${CKPT}/sd/512-base-ema.ckpt"                    | sha256sum -c

## inception (needed for validation)
FID_WEIGHTS_URL='https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth'
FID_WEIGHTS_SHA1="bd836944fd6db519dfd8d924aa457f5b3c8357ff"
wget -N -P ${CKPT}/inception ${FID_WEIGHTS_URL}
echo "${FID_WEIGHTS_SHA1}  ${CKPT}/inception/pt_inception-2015-12-05-6726825d.pth"                    | sha1sum -c