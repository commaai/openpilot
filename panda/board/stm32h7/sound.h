#define SOUND_RX_BUF_SIZE 1000U
#define SOUND_TX_BUF_SIZE (SOUND_RX_BUF_SIZE/2U)
#define MIC_RX_BUF_SIZE 512U
#define MIC_TX_BUF_SIZE (MIC_RX_BUF_SIZE * 2U)
__attribute__((section(".sram4"))) static uint16_t sound_rx_buf[2][SOUND_RX_BUF_SIZE];
__attribute__((section(".sram4"))) static uint16_t sound_tx_buf[2][SOUND_TX_BUF_SIZE];
__attribute__((section(".sram4"))) static uint32_t mic_rx_buf[2][MIC_RX_BUF_SIZE];

#define SOUND_IDLE_TIMEOUT 4U
static uint8_t sound_idle_count;
static uint8_t mic_idle_count;

void sound_tick(void) {
  if (sound_idle_count > 0U) {
    sound_idle_count--;
    if (sound_idle_count == 0U) {
      current_board->set_amp_enabled(false);
      register_clear_bits(&DMA1_Stream1->CR, DMA_SxCR_EN);
    }
  }

  if (mic_idle_count > 0U) {
    mic_idle_count--;
    if (mic_idle_count == 0U) {
      register_clear_bits(&DFSDM1_Channel0->CHCFGR1, DFSDM_CHCFGR1_DFSDMEN);
    }
  }
}

// Recording processing
static void DMA1_Stream0_IRQ_Handler(void) {
  __attribute__((section(".sram4"))) static uint16_t tx_buf[MIC_TX_BUF_SIZE];

  DMA1->LIFCR |= 0x7D; // clear flags

  // process samples
  uint8_t buf_idx = (((DMA1_Stream0->CR & DMA_SxCR_CT) >> DMA_SxCR_CT_Pos) == 1U) ? 0U : 1U;
  for (uint16_t i=0U; i < MIC_RX_BUF_SIZE; i++) {
    tx_buf[2U*i] = ((mic_rx_buf[buf_idx][i] >> 16U) & 0xFFFFU);
    tx_buf[(2U*i)+1U] = tx_buf[2U*i];
  }

  BDMA->IFCR |= BDMA_IFCR_CGIF1;
  BDMA_Channel1->CCR &= ~BDMA_CCR_EN;
  register_set(&BDMA_Channel1->CM0AR, (uint32_t) tx_buf, 0xFFFFFFFFU);
  BDMA_Channel1->CNDTR = MIC_TX_BUF_SIZE;
  BDMA_Channel1->CCR |= BDMA_CCR_EN;
}

// Playback processing
static void BDMA_Channel0_IRQ_Handler(void) {
  static uint8_t playback_buf = 0U;

  BDMA->IFCR |= BDMA_IFCR_CGIF0; // clear flag

  uint8_t rx_buf_idx = (((BDMA_Channel0->CCR & BDMA_CCR_CT) >> BDMA_CCR_CT_Pos) == 1U) ? 0U : 1U;
  playback_buf = 1U - playback_buf;

  // wait until we're done playing the current buf
  for (uint32_t timeout_counter = 100000U; timeout_counter > 0U; timeout_counter--){
    uint8_t playing_buf = (DMA1_Stream1->CR & DMA_SxCR_CT) >> DMA_SxCR_CT_Pos;
    if ((playing_buf != playback_buf) || (!(DMA1_Stream1->CR & DMA_SxCR_EN))) {
      break;
    }
  }

  // process samples (shift to 12b and bias to be unsigned)
  bool sound_playing = false;
  for (uint16_t i=0U; i < SOUND_RX_BUF_SIZE; i += 2U) {
    // since we are playing mono and receiving stereo, we take every other sample
    sound_tx_buf[playback_buf][i/2U] = ((sound_rx_buf[rx_buf_idx][i] + (1UL << 14)) >> 3);
    if (sound_rx_buf[rx_buf_idx][i] > 0U) {
      sound_playing = true;
    }
  }

  // manage amp state
  if (sound_playing) {
    if (sound_idle_count == 0U) {
      current_board->set_amp_enabled(true);

      // empty the other buf and start playing that
      for (uint16_t i=0U; i < SOUND_TX_BUF_SIZE; i++) {
        sound_tx_buf[1U - playback_buf][i] = (1UL << 11);
      }
      register_clear_bits(&DMA1_Stream1->CR, DMA_SxCR_EN);
      DMA1_Stream1->CR = (DMA1_Stream1->CR & ~DMA_SxCR_CT_Msk) | ((1UL - playback_buf) << DMA_SxCR_CT_Pos);
      register_set_bits(&DMA1_Stream1->CR, DMA_SxCR_EN);
    }
    sound_idle_count = SOUND_IDLE_TIMEOUT;
  }

  // manage mic state
  if (mic_idle_count == 0U) {
    register_set_bits(&DFSDM1_Channel0->CHCFGR1, DFSDM_CHCFGR1_DFSDMEN);
    DFSDM1_Filter0->FLTCR1 |= DFSDM_FLTCR1_RSWSTART;
  }
  mic_idle_count = SOUND_IDLE_TIMEOUT;

  sound_tick();
}

void sound_init(void) {
  REGISTER_INTERRUPT(BDMA_Channel0_IRQn, BDMA_Channel0_IRQ_Handler, 128U, FAULT_INTERRUPT_RATE_SOUND_DMA)
  REGISTER_INTERRUPT(DMA1_Stream0_IRQn, DMA1_Stream0_IRQ_Handler, 128U, FAULT_INTERRUPT_RATE_SOUND_DMA)

  // *** tmp, remove soon ***
  static const uint8_t olds[][12] = {
    {0x44, 0x00, 0x10, 0x00, 0x19, 0x51, 0x32, 0x34, 0x39, 0x37, 0x37, 0x30},
    {0x14, 0x00, 0x13, 0x00, 0x18, 0x51, 0x32, 0x34, 0x39, 0x37, 0x37, 0x30},
    {0x04, 0x00, 0x30, 0x00, 0x18, 0x51, 0x32, 0x34, 0x39, 0x37, 0x37, 0x30},
    {0x2f, 0x00, 0x14, 0x00, 0x18, 0x51, 0x32, 0x34, 0x39, 0x37, 0x37, 0x30},
    {0x1e, 0x00, 0x2f, 0x00, 0x18, 0x51, 0x32, 0x34, 0x39, 0x37, 0x37, 0x30},
    {0x26, 0x00, 0x15, 0x00, 0x19, 0x51, 0x32, 0x34, 0x39, 0x37, 0x37, 0x30},
    {0x35, 0x00, 0x32, 0x00, 0x18, 0x51, 0x32, 0x34, 0x39, 0x37, 0x37, 0x30},
    {0x37, 0x00, 0x2f, 0x00, 0x18, 0x51, 0x32, 0x34, 0x39, 0x37, 0x37, 0x30},
  };
  bool is_old = false;
  for (uint8_t i = 0U; i < (sizeof(olds) / sizeof(olds[0])); i++) {
    is_old |= (memcmp(olds[i], ((uint8_t *)UID_BASE), 12) == 0);
  }
  // *** tmp end ***

  // Init DAC
  DAC1->DHR12R1 = (1UL << 11);
  if (!is_old) DAC1->DHR12R2 = (1UL << 11);
  register_set(&DAC1->MCR, 0U, 0xFFFFFFFFU);
  register_set(&DAC1->CR, DAC_CR_TEN1 | (4U << DAC_CR_TSEL1_Pos) | DAC_CR_DMAEN1, 0xFFFFFFFFU);
  if (is_old) {
    register_set_bits(&DAC1->CR, DAC_CR_EN1);
  } else {
    register_set_bits(&DAC1->CR, DAC_CR_EN1 | DAC_CR_EN2);
  }

  // Setup DMAMUX (DAC_CH1_DMA as input)
  register_set(&DMAMUX1_Channel1->CCR, 67U, DMAMUX_CxCR_DMAREQ_ID_Msk);

  // Setup DMA
  register_set(&DMA1_Stream1->PAR, (uint32_t) &(DAC1->DHR12R1), 0xFFFFFFFFU);
  register_set(&DMA1_Stream1->M0AR, (uint32_t) sound_tx_buf[0], 0xFFFFFFFFU);
  register_set(&DMA1_Stream1->M1AR, (uint32_t) sound_tx_buf[1], 0xFFFFFFFFU);
  register_set(&DMA1_Stream1->FCR, 0U, 0x00000083U);
  DMA1_Stream1->NDTR = SOUND_TX_BUF_SIZE;
  DMA1_Stream1->CR = DMA_SxCR_DBM | (0b11UL << DMA_SxCR_PL_Pos) | (0b01UL << DMA_SxCR_MSIZE_Pos) | (0b01UL << DMA_SxCR_PSIZE_Pos) | DMA_SxCR_MINC | (1U << DMA_SxCR_DIR_Pos);

  // Init trigger timer (little slower than 48kHz, pulled in sync by SAI4_FS_B)
  register_set(&TIM5->PSC, 2600U, 0xFFFFU);
  register_set(&TIM5->ARR, 100U, 0xFFFFFFFFU); // not important
  register_set(&TIM5->AF1, (0b0010UL << TIM5_AF1_ETRSEL_Pos), TIM5_AF1_ETRSEL_Msk);
  register_set(&TIM5->CR2, (0b010U << TIM_CR2_MMS_Pos), TIM_CR2_MMS_Msk);
  register_set(&TIM5->SMCR, TIM_SMCR_ECE | (0b00111UL << TIM_SMCR_TS_Pos)| (0b0100UL << TIM_SMCR_SMS_Pos), 0x31FFF7U);
  TIM5->CNT = 0U; TIM5->SR = 0U;
  TIM5->CR1 |= TIM_CR1_CEN;

  // sync both SAIs
  register_set(&SAI4->GCR, (0b10UL << SAI_GCR_SYNCOUT_Pos), SAI_GCR_SYNCIN_Msk | SAI_GCR_SYNCOUT_Msk);

  // stereo audio in
  register_set(&SAI4_Block_B->CR1, SAI_xCR1_DMAEN | (0b00UL << SAI_xCR1_SYNCEN_Pos) | (0b100U << SAI_xCR1_DS_Pos) | (0b11U << SAI_xCR1_MODE_Pos), 0x0FFB3FEFU);
  register_set(&SAI4_Block_B->CR2, (0b001U << SAI_xCR2_FTH_Pos), 0xFFFBU);
  register_set(&SAI4_Block_B->FRCR, (31U << SAI_xFRCR_FRL_Pos), 0x7FFFFU);
  register_set(&SAI4_Block_B->SLOTR, (0b11UL << SAI_xSLOTR_SLOTEN_Pos) | (1UL << SAI_xSLOTR_NBSLOT_Pos) | (0b01UL << SAI_xSLOTR_SLOTSZ_Pos), 0xFFFF0FDFU); // NBSLOT definition is vague

  // init sound DMA (SAI4_B -> memory, double buffers)
  register_set(&BDMA_Channel0->CPAR, (uint32_t) &(SAI4_Block_B->DR), 0xFFFFFFFFU);
  register_set(&BDMA_Channel0->CM0AR, (uint32_t) sound_rx_buf[0], 0xFFFFFFFFU);
  register_set(&BDMA_Channel0->CM1AR, (uint32_t) sound_rx_buf[1], 0xFFFFFFFFU);
  BDMA_Channel0->CNDTR = SOUND_RX_BUF_SIZE;
  register_set(&BDMA_Channel0->CCR, BDMA_CCR_DBM | (0b01UL << BDMA_CCR_MSIZE_Pos) | (0b01UL << BDMA_CCR_PSIZE_Pos) | BDMA_CCR_MINC | BDMA_CCR_CIRC | BDMA_CCR_TCIE, 0xFFFFU);
  register_set(&DMAMUX2_Channel0->CCR, 16U, DMAMUX_CxCR_DMAREQ_ID_Msk); // SAI4_B_DMA
  register_set_bits(&BDMA_Channel0->CCR, BDMA_CCR_EN);

  // mic output
  register_set(&SAI4_Block_A->CR1, SAI_xCR1_DMAEN | (0b01UL << SAI_xCR1_SYNCEN_Pos) | (0b100UL << SAI_xCR1_DS_Pos) | (0b10UL << SAI_xCR1_MODE_Pos), 0x0FFB3FEFU);
  register_set(&SAI4_Block_A->CR2, 0U, 0xFFFBU);
  register_set(&SAI4_Block_A->FRCR, (31U << SAI_xFRCR_FRL_Pos), 0x7FFFFU);
  register_set(&SAI4_Block_A->SLOTR, (0b11UL << SAI_xSLOTR_SLOTEN_Pos) | (1UL << SAI_xSLOTR_NBSLOT_Pos) | (0b01U << SAI_xSLOTR_SLOTSZ_Pos), 0xFFFF0FDFU); // NBSLOT definition is vague

  // init DFSDM for PDM mic
  register_set(&DFSDM1_Channel0->CHCFGR1, (76UL << DFSDM_CHCFGR1_CKOUTDIV_Pos) | DFSDM_CHCFGR1_CHEN, 0xC0FFF1EFU); // CH0 controls the clock
  register_set(&DFSDM1_Channel3->CHCFGR1, (0b01UL << DFSDM_CHCFGR1_SPICKSEL_Pos) | (0b00U << DFSDM_CHCFGR1_SITP_Pos) | DFSDM_CHCFGR1_CHEN, 0x0000F1EFU); // SITP determines sample edge
  register_set(&DFSDM1_Filter0->FLTFCR, (0U << DFSDM_FLTFCR_IOSR_Pos) | (64UL << DFSDM_FLTFCR_FOSR_Pos) | (4UL << DFSDM_FLTFCR_FORD_Pos), 0xE3FF00FFU);
  register_set(&DFSDM1_Filter0->FLTCR1, DFSDM_FLTCR1_FAST | (3UL << DFSDM_FLTCR1_RCH_Pos) | DFSDM_FLTCR1_RDMAEN | DFSDM_FLTCR1_RCONT | DFSDM_FLTCR1_DFEN, 0x672E7F3BU);

  // DMA (DFSDM1 -> memory)
  register_set(&DMA1_Stream0->PAR, (uint32_t) &DFSDM1_Filter0->FLTRDATAR, 0xFFFFFFFFU);
  register_set(&DMA1_Stream0->M0AR, (uint32_t)mic_rx_buf[0], 0xFFFFFFFFU);
  register_set(&DMA1_Stream0->M1AR, (uint32_t)mic_rx_buf[1], 0xFFFFFFFFU);
  DMA1_Stream0->NDTR = MIC_RX_BUF_SIZE;
  register_set(&DMA1_Stream0->CR, DMA_SxCR_DBM | (0b10UL << DMA_SxCR_MSIZE_Pos) | (0b10UL << DMA_SxCR_PSIZE_Pos) | DMA_SxCR_MINC | DMA_SxCR_CIRC | DMA_SxCR_TCIE, 0x01F7FFFFU);
  register_set(&DMAMUX1_Channel0->CCR, 101U, DMAMUX_CxCR_DMAREQ_ID_Msk); // DFSDM1_DMA0
  register_set_bits(&DMA1_Stream0->CR, DMA_SxCR_EN);
  DMA1->LIFCR |= 0x7D; // clear flags

  // DMA (memory -> SAI4)
  register_set(&BDMA_Channel1->CPAR, (uint32_t) &(SAI4_Block_A->DR), 0xFFFFFFFFU);
  register_set(&BDMA_Channel1->CCR, (0b01UL << BDMA_CCR_MSIZE_Pos) | (0b01UL << BDMA_CCR_PSIZE_Pos) | BDMA_CCR_MINC | (0b1U << BDMA_CCR_DIR_Pos), 0xFFFEU);
  register_set(&DMAMUX2_Channel1->CCR, 15U, DMAMUX_CxCR_DMAREQ_ID_Msk); // SAI4_A_DMA

  // enable all initted blocks
  register_set_bits(&SAI4_Block_A->CR1, SAI_xCR1_SAIEN);
  register_set_bits(&SAI4_Block_B->CR1, SAI_xCR1_SAIEN);
  NVIC_EnableIRQ(BDMA_Channel0_IRQn);
  NVIC_EnableIRQ(DMA1_Stream0_IRQn);
}
