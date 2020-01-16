#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <csignal>
#include <unistd.h>
#include <semaphore.h>
#include <time.h>

#include <pthread.h>
#include <capnp/serialize.h>

#include "cereal/gen/cpp/log.capnp.h"

#include "messaging.hpp"
#include "common/timing.h"
#include "common/util.h"
#include "common/swaglog.h"

#include "libdiag.h"

#define NV_GNSS_OEM_FEATURE_MASK 7165
# define NV_GNSS_OEM_FEATURE_MASK_OEMDRE 1
#define NV_CGPS_DPO_CONTROL 5596

#define DIAG_NV_READ_F 38
#define DIAG_NV_WRITE_F 39

#define DIAG_SUBSYS_CMD 75
#define DIAG_SUBSYS_CMD_VER_2 128

#define DIAG_SUBSYS_GPS 13
#define DIAG_SUBSYS_FS 19

#define CGPS_DIAG_PDAPI_CMD 100
#define CGPS_OEM_CONTROL 202

#define GPSDIAG_OEMFEATURE_DRE 1
#define GPSDIAG_OEM_DRE_ON 1

#define FEATURE_OEMDRE_NOT_SUPPORTED 1
#define FEATURE_OEMDRE_ON 2
#define FEATURE_OEMDRE_ALREADY_ON 4

#define TM_DIAG_NAV_CONFIG_CMD 0x6E

#define EFS2_DIAG_SYNC_NO_WAIT 48

struct __attribute__((packed)) NvPacket {
  uint8_t cmd_code;
  uint16_t nv_id;
  uint8_t data[128];
  uint16_t status;
};

enum NvStatus {
  NV_DONE,
  NV_BUSY,
  NV_FULL,
  NV_FAIL,
  NV_NOTACTIVE,
  NV_BADPARAM,
  NV_READONLY,
  NV_BADRG,
  NV_NOMEM,
  NV_NOTALLOC,
};

struct __attribute__((packed)) Efs2DiagSyncReq {
  uint8_t cmd_code;
  uint8_t subsys_id;
  uint16_t subsys_cmd_code;
  uint16_t sequence_num;
  char path[8];
};


struct __attribute__((packed)) Efs2DiagSyncResp {
  uint8_t cmd_code;
  uint8_t subsys_id;
  uint16_t subsys_cmd_code;
  uint16_t sequence_num;
  uint32_t sync_token;
  int32_t diag_errno;
};



struct __attribute__((packed)) GpsOemControlReq {
  uint8_t cmd_code;
  uint8_t subsys_id;
  uint16_t subsys_cmd_code;
  uint8_t gps_cmd_code;
  uint8_t version;

  uint32_t oem_feature;
  uint32_t oem_command;
  uint32_t reserved[2];
};


struct __attribute__((packed)) GpsOemControlResp {
  uint8_t cmd_code;
  uint8_t subsys_id;
  uint16_t subsys_cmd_code;
  uint8_t gps_cmd_code;
  uint8_t version;

  uint32_t oem_feature;
  uint32_t oem_command;
  uint32_t resp_result;
  uint32_t reserved[2];
};

struct __attribute__((packed)) GpsNavConfigReq {
  uint8_t cmd_code;
  uint8_t subsys_id;
  uint16_t subsys_cmd_code;
  uint32_t subsys_status;
  uint16_t subsys_delayed_resp_id;
  uint16_t subsys_rsp_cnt;

  uint8_t desired_config;
};

struct __attribute__((packed)) GpsNavConfigResp {
  uint8_t cmd_code;
  uint8_t subsys_id;
  uint16_t subsys_cmd_code;
  uint32_t subsys_status;
  uint16_t subsys_delayed_resp_id;
  uint16_t subsys_rsp_cnt;

  uint8_t supported_config;
  uint8_t actual_config;
};


#define LOG_GNSS_POSITION_REPORT            0x1476
#define LOG_GNSS_GPS_MEASUREMENT_REPORT     0x1477
#define LOG_GNSS_CLOCK_REPORT               0x1478
#define LOG_GNSS_GLONASS_MEASUREMENT_REPORT 0x1480
#define LOG_GNSS_BDS_MEASUREMENT_REPORT     0x1756
#define LOG_GNSS_GAL_MEASUREMENT_REPORT     0x1886

#define LOG_GNSS_OEMDRE_MEASUREMENMT_REPORT 0x14DE
#define LOG_GNSS_OEMDRE_SVPOLY_REPORT       0x14E1

#define LOG_GNSS_ME_DPO_STATUS              0x1838
#define LOG_GNSS_CD_DB_REPORT               0x147B
#define LOG_GNSS_PRX_RF_HW_STATUS_REPORT    0x147E
#define LOG_CGPS_SLOW_CLOCK_CLIB_REPORT     0x1488
#define LOG_GNSS_CONFIGURATION_STATE        0x1516

struct __attribute__((packed)) log_header_type {
  uint16_t len;
  uint16_t code;
  uint64_t ts;
};



enum SVObservationStates {
  SV_IDLE,
  SV_SEARCH,
  SV_SEACH_VERIFY,
  SV_BIT_EDGE,
  SV_TRACK_VERIFY,
  SV_TRACK,
  SV_RESTART,
  SV_DPO,
  SV_GLO_10ms_BE,
  SV_GLO_10ms_AT,
};

struct __attribute__((packed)) GNSSGpsMeasurementReportv0_SV{
  uint8_t sv_id;
  uint8_t observation_state; // SVObservationStates
  uint8_t observations;
  uint8_t good_observations;
  uint16_t parity_error_count;
  uint8_t filter_stages;
  uint16_t carrier_noise;
  int16_t latency;
  uint8_t predetect_interval;
  uint16_t postdetections;
  uint32_t unfiltered_measurement_integral;
  float unfiltered_measurement_fraction;
  float unfiltered_time_uncertainty;
  float unfiltered_speed;
  float unfiltered_speed_uncertainty;
  uint32_t measurement_status;
  uint8_t misc_status;
  uint32_t multipath_estimate;
  float azimuth;
  float elevation;
  int32_t carrier_phase_cycles_integral;
  uint16_t carrier_phase_cycles_fraction;
  float fine_speed;
  float fine_speed_uncertainty;
  uint8_t cycle_slip_count;
  uint32_t pad;
};

_Static_assert(sizeof(GNSSGpsMeasurementReportv0_SV) == 70, "error");

struct __attribute__((packed)) GNSSGpsMeasurementReportv0{
  log_header_type header;
  uint8_t version;
  uint32_t f_count;
  uint16_t week;
  uint32_t milliseconds;
  float time_bias;
  float clock_time_uncertainty;
  float clock_frequency_bias;
  float clock_frequency_uncertainty;
  uint8_t sv_count;
  GNSSGpsMeasurementReportv0_SV sv[];
};



struct __attribute__((packed)) GNSSGlonassMeasurementReportv0_SV {
  uint8_t sv_id;
  int8_t frequency_index;
  uint8_t observation_state; // SVObservationStates
  uint8_t observations;
  uint8_t good_observations;
  uint8_t hemming_error_count;
  uint8_t filter_stages;
  uint16_t carrier_noise;
  int16_t latency;
  uint8_t predetect_interval;
  uint16_t postdetections;
  uint32_t unfiltered_measurement_integral;
  float unfiltered_measurement_fraction;
  float unfiltered_time_uncertainty;
  float unfiltered_speed;
  float unfiltered_speed_uncertainty;
  uint32_t measurement_status;
  uint8_t misc_status;
  uint32_t multipath_estimate;
  float azimuth;
  float elevation;
  int32_t carrier_phase_cycles_integral;
  uint16_t carrier_phase_cycles_fraction;
  float fine_speed;
  float fine_speed_uncertainty;
  uint8_t cycle_slip_count;
  uint32_t pad;
};

_Static_assert(sizeof(GNSSGlonassMeasurementReportv0_SV) == 70, "error");

struct __attribute__((packed)) GNSSGlonassMeasurementReportv0 {
  log_header_type header;
  uint8_t version;
  uint32_t f_count;
  uint8_t glonass_cycle_number;
  uint16_t glonass_number_of_days;
  uint32_t milliseconds;
  float time_bias;
  float clock_time_uncertainty;
  float clock_frequency_bias;
  float clock_frequency_uncertainty;
  uint8_t sv_count;
  GNSSGlonassMeasurementReportv0_SV sv[];
};


struct __attribute__((packed)) GNSSClockReportv2 {
  log_header_type header;
  uint8_t version;
  uint16_t valid_flags;

  uint32_t f_count;

  uint16_t gps_week;
  uint32_t gps_milliseconds;
  float gps_time_bias;
  float gps_clock_time_uncertainty;
  uint8_t gps_clock_source;

  uint8_t glonass_year;
  uint16_t glonass_day;
  uint32_t glonass_milliseconds;
  float glonass_time_bias;
  float glonass_clock_time_uncertainty;
  uint8_t glonass_clock_source;

  uint16_t bds_week;
  uint32_t bds_milliseconds;
  float bds_time_bias;
  float bds_clock_time_uncertainty;
  uint8_t bds_clock_source;

  uint16_t gal_week;
  uint32_t gal_milliseconds;
  float gal_time_bias;
  float gal_clock_time_uncertainty;
  uint8_t gal_clock_source;

  float clock_frequency_bias;
  float clock_frequency_uncertainty;
  uint8_t frequency_source;
  uint8_t gps_leap_seconds;
  uint8_t gps_leap_seconds_uncertainty;
  uint8_t gps_leap_seconds_source;

  float gps_to_glonass_time_bias_milliseconds;
  float gps_to_glonass_time_bias_milliseconds_uncertainty;
  float gps_to_bds_time_bias_milliseconds;
  float gps_to_bds_time_bias_milliseconds_uncertainty;
  float bds_to_glo_time_bias_milliseconds;
  float bds_to_glo_time_bias_milliseconds_uncertainty;
  float gps_to_gal_time_bias_milliseconds;
  float gps_to_gal_time_bias_milliseconds_uncertainty;
  float gal_to_glo_time_bias_milliseconds;
  float gal_to_glo_time_bias_milliseconds_uncertainty;
  float gal_to_bds_time_bias_milliseconds;
  float gal_to_bds_time_bias_milliseconds_uncertainty;

  uint32_t system_rtc_time;
  uint32_t f_count_offset;
  uint32_t lpm_rtc_count;
  uint32_t clock_resets;

  uint32_t reserved[3];

};


enum GNSSMeasurementSource {
  SOURCE_GPS,
  SOURCE_GLONASS,
  SOURCE_BEIDOU,
};

struct __attribute__((packed)) GNSSOemdreMeasurement {
  uint8_t sv_id;
  uint8_t unkn;
  int8_t glonass_frequency_index;
  uint32_t observation_state;
  uint8_t observations;
  uint8_t good_observations;
  uint8_t filter_stages;
  uint8_t predetect_interval;
  uint8_t cycle_slip_count;
  uint16_t postdetections;

  uint32_t measurement_status;
  uint32_t measurement_status2;

  uint16_t carrier_noise;
  uint16_t rf_loss;
  int16_t latency;

  float filtered_measurement_fraction;
  uint32_t filtered_measurement_integral;
  float filtered_time_uncertainty;
  float filtered_speed;
  float filtered_speed_uncertainty;

  float unfiltered_measurement_fraction;
  uint32_t unfiltered_measurement_integral;
  float unfiltered_time_uncertainty;
  float unfiltered_speed;
  float unfiltered_speed_uncertainty;

  uint8_t multipath_estimate_valid;
  uint32_t multipath_estimate;
  uint8_t direction_valid;
  float azimuth;
  float elevation;
  float doppler_acceleration;
  float fine_speed;
  float fine_speed_uncertainty;

  uint64_t carrier_phase;
  uint32_t f_count;

  uint16_t parity_error_count;
  uint8_t good_parity;

};

_Static_assert(sizeof(GNSSOemdreMeasurement) == 109, "error");

struct __attribute__((packed)) GNSSOemdreMeasurementReportv2 {
  log_header_type header;
  uint8_t version;
  uint8_t reason;
  uint8_t sv_count;
  uint8_t seq_num;
  uint8_t seq_max;
  uint16_t rf_loss;

  uint8_t system_rtc_valid;
  uint32_t f_count;
  uint32_t clock_resets;
  uint64_t system_rtc_time;

  uint8_t gps_leap_seconds;
  uint8_t gps_leap_seconds_uncertainty;
  float gps_to_glonass_time_bias_milliseconds;
  float gps_to_glonass_time_bias_milliseconds_uncertainty;

  uint16_t gps_week;
  uint32_t gps_milliseconds;
  uint32_t gps_time_bias;
  uint32_t gps_clock_time_uncertainty;
  uint8_t gps_clock_source;

  uint8_t glonass_clock_source;
  uint8_t glonass_year;
  uint16_t glonass_day;
  uint32_t glonass_milliseconds;
  float glonass_time_bias;
  float glonass_clock_time_uncertainty;

  float clock_frequency_bias;
  float clock_frequency_uncertainty;
  uint8_t frequency_source;

  uint32_t cdma_clock_info[5];

  uint8_t source;

  GNSSOemdreMeasurement measurements[16];

};

_Static_assert(sizeof(GNSSOemdreMeasurementReportv2) == 1851, "error");


struct __attribute__((packed)) GNSSOemdreSVPolyReportv2 {
  log_header_type header;
  uint8_t version;
  uint16_t sv_id;
  int8_t frequency_index;
  uint8_t flags;
  uint16_t iode;
  double t0;
  double xyz0[3];
  double xyzN[9];
  float other[4];
  float position_uncertainty;
  float iono_delay;
  float iono_dot;
  float sbas_iono_delay;
  float sbas_iono_dot;
  float tropo_delay;
  float elevation;
  float elevation_dot;
  float elevation_uncertainty;
  double velocity_coeff[12];
};

_Static_assert(sizeof(GNSSOemdreSVPolyReportv2) == 271, "error");


static Context *rawgps_context;
static PubSocket *rawgps_publisher;
static int client_id = 0;

static void hexdump(uint8_t* d, size_t len) {
  for (int i=0; i<len; i++) {
    printf("%02x ", d[i]);
  }
  printf("\n");
}


static void parse_measurement_status_common(uint32_t measurement_status,
    cereal::QcomGnss::MeasurementStatus::Builder status) {
  status.setSubMillisecondIsValid(measurement_status & (1 << 0));
  status.setSubBitTimeIsKnown(measurement_status & (1 << 1));
  status.setSatelliteTimeIsKnown(measurement_status & (1 << 2));
  status.setBitEdgeConfirmedFromSignal(measurement_status & (1 << 3));
  status.setMeasuredVelocity(measurement_status & (1 << 4));
  status.setFineOrCoarseVelocity(measurement_status & (1 << 5));
  status.setLockPointValid(measurement_status & (1 << 6));
  status.setLockPointPositive(measurement_status & (1 << 7));

  status.setLastUpdateFromDifference(measurement_status & (1 << 9));
  status.setLastUpdateFromVelocityDifference(measurement_status & (1 << 10));
  status.setStrongIndicationOfCrossCorelation(measurement_status & (1 << 11));
  status.setTentativeMeasurement(measurement_status & (1 << 12));
  status.setMeasurementNotUsable(measurement_status & (1 << 13));
  status.setSirCheckIsNeeded(measurement_status & (1 << 14));
  status.setProbationMode(measurement_status & (1 << 15));

  status.setMultipathIndicator(measurement_status & (1 << 24));
  status.setImdJammingIndicator(measurement_status & (1 << 25));
  status.setLteB13TxJammingIndicator(measurement_status & (1 << 26));
  status.setFreshMeasurementIndicator(measurement_status & (1 << 27));
}

static void handle_log(uint8_t *ptr, int len) {
  assert(len >= sizeof(log_header_type)+1);
  log_header_type* log_header = (log_header_type*)ptr;
  uint8_t* log_data = ptr + sizeof(log_header_type);

#ifdef RAWGPS_TEST
  printf("%04x\n", log_header->code);
#endif

  capnp::MallocMessageBuilder msg;
  cereal::Event::Builder event = msg.initRoot<cereal::Event>();
  event.setLogMonoTime(nanos_since_boot());
  auto qcomGnss = event.initQcomGnss();
  qcomGnss.setLogTs(log_header->ts);

  switch (log_header->code) {

  case LOG_GNSS_CLOCK_REPORT: {
    uint8_t version = log_data[0];
    assert(version == 2);

    assert(len >= sizeof(GNSSClockReportv2));
    const GNSSClockReportv2* report = (const GNSSClockReportv2*)ptr;

    auto lreport = qcomGnss.initClockReport();
    lreport.setHasFCount(report->valid_flags & (1 << 0));
    lreport.setFCount(report->f_count);

    lreport.setHasGpsWeek(report->valid_flags & (1 << 2));
    lreport.setGpsWeek(report->gps_week);
    lreport.setHasGpsMilliseconds(report->valid_flags & (1 << 1));
    lreport.setGpsMilliseconds(report->gps_milliseconds);
    lreport.setGpsTimeBias(report->gps_time_bias);
    lreport.setGpsClockTimeUncertainty(report->gps_clock_time_uncertainty);
    lreport.setGpsClockSource(report->gps_clock_source);

    lreport.setHasGlonassYear(report->valid_flags & (1 << 6));
    lreport.setGlonassYear(report->glonass_year);
    lreport.setHasGlonassDay(report->valid_flags & (1 << 5));
    lreport.setGlonassDay(report->glonass_day);
    lreport.setHasGlonassMilliseconds(report->valid_flags & (1 << 4));
    lreport.setGlonassMilliseconds(report->glonass_milliseconds);
    lreport.setGlonassTimeBias(report->glonass_time_bias);
    lreport.setGlonassClockTimeUncertainty(report->glonass_clock_time_uncertainty);
    lreport.setGlonassClockSource(report->glonass_clock_source);

    lreport.setBdsWeek(report->bds_week);
    lreport.setBdsMilliseconds(report->bds_milliseconds);
    lreport.setBdsTimeBias(report->bds_time_bias);
    lreport.setBdsClockTimeUncertainty(report->bds_clock_time_uncertainty);
    lreport.setBdsClockSource(report->bds_clock_source);

    lreport.setGalWeek(report->gal_week);
    lreport.setGalMilliseconds(report->gal_milliseconds);
    lreport.setGalTimeBias(report->gal_time_bias);
    lreport.setGalClockTimeUncertainty(report->gal_clock_time_uncertainty);
    lreport.setGalClockSource(report->gal_clock_source);

    lreport.setClockFrequencyBias(report->clock_frequency_bias);
    lreport.setClockFrequencyUncertainty(report->clock_frequency_uncertainty);
    lreport.setFrequencySource(report->frequency_source);
    lreport.setGpsLeapSeconds(report->gps_leap_seconds);
    lreport.setGpsLeapSecondsUncertainty(report->gps_leap_seconds_uncertainty);
    lreport.setGpsLeapSecondsSource(report->gps_leap_seconds_source);

    lreport.setGpsToGlonassTimeBiasMilliseconds(report->gps_to_glonass_time_bias_milliseconds);
    lreport.setGpsToGlonassTimeBiasMillisecondsUncertainty(report->gps_to_glonass_time_bias_milliseconds_uncertainty);
    lreport.setGpsToBdsTimeBiasMilliseconds(report->gps_to_bds_time_bias_milliseconds);
    lreport.setGpsToBdsTimeBiasMillisecondsUncertainty(report->gps_to_bds_time_bias_milliseconds_uncertainty);
    lreport.setBdsToGloTimeBiasMilliseconds(report->bds_to_glo_time_bias_milliseconds);
    lreport.setBdsToGloTimeBiasMillisecondsUncertainty(report->bds_to_glo_time_bias_milliseconds_uncertainty);
    lreport.setGpsToGalTimeBiasMilliseconds(report->gps_to_gal_time_bias_milliseconds);
    lreport.setGpsToGalTimeBiasMillisecondsUncertainty(report->gps_to_gal_time_bias_milliseconds_uncertainty);
    lreport.setGalToGloTimeBiasMilliseconds(report->gal_to_glo_time_bias_milliseconds);
    lreport.setGalToGloTimeBiasMillisecondsUncertainty(report->gal_to_glo_time_bias_milliseconds_uncertainty);
    lreport.setGalToBdsTimeBiasMilliseconds(report->gal_to_bds_time_bias_milliseconds);
    lreport.setGalToBdsTimeBiasMillisecondsUncertainty(report->gal_to_bds_time_bias_milliseconds_uncertainty);

    lreport.setHasRtcTime(report->valid_flags & (1 << 3));
    lreport.setSystemRtcTime(report->system_rtc_time);
    lreport.setFCountOffset(report->f_count_offset);
    lreport.setLpmRtcCount(report->lpm_rtc_count);
    lreport.setClockResets(report->clock_resets);

    break;
  }
  case LOG_GNSS_GPS_MEASUREMENT_REPORT: {
    uint8_t version = log_data[0];
    assert(version == 0);

    assert(len >= sizeof(GNSSGpsMeasurementReportv0));
    const GNSSGpsMeasurementReportv0* report = (const GNSSGpsMeasurementReportv0*)ptr;
    assert(len >= sizeof(sizeof(GNSSGpsMeasurementReportv0))+sizeof(GNSSGpsMeasurementReportv0_SV) * report->sv_count);

    auto lreport = qcomGnss.initMeasurementReport();
    lreport.setSource(cereal::QcomGnss::MeasurementSource::GPS);
    lreport.setFCount(report->f_count);
    lreport.setGpsWeek(report->week);
    lreport.setMilliseconds(report->milliseconds);
    lreport.setTimeBias(report->time_bias);
    lreport.setClockTimeUncertainty(report->clock_time_uncertainty);
    lreport.setClockFrequencyBias(report->clock_frequency_bias);
    lreport.setClockFrequencyUncertainty(report->clock_frequency_uncertainty);

    auto lsvs = lreport.initSv(report->sv_count);
    for (int i=0; i<report->sv_count; i++) {
      auto lsv = lsvs[i];
      const GNSSGpsMeasurementReportv0_SV *sv = &report->sv[i];

      lsv.setSvId(sv->sv_id);
      lsv.setObservationState(cereal::QcomGnss::SVObservationState(sv->observation_state));
      lsv.setObservations(sv->observations);
      lsv.setGoodObservations(sv->good_observations);
      lsv.setGpsParityErrorCount(sv->parity_error_count);
      lsv.setFilterStages(sv->filter_stages);
      lsv.setCarrierNoise(sv->carrier_noise);
      lsv.setLatency(sv->latency);
      lsv.setPredetectInterval(sv->predetect_interval);
      lsv.setPostdetections(sv->postdetections);
      lsv.setUnfilteredMeasurementIntegral(sv->unfiltered_measurement_integral);
      lsv.setUnfilteredMeasurementFraction(sv->unfiltered_measurement_fraction);
      lsv.setUnfilteredTimeUncertainty(sv->unfiltered_time_uncertainty);
      lsv.setUnfilteredSpeed(sv->unfiltered_speed);
      lsv.setUnfilteredSpeedUncertainty(sv->unfiltered_speed_uncertainty);

      lsv.setMultipathEstimate(sv->multipath_estimate);
      lsv.setAzimuth(sv->azimuth);
      lsv.setElevation(sv->elevation);
      lsv.setCarrierPhaseCyclesIntegral(sv->carrier_phase_cycles_integral);
      lsv.setCarrierPhaseCyclesFraction(sv->carrier_phase_cycles_fraction);
      lsv.setFineSpeed(sv->fine_speed);
      lsv.setFineSpeedUncertainty(sv->fine_speed_uncertainty);
      lsv.setCycleSlipCount(sv->cycle_slip_count);

      auto status = lsv.initMeasurementStatus();
      parse_measurement_status_common(sv->measurement_status, status);

      status.setGpsRoundRobinRxDiversity(sv->measurement_status & (1 << 18));
      status.setGpsRxDiversity(sv->measurement_status & (1 << 19));
      status.setGpsLowBandwidthRxDiversityCombined(sv->measurement_status & (1 << 20));
      status.setGpsHighBandwidthNu4(sv->measurement_status & (1 << 21));
      status.setGpsHighBandwidthNu8(sv->measurement_status & (1 << 22));
      status.setGpsHighBandwidthUniform(sv->measurement_status & (1 << 23));

      status.setMultipathEstimateIsValid(sv->misc_status & (1 << 0));
      status.setDirectionIsValid(sv->misc_status & (1 << 1));

#ifdef RAWGPS_TEST
      // if (sv->measurement_status & (1 << 27)) printf("%d\n", sv->unfiltered_measurement_integral);
      printf("GPS %03d %d %d 0x%08X    o: %02x go: %02x fs: %02x cn: %02x pd: %02x cs: %02x po: %02x ms: %08x ms2: %08x me: %08x  az: %08x el: %08x  fc: %08x\n",
        sv->sv_id,
        !!(sv->measurement_status & (1 << 27)),
        sv->unfiltered_measurement_integral, sv->unfiltered_measurement_integral,
        sv->observations,
        sv->good_observations,
        sv->filter_stages,
        sv->carrier_noise,
        sv->predetect_interval,
        sv->cycle_slip_count,
        sv->postdetections,
        sv->measurement_status,
        sv->misc_status,
        sv->multipath_estimate,
        *(uint32_t*)&sv->azimuth,
        *(uint32_t*)&sv->elevation,
        report->f_count
        );
#endif

    }

    break;
  }
  case LOG_GNSS_GLONASS_MEASUREMENT_REPORT: {
    uint8_t version = log_data[0];
    assert(version == 0);

    assert(len >= sizeof(GNSSGlonassMeasurementReportv0));
    const GNSSGlonassMeasurementReportv0* report = (const GNSSGlonassMeasurementReportv0*)ptr;

    auto lreport = qcomGnss.initMeasurementReport();
    lreport.setSource(cereal::QcomGnss::MeasurementSource::GLONASS);
    lreport.setFCount(report->f_count);
    lreport.setGlonassCycleNumber(report->glonass_cycle_number);
    lreport.setGlonassNumberOfDays(report->glonass_number_of_days);
    lreport.setMilliseconds(report->milliseconds);
    lreport.setTimeBias(report->time_bias);
    lreport.setClockTimeUncertainty(report->clock_time_uncertainty);
    lreport.setClockFrequencyBias(report->clock_frequency_bias);
    lreport.setClockFrequencyUncertainty(report->clock_frequency_uncertainty);

    auto lsvs = lreport.initSv(report->sv_count);
    for (int i=0; i<report->sv_count; i++) {
      auto lsv = lsvs[i];
      const GNSSGlonassMeasurementReportv0_SV *sv = &report->sv[i];

      lsv.setSvId(sv->sv_id);
      lsv.setObservationState(cereal::QcomGnss::SVObservationState(sv->observation_state));
      lsv.setObservations(sv->observations);
      lsv.setGoodObservations(sv->good_observations);
      lsv.setGlonassFrequencyIndex(sv->frequency_index);
      lsv.setGlonassHemmingErrorCount(sv->hemming_error_count);
      lsv.setFilterStages(sv->filter_stages);
      lsv.setCarrierNoise(sv->carrier_noise);
      lsv.setLatency(sv->latency);
      lsv.setPredetectInterval(sv->predetect_interval);
      lsv.setPostdetections(sv->postdetections);
      lsv.setUnfilteredMeasurementIntegral(sv->unfiltered_measurement_integral);
      lsv.setUnfilteredMeasurementFraction(sv->unfiltered_measurement_fraction);
      lsv.setUnfilteredTimeUncertainty(sv->unfiltered_time_uncertainty);
      lsv.setUnfilteredSpeed(sv->unfiltered_speed);
      lsv.setUnfilteredSpeedUncertainty(sv->unfiltered_speed_uncertainty);

      lsv.setMultipathEstimate(sv->multipath_estimate);
      lsv.setAzimuth(sv->azimuth);
      lsv.setElevation(sv->elevation);
      lsv.setCarrierPhaseCyclesIntegral(sv->carrier_phase_cycles_integral);
      lsv.setCarrierPhaseCyclesFraction(sv->carrier_phase_cycles_fraction);
      lsv.setFineSpeed(sv->fine_speed);
      lsv.setFineSpeedUncertainty(sv->fine_speed_uncertainty);
      lsv.setCycleSlipCount(sv->cycle_slip_count);

      auto status = lsv.initMeasurementStatus();
      parse_measurement_status_common(sv->measurement_status, status);

      status.setGlonassMeanderBitEdgeValid(sv->measurement_status & (1 << 16));
      status.setGlonassTimeMarkValid(sv->measurement_status & (1 << 17));

      status.setMultipathEstimateIsValid(sv->misc_status & (1 << 0));
      status.setDirectionIsValid(sv->misc_status & (1 << 1));


#ifdef RAWGPS_TEST
      // if (sv->measurement_status & (1 << 27)) printf("%d\n", sv->unfiltered_measurement_integral);
      printf("GLO %03d %02x %d %d 0x%08X    o: %02x go: %02x fs: %02x cn: %02x pd: %02x cs: %02x po: %02x ms: %08x ms2: %08x me: %08x  az: %08x el: %08x  fc: %08x\n",
        sv->sv_id, sv->frequency_index & 0xff,
        !!(sv->measurement_status & (1 << 27)),
        sv->unfiltered_measurement_integral, sv->unfiltered_measurement_integral,
        sv->observations,
        sv->good_observations,
        sv->filter_stages,
        sv->carrier_noise,
        sv->predetect_interval,
        sv->cycle_slip_count,
        sv->postdetections,
        sv->measurement_status,
        sv->misc_status,
        sv->multipath_estimate,
        *(uint32_t*)&sv->azimuth,
        *(uint32_t*)&sv->elevation,
        report->f_count
        );
#endif

    }
    break;
  }
  case LOG_GNSS_OEMDRE_MEASUREMENMT_REPORT: {
    // hexdump(ptr, len);

    uint8_t version = log_data[0];
    assert(version == 2);

    assert(len >= sizeof(GNSSOemdreMeasurementReportv2));
    const GNSSOemdreMeasurementReportv2* report = (const GNSSOemdreMeasurementReportv2*)ptr;


    auto lreport = qcomGnss.initDrMeasurementReport();

    lreport.setReason(report->reason);
    lreport.setSeqNum(report->seq_num);
    lreport.setSeqMax(report->seq_max);
    lreport.setRfLoss(report->rf_loss);
    lreport.setSystemRtcValid(report->system_rtc_valid);
    lreport.setFCount(report->f_count);
    lreport.setClockResets(report->clock_resets);
    lreport.setSystemRtcTime(report->system_rtc_time);

    lreport.setGpsLeapSeconds(report->gps_leap_seconds);
    lreport.setGpsLeapSecondsUncertainty(report->gps_leap_seconds_uncertainty);
    lreport.setGpsToGlonassTimeBiasMilliseconds(report->gps_to_glonass_time_bias_milliseconds);
    lreport.setGpsToGlonassTimeBiasMillisecondsUncertainty(report->gps_to_glonass_time_bias_milliseconds_uncertainty);

    lreport.setGpsWeek(report->gps_week);
    lreport.setGpsMilliseconds(report->gps_milliseconds);
    lreport.setGpsTimeBiasMs(report->gps_time_bias);
    lreport.setGpsClockTimeUncertaintyMs(report->gps_clock_time_uncertainty);
    lreport.setGpsClockSource(report->gps_clock_source);

    lreport.setGlonassClockSource(report->glonass_clock_source);
    lreport.setGlonassYear(report->glonass_year);
    lreport.setGlonassDay(report->glonass_day);
    lreport.setGlonassMilliseconds(report->glonass_milliseconds);
    lreport.setGlonassTimeBias(report->glonass_time_bias);
    lreport.setGlonassClockTimeUncertainty(report->glonass_clock_time_uncertainty);

    lreport.setClockFrequencyBias(report->clock_frequency_bias);
    lreport.setClockFrequencyUncertainty(report->clock_frequency_uncertainty);
    lreport.setFrequencySource(report->frequency_source);

    lreport.setSource(cereal::QcomGnss::MeasurementSource(report->source));

    auto lsvs = lreport.initSv(report->sv_count);

    // for (int i=0; i<report->sv_count; i++) {
    //   GNSSOemdreMeasurement *sv = &report->gps[i];
    //   if (!(sv->measurement_status & (1 << 27))) continue;
    //   printf("oemdre %03d %d %f\n", sv->sv_id, sv->unfiltered_measurement_integral, sv->unfiltered_measurement_fraction);
    // }
    for (int i=0; i<report->sv_count; i++) {
      auto lsv = lsvs[i];
      const GNSSOemdreMeasurement *sv = &report->measurements[i];

      lsv.setSvId(sv->sv_id);
      lsv.setGlonassFrequencyIndex(sv->glonass_frequency_index);
      lsv.setObservationState(cereal::QcomGnss::SVObservationState(sv->observation_state));
      lsv.setObservations(sv->observations);
      lsv.setGoodObservations(sv->good_observations);
      lsv.setFilterStages(sv->filter_stages);
      lsv.setPredetectInterval(sv->predetect_interval);
      lsv.setCycleSlipCount(sv->cycle_slip_count);
      lsv.setPostdetections(sv->postdetections);

      auto status = lsv.initMeasurementStatus();
      parse_measurement_status_common(sv->measurement_status, status);

      status.setMultipathEstimateIsValid(sv->multipath_estimate_valid);
      status.setDirectionIsValid(sv->direction_valid);

      lsv.setCarrierNoise(sv->carrier_noise);
      lsv.setRfLoss(sv->rf_loss);
      lsv.setLatency(sv->latency);

      lsv.setFilteredMeasurementFraction(sv->filtered_measurement_fraction);
      lsv.setFilteredMeasurementIntegral(sv->filtered_measurement_integral);
      lsv.setFilteredTimeUncertainty(sv->filtered_time_uncertainty);
      lsv.setFilteredSpeed(sv->filtered_speed);
      lsv.setFilteredSpeedUncertainty(sv->filtered_speed_uncertainty);

      lsv.setUnfilteredMeasurementFraction(sv->unfiltered_measurement_fraction);
      lsv.setUnfilteredMeasurementIntegral(sv->unfiltered_measurement_integral);
      lsv.setUnfilteredTimeUncertainty(sv->unfiltered_time_uncertainty);
      lsv.setUnfilteredSpeed(sv->unfiltered_speed);
      lsv.setUnfilteredSpeedUncertainty(sv->unfiltered_speed_uncertainty);

      lsv.setMultipathEstimate(sv->multipath_estimate);
      lsv.setAzimuth(sv->azimuth);
      lsv.setElevation(sv->elevation);
      lsv.setDopplerAcceleration(sv->doppler_acceleration);
      lsv.setFineSpeed(sv->fine_speed);
      lsv.setFineSpeedUncertainty(sv->fine_speed_uncertainty);

      lsv.setCarrierPhase(sv->carrier_phase);
      lsv.setFCount(sv->f_count);

      lsv.setParityErrorCount(sv->parity_error_count);
      lsv.setGoodParity(sv->good_parity);

    }
    break;
  }
  case LOG_GNSS_OEMDRE_SVPOLY_REPORT: {
    uint8_t version = log_data[0];
    assert(version == 2);

    assert(len >= sizeof(GNSSOemdreSVPolyReportv2));
    const GNSSOemdreSVPolyReportv2* report = (const GNSSOemdreSVPolyReportv2*)ptr;

    auto lreport = qcomGnss.initDrSvPoly();

    lreport.setSvId(report->sv_id);
    lreport.setFrequencyIndex(report->frequency_index);

    lreport.setHasPosition(report->flags & 1);
    lreport.setHasIono(report->flags & 2);
    lreport.setHasTropo(report->flags & 4);
    lreport.setHasElevation(report->flags & 8);
    lreport.setPolyFromXtra(report->flags & 16);
    lreport.setHasSbasIono(report->flags & 32);

    lreport.setIode(report->iode);
    lreport.setT0(report->t0);

    kj::ArrayPtr<const double> xyz0(report->xyz0, 3);
    lreport.setXyz0(xyz0);

    kj::ArrayPtr<const double> xyzN(report->xyzN, 9);
    lreport.setXyzN(xyzN);

    kj::ArrayPtr<const float> other(report->other, 4);
    lreport.setOther(other);

    lreport.setPositionUncertainty(report->position_uncertainty);
    lreport.setIonoDelay(report->iono_delay);
    lreport.setIonoDot(report->iono_dot);
    lreport.setSbasIonoDelay(report->sbas_iono_delay);
    lreport.setSbasIonoDot(report->sbas_iono_dot);
    lreport.setTropoDelay(report->tropo_delay);
    lreport.setElevation(report->elevation);
    lreport.setElevationDot(report->elevation_dot);
    lreport.setElevationUncertainty(report->elevation_uncertainty);

    kj::ArrayPtr<const double> velocity_coeff(report->velocity_coeff, 12);
    lreport.setVelocityCoeff(velocity_coeff);

    break;
  }
  default:
    // printf("%04x\n", log_header->code);
    // hexdump(ptr, len);

    qcomGnss.setRawLog(kj::arrayPtr(ptr, len));

    break;
  }

  auto words = capnp::messageToFlatArray(msg);
  auto bytes = words.asBytes();
  rawgps_publisher->send((char*)bytes.begin(), bytes.size());
}

static void handle_event(unsigned char *ptr, int len) {
  // printf("EVENT\n");
}

static uint16_t log_codes[] = {
  LOG_GNSS_CLOCK_REPORT,
  LOG_GNSS_GPS_MEASUREMENT_REPORT,
  LOG_GNSS_GLONASS_MEASUREMENT_REPORT,
  LOG_GNSS_OEMDRE_MEASUREMENMT_REPORT,
  LOG_GNSS_OEMDRE_SVPOLY_REPORT,

  // unparsed:
  LOG_GNSS_POSITION_REPORT,

  LOG_GNSS_BDS_MEASUREMENT_REPORT, // these are missing by might as well try to catch them anyway
  LOG_GNSS_GAL_MEASUREMENT_REPORT,

  LOG_GNSS_ME_DPO_STATUS,
  LOG_GNSS_CD_DB_REPORT,
  LOG_GNSS_PRX_RF_HW_STATUS_REPORT,
  LOG_CGPS_SLOW_CLOCK_CLIB_REPORT,
  LOG_GNSS_CONFIGURATION_STATE,
};


struct SendDiagSyncState {
  sem_t sem;
  int len;
};

static void diag_send_sync_cb(unsigned char *ptr, int len, void *data_ptr) {
  SendDiagSyncState *s = (SendDiagSyncState*)data_ptr;
  s->len = len;
  sem_post(&s->sem);
}

static int diag_send_sync(int client_id, unsigned char* req_pkt, size_t pkt_len,
                           unsigned char* res_pkt, size_t res_pkt_size) {

  SendDiagSyncState s = {0};
  sem_init(&s.sem, 0, 0);

  int err = diag_send_dci_async_req(client_id, req_pkt, pkt_len, res_pkt, res_pkt_size,
                                    diag_send_sync_cb, &s);
  assert(err == DIAG_DCI_NO_ERROR);

  sem_wait(&s.sem);
  return s.len;
}

static int oemdre_on(int client_id) {
  // enable OEM DR
  unsigned char res_pkt[DIAG_MAX_RX_PKT_SIZ];

  GpsOemControlReq req_pkt = {
    .cmd_code = DIAG_SUBSYS_CMD,
    .subsys_id = DIAG_SUBSYS_GPS,
    .subsys_cmd_code = CGPS_DIAG_PDAPI_CMD,
    .gps_cmd_code = CGPS_OEM_CONTROL,
    .version = 1,

    .oem_feature = GPSDIAG_OEMFEATURE_DRE,
    .oem_command = GPSDIAG_OEM_DRE_ON,
  };

  int res_len = diag_send_sync(client_id, (unsigned char*)&req_pkt, sizeof(req_pkt),
                                res_pkt, sizeof(res_pkt));
  GpsOemControlResp *resp = (GpsOemControlResp*)res_pkt;

  if (res_len != sizeof(GpsOemControlResp)
      || resp->cmd_code != DIAG_SUBSYS_CMD
      || resp->subsys_id != DIAG_SUBSYS_GPS
      || resp->subsys_cmd_code != CGPS_DIAG_PDAPI_CMD
      || resp->gps_cmd_code != CGPS_OEM_CONTROL
      || resp->oem_feature != GPSDIAG_OEMFEATURE_DRE
      || resp->oem_command != GPSDIAG_OEM_DRE_ON) {
    LOGW("oemdre_on: bad response!");
    return -1;
  }

  return resp->resp_result;
}

static void efs_sync(int client_id) {
  unsigned char res_pkt[DIAG_MAX_RX_PKT_SIZ];

  Efs2DiagSyncReq req_pkt = {
    .cmd_code = DIAG_SUBSYS_CMD,
    .subsys_id = DIAG_SUBSYS_FS,
    .subsys_cmd_code = EFS2_DIAG_SYNC_NO_WAIT,
    .sequence_num = (uint16_t)(rand() % 100),
  };
  req_pkt.path[0] = '/';
  req_pkt.path[1] = 0;

  int res_len = diag_send_sync(client_id, (unsigned char*)&req_pkt, sizeof(req_pkt),
                                res_pkt, sizeof(res_pkt));
  Efs2DiagSyncResp *resp = (Efs2DiagSyncResp*)res_pkt;

  if (res_len != sizeof(Efs2DiagSyncResp)
      || resp->cmd_code != DIAG_SUBSYS_CMD
      || resp->subsys_id != DIAG_SUBSYS_FS
      || resp->subsys_cmd_code != EFS2_DIAG_SYNC_NO_WAIT) {
    LOGW("efs_sync: bad response!");
    return;
  }
  if (resp->diag_errno != 0) {
    LOGW("efs_sync: error %d", resp->diag_errno);
  }
}

static uint32_t nv_read_u32(int client_id, uint16_t nv_id) {
  NvPacket req = {
    .cmd_code = DIAG_NV_READ_F,
    .nv_id = nv_id,
  };
  NvPacket resp = {0};

  int res_len = diag_send_sync(client_id, (unsigned char*)&req, sizeof(req),
                               (unsigned char*)&resp, sizeof(resp));

  // hexdump((uint8_t*)&resp, res_len);

  if (resp.cmd_code != DIAG_NV_READ_F
      || resp.nv_id != nv_id) {
    LOGW("nv_read_u32: diag command failed");
    return 0;
  }

  if (resp.status != NV_DONE) {
    LOGW("nv_read_u32: read failed: %d", resp.status);
    return 0;
  }
  return *(uint32_t*)resp.data;
}

static bool nv_write_u32(int client_id, uint16_t nv_id, uint32_t val) {
  NvPacket req = {
    .cmd_code = DIAG_NV_WRITE_F,
    .nv_id = nv_id,
  };
  *(uint32_t*)req.data = val;

  NvPacket resp = {0};
  int res_len = diag_send_sync(client_id, (unsigned char*)&req, sizeof(req),
                               (unsigned char*)&resp, sizeof(resp));

  // hexdump((uint8_t*)&resp, res_len);

  if (resp.cmd_code != DIAG_NV_WRITE_F
      || resp.nv_id != nv_id) {
    LOGW("nv_write_u32: diag command failed");
    return false;
  }

  if (resp.status != NV_DONE) {
    LOGW("nv_write_u32: write failed: %d", resp.status);
    return false;
  }

  return true;
}

static void nav_config(int client_id, uint8_t config) {
  unsigned char res_pkt[DIAG_MAX_RX_PKT_SIZ];

  GpsNavConfigReq req_pkt = {
    .cmd_code = DIAG_SUBSYS_CMD_VER_2,
    .subsys_id = DIAG_SUBSYS_GPS,
    .subsys_cmd_code = TM_DIAG_NAV_CONFIG_CMD,
    .desired_config = config,
  };

  int res_len = diag_send_sync(client_id, (unsigned char*)&req_pkt, sizeof(req_pkt),
                                res_pkt, sizeof(res_pkt));
  GpsNavConfigResp *resp = (GpsNavConfigResp*)res_pkt;

  if (res_len != sizeof(GpsNavConfigResp)
      || resp->cmd_code != DIAG_SUBSYS_CMD_VER_2
      || resp->subsys_id != DIAG_SUBSYS_GPS
      || resp->subsys_cmd_code != TM_DIAG_NAV_CONFIG_CMD) {
    LOGW("nav_config: bad response!");
    return;
  }
  LOG("nav config: %04x %04x", resp->supported_config, resp->actual_config);
}

void rawgps_init() {
  int err;

  rawgps_context = Context::create();
  rawgps_publisher = PubSocket::create(rawgps_context, "qcomGnss");
  assert(rawgps_publisher != NULL);

  bool init_success = Diag_LSM_Init(NULL);
  assert(init_success);

  uint16_t list = DIAG_CON_APSS | DIAG_CON_MPSS;
  int signal_type = SIGCONT;
  err = diag_register_dci_client(&client_id, &list, 0, &signal_type);
  assert(err == DIAG_DCI_NO_ERROR);

  {
    uint32_t oem_features = nv_read_u32(client_id, NV_GNSS_OEM_FEATURE_MASK);
    LOG("oem features: %08x", oem_features);

    if (!(oem_features & NV_GNSS_OEM_FEATURE_MASK_OEMDRE)) {
      LOG("OEMDRE feature disabled, enabling...");
      nv_write_u32(client_id, NV_GNSS_OEM_FEATURE_MASK, NV_GNSS_OEM_FEATURE_MASK_OEMDRE);
      efs_sync(client_id);
    }

    int oemdre_status = oemdre_on(client_id);
    LOG("oemdre status: %d", oemdre_status);
  }

  {
    // make sure GNSS duty cycling is off
    uint32_t dpo = nv_read_u32(client_id, NV_CGPS_DPO_CONTROL);
    LOG("dpo: %d", dpo);
    if (dpo != 0) {
      nv_write_u32(client_id, NV_CGPS_DPO_CONTROL, 0);
      efs_sync(client_id);
    }
  }

  // enable beidou
  // nav_config(client_id, 0x13); // 0b10011

  err = diag_register_dci_stream(handle_log, handle_event);
  assert(err == DIAG_DCI_NO_ERROR);

  err = diag_log_stream_config(client_id, true, log_codes, ARRAYSIZE(log_codes));
  assert(err == DIAG_DCI_NO_ERROR);
}

void rawgps_destroy() {

  int err;

  err = diag_log_stream_config(client_id, false, log_codes, ARRAYSIZE(log_codes));
  assert(err == DIAG_DCI_NO_ERROR);

  err = diag_release_dci_client(&client_id);
  assert(err == DIAG_DCI_NO_ERROR);

  Diag_LSM_DeInit();

  delete rawgps_publisher;
  delete rawgps_context;
}


#ifdef RAWGPS_TEST
int main() {
  int err = 0;
  rawgps_init();

  while(1) {
    usleep(100000);
  }

  rawgps_destroy();
  return 0;
}
#endif
