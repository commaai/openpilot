#define _CRT_SECURE_NO_WARNINGS

#include "limegps.h"
#include <math.h>

// for _getch used in Windows runtime.
#ifdef WIN32
#include <conio.h>
#include "getopt.h"
#else
#include <unistd.h>
#endif

void init_sim(sim_t *s)
{
	pthread_mutex_init(&(s->tx.lock), NULL);
	//s->tx.error = 0;

	pthread_mutex_init(&(s->gps.lock), NULL);
	//s->gps.error = 0;
	s->gps.ready = 0;
	pthread_cond_init(&(s->gps.initialization_done), NULL);

	s->status = 0;
	s->head = 0;
	s->tail = 0;
	s->sample_length = 0;

	pthread_cond_init(&(s->fifo_write_ready), NULL);
	pthread_cond_init(&(s->fifo_read_ready), NULL);

	s->time = 0.0;
}

size_t get_sample_length(sim_t *s)
{
	long length;

	length = s->head - s->tail;
	if (length < 0)
		length += FIFO_LENGTH;

	return((size_t)length);
}

size_t fifo_read(int16_t *buffer, size_t samples, sim_t *s)
{
	size_t length;
	size_t samples_remaining;
	int16_t *buffer_current = buffer;

	length = get_sample_length(s);

	if (length < samples)
		samples = length;

	length = samples; // return value

	samples_remaining = FIFO_LENGTH - s->tail;

	if (samples > samples_remaining) {
		memcpy(buffer_current, &(s->fifo[s->tail * 2]), samples_remaining * sizeof(int16_t) * 2);
		s->tail = 0;
		buffer_current += samples_remaining * 2;
		samples -= samples_remaining;
	}

	memcpy(buffer_current, &(s->fifo[s->tail * 2]), samples * sizeof(int16_t) * 2);
	s->tail += (long)samples;
	if (s->tail >= FIFO_LENGTH)
		s->tail -= FIFO_LENGTH;

	return(length);
}

bool is_finished_generation(sim_t *s)
{
	return s->finished;
}

int is_fifo_write_ready(sim_t *s)
{
	int status = 0;

	s->sample_length = get_sample_length(s);
	if (s->sample_length < NUM_IQ_SAMPLES)
		status = 1;

	return(status);
}

void *tx_task(void *arg)
{
	sim_t *s = (sim_t *)arg;
	size_t samples_populated;

	while (1) {
		int16_t *tx_buffer_current = s->tx.buffer;
		unsigned int buffer_samples_remaining = SAMPLES_PER_BUFFER;

		while (buffer_samples_remaining > 0) {
			
			pthread_mutex_lock(&(s->gps.lock));
			while (get_sample_length(s) == 0)
			{
				pthread_cond_wait(&(s->fifo_read_ready), &(s->gps.lock));
			}
//			assert(get_sample_length(s) > 0);

			samples_populated = fifo_read(tx_buffer_current,
				buffer_samples_remaining,
				s);
			pthread_mutex_unlock(&(s->gps.lock));

			pthread_cond_signal(&(s->fifo_write_ready));
#if 0
			if (is_fifo_write_ready(s)) {
				/*
				printf("\rTime = %4.1f", s->time);
				s->time += 0.1;
				fflush(stdout);
				*/
			}
			else if (is_finished_generation(s))
			{
				goto out;
			}
#endif
			// Advance the buffer pointer.
			buffer_samples_remaining -= (unsigned int)samples_populated;
			tx_buffer_current += (2 * samples_populated);
		}

		// If there were no errors, transmit the data buffer.
		LMS_SendStream(&s->tx.stream, s->tx.buffer, SAMPLES_PER_BUFFER, NULL, 1000);
		if (is_fifo_write_ready(s)) {
			/*
			printf("\rTime = %4.1f", s->time);
			s->time += 0.1;
			fflush(stdout);
			*/
		}
		else if (is_finished_generation(s))
		{
			goto out;
		}

	}
out:
	return NULL;
}

int start_tx_task(sim_t *s)
{
	int status;

	status = pthread_create(&(s->tx.thread), NULL, tx_task, s);

	return(status);
}

int start_gps_task(sim_t *s)
{
	int status;

	status = pthread_create(&(s->gps.thread), NULL, gps_task, s);

	return(status);
}

void usage(char *progname)
{
	printf("Usage: %s [options]\n"
		"Options:\n"
		"  -e <gps_nav>     RINEX navigation file for GPS ephemerides (required)\n"
		"  -u <user_motion> User motion file (dynamic mode)\n"
		"  -g <nmea_gga>    NMEA GGA stream (dynamic mode)\n"
		"  -l <location>    Lat,Lon,Hgt (static mode) e.g. 35.274,137.014,100\n"
		"  -t <date,time>   Scenario start time YYYY/MM/DD,hh:mm:ss\n"
		"  -T <date,time>   Overwrite TOC and TOE to scenario start time\n"
		"  -d <duration>    Duration [sec] (max: %.0f)\n"
		"  -a <rf_gain>     Normalized RF gain in [0.0 ... 1.0] (default: 0.1)\n"
#ifdef WIN32
		"  -i               Interactive mode: North='%c', South='%c', East='%c', West='%c'\n"
#ifdef USE_GAMEPAD
		"                   (Xbox gamepad: Turn Left=<, Turn Right=>, Forward=B)\n"
#endif
#endif
		"  -I               Disable ionospheric delay for spacecraft scenario\n",
		progname,
#ifdef WIN32
		((double)USER_MOTION_SIZE)/10.0, 
		NORTH_KEY, SOUTH_KEY, EAST_KEY, WEST_KEY);
#else
		((double)USER_MOTION_SIZE)/10.0);
#endif
	return;
}

int main(int argc, char *argv[])
{
	if (argc<3)
	{
		usage(argv[0]);
		exit(1);
	}

	// Set default values
	sim_t s;

	s.finished = false;
	s.opt.navfile[0] = 0;
	s.opt.umfile[0] = 0;
	s.opt.g0.week = -1;
	s.opt.g0.sec = 0.0;
	s.opt.iduration = USER_MOTION_SIZE;
	s.opt.verb = TRUE;
	s.opt.nmeaGGA = FALSE;
	s.opt.staticLocationMode = TRUE;
	//s.opt.llh[0] = 35.6811673 / R2D;
	//s.opt.llh[1] = 139.7648576 / R2D;
	s.opt.llh[0] = 40.7850916 / R2D;
	s.opt.llh[1] = -73.968285 / R2D;
	s.opt.llh[2] = 10.0;
	s.opt.interactive = FALSE;
	s.opt.timeoverwrite = FALSE;
	s.opt.iono_enable = TRUE;

	// Options
	int result;
	double duration;
	datetime_t t0;
	double gain = 0.1;

	while ((result=getopt(argc,argv,"e:u:g:l:T:t:d:a:iI"))!=-1)
	{
		switch (result)
		{
		case 'e':
			strcpy(s.opt.navfile, optarg);
			break;
		case 'u':
			strcpy(s.opt.umfile, optarg);
			s.opt.nmeaGGA = FALSE;
			s.opt.staticLocationMode = FALSE;
			break;
		case 'g':
			strcpy(s.opt.umfile, optarg);
			s.opt.nmeaGGA = TRUE;
			s.opt.staticLocationMode = FALSE;
			break;
		case 'l':
			// Static geodetic coordinates input mode
			// Added by scateu@gmail.com
			s.opt.nmeaGGA = FALSE;
			s.opt.staticLocationMode = TRUE;
			sscanf(optarg,"%lf,%lf,%lf",&s.opt.llh[0],&s.opt.llh[1],&s.opt.llh[2]);
			s.opt.llh[0] /= R2D; // convert to RAD
			s.opt.llh[1] /= R2D; // convert to RAD
			break;
		case 'T':
			s.opt.timeoverwrite = TRUE;
			if (strncmp(optarg, "now", 3)==0)
			{
				time_t timer;
				struct tm *gmt;
				
				time(&timer);
				gmt = gmtime(&timer);

				t0.y = gmt->tm_year+1900;
				t0.m = gmt->tm_mon+1;
				t0.d = gmt->tm_mday;
				t0.hh = gmt->tm_hour;
				t0.mm = gmt->tm_min;
				t0.sec = (double)gmt->tm_sec;

				date2gps(&t0, &s.opt.g0);

				break;
			}
		case 't':
			sscanf(optarg, "%d/%d/%d,%d:%d:%lf", &t0.y, &t0.m, &t0.d, &t0.hh, &t0.mm, &t0.sec);
			if (t0.y<=1980 || t0.m<1 || t0.m>12 || t0.d<1 || t0.d>31 ||
				t0.hh<0 || t0.hh>23 || t0.mm<0 || t0.mm>59 || t0.sec<0.0 || t0.sec>=60.0)
			{
				printf("ERROR: Invalid date and time.\n");
				exit(1);
			}
			t0.sec = floor(t0.sec);
			date2gps(&t0, &s.opt.g0);
			break;
		case 'd':
			duration = atof(optarg);
			if (duration<0.0 || duration>((double)USER_MOTION_SIZE)/10.0)
			{
				printf("ERROR: Invalid duration.\n");
				exit(1);
			}
			s.opt.iduration = (int)(duration*10.0+0.5);
			break;
		case 'a':
			gain = atof(optarg);
			if (gain < 0.0)
				gain = 0.0;
			if (gain > 1.0)
				gain = 1.0;
			break;
		case 'i':
			s.opt.interactive = TRUE;
			break;
		case 'I':
			s.opt.iono_enable = FALSE; // Disable ionospheric correction
			break;
		case ':':
		case '?':
			usage(argv[0]);
			exit(1);
		default:
			break;
		}
	}

	if (s.opt.navfile[0]==0)
	{
		printf("ERROR: GPS ephemeris file is not specified.\n");
		exit(1);
	}

	if (s.opt.umfile[0]==0 && !s.opt.staticLocationMode)
	{
		printf("ERROR: User motion file / NMEA GGA stream is not specified.\n");
		printf("You may use -l to specify the static location directly.\n");
		exit(1);
	}

	// Find device
	int device_count = LMS_GetDeviceList(NULL);

	if (device_count < 1)
	{
		printf("ERROR: No device was found.\n");
		exit(1);
	}
	else if (device_count > 1)
	{
		printf("ERROR: Found more than one device.\n");
		exit(1);
	}

	lms_info_str_t *device_list = malloc(sizeof(lms_info_str_t) * device_count);
	device_count = LMS_GetDeviceList(device_list);

	// Initialize simulator
	init_sim(&s);

	// Allocate TX buffer to hold each block of samples to transmit.
	s.tx.buffer = (int16_t *)malloc(SAMPLES_PER_BUFFER * sizeof(int16_t) * 2);
	// for 16-bit I and Q samples
	
	if (s.tx.buffer == NULL) 
	{
		printf("ERROR: Failed to allocate TX buffer.\n");
		goto out;
	}

	// Allocate FIFOs to hold 0.1 seconds of I/Q samples each.
	s.fifo = (int16_t *)malloc(FIFO_LENGTH * sizeof(int16_t) * 2); // for 16-bit I and Q samples

	if (s.fifo == NULL)
	{
		printf("ERROR: Failed to allocate I/Q sample buffer.\n");
		goto out;
	}

	// Initializing device
	printf("Opening and initializing device...\n");

	lms_device_t *device = NULL;

	if (LMS_Open(&device, device_list[0], NULL))
	{
		printf("ERROR: Failed to open device: %s\n", device_list[0]);
		goto out;
	}

	const lms_dev_info_t *devinfo =  LMS_GetDeviceInfo(device);

	if (devinfo == NULL)
	{
		printf("ERROR: Failed to read device info: %s\n", LMS_GetLastErrorMessage());
		goto out;
	}

    printf("deviceName: %s\n", devinfo->deviceName);
    printf("expansionName: %s\n", devinfo->expansionName);
    printf("firmwareVersion: %s\n", devinfo->firmwareVersion);
    printf("hardwareVersion: %s\n", devinfo->hardwareVersion);
    printf("protocolVersion: %s\n", devinfo->protocolVersion);
    printf("gatewareVersion: %s\n", devinfo->gatewareVersion);
    printf("gatewareTargetBoard: %s\n", devinfo->gatewareTargetBoard);

    int limeOversample = 1;
    if(strncmp(devinfo->deviceName, "LimeSDR-USB", 11) == 0)
    {
        limeOversample = 0;    // LimeSDR-USB works best with default oversampling
        printf("Found a LimeSDR-USB\n");
    }
    else
    {
        printf("Found a LimeSDR-Mini\n");
    }

	int lmsReset = LMS_Reset(device);
	if (lmsReset)
	{
		printf("ERROR: Failed to reset device: %s\n", LMS_GetLastErrorMessage());
		goto out;
	}

	// use default configuration
	int lmsInit = LMS_Init(device);
	if (lmsInit)
	{
		printf("ERROR: Failed to linitialize device: %s\n", LMS_GetLastErrorMessage());
		goto out;
	}

	// Select channel
	int32_t channel = 0;
	//int channel_count = LMS_GetNumChannels(device, LMS_CH_TX);

	// Select antenna
	//int32_t antenna = 1;
	int antenna_count = LMS_GetAntennaList(device, LMS_CH_TX, channel, NULL);
	printf("Antenna Count: %d\n", antenna_count);
	lms_name_t *antenna_name = malloc(sizeof(lms_name_t) * antenna_count);

	if (antenna_count > 0)
	{
		int i = 0;
		lms_range_t *antenna_bw = malloc(sizeof(lms_range_t) * antenna_count);
		LMS_GetAntennaList(device, LMS_CH_TX, channel, antenna_name);
		for (i = 0; i < antenna_count; i++)
		{
			LMS_GetAntennaBW(device, LMS_CH_TX, channel, i, antenna_bw + i);
			printf("Channel %d, antenna [%s] has BW [%lf .. %lf] (step %lf)" "\n", channel, antenna_name[i], antenna_bw[i].min, antenna_bw[i].max, antenna_bw[i].step);
		}
	}

	LMS_SetNormalizedGain(device, LMS_CH_TX, channel, gain);
	// Disable all other channels
	LMS_EnableChannel(device, LMS_CH_TX, 1 - channel, false);
	LMS_EnableChannel(device, LMS_CH_RX, 0, false);
	LMS_EnableChannel(device, LMS_CH_RX, 1, false);
	// Enable our Tx channel
	LMS_EnableChannel(device, LMS_CH_TX, channel, true);

	int setLOFrequency = LMS_SetLOFrequency(device, LMS_CH_TX, channel, (double)TX_FREQUENCY);
	if (setLOFrequency)
	{
		printf("ERROR: Failed to set TX frequency: %s\n", LMS_GetLastErrorMessage());
		goto out;
	}

	// Set sample rate
	lms_range_t sampleRateRange;
	int getSampleRateRange = LMS_GetSampleRateRange(device, LMS_CH_TX, &sampleRateRange);
	if (getSampleRateRange)
		printf("Warning: Failed to get sample rate range: %s\n", LMS_GetLastErrorMessage());

	int setSampleRate = LMS_SetSampleRate(device, (double)TX_SAMPLERATE, limeOversample); 

	if (setSampleRate)
	{
		printf("ERROR: Failed to set sample rate: %s\n", LMS_GetLastErrorMessage());
		goto out;
	}

	double actualHostSampleRate = 0.0;
	double actualRFSampleRate = 0.0;
	int getSampleRate = LMS_GetSampleRate(device, LMS_CH_TX, channel, &actualHostSampleRate, &actualRFSampleRate);
	if (getSampleRate)
		printf("Warnig: Failed to get sample rate: %s\n", LMS_GetLastErrorMessage());
	else
		printf("Sample rate: %.1lf Hz (Host) / %.1lf Hz (RF)" "\n", actualHostSampleRate, actualRFSampleRate);

	// Automatic calibration
	printf("Calibrating...\n");
	int calibrate = LMS_Calibrate(device, LMS_CH_TX, channel, (double)TX_BANDWIDTH, 0);
	if (calibrate)
		printf("Warning: Failed to calibrate device: %s\n", LMS_GetLastErrorMessage());

	// Setup TX stream
	printf("Setup TX stream...\n");
	s.tx.stream.channel = channel;
	s.tx.stream.fifoSize = 1024 * 1024;
	s.tx.stream.throughputVsLatency = 0.5;
	s.tx.stream.isTx = true;
	s.tx.stream.dataFmt = LMS_FMT_I12;
	int setupStream = LMS_SetupStream(device, &s.tx.stream);
	if (setupStream)
	{
		printf("ERROR: Failed to setup TX stream: %s\n", LMS_GetLastErrorMessage());
		goto out;
	}

	// Start TX stream
	LMS_StartStream(&s.tx.stream);

	// Start GPS task.
	s.status = start_gps_task(&s);
	if (s.status < 0) {
		fprintf(stderr, "Failed to start GPS task.\n");
		goto out;
	}
	else
		printf("Creating GPS task...\n");

	// Wait until GPS task is initialized
	pthread_mutex_lock(&(s.tx.lock));
	while (!s.gps.ready)
		pthread_cond_wait(&(s.gps.initialization_done), &(s.tx.lock));
	pthread_mutex_unlock(&(s.tx.lock));

	// Fillfull the FIFO.
	if (is_fifo_write_ready(&s))
		pthread_cond_signal(&(s.fifo_write_ready));

	// Start TX task
	s.status = start_tx_task(&s);
	if (s.status < 0) {
		fprintf(stderr, "Failed to start TX task.\n");
		goto out;
	}
	else
		printf("Creating TX task...\n");

	// Running...
#ifdef WIN32
	printf("Running...\n" "Press 'q' to abort.\n");
#else
	printf("Running...\n" "Press Ctrl+C to abort.\n");
#endif

	// Wainting for TX task to complete.
	pthread_join(s.tx.thread, NULL);
	printf("\nDone!\n");

out:
	// Disable TX module and shut down underlying TX stream.
	LMS_StopStream(&s.tx.stream);
	LMS_DestroyStream(device, &s.tx.stream);

	// Free up resources
	if (s.tx.buffer != NULL)
		free(s.tx.buffer);

	if (s.fifo != NULL)
		free(s.fifo);

	printf("Closing device...\n");
	LMS_EnableChannel(device, LMS_CH_TX, channel, false);
	LMS_Close(device);

	return(0);
}
