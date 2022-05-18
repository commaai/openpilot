#pragma once

#define STATLOG_GAUGE "g"
#define STATLOG_SAMPLE "sa"

void statlog_log(const char* metric_type, const char* metric, int value);
void statlog_log(const char* metric_type, const char* metric, float value);

#define statlog_gauge(metric, value) statlog_log(STATLOG_GAUGE, metric, value)
#define statlog_sample(metric, value) statlog_log(STATLOG_SAMPLE, metric, value)
