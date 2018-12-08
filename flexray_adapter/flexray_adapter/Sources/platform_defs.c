#include <FreeRTOS.h>
#include <task.h>
#include "platform_defs.h"

void sleep(uint32_t ms) {
	TickType_t xDelay = ms / portTICK_PERIOD_MS;
	vTaskDelay( xDelay );
}
