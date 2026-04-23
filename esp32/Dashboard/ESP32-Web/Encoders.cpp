#include "Encoders.h"
#include "Globals.h"

void IRAM_ATTR leftEncoderISR() { pulseLeft++; }
void IRAM_ATTR rightEncoderISR() { pulseRight++; }
